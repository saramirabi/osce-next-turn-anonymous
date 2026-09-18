"""Cohen's kappa for OSCE validation annotation kit (150 items)."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from analyze_candidate_turns_v3 import classify_primary_turn_function  # noqa: E402
from analyze_candidate_turns_v2 import classify_information_seeking  # noqa: E402

A1_PATH = ROOT / "annotation" / "annotator1.xlsx"
A2_PATH = ROOT / "annotation" / "annotator2.xlsx"
FROZEN = ROOT / "final_1500_TRULY_FROZEN.csv"
CAND = ROOT / "candidate_characteristics_v3.csv"

Q1C = "Q1: number of information requests"
Q2C = "Q2: primary function"
Q3C = "Q3: uncertain? (Y/N)"
TURN = "CLINICIAN TURN TO CODE"

FN_TO_Q2 = {
    "history_information_seeking": "HIST",
    "acknowledgement_only": "ACK",
    "closing_farewell": "CLOSE",
    "examination_procedure": "EXAM",
    "diagnosis_explanation": "INTERP",
    "treatment_management": "MGMT",
}


def load_sheet(path: Path, name: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="Coding Sheet", header=2)
    df = df[df["Item ID"].astype(str).str.upper() != "EXAMPLE"].copy()
    df["annotator"] = name
    df["item"] = df["Item ID"].astype(str).str.strip()

    def norm_q1(x):
        if pd.isna(x):
            return np.nan
        s = str(x).strip()
        s = re.sub(r"\.0$", "", s)
        if s in {"5+", "5 plus", ">=5"}:
            return "5+"
        return s

    df["Q1"] = df[Q1C].map(norm_q1)
    df["Q2"] = df[Q2C].astype(str).str.strip().str.upper()
    df["Q2"] = df["Q2"].replace({"NAN": np.nan, "NONE": np.nan})
    df["Q3"] = df[Q3C].astype(str).str.strip().str.upper().str[0]
    df["Q3"] = df["Q3"].where(df["Q3"].isin(["Y", "N"]))
    df["turn"] = df[TURN].astype(str)
    df["turn_norm"] = df["turn"].map(norm_text)
    return df


def norm_text(s: str) -> str:
    s = str(s).replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()


def q1_ord(s):
    if pd.isna(s):
        return np.nan
    if str(s) == "5+":
        return 5.0
    try:
        return float(str(s))
    except ValueError:
        return np.nan


def report(title: str, x: pd.Series, y: pd.Series, weights=None) -> None:
    mask = x.notna() & y.notna()
    xx = x[mask].astype(str)
    yy = y[mask].astype(str)
    n = int(mask.sum())
    if n == 0:
        print(f"{title}: n=0")
        return
    agree = float((xx == yy).mean())
    k = cohen_kappa_score(xx, yy, weights=weights)
    extra = f"  weights={weights}" if weights else ""
    print(f"{title}: n={n}  %agree={agree:.3f}  kappa={k:.3f}{extra}")


def auto_q2(text: str) -> str:
    fn, _, _ = classify_primary_turn_function(text)
    return FN_TO_Q2.get(fn, "OTHER")


def auto_seeking(text: str) -> str:
    hit, _ = classify_information_seeking(text)
    return "Y" if hit else "N"


def main() -> None:
    a1 = load_sheet(A1_PATH, "A1")
    a2 = load_sheet(A2_PATH, "A2_annotated")
    m = a1.merge(a2, on="item", suffixes=("_1", "_2"))
    print("Annotator 1 = annotator1.xlsx")
    print("Annotator 2 = annotator2.xlsx")
    print(f"Items: A1={len(a1)} A2={len(a2)} paired={len(m)}")
    print()
    print("=== Inter-annotator agreement (A1 vs A2) ===")
    report("Q1 (nominal)", m["Q1_1"], m["Q1_2"])
    o1 = m["Q1_1"].map(q1_ord)
    o2 = m["Q1_2"].map(q1_ord)
    mask = o1.notna() & o2.notna()
    if mask.any():
        print(
            f"Q1 (linear weighted): n={int(mask.sum())}  kappa="
            f"{cohen_kappa_score(o1[mask], o2[mask], weights='linear'):.3f}"
        )
        print(
            f"Q1 (quadratic weighted): n={int(mask.sum())}  kappa="
            f"{cohen_kappa_score(o1[mask], o2[mask], weights='quadratic'):.3f}"
        )
    report("Q2 primary function", m["Q2_1"], m["Q2_2"])
    report("Q3 uncertain Y/N", m["Q3_1"], m["Q3_2"])

    labs = sorted(set(m["Q2_1"].dropna()) | set(m["Q2_2"].dropna()))
    cm = pd.DataFrame(
        confusion_matrix(m["Q2_1"].fillna("NA"), m["Q2_2"].fillna("NA"), labels=labs),
        index=labs,
        columns=labs,
    )
    print("\nQ2 confusion (rows=A1, cols=A2):")
    print(cm.to_string())

    # Automatic labels on the blinded turn (same v3 function labels)
    m["auto_q2"] = m["turn_1"].map(auto_q2)
    m["auto_seek"] = m["turn_1"].map(auto_seeking)
    m["a1_seek"] = m["Q1_1"].map(q1_ord).gt(0).map({True: "Y", False: "N"})
    m["a2_seek"] = m["Q1_2"].map(q1_ord).gt(0).map({True: "Y", False: "N"})
    m.loc[m["Q1_1"].map(q1_ord).isna(), "a1_seek"] = np.nan
    m.loc[m["Q1_2"].map(q1_ord).isna(), "a2_seek"] = np.nan

    print("\n=== Annotators vs automatic codebook (v3 primary function on coded turn) ===")
    report("A1 vs auto Q2", m["Q2_1"], m["auto_q2"])
    report("A2 vs auto Q2", m["Q2_2"], m["auto_q2"])
    both_agree = m["Q2_1"] == m["Q2_2"]
    print(f"Items where A1=A2: {int(both_agree.sum())}")
    report("Consensus (A1=A2) vs auto Q2", m.loc[both_agree, "Q2_1"], m.loc[both_agree, "auto_q2"])

    print("\n=== Information-seeking (Q1>0 vs v2 binary seek flag) ===")
    report("A1 vs auto seek", m["a1_seek"], m["auto_seek"])
    report("A2 vs auto seek", m["a2_seek"], m["auto_seek"])

    # Human vs model: match turn to frozen human target and generated CSVs
    frozen = pd.read_csv(FROZEN)
    frozen["tn"] = frozen["human_doctor_target"].map(norm_text)
    human_set = set(frozen["tn"].dropna())

    model_files = {
        "Claude": ROOT / "final_Claude_1500_NEXT_TURN_V1.csv",
        "Claude_recovered": ROOT / "final_Claude_1500_NEXT_TURN_V1.csv",
        "Gemini": ROOT / "." / "final_Gemini_1500_NEXT_TURN_V1.csv",
        "GPT": ROOT / "." / "final_GPT_1500_NEXT_TURN_V1.csv",
        "MedGemma": ROOT / "." / "final_MedGemma_1500_NEXT_TURN_V1.csv",
        "MedLlama": ROOT / "." / "final_MedLlama_1500_NEXT_TURN_V1.csv",
        "Qwen36": ROOT / "final_Qwen_1500_NEXT_TURN_V1.csv",
        "Qwen38": ROOT / "final_Qwen_1500_NEXT_TURN_V1.csv",
        "DeepSeek": ROOT / "final_DeepSeek_1500_NEXT_TURN_V1.csv",
    }
    gen_sets: dict[str, set[str]] = {}
    for name, p in model_files.items():
        if not p.exists():
            continue
        df = pd.read_csv(p)
        col = "generated_doctor_turn" if "generated_doctor_turn" in df.columns else None
        if col is None:
            continue
        gen_sets[name] = set(df[col].dropna().map(norm_text))

    def source_row(tn: str) -> str:
        hits = [k for k, s in gen_sets.items() if tn in s]
        is_h = tn in human_set
        if is_h and hits:
            return "both_human_and_model"
        if is_h:
            return "human"
        if hits:
            return "model:" + "+".join(sorted(hits))
        return "unmatched"

    m["source"] = m["turn_norm_1"].map(source_row)
    print("\n=== Source of coded turn (match to frozen human / model CSVs) ===")
    print(m["source"].value_counts().to_string())

    m["src_bin"] = np.where(
        m["source"].eq("human"),
        "human",
        np.where(m["source"].str.startswith("model:"), "model", m["source"]),
    )
    print("\n--- Kappa A1 vs A2 by source ---")
    for src, g in m.groupby("src_bin"):
        print(f"\n[{src}] n={len(g)}")
        report("  Q2", g["Q2_1"], g["Q2_2"])
        report("  Q1 nominal", g["Q1_1"], g["Q1_2"])

    print("\n--- Kappa vs auto Q2 by source ---")
    for src, g in m.groupby("src_bin"):
        print(f"\n[{src}] n={len(g)}")
        report("  A1 vs auto", g["Q2_1"], g["auto_q2"])
        report("  A2 vs auto", g["Q2_2"], g["auto_q2"])

    # Also match candidate_characteristics_v3 if turn texts exist there
    if CAND.exists():
        cand = pd.read_csv(CAND)
        tcol = None
        for c in cand.columns:
            if "turn" in c.lower() and "generated" not in c.lower():
                tcol = c
                break
        print("\ncandidate_characteristics_v3 columns:", list(cand.columns)[:20])
        if "primary_turn_function" in cand.columns:
            # try doctor_turn / candidate_turn
            for c in cand.columns:
                if cand[c].dtype == object:
                    cmap = dict(zip(cand[c].map(norm_text), cand["primary_turn_function"]))
                    mapped = m["turn_norm_1"].map(cmap)
                    n = mapped.notna().sum()
                    if n > 20:
                        print(f"Matched {n} items via column {c} to v3 primary_turn_function")
                        m["v3_fn"] = mapped.map(lambda x: FN_TO_Q2.get(x, "OTHER") if pd.notna(x) else np.nan)
                        report("A1 vs v3 CSV Q2", m["Q2_1"], m["v3_fn"])
                        report("A2 vs v3 CSV Q2", m["Q2_2"], m["v3_fn"])
                        break

    out = ROOT / "out" / "annotation_kappa_pairs.csv"
    out.parent.mkdir(exist_ok=True)
    m.to_csv(out, index=False)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
