"""Build the annotation mastery file: 150 coded items x Human + every model."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
A1_PATH = ROOT / "annotation" / "annotator1.xlsx"
A2_PATH = ROOT / "annotation" / "annotator2.xlsx"
FROZEN = ROOT / "final_1500_TRULY_FROZEN.csv"
OUT_XLSX = ROOT / "annotation" / "annotation_mastery_150.xlsx"
OUT_CSV = ROOT / "annotation" / "annotation_mastery_150_wide.csv"

Q1C = "Q1: number of information requests"
Q2C = "Q2: primary function"
Q3C = "Q3: uncertain? (Y/N)"
NOTES = "Notes (optional)"
TURN = "CLINICIAN TURN TO CODE"
PATIENT = "Latest patient turn"
CTX = "Full preceding context (reference only)"
POS = "Position in consultation"
PRECEDING = "Preceding conversation (final exchanges)"
KIT_COLS = [
    "Item ID",
    "Position in consultation",
    PRECEDING,
    PATIENT,
    TURN,
    Q1C,
    Q2C,
    Q3C,
    NOTES,
    CTX,
]

MODEL_FILES = {
    "Gemini": ROOT / "." / "final_Gemini_1500_NEXT_TURN_V1.csv",
    "Claude": ROOT / "final_Claude_1500_NEXT_TURN_V1.csv",
    "GPT": ROOT / "." / "final_GPT_1500_NEXT_TURN_V1.csv",
    "Qwen": ROOT / "final_Qwen_1500_NEXT_TURN_V1.csv",
    "MedGemma": ROOT / "." / "final_MedGemma_1500_NEXT_TURN_V1.csv",
    "MedLlama": ROOT / "." / "final_MedLlama_1500_NEXT_TURN_V1.csv",
}
SOURCE_ORDER = ["Human", "Gemini", "Claude", "GPT", "Qwen", "MedGemma", "MedLlama"]


def norm_text(s) -> str:
    if pd.isna(s):
        return ""
    t = str(s).replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"[ \t]+", " ", t).strip()
    t = re.sub(r"^(doctor|dr|clinician|physician)\s*:\s*", "", t, flags=re.I)
    return t


def load_kit(path: Path, suffix: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="Coding Sheet", header=2)
    df = df[df["Item ID"].astype(str).str.upper() != "EXAMPLE"].copy()
    df["item_id"] = df["Item ID"].astype(str).str.strip()
    keep = df[
        ["item_id", POS, PRECEDING, PATIENT, TURN, Q1C, Q2C, Q3C, NOTES, CTX]
    ].rename(
        columns={
            POS: "position_category",
            PRECEDING: "preceding",
            PATIENT: "latest_patient_turn",
            TURN: "coded_turn",
            Q1C: f"Q1_{suffix}",
            Q2C: f"Q2_{suffix}",
            Q3C: f"Q3_{suffix}",
            NOTES: f"Notes_{suffix}",
            CTX: "full_context",
        }
    )
    return keep


def load_model(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["context_id", "generated_doctor_turn", "api_request_status"])
    return df


def match_context_id(kit: pd.DataFrame, frozen: pd.DataFrame, models: dict[str, pd.DataFrame]) -> pd.Series:
    frozen = frozen.copy()
    frozen["ctx_n"] = frozen["context"].map(norm_text)
    frozen["pat_n"] = frozen["latest_patient_turn"].map(norm_text)
    frozen["hum_n"] = frozen["human_doctor_target"].map(norm_text)
    ctx_map: dict[str, list[str]] = {}
    for cid, c in zip(frozen["context_id"], frozen["ctx_n"]):
        ctx_map.setdefault(c, []).append(cid)

    turn_map: dict[str, list[tuple[str, str]]] = {}
    for cid, t in zip(frozen["context_id"], frozen["hum_n"]):
        if t:
            turn_map.setdefault(t, []).append(("Human", cid))
    for src, mdf in models.items():
        tmp = mdf.copy()
        tmp["tn"] = tmp["generated_doctor_turn"].map(norm_text)
        for cid, t, st in zip(tmp["context_id"], tmp["tn"], tmp["api_request_status"]):
            if t and str(st).lower() == "success":
                turn_map.setdefault(t, []).append((src, cid))

    ids = []
    how = []
    for _, row in kit.iterrows():
        ctx_n = norm_text(row["full_context"])
        turn_n = norm_text(row["coded_turn"])
        pat_n = norm_text(row["latest_patient_turn"])
        cid = None
        method = "unmatched"
        hits = ctx_map.get(ctx_n, [])
        if len(hits) == 1:
            cid, method = hits[0], "full_context"
        elif len(hits) > 1:
            sub = frozen[frozen["context_id"].isin(hits)]
            sub2 = sub[sub["pat_n"] == pat_n]
            if len(sub2) == 1:
                cid, method = sub2.iloc[0]["context_id"], "full_context+patient"
            else:
                sub3 = sub2[sub2["hum_n"] == turn_n] if len(sub2) else sub[sub["hum_n"] == turn_n]
                if len(sub3) == 1:
                    cid, method = sub3.iloc[0]["context_id"], "full_context+turn"
        if cid is None:
            th = turn_map.get(turn_n, [])
            uniq = list(dict.fromkeys(c for _, c in th))
            if len(uniq) == 1:
                cid, method = uniq[0], "coded_turn"
            elif len(uniq) > 1:
                sub = frozen[frozen["context_id"].isin(uniq)]
                sub2 = sub[(sub["pat_n"] == pat_n) & (sub["position_category"] == row["position_category"])]
                if len(sub2) == 1:
                    cid, method = sub2.iloc[0]["context_id"], "coded_turn+patient+position"
                else:
                    sub3 = sub[sub["pat_n"] == pat_n]
                    if len(sub3) == 1:
                        cid, method = sub3.iloc[0]["context_id"], "coded_turn+patient"
        ids.append(cid)
        how.append(method)
    kit = kit.copy()
    kit["context_id"] = ids
    kit["match_method"] = how
    return kit


def coded_source(turn: str, row: pd.Series) -> str:
    t = norm_text(turn)
    hits = []
    if t and t == norm_text(row.get("Human")):
        hits.append("Human")
    for src in SOURCE_ORDER[1:]:
        if t and t == norm_text(row.get(src)):
            hits.append(src)
    if not hits:
        return "unmatched"
    if len(hits) == 1:
        return hits[0]
    return "ambiguous:" + "+".join(hits)


def main() -> None:
    a1 = load_kit(A1_PATH, "A1")
    a2 = load_kit(A2_PATH, "A2")
    a2_codes = a2[["item_id", "Q1_A2", "Q2_A2", "Q3_A2", "Notes_A2"]]
    kit = a1.merge(a2_codes, on="item_id", how="left")
    assert len(kit) == 150, len(kit)

    frozen = pd.read_csv(FROZEN)
    models = {name: load_model(path) for name, path in MODEL_FILES.items() if path.exists()}
    missing = [n for n in MODEL_FILES if n not in models]
    if missing:
        raise FileNotFoundError(f"Missing model CSVs: {missing}")

    kit = match_context_id(kit, frozen, models)
    print("Match methods:")
    print(kit["match_method"].value_counts().to_string())
    print("unmatched", kit["context_id"].isna().sum())

    meta = frozen[
        [
            "context_id",
            "dialogue_id",
            "specialty",
            "target_turn_id",
            "position_category",
            "normalized_target_position",
            "human_doctor_target",
        ]
    ].rename(columns={"position_category": "position_frozen", "human_doctor_target": "Human"})
    wide = kit.merge(meta, on="context_id", how="left")
    for src, mdf in models.items():
        tmp = mdf.rename(columns={"generated_doctor_turn": src})[["context_id", src]]
        wide = wide.merge(tmp, on="context_id", how="left")

    wide["coded_turn_source"] = wide.apply(lambda r: coded_source(r["coded_turn"], r), axis=1)

    col_order = [
        "item_id",
        "context_id",
        "dialogue_id",
        "specialty",
        "target_turn_id",
        "position_category",
        "position_frozen",
        "normalized_target_position",
        "match_method",
        "coded_turn_source",
        "latest_patient_turn",
        "coded_turn",
        "preceding",
        "Q1_A1",
        "Q2_A1",
        "Q3_A1",
        "Notes_A1",
        "Q1_A2",
        "Q2_A2",
        "Q3_A2",
        "Notes_A2",
        *SOURCE_ORDER,
        "full_context",
    ]
    wide = wide[col_order].sort_values("item_id")

    long_rows = []
    for _, r in wide.iterrows():
        coded_src = str(r["coded_turn_source"])
        for src in SOURCE_ORDER:
            is_coded = src == coded_src or (
                coded_src.startswith("ambiguous:") and src in coded_src
            )
            long_rows.append(
                {
                    "Item ID": r["item_id"],
                    "Position in consultation": r["position_category"],
                    PRECEDING: r["preceding"],
                    PATIENT: r["latest_patient_turn"],
                    TURN: r[src],
                    Q1C: r["Q1_A1"] if is_coded else pd.NA,
                    Q2C: r["Q2_A1"] if is_coded else pd.NA,
                    Q3C: r["Q3_A1"] if is_coded else pd.NA,
                    NOTES: r["Notes_A1"] if is_coded else pd.NA,
                    CTX: r["full_context"],
                    "_source": src,
                }
            )
    long_raw = pd.DataFrame(long_rows)
    src_cat = pd.Categorical(long_raw["_source"], categories=SOURCE_ORDER, ordered=True)
    long_raw["_source"] = src_cat
    long_raw = long_raw.sort_values(["_source", "Item ID"]).reset_index(drop=True)
    long_df = long_raw[KIT_COLS].copy()

    n_per_source = (
        long_raw.groupby("_source", observed=True)
        .agg(n=(TURN, "size"), n_nonempty=(TURN, lambda s: int(s.astype(str).str.strip().ne("").sum())))
        .reset_index()
        .rename(columns={"_source": "source"})
    )

    OUT_LONG_CSV = ROOT / "annotation" / "annotation_mastery_long_150.csv"
    OUT_XLSX.parent.mkdir(exist_ok=True)
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
        long_df.to_excel(writer, sheet_name="long_150", index=False)
        for src in SOURCE_ORDER:
            long_raw.loc[long_raw["_source"] == src, KIT_COLS].to_excel(
                writer, sheet_name=src[:31], index=False
            )
    long_df.to_csv(OUT_LONG_CSV, index=False)
    print(f"Wrote {OUT_XLSX}")
    print(f"Wrote {OUT_LONG_CSV}")
    print("long shape", long_df.shape, "columns", list(long_df.columns))
    print(n_per_source.to_string(index=False))


if __name__ == "__main__":
    main()
