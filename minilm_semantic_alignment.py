"""
Semantic-alignment feature for "Same Context, Different Clinician".

Computes, per generated/human turn:
    cos_H = cos( embed(nu(turn)), embed(H_i) )   # vs clinician's ACTUAL next turn
    cos_p = cos( embed(nu(turn)), embed(p_i) )   # vs patient's last turn

Encoder : all-MiniLM-L6-v2 (sentence-transformers), frozen, score-only.
Inputs  : the seven final_<Model>_1500_NEXT_TURN_V1.csv files.
          Columns used: context_id, dialogue_id, generated_doctor_turn,
          human_doctor_target, latest_patient_turn.
Turns are normalised with nu() before scoring, matching the paper's described
normalisation (strip a leading role label; cut at a hallucinated multi-turn
"Patient:" marker; drop trailing ###/Explanation: scaffolding).

NOTE (kept honest): the Human source's cos_H is 1.0 by construction (the scored
turn IS H_i); it is set to NaN and excluded from reporting. cos_p is valid for
Human. cos_H / cos_p are embedding-derived and MUST NOT be added to the
source-recovery classifier, which the paper describes as using pragmatic
features "alone (no text embeddings)".

Run where huggingface.co is reachable:
    pip install "sentence-transformers>=3,<7" pandas numpy
    python minilm_semantic_alignment.py --data_dir /path/to/csvs --out_dir ./out
"""

import argparse
import glob
import os
import re
import numpy as np
import pandas as pd

FILE_TO_SOURCE = {
    "final_Claude_1500_NEXT_TURN_V1.csv":   "Claude",
    "final_DeepSeek_1500_NEXT_TURN_V1.csv": "DeepSeek",   # not in paper draft; drop if unused
    "final_Gemini_1500_NEXT_TURN_V1.csv":   "Gemini",
    "final_GPT_1500_NEXT_TURN_V1.csv":      "GPT",
    "final_MedGemma_1500_NEXT_TURN_V1.csv": "MedGemma",
    "final_MedLlama_1500_NEXT_TURN_V1.csv": "MedLlama",
    "final_Qwen_1500_NEXT_TURN_V1.csv":   "Qwen",
}
SOURCE_ORDER = ["Human", "Gemini", "Claude", "GPT", "Qwen",
                "MedGemma", "MedLlama", "DeepSeek"]

# --- nu(): normalisation matching the paper's description --------------------
ROLE_LABEL_RE = re.compile(r"^\s*(doctor|dr|clinician|physician)\s*:\s*", re.I)
PATIENT_MARKER_RE = re.compile(r"\n\s*patient\s*:", re.I)
SCAFFOLD_RE = re.compile(r"(\n\s*#{2,}.*$)|(\n\s*explanation\s*:.*$)", re.I | re.S)

def nu(text):
    if not isinstance(text, str):
        return None
    t = text.strip()
    m = PATIENT_MARKER_RE.search(t)          # (iii) multi-turn: keep first utterance
    if m:
        t = t[:m.start()]
    t = SCAFFOLD_RE.sub("", t)               # (ii) trailing scaffolding
    t = ROLE_LABEL_RE.sub("", t).strip()     # (i) leading role label
    return t if t else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=".")
    ap.add_argument("--out_dir", default="./out")
    ap.add_argument("--model", default="all-MiniLM-L6-v2")
    ap.add_argument("--drop_deepseek", action="store_true",
                    help="exclude the DeepSeek file (not in the paper draft)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    from sentence_transformers import SentenceTransformer
    import sentence_transformers

    # --- load -------------------------------------------------------------
    rows, human_added = [], False
    files = sorted(glob.glob(os.path.join(args.data_dir,
                                           "final_*_1500_NEXT_TURN_V1.csv")))
    if not files:
        raise SystemExit(f"No input CSVs found in {args.data_dir}")
    for f in files:
        name = os.path.basename(f)
        src = FILE_TO_SOURCE.get(name)
        if src is None:
            print(f"[skip] unrecognised file {name}")
            continue
        if src == "DeepSeek" and args.drop_deepseek:
            continue
        df = pd.read_csv(f)
        for _, r in df.iterrows():
            rows.append(dict(source=src,
                             context_id=r["context_id"],
                             consultation=r["dialogue_id"],
                             turn_raw=r["generated_doctor_turn"],
                             human_turn=r["human_doctor_target"],
                             patient_turn=r["latest_patient_turn"]))
        if not human_added:
            for _, r in df.iterrows():
                rows.append(dict(source="Human",
                                 context_id=r["context_id"],
                                 consultation=r["dialogue_id"],
                                 turn_raw=r["human_doctor_target"],
                                 human_turn=r["human_doctor_target"],
                                 patient_turn=r["latest_patient_turn"]))
            human_added = True

    data = pd.DataFrame(rows)
    data["turn_norm"] = data["turn_raw"].map(nu)
    data["human_norm"] = data["human_turn"].map(nu)
    data["patient_norm"] = data["patient_turn"].map(nu)

    # --- embed (each unique normalised string once) -----------------------
    model = SentenceTransformer(args.model)   # loads from HF; frozen, score-only
    texts = pd.unique(pd.concat(
        [data["turn_norm"], data["human_norm"], data["patient_norm"]]).dropna())
    emb = model.encode(list(texts), normalize_embeddings=True,
                       batch_size=256, show_progress_bar=True)
    vec = {t: e for t, e in zip(texts, emb)}

    def cos(a, b):
        if a is None or b is None or a not in vec or b not in vec:
            return np.nan
        return float(np.dot(vec[a], vec[b]))   # L2-normalised already

    data["cos_H"] = [cos(t, h) for t, h in zip(data["turn_norm"], data["human_norm"])]
    data["cos_p"] = [cos(t, p) for t, p in zip(data["turn_norm"], data["patient_norm"])]
    data.loc[data["source"] == "Human", "cos_H"] = np.nan   # 1.0 by construction

    # --- per-source summary ----------------------------------------------
    def summ(g):
        ch, cp = g["cos_H"].dropna(), g["cos_p"].dropna()
        return pd.Series({"n_scored": int(g["turn_norm"].notna().sum()),
                          "cos_H_mean": ch.mean(), "cos_H_sd": ch.std(),
                          "cos_p_mean": cp.mean(), "cos_p_sd": cp.std()})

    present = [s for s in SOURCE_ORDER if s in set(data["source"])]
    table = (data.groupby("source").apply(summ, include_groups=False)
                 .reindex(present).round(3))
    table["n_scored"] = table["n_scored"].astype(int)

    print(f"\nEncoder: {args.model} | sentence-transformers "
          f"{sentence_transformers.__version__}")
    print("cos_H excludes Human (1.0 by construction); turns normalised via nu()\n")
    print(table.to_string())

    table.to_csv(os.path.join(args.out_dir, "semantic_alignment_by_source.csv"))
    keep = ["source", "context_id", "consultation",
            "turn_norm", "cos_H", "cos_p"]
    data[keep].to_csv(os.path.join(args.out_dir,
                                   "semantic_alignment_per_turn.csv"), index=False)
    print(f"\nWrote:\n  {args.out_dir}/semantic_alignment_by_source.csv"
          f"\n  {args.out_dir}/semantic_alignment_per_turn.csv")


if __name__ == "__main__":
    main()
