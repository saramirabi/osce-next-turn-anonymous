# Same context, different clinician (anonymous artifact)

Anonymous code and **actual model outputs** for double-blind conference review.

Do not put author names, affiliations, acknowledgements, emails, or local
usernames in this repository until the reviewing period ends.

## What is included

- `final_1500_TRULY_FROZEN.csv` — 1,500 frozen contexts (272 dialogues)
- `final_<Source>_1500_NEXT_TURN_V1.csv` — generated next clinician turns
  (Human target is in the frozen file; Claude is the completed 1,500-row set)
- Generation scripts (`generate_*_next_turn_v1.py`) and `NEXT_TURN_V1` configs
- Pragmatic coding (`analyze_candidate_turns_v2.py`, `analyze_candidate_turns_v3.py`)
- MiniLM alignment (`minilm_semantic_alignment.py`, tables in `out/`)
- Validation kit: `annotation/annotator1.xlsx`, `annotation/annotator2.xlsx`
  (150 double-coded items), plus `annotation/annotation_mastery_150.xlsx`
  (same 150 contexts with Human + 6 model turns)

Six models: GPT, Gemini, Claude, Qwen (`qwen/qwen3.6-35b-a3b`), MedGemma, MedLlama.
Total: 9,000 model + 1,500 human = 10,500 turns.

Consultations originate in the Fareez et al. OSCE dialogue corpus. This release
contains the experimental sample rows, not the full original transcript archive.

## Prompt (NEXT_TURN_V1)

System: simulated consultation; emit only the next clinician utterance.

User: `CONVERSATION:\n\n{context}`. Temperature 0 where supported.

## Reproduce reported scores (no API keys required)

```bash
pip install -r requirements.txt
python minilm_semantic_alignment.py --data_dir . --out_dir ./out --drop_deepseek
python compute_annotation_kappa.py
```

Encoder: frozen `sentence-transformers/all-MiniLM-L6-v2` (library 4.1.0).

## Re-run generation (optional)

Copy `.env.example` to `.env`. Cloud keys: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`,
`GOOGLE_API_KEY`. Local models: LM Studio at `http://127.0.0.1:1234/v1`.
Never commit `.env`.

## Double-blind checklist before `git push`

- [ ] No author names in README, comments, Excel filenames, or commit messages
- [ ] Git user.name / user.email not identifying (or use a throwaway)
- [ ] No API keys, no private IPs, no `*_raw.jsonl`
