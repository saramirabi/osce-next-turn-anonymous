#!/usr/bin/env python3
"""Descriptive characterization of position-matched next-doctor-turn candidates."""

from __future__ import annotations

import csv
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median, pstdev

ROOT = Path(__file__).resolve().parent
INPUT_CSV = ROOT / "all_candidate_turns.csv"
OUTPUT_CHARACTERISTICS = ROOT / "candidate_characteristics.csv"
OUTPUT_REPORT = ROOT / "candidate_analysis_report.md"
OUTPUT_EXAMPLES = ROOT / "candidate_examples_for_review.csv"

RANDOM_SEED = 42
EXAMPLE_TARGET_COUNT = 100

WORD_RE = re.compile(r"[A-Za-z0-9']+")

# ---------------------------------------------------------------------------
# Rule definitions (documented in report and applied transparently below)
# ---------------------------------------------------------------------------

WH_START_RE = re.compile(
    r"^\s*(what|when|where|why|who|which|how)\b", re.IGNORECASE
)
YES_NO_START_RE = re.compile(
    r"^\s*(do|does|did|are|is|was|were|have|has|had|can|could|will|would|"
    r"should|shall|may|might|any|have you|has there|is there|are there|"
    r"did you|do you|does it|can you|could you|would you|will you)\b",
    re.IGNORECASE,
)

ACKNOWLEDGEMENT_PHRASES = (
    r"ok(?:ay)?|alright|right|sure|yes|yeah|yep|mm(?:-?hmm)?|uh(?:-|\s)?huh|"
    r"i see|got it|understood|thank you|thanks|good|great|perfect|"
    r"sounds good|no problem|of course|mhm"
)
ACKNOWLEDGEMENT_ONLY_RE = re.compile(
    rf"^\s*(?:{ACKNOWLEDGEMENT_PHRASES})(?:[\s,.!?-]*(?:{ACKNOWLEDGEMENT_PHRASES}))*[\s.!?-]*$",
    re.IGNORECASE,
)

GREETING_RE = re.compile(
    r"\b(hi|hello|hey|good morning|good afternoon|good evening|nice to meet you|"
    r"welcome|my name is)\b",
    re.IGNORECASE,
)
CLOSING_RE = re.compile(
    r"\b(goodbye|bye|see you|take care|have a (?:good|nice) day|"
    r"that(?:'s| is) (?:all|everything)|we(?:'re| are) (?:done|finished)|"
    r"thank you for (?:coming|your time)|anything else(?:\?| before we)|"
    r"do you have any (?:other )?questions)\b",
    re.IGNORECASE,
)
EXAMINATION_RE = re.compile(
    r"\b(examine|examination|physical exam|let me (?:look|check|listen|feel|palpate|"
    r"have a look|take a look)|i(?:'ll| will) (?:look|check|listen|examine|palpate|"
    r"need to examine|do an exam)|listen to your|check your (?:blood pressure|vitals|"
    r"pulse|heart|lungs|abdomen|back|leg|skin|reflexes)|"
    r"going to (?:examine|check|look at|inspect|palpate)|"
    r"move on to the physical|start the physical|during the exam|"
    r"can you (?:lie down|sit up|stand up|remove|take off|lift|bend|move))\b",
    re.IGNORECASE,
)
DIAGNOSIS_RE = re.compile(
    r"\b(i think|it (?:sounds|looks|seems) like|likely|probably|possibly|"
    r"this (?:is|could be|may be|might be)|appears to be|diagnosis|"
    r"based on what you(?:'ve| have) told me|from what you(?:'ve| have) said|"
    r"suggest(?:s|ing)? (?:that )?you (?:have|may have|might have)|"
    r"consistent with|indicative of|concerned about|worried about)\b",
    re.IGNORECASE,
)
TREATMENT_RE = re.compile(
    r"\b(recommend|prescribe|prescription|medication|treatment|manage(?:ment)?|"
    r"you should (?:take|try|use|start|stop|avoid|rest|drink|eat)|"
    r"we(?:'ll| will) (?:start|give|prescribe|order|refer|arrange|book)|"
    r"follow up|follow-up|refer you|send you for|"
    r"take (?:this|these|it|them)|over the counter|"
    r"apply (?:this|a)|come back if|return if|go to (?:the )?(?:er|emergency))\b",
    re.IGNORECASE,
)
ADMINISTRATIVE_RE = re.compile(
    r"\b(appointment|schedule|paperwork|form|insurance|billing|registration|"
    r"wait here|waiting room|front desk|referral letter|fill out|sign this|"
    r"contact (?:the )?clinic|book (?:you )?(?:in|an appointment))\b",
    re.IGNORECASE,
)


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def split_sentences(text: str) -> list[str]:
    parts = [
        part.strip()
        for part in re.findall(r"[^.!?]+(?:[.!?]|$)", text.strip())
        if part.strip()
    ]
    if not parts and text.strip():
        return [text.strip()]
    return parts


def count_interrogative_sentences(text: str) -> int:
    count = 0
    for sentence in split_sentences(text):
        if "?" in sentence:
            count += 1
            continue
        core = sentence.rstrip(".!?").strip()
        if WH_START_RE.search(core) or YES_NO_START_RE.search(core):
            count += 1
    return count


def is_yes_no_question(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if YES_NO_START_RE.search(stripped):
        return True
    if re.search(r"\b(or not)\b.*\?$", stripped, re.IGNORECASE):
        return True
    if re.search(r",?\s*(?:right|correct|yes|no)\?\s*$", stripped, re.IGNORECASE):
        return True
    return False


def count_context_turns(context: str) -> tuple[int, int, int]:
    doctor_turns = 0
    patient_turns = 0
    for block in context.split("\n\n"):
        block = block.strip()
        if block.startswith("Doctor:"):
            doctor_turns += 1
        elif block.startswith("Patient:"):
            patient_turns += 1
    return doctor_turns, patient_turns, doctor_turns + patient_turns


def pct(part: int, whole: int) -> float:
    return (part / whole * 100) if whole else 0.0


def fmt_num(value: float) -> str:
    return f"{value:.2f}"


def load_candidates() -> list[dict[str, str]]:
    with INPUT_CSV.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def enrich_candidate(row: dict[str, str]) -> dict[str, object]:
    target = row["human_doctor_target"]
    context = row["context"]
    latest_patient = row["latest_patient_turn"]

    target_words = word_count(target)
    context_words = word_count(context)
    latest_patient_words = word_count(latest_patient)
    question_marks = target.count("?")
    sentences = split_sentences(target)
    sentence_count = len(sentences)

    preceding_doctor, preceding_patient, preceding_total = count_context_turns(context)
    completed_exchanges = preceding_patient

    normalized_position = float(row["target_position"])
    position_category = row["target_position_third"]

    flags = {
        "flag_question_containing": question_marks > 0,
        "flag_acknowledgement_only": bool(
            ACKNOWLEDGEMENT_ONLY_RE.match(target.strip())
            and question_marks == 0
        ),
        "flag_greeting_opening": bool(GREETING_RE.search(target)),
        "flag_closing_farewell": bool(CLOSING_RE.search(target)),
        "flag_examination_procedure": bool(EXAMINATION_RE.search(target)),
        "flag_diagnosis_explanation": bool(DIAGNOSIS_RE.search(target)),
        "flag_treatment_management": bool(TREATMENT_RE.search(target)),
        "flag_administrative": bool(ADMINISTRATIVE_RE.search(target)),
        "flag_very_short_le2": target_words <= 2,
        "flag_very_short_le3": target_words <= 3,
        "flag_very_short_le5": target_words <= 5,
    }

    enriched: dict[str, object] = {
        **row,
        "target_word_count": target_words,
        "context_word_count": context_words,
        "latest_patient_word_count": latest_patient_words,
        "context_turn_count": preceding_total,
        "preceding_doctor_turns": preceding_doctor,
        "preceding_patient_turns": preceding_patient,
        "preceding_total_turns": preceding_total,
        "completed_exchanges_before_target": completed_exchanges,
        "absolute_target_doctor_turn_index": int(row["doctor_turn_number"]),
        "normalized_target_position": normalized_position,
        "position_category": position_category,
        "contains_question_mark": question_marks > 0,
        "question_mark_count": question_marks,
        "sentence_count": sentence_count,
        "interrogative_sentence_count": count_interrogative_sentences(target),
        "starts_with_wh_word": bool(WH_START_RE.search(target)),
        "appears_yes_no_question": is_yes_no_question(target),
        "contains_multiple_questions": question_marks > 1,
        **flags,
    }
    return enriched


def candidates_per_dialogue_stats(rows: list[dict[str, object]]) -> dict[str, float]:
    per_dialogue = Counter(row["dialogue_id"] for row in rows)
    values = list(per_dialogue.values())
    return {
        "mean": mean(values),
        "median": median(values),
        "sd": pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def count_where(rows: list[dict[str, object]], predicate) -> int:
    return sum(1 for row in rows if predicate(row))


def combo_counts(rows: list[dict[str, object]]) -> dict[str, int]:
    def q(row):
        return row["flag_question_containing"]

    def ge2(row):
        return row["completed_exchanges_before_target"] >= 2

    def ge3(row):
        return row["completed_exchanges_before_target"] >= 3

    def not_ack(row):
        return not row["flag_acknowledgement_only"]

    def not_dte(row):
        return not (
            row["flag_diagnosis_explanation"]
            or row["flag_treatment_management"]
            or row["flag_examination_procedure"]
        )

    return {
        "question_and_ge2_exchanges": count_where(rows, lambda r: q(r) and ge2(r)),
        "question_and_ge3_exchanges": count_where(rows, lambda r: q(r) and ge3(r)),
        "question_and_not_ack_only": count_where(rows, lambda r: q(r) and not_ack(r)),
        "question_and_not_dte": count_where(rows, lambda r: q(r) and not_dte(r)),
        "question_and_ge2_and_not_dte": count_where(
            rows, lambda r: q(r) and ge2(r) and not_dte(r)
        ),
    }


def breakdown_table(
    rows: list[dict[str, object]], key_field: str, combo_names: list[str]
) -> list[str]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key_field])].append(row)

    lines = [
        f"| {key_field} | "
        + " | ".join(combo_names)
        + " |",
        "| --- | " + " | ".join(["---:"] * len(combo_names)) + " |",
    ]
    for key in sorted(grouped):
        combos = combo_counts(grouped[key])
        lines.append(
            f"| {key} | "
            + " | ".join(str(combos[name]) for name in combo_names)
            + " |"
        )
    return lines


def histogram_lines(values: list[float], bins: list[tuple[str, float, float]]) -> list[str]:
    lines = ["| Bin | Count | Share |", "| --- | ---: | ---: |"]
    total = len(values)
    for label, low, high in bins:
        count = sum(1 for value in values if low <= value < high or (high == float("inf") and value >= low))
        lines.append(f"| {label} | {count} | {pct(count, total):.2f}% |")
    return lines


def position_histogram(rows: list[dict[str, object]]) -> list[str]:
    bins = [
        ("0.00-0.10", 0.0, 0.10),
        ("0.10-0.20", 0.10, 0.20),
        ("0.20-0.30", 0.20, 0.30),
        ("0.30-0.40", 0.30, 0.40),
        ("0.40-0.50", 0.40, 0.50),
        ("0.50-0.60", 0.50, 0.60),
        ("0.60-0.70", 0.60, 0.70),
        ("0.70-0.80", 0.70, 0.80),
        ("0.80-0.90", 0.80, 0.90),
        ("0.90-1.01", 0.90, 1.01),
    ]
    values = [float(row["normalized_target_position"]) for row in rows]
    lines = ["| Position bin | Count | Share |", "| --- | ---: | ---: |"]
    total = len(values)
    for label, low, high in bins:
        if high == 1.01:
            count = sum(1 for value in values if low <= value <= 1.0)
        else:
            count = sum(1 for value in values if low <= value < high)
        lines.append(f"| {label} | {count} | {pct(count, total):.2f}% |")
    return lines


def stratified_examples(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rng = random.Random(RANDOM_SEED)
    buckets: dict[tuple[str, str, str, str], list[dict[str, object]]] = defaultdict(list)

    for row in rows:
        length_bucket = "short" if row["target_word_count"] <= 5 else "longer"
        question_bucket = "question" if row["flag_question_containing"] else "non_question"
        key = (
            str(row["specialty"]),
            str(row["position_category"]),
            question_bucket,
            length_bucket,
        )
        buckets[key].append(row)

    selected: list[dict[str, object]] = []
    selected_ids: set[str] = set()
    bucket_keys = sorted(buckets)
    per_bucket = max(1, EXAMPLE_TARGET_COUNT // max(1, len(bucket_keys)))

    for key in bucket_keys:
        pool = buckets[key][:]
        rng.shuffle(pool)
        for row in pool[:per_bucket]:
            if row["target_turn_id"] not in selected_ids:
                selected.append(row)
                selected_ids.add(str(row["target_turn_id"]))

    if len(selected) < EXAMPLE_TARGET_COUNT:
        remaining = [row for row in rows if row["target_turn_id"] not in selected_ids]
        rng.shuffle(remaining)
        for row in remaining:
            selected.append(row)
            selected_ids.add(str(row["target_turn_id"]))
            if len(selected) >= EXAMPLE_TARGET_COUNT:
                break

    return selected[:EXAMPLE_TARGET_COUNT]


def write_characteristics_csv(rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with OUTPUT_CHARACTERISTICS.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def example_fieldnames(rows: list[dict[str, object]]) -> list[str]:
    preferred = [
        "dialogue_id",
        "specialty",
        "target_turn_id",
        "normalized_target_position",
        "position_category",
        "context",
        "latest_patient_turn",
        "human_doctor_target",
        "target_word_count",
        "context_word_count",
        "context_turn_count",
        "preceding_doctor_turns",
        "preceding_patient_turns",
        "completed_exchanges_before_target",
        "contains_question_mark",
        "question_mark_count",
        "sentence_count",
        "interrogative_sentence_count",
        "starts_with_wh_word",
        "appears_yes_no_question",
        "contains_multiple_questions",
        "flag_question_containing",
        "flag_acknowledgement_only",
        "flag_greeting_opening",
        "flag_closing_farewell",
        "flag_examination_procedure",
        "flag_diagnosis_explanation",
        "flag_treatment_management",
        "flag_administrative",
        "flag_very_short_le2",
        "flag_very_short_le3",
        "flag_very_short_le5",
    ]
    available = set(rows[0].keys()) if rows else set()
    return [name for name in preferred if name in available]


def write_examples_csv(rows: list[dict[str, object]]) -> None:
    fieldnames = example_fieldnames(rows)
    with OUTPUT_EXAMPLES.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def write_report(rows: list[dict[str, object]], combos: dict[str, int]) -> None:
    total = len(rows)
    per_dialogue_stats = candidates_per_dialogue_stats(rows)
    specialty_counts = Counter(row["specialty"] for row in rows)
    position_counts = Counter(row["position_category"] for row in rows)
    combo_names = list(combos.keys())

    flag_fields = [
        ("flag_question_containing", "Question-containing (`?` present)"),
        ("flag_acknowledgement_only", "Acknowledgement-only"),
        ("flag_greeting_opening", "Greeting/opening"),
        ("flag_closing_farewell", "Closing/farewell"),
        ("flag_examination_procedure", "Examination/procedure instruction"),
        ("flag_diagnosis_explanation", "Diagnosis/explanation"),
        ("flag_treatment_management", "Treatment/management"),
        ("flag_administrative", "Administrative"),
        ("flag_very_short_le2", "Very short (<=2 words)"),
        ("flag_very_short_le3", "Very short (<=3 words)"),
        ("flag_very_short_le5", "Very short (<=5 words)"),
    ]

    context_words = [int(row["context_word_count"]) for row in rows]
    context_turns = [int(row["context_turn_count"]) for row in rows]
    target_words = [int(row["target_word_count"]) for row in rows]

    exchange_thresholds = [
        (">=1 completed exchange", lambda r: r["completed_exchanges_before_target"] >= 1),
        (">=2 completed exchanges", lambda r: r["completed_exchanges_before_target"] >= 2),
        (">=3 completed exchanges", lambda r: r["completed_exchanges_before_target"] >= 3),
        (">=4 completed exchanges", lambda r: r["completed_exchanges_before_target"] >= 4),
    ]

    lines = [
        "# Candidate Analysis Report",
        "",
        "Descriptive characterization of all position-matched next-doctor-turn candidates.",
        "No instances were removed, sampled for the final experiment, or classified with an LLM.",
        "",
        "## 1. Basic Corpus Statistics",
        "",
        f"- Total candidate instances: **{total:,}**",
        f"- Unique dialogues: **{len(set(row['dialogue_id'] for row in rows)):,}**",
        f"- Candidate turns per dialogue (mean): **{per_dialogue_stats['mean']:.2f}**",
        f"- Candidate turns per dialogue (median): **{per_dialogue_stats['median']:.1f}**",
        f"- Candidate turns per dialogue (SD): **{per_dialogue_stats['sd']:.2f}**",
        f"- Candidate turns per dialogue (min/max): **{per_dialogue_stats['min']} / {per_dialogue_stats['max']}**",
        "",
        "### Candidate Turns by Specialty",
        "",
        "| Specialty | Count | Percentage |",
        "| --- | ---: | ---: |",
    ]

    for specialty in sorted(specialty_counts):
        count = specialty_counts[specialty]
        lines.append(f"| {specialty} | {count:,} | {pct(count, total):.2f}% |")

    lines.extend(
        [
            "",
            "### Target Position Distribution Within Dialogues",
            "",
            *position_histogram(rows),
            "",
            "### Context Length",
            "",
            f"- Context words (mean / median / SD): **{mean(context_words):.2f} / {median(context_words):.1f} / {pstdev(context_words):.2f}**",
            f"- Context words (min / max): **{min(context_words)} / {max(context_words)}**",
            f"- Context turns (mean / median / SD): **{mean(context_turns):.2f} / {median(context_turns):.1f} / {pstdev(context_turns):.2f}**",
            f"- Context turns (min / max): **{min(context_turns)} / {max(context_turns)}**",
            "",
            "### Human Target Length",
            "",
            f"- Target words (mean / median / SD): **{mean(target_words):.2f} / {median(target_words):.1f} / {pstdev(target_words):.2f}**",
            f"- Target words (min / max): **{min(target_words)} / {max(target_words)}**",
            "",
            "## 2. Question Characteristics",
            "",
            "Automatic question features are heuristic only, not pragmatic ground truth.",
            "",
            f"- Contains question mark: **{count_where(rows, lambda r: r['contains_question_mark']):,}** ({pct(count_where(rows, lambda r: r['contains_question_mark']), total):.2f}%)",
            f"- Multiple question marks: **{count_where(rows, lambda r: r['contains_multiple_questions']):,}** ({pct(count_where(rows, lambda r: r['contains_multiple_questions']), total):.2f}%)",
            f"- Starts with WH-word: **{count_where(rows, lambda r: r['starts_with_wh_word']):,}** ({pct(count_where(rows, lambda r: r['starts_with_wh_word']), total):.2f}%)",
            f"- Appears yes/no by rule: **{count_where(rows, lambda r: r['appears_yes_no_question']):,}** ({pct(count_where(rows, lambda r: r['appears_yes_no_question']), total):.2f}%)",
            f"- Mean question marks per target: **{mean(float(r['question_mark_count']) for r in rows):.2f}**",
            f"- Mean sentences per target: **{mean(float(r['sentence_count']) for r in rows):.2f}**",
            f"- Mean interrogative sentences per target: **{mean(float(r['interrogative_sentence_count']) for r in rows):.2f}**",
            "",
            "## 3. Exploratory Target Flags",
            "",
            "These flags are descriptive only. No instances were excluded based on them.",
            "",
            "| Flag | Count | Percentage |",
            "| --- | ---: | ---: |",
        ]
    )

    for field, label in flag_fields:
        count = count_where(rows, lambda r, f=field: bool(r[f]))
        lines.append(f"| {label} | {count:,} | {pct(count, total):.2f}% |")

    lines.extend(
        [
            "",
            "### Flag Rule Documentation",
            "",
            "#### Question-containing",
            "- True when `human_doctor_target` contains at least one `?`.",
            "",
            "#### Acknowledgement-only",
            "- True when there is no `?` and the full target matches:",
            f"  - regex: `{ACKNOWLEDGEMENT_ONLY_RE.pattern}`",
            "- Intended for short acknowledgement/backchannel turns built from phrases such as `ok`, `alright`, `I see`, `thank you`, `got it`, `mhm`.",
            "",
            "#### Greeting/opening",
            "- True when target matches:",
            f"  - regex: `{GREETING_RE.pattern}`",
            "",
            "#### Closing/farewell",
            "- True when target matches:",
            f"  - regex: `{CLOSING_RE.pattern}`",
            "",
            "#### Examination/procedure instruction",
            "- True when target matches:",
            f"  - regex: `{EXAMINATION_RE.pattern}`",
            "",
            "#### Diagnosis/explanation",
            "- True when target matches:",
            f"  - regex: `{DIAGNOSIS_RE.pattern}`",
            "",
            "#### Treatment/management",
            "- True when target matches:",
            f"  - regex: `{TREATMENT_RE.pattern}`",
            "",
            "#### Administrative",
            "- True when target matches:",
            f"  - regex: `{ADMINISTRATIVE_RE.pattern}`",
            "",
            "#### Very short targets",
            "- `flag_very_short_le2`: target word count <= 2",
            "- `flag_very_short_le3`: target word count <= 3",
            "- `flag_very_short_le5`: target word count <= 5",
            "- Word count uses regex `[A-Za-z0-9']+`.",
            "",
            "#### WH-word start",
            f"- regex: `{WH_START_RE.pattern}`",
            "",
            "#### Yes/no question heuristic",
            "- True when target starts with auxiliary/modal/question opener regex:",
            f"  - `{YES_NO_START_RE.pattern}`",
            "- Or ends with tag-like patterns such as `, right?`, `, correct?`, `, yes?`, `, no?`, or contains `or not?`.",
            "",
            "#### Interrogative sentence count",
            "- A sentence is counted as interrogative if it contains `?`, starts with a WH-word, or starts with the yes/no opener regex above.",
            "- Sentences are split on `.`, `!`, and `?`.",
            "",
            "## 4. Dialogue Position",
            "",
            "- `absolute_target_doctor_turn_index` = doctor turn number within the dialogue.",
            "- `normalized_target_position` = candidate turn index / candidate turns in dialogue, in `[0, 1]`.",
            "- `position_category`: Early = first third, Middle = second third, Late = final third.",
            "",
            "### Position Category Counts",
            "",
            "| Category | Count | Percentage |",
            "| --- | ---: | ---: |",
        ]
    )

    for category in ("early", "middle", "late"):
        count = position_counts[category]
        lines.append(f"| {category} | {count:,} | {pct(count, total):.2f}% |")

    lines.extend(
        [
            "",
            "## 5. Context Depth",
            "",
            f"- Mean preceding doctor turns: **{mean(float(r['preceding_doctor_turns']) for r in rows):.2f}**",
            f"- Mean preceding patient turns: **{mean(float(r['preceding_patient_turns']) for r in rows):.2f}**",
            f"- Mean preceding total turns: **{mean(float(r['preceding_total_turns']) for r in rows):.2f}**",
            f"- Mean latest patient-turn word count: **{mean(float(r['latest_patient_word_count']) for r in rows):.2f}**",
            "",
            "Completed exchange count before target = number of preceding patient turns in context.",
            "",
            "| Threshold | Count | Percentage |",
            "| --- | ---: | ---: |",
        ]
    )

    for label, predicate in exchange_thresholds:
        count = count_where(rows, predicate)
        lines.append(f"| {label} | {count:,} | {pct(count, total):.2f}% |")

    lines.extend(
        [
            "",
            "## 6. Target-Type Combinations",
            "",
            "Counts under alternative possible inclusion rules. These are not final decisions.",
            "",
            "| Rule | Count | Percentage |",
            "| --- | ---: | ---: |",
        ]
    )

    for name, count in combos.items():
        pretty = name.replace("_", " ")
        lines.append(f"| {pretty} | {count:,} | {pct(count, total):.2f}% |")

    lines.extend(
        [
            "",
            "### By Specialty",
            "",
            *breakdown_table(rows, "specialty", combo_names),
            "",
            "### By Position Category",
            "",
            *breakdown_table(rows, "position_category", combo_names),
            "",
            "## 7. Random Examples for Manual Inspection",
            "",
            f"- Output file: `{OUTPUT_EXAMPLES.name}`",
            f"- Random seed: **{RANDOM_SEED}**",
            f"- Target sample size: **{EXAMPLE_TARGET_COUNT}**",
            "- Stratification buckets: specialty x Early/Middle/Late x question/non-question x short(<=5 words)/longer.",
            "- Examples were selected automatically; no manual classification was applied.",
            "",
            "## 8. Output Files",
            "",
            f"- `{OUTPUT_CHARACTERISTICS.name}`",
            f"- `{OUTPUT_REPORT.name}`",
            f"- `{OUTPUT_EXAMPLES.name}`",
            f"- `{Path(__file__).name}`",
            f"- Source input (unchanged): `{INPUT_CSV.name}`",
        ]
    )

    OUTPUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    raw_rows = load_candidates()
    enriched_rows = [enrich_candidate(row) for row in raw_rows]
    combos = combo_counts(enriched_rows)
    examples = stratified_examples(enriched_rows)

    write_characteristics_csv(enriched_rows)
    write_examples_csv(examples)
    write_report(enriched_rows, combos)

    print(f"Wrote {len(enriched_rows):,} rows to {OUTPUT_CHARACTERISTICS}")
    print(f"Wrote report to {OUTPUT_REPORT}")
    print(f"Wrote {len(examples)} examples to {OUTPUT_EXAMPLES}")


if __name__ == "__main__":
    main()
