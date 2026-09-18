#!/usr/bin/env python3
"""Pre-generation QC: v2 sampling and targeted eligibility audit."""

from __future__ import annotations

import csv
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median, pstdev

import analyze_candidate_turns_v3 as v3

ROOT = Path(__file__).resolve().parent
INPUT_CHARACTERISTICS = ROOT / "candidate_characteristics_v3.csv"
OUTPUT_SAMPLE = ROOT / "final_1500_contexts_v2.csv"
OUTPUT_REPORT = ROOT / "final_sampling_report_v2.md"
OUTPUT_SAMPLING_AUDIT = ROOT / "final_sampling_audit_v2.csv"
OUTPUT_ELIGIBILITY_AUDIT = ROOT / "final_eligibility_audit_for_review.csv"

RANDOM_SEED = 42
TARGET_SIZE = 1500
DIALOGUE_COUNT = 272
BASE_TARGETS_PER_DIALOGUE = 5
PREFERRED_ALLOCATION = {"early": 2, "middle": 2, "late": 1}
EXTRA_LATE_ALLOCATION = 2

NON_HISTORY_SENTENCE_FUNCTIONS = {
    "diagnosis_explanation",
    "treatment_management",
    "examination_procedure",
    "conversational_transition",
    "closing_farewell",
}

SAMPLE_COLUMNS = [
    "context_id",
    "dialogue_id",
    "specialty",
    "target_turn_id",
    "context",
    "latest_patient_turn",
    "human_doctor_target",
    "completed_exchanges_before_target",
    "normalized_target_position",
    "position_category",
    "primary_turn_function",
    "target_word_count",
    "context_word_count",
]

SAMPLING_AUDIT_COLUMNS = [
    "context_id",
    "dialogue_id",
    "target_turn_id",
    "position_category",
    "selection_stratum",
    "selection_reason",
    "normalized_target_position",
    "completed_exchanges_before_target",
    "extra_late_dialogue",
]

ELIGIBILITY_AUDIT_COLUMNS = [
    "context_id",
    "dialogue_id",
    "specialty",
    "target_turn_id",
    "normalized_target_position",
    "position_category",
    "primary_turn_function",
    "primary_function_reason",
    "sentence_level_classifications",
    "audit_flag_reasons",
    "context",
    "latest_patient_turn",
    "human_doctor_target",
    "target_word_count",
    "sentence_count",
    "contains_question_mark",
    "flag_question_containing",
    "flag_information_seeking",
    "flag_acknowledgement_only",
    "flag_greeting_opening",
    "flag_closing_farewell",
    "flag_conversational_transition",
    "flag_examination_procedure",
    "flag_diagnosis_explanation",
    "flag_treatment_management",
    "flag_administrative",
]

FLAG_FIELDS = [
    "contains_question_mark",
    "flag_question_containing",
    "flag_information_seeking",
    "flag_acknowledgement_only",
    "flag_greeting_opening",
    "flag_closing_farewell",
    "flag_conversational_transition",
    "flag_examination_procedure",
    "flag_diagnosis_explanation",
    "flag_treatment_management",
    "flag_administrative",
]


def pct(part: int, whole: int) -> float:
    return (part / whole * 100) if whole else 0.0


def load_characteristics() -> list[dict[str, str]]:
    with INPUT_CHARACTERISTICS.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def eligible_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if row["primary_turn_function"] == "history_information_seeking"
        and int(row["completed_exchanges_before_target"]) >= 2
    ]


def choose_extra_late_dialogues(
    by_dialogue: dict[str, dict[str, list[dict[str, str]]]], extra_needed: int, rng: random.Random
) -> set[str]:
    candidates = [
        dialogue_id
        for dialogue_id, strata in sorted(by_dialogue.items())
        if len(strata["late"]) >= EXTRA_LATE_ALLOCATION
    ]
    if len(candidates) < extra_needed:
        raise ValueError(
            f"Only {len(candidates)} dialogues have >= {EXTRA_LATE_ALLOCATION} eligible late targets."
        )
    return set(rng.sample(candidates, extra_needed))


def select_primary_sample(
    eligible: list[dict[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[str]]:
    rng = random.Random(RANDOM_SEED)
    by_dialogue: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in eligible:
        by_dialogue[row["dialogue_id"]][row["position_category"]].append(row)

    dialogue_ids = sorted(by_dialogue)
    extra_needed = TARGET_SIZE - (DIALOGUE_COUNT * BASE_TARGETS_PER_DIALOGUE)
    extra_dialogues = choose_extra_late_dialogues(by_dialogue, extra_needed, rng)
    extra_dialogue_list = sorted(extra_dialogues)

    selected: list[dict[str, str]] = []
    audit_rows: list[dict[str, str]] = []
    selected_ids: set[str] = set()

    for dialogue_id in dialogue_ids:
        allocation = dict(PREFERRED_ALLOCATION)
        is_extra_late = dialogue_id in extra_dialogues
        if is_extra_late:
            allocation["late"] = EXTRA_LATE_ALLOCATION

        for position_category in ("early", "middle", "late"):
            quota = allocation[position_category]
            pool = by_dialogue[dialogue_id][position_category][:]
            rng.shuffle(pool)
            if len(pool) < quota:
                raise ValueError(
                    f"Dialogue {dialogue_id} lacks {quota} eligible {position_category} targets."
                )
            for row in pool[:quota]:
                if row["target_turn_id"] in selected_ids:
                    raise ValueError(f"Duplicate target selected: {row['target_turn_id']}")
                selected_ids.add(row["target_turn_id"])
                context_id = f"CTX_{len(selected):04d}"
                enriched = dict(row)
                enriched["context_id"] = context_id
                selected.append(enriched)
                audit_rows.append(
                    {
                        "context_id": context_id,
                        "dialogue_id": dialogue_id,
                        "target_turn_id": row["target_turn_id"],
                        "position_category": position_category,
                        "selection_stratum": f"{dialogue_id}:{position_category}",
                        "selection_reason": (
                            f"primary_allocation {quota} {position_category} "
                            f"(dialogue_total={sum(allocation.values())}; "
                            f"extra_late={'yes' if is_extra_late else 'no'})"
                        ),
                        "normalized_target_position": row["normalized_target_position"],
                        "completed_exchanges_before_target": row[
                            "completed_exchanges_before_target"
                        ],
                        "extra_late_dialogue": "yes" if is_extra_late else "no",
                    }
                )

    if len(selected) != TARGET_SIZE:
        raise ValueError(f"Expected {TARGET_SIZE} selections, got {len(selected)}.")

    return selected, audit_rows, extra_dialogue_list


def sentence_pattern_flags(target: str) -> tuple[list[str], list[tuple[str, str]]]:
    reasons: list[str] = []
    sentences = v3.v2.split_sentences(target)
    sentence_classes: list[tuple[str, str]] = []

    for index, sentence in enumerate(sentences):
        function, reason = v3.classify_sentence(sentence, index)
        sentence_classes.append((function, reason))

    normalized = v3.v2.normalize_for_rules(target)

    if v3.DIAGNOSIS_STATEMENT_RE.search(normalized) or v3.v2.DIAGNOSIS_RE.search(normalized):
        reasons.append("diagnosis_explanation_phrase")

    if v3.TREATMENT_DIRECTIVE_RE.search(normalized) or (
        v3.v2.TREATMENT_RE.search(normalized)
        and not v3.HISTORY_MEDICATION_RE.search(normalized)
    ):
        reasons.append("treatment_management_phrase")

    if v3.EXAMINATION_SENTENCE_RE.search(normalized) or v3.v2.EXAMINATION_RE.search(normalized):
        reasons.append("examination_procedure_phrase")

    if v3.CLOSING_STATEMENT_RE.search(normalized) or v3.v2.CLOSING_RE.search(normalized):
        reasons.append("closing_or_any_questions_phrase")
    for sentence in sentences:
        sent_norm = v3.v2.normalize_for_rules(sentence)
        if v3.INCIDENTAL_CLOSING_Q_RE.match(sent_norm):
            reasons.append("closing_or_any_questions_phrase")
            break

    if v3.v2.TRANSITION_RE.search(normalized):
        reasons.append("conversational_transition_phrase")

    non_history_sentences = [
        function
        for function, _ in sentence_classes
        if function in NON_HISTORY_SENTENCE_FUNCTIONS
    ]
    if len(sentences) > 1 and non_history_sentences:
        reasons.append(
            "multi_sentence_with_non_history_pattern:"
            + ",".join(sorted(set(non_history_sentences)))
        )

    return sorted(set(reasons)), sentence_classes


def audit_flags_for_row(row: dict[str, str]) -> list[str]:
    reasons: list[str] = []
    target = row["human_doctor_target"]
    target_words = int(row.get("target_word_count") or v3.v2.word_count(target))

    if target_words > 30:
        reasons.append("target_length_gt_30_words")

    pattern_reasons, _ = sentence_pattern_flags(target)
    reasons.extend(pattern_reasons)
    return sorted(set(reasons))


def build_eligibility_audit(
    selected: list[dict[str, str]],
    characteristics_by_target: dict[str, dict[str, str]],
) -> list[dict[str, str]]:
    audit_rows: list[dict[str, str]] = []

    for row in selected:
        flags = audit_flags_for_row(row)
        if not flags:
            continue

        source = characteristics_by_target.get(row["target_turn_id"], {})
        sentences = v3.v2.split_sentences(row["human_doctor_target"])
        audit_rows.append(
            {
                "context_id": row["context_id"],
                "dialogue_id": row["dialogue_id"],
                "specialty": row["specialty"],
                "target_turn_id": row["target_turn_id"],
                "normalized_target_position": row["normalized_target_position"],
                "position_category": row["position_category"],
                "primary_turn_function": row["primary_turn_function"],
                "primary_function_reason": source.get("primary_function_reason", ""),
                "sentence_level_classifications": source.get(
                    "sentence_level_classifications", ""
                ),
                "audit_flag_reasons": "|".join(flags),
                "context": row["context"],
                "latest_patient_turn": row["latest_patient_turn"],
                "human_doctor_target": row["human_doctor_target"],
                "target_word_count": row["target_word_count"],
                "sentence_count": str(len(sentences)),
                **{field: source.get(field, row.get(field, "")) for field in FLAG_FIELDS},
            }
        )

    return audit_rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def write_report(
    eligible: list[dict[str, str]],
    selected: list[dict[str, str]],
    extra_dialogues: list[str],
    eligibility_audit_count: int,
) -> None:
    per_dialogue = Counter(row["dialogue_id"] for row in selected)
    per_dialogue_values = list(per_dialogue.values())
    position_counts = Counter(row["position_category"] for row in selected)
    specialty_counts = Counter(row["specialty"] for row in selected)
    context_words = [int(row["context_word_count"]) for row in selected]
    target_words = [int(row["target_word_count"]) for row in selected]
    exchanges = [int(row["completed_exchanges_before_target"]) for row in selected]
    extra_needed = TARGET_SIZE - (DIALOGUE_COUNT * BASE_TARGETS_PER_DIALOGUE)

    lines = [
        "# Final Sampling Report v2",
        "",
        "Frozen experimental sample reconstructed with random extra-late dialogue allocation.",
        "V3 classification rules were not modified.",
        "",
        "## Eligibility Criteria",
        "",
        "1. `primary_turn_function == history_information_seeking`",
        "2. `completed_exchanges_before_target >= 2`",
        "",
        "## Pool and Sample Size",
        "",
        f"- Eligible pool before sampling: **{len(eligible):,}**",
        f"- Final frozen sample size: **{len(selected):,}**",
        f"- Unique source dialogues represented: **{len(per_dialogue):,}**",
        "",
        "## Targets per Dialogue",
        "",
        f"- Mean: **{mean(per_dialogue_values):.2f}**",
        f"- Median: **{median(per_dialogue_values):.1f}**",
        f"- SD: **{pstdev(per_dialogue_values):.2f}**",
        f"- Min / max: **{min(per_dialogue_values)} / {max(per_dialogue_values)}**",
        "",
        "## Allocation Design",
        "",
        f"- Base allocation for every dialogue: **2 Early + 2 Middle + 1 Late = {BASE_TARGETS_PER_DIALOGUE}**",
        f"- Additional Late target added to **{extra_needed}** dialogues randomly selected among "
        f"dialogues with at least **{EXTRA_LATE_ALLOCATION}** eligible Late targets.",
        f"- Random seed for extra-late dialogue selection and within-stratum sampling: **{RANDOM_SEED}**",
        "",
        "## Position Category Distribution",
        "",
        "| Category | Count | Percentage |",
        "| --- | ---: | ---: |",
    ]

    for category in ("early", "middle", "late"):
        count = position_counts[category]
        lines.append(f"| {category} | {count:,} | {pct(count, len(selected)):.2f}% |")

    lines.extend(
        [
            "",
            "## Specialty Distribution",
            "",
            "| Specialty | Count | Percentage |",
            "| --- | ---: | ---: |",
        ]
    )
    for specialty in sorted(specialty_counts):
        count = specialty_counts[specialty]
        lines.append(f"| {specialty} | {count:,} | {pct(count, len(selected)):.2f}% |")

    lines.extend(
        [
            "",
            "## Context and Target Length",
            "",
            f"- Context words mean / median / SD: **{mean(context_words):.2f} / {median(context_words):.1f} / {pstdev(context_words):.2f}**",
            f"- Target words mean / median / SD: **{mean(target_words):.2f} / {median(target_words):.1f} / {pstdev(target_words):.2f}**",
            "",
            "## Completed Exchange Distribution",
            "",
            f"- Mean completed exchanges before target: **{mean(exchanges):.2f}**",
            "",
            "## Sampling Algorithm",
            "",
            "1. Filter eligible rows from `candidate_characteristics_v3.csv`.",
            "2. Group by `dialogue_id` and `position_category`.",
            "3. Randomly choose 140 dialogues with at least 2 eligible Late targets for the extra Late slot.",
            "4. For each dialogue, allocate 2 Early, 2 Middle, and 1 or 2 Late targets.",
            "5. Randomly sample within each dialogue-position stratum without replacement using seed 42.",
            "6. Assign `context_id` values in selection order.",
            "",
            "## Pre-Generation Eligibility Audit",
            "",
            f"- Flagged targets for manual review: **{eligibility_audit_count:,}**",
            f"- Output file: `{OUTPUT_ELIGIBILITY_AUDIT.name}`",
            "- Flagged rows were **not** automatically removed.",
            "",
            "## Extra-Late Dialogues (Random Selection)",
            "",
            ", ".join(extra_dialogues),
            "",
            "## Output Files",
            "",
            f"- `{OUTPUT_SAMPLE.name}`",
            f"- `{OUTPUT_REPORT.name}`",
            f"- `{OUTPUT_SAMPLING_AUDIT.name}`",
            f"- `{OUTPUT_ELIGIBILITY_AUDIT.name}`",
            f"- `{Path(__file__).name}`",
        ]
    )

    OUTPUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    characteristics = load_characteristics()
    characteristics_by_target = {
        row["target_turn_id"]: row for row in characteristics
    }
    eligible = eligible_rows(characteristics)
    selected, sampling_audit, extra_dialogues = select_primary_sample(eligible)
    eligibility_audit = build_eligibility_audit(selected, characteristics_by_target)

    write_csv(OUTPUT_SAMPLE, SAMPLE_COLUMNS, selected)
    write_csv(OUTPUT_SAMPLING_AUDIT, SAMPLING_AUDIT_COLUMNS, sampling_audit)
    write_csv(OUTPUT_ELIGIBILITY_AUDIT, ELIGIBILITY_AUDIT_COLUMNS, eligibility_audit)
    write_report(eligible, selected, extra_dialogues, len(eligibility_audit))

    print(f"Eligible pool: {len(eligible):,}")
    print(f"Wrote {len(selected):,} contexts to {OUTPUT_SAMPLE}")
    print(f"Wrote sampling audit to {OUTPUT_SAMPLING_AUDIT}")
    print(f"Wrote {len(eligibility_audit)} eligibility audit rows to {OUTPUT_ELIGIBILITY_AUDIT}")
    print(f"Wrote report to {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()
