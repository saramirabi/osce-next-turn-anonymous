#!/usr/bin/env python3
"""Apply manual eligibility decisions and produce final_1500_TRULY_FROZEN.csv."""

from __future__ import annotations

import csv
import random
from collections import Counter
from pathlib import Path
from statistics import mean, median, pstdev

import sample_final_1500_v2 as qc

ROOT = Path(__file__).resolve().parent
REVIEWED_AUDIT = ROOT / "final_eligibility_audit_for_review.csv"
INPUT_FROZEN = ROOT / "final_1500_FROZEN.csv"
INPUT_CHARACTERISTICS = ROOT / "candidate_characteristics_v3.csv"
OUTPUT_FROZEN = ROOT / "final_1500_TRULY_FROZEN.csv"
OUTPUT_REPLACEMENT_AUDIT = ROOT / "final_replacement_audit_completed.csv"
OUTPUT_REPORT = ROOT / "final_frozen_validation_report.md"

RANDOM_SEED = 42
SAMPLE_COLUMNS = qc.SAMPLE_COLUMNS
REPLACEMENT_AUDIT_COLUMNS = [
    "original_context_id",
    "removed_target_turn_id",
    "manual_exclusion_reason",
    "replacement_target_turn_id",
    "dialogue_id",
    "position_category",
]

# Manual review of all 61 flagged audit rows.
MANUAL_DECISIONS: dict[str, tuple[str, str]] = {
    "CTX_0021": ("KEEP", "Substantive history: recent changes/triggers for chest pain."),
    "CTX_0022": ("KEEP", "Substantive history: activity-related symptom pattern."),
    "CTX_0027": ("KEEP", "Substantive history: elicits additional new symptoms."),
    "CTX_0032": ("KEEP", "Substantive history: urinary frequency and related symptoms."),
    "CTX_0040": ("KEEP", "Substantive history: alcohol use screening (CAGE-style)."),
    "CTX_0114": (
        "EXCLUDE",
        "Substantive purpose is examination/procedure (Homan test during dorsiflexion).",
    ),
    "CTX_0151": ("KEEP", "Substantive history: pain quality characterization."),
    "CTX_0154": (
        "EXCLUDE",
        "Substantive purpose is examination finding plus closing/any-questions (Phalen's test).",
    ),
    "CTX_0160": (
        "EXCLUDE",
        "Substantive purpose is examination/procedure (active knee movement testing).",
    ),
    "CTX_0176": (
        "EXCLUDE",
        "Substantive purpose is examination summary and closing/any-questions.",
    ),
    "CTX_0182": ("KEEP", "Substantive history: pain in finger joints."),
    "CTX_0186": ("KEEP", "Substantive history: deformity and functional limitation."),
    "CTX_0218": ("KEEP", "Substantive history: shoulder range-of-motion/function."),
    "CTX_0221": ("KEEP", "Substantive history: pain radiation pattern."),
    "CTX_0224": (
        "EXCLUDE",
        "Substantive purpose is examination/procedure (palpation/temperature check).",
    ),
    "CTX_0231": ("KEEP", "Substantive history: leg alignment/appearance comparison."),
    "CTX_0275": (
        "EXCLUDE",
        "Substantive purpose is closing and transition to physical examination.",
    ),
    "CTX_0285": (
        "EXCLUDE",
        "Substantive purpose is examination findings and range-of-motion testing.",
    ),
    "CTX_0304": ("KEEP", "Substantive history: fever/chills and related symptoms."),
    "CTX_0314": (
        "EXCLUDE",
        "Substantive purpose is treatment/management (splint, physiotherapy, injection plan).",
    ),
    "CTX_0315": (
        "EXCLUDE",
        "Substantive purpose is examination/procedure (Finkelstein test).",
    ),
    "CTX_0326": (
        "EXCLUDE",
        "Substantive purpose is examination/procedure (Tinel's test).",
    ),
    "CTX_0364": ("KEEP", "Substantive history: seasonal allergy symptoms."),
    "CTX_0365": ("KEEP", "Substantive history: lifestyle and personal habits."),
    "CTX_0394": (
        "EXCLUDE",
        "Substantive purpose is diagnosis/explanation rather than history elicitation.",
    ),
    "CTX_0421": ("KEEP", "Substantive history: occupational and exposure history."),
    "CTX_0438": (
        "EXCLUDE",
        "Substantive purpose is examination/procedure (in-visit temperature measurement).",
    ),
    "CTX_0538": (
        "EXCLUDE",
        "Substantive purpose is diagnosis/explanation and treatment/investigation planning.",
    ),
    "CTX_0561": ("KEEP", "Substantive history: cough quality characterization."),
    "CTX_0562": ("KEEP", "Substantive history: cough triggers."),
    "CTX_0651": ("KEEP", "Substantive history: weight loss and night sweats."),
    "CTX_0712": ("KEEP", "Substantive history: whether temperature was measured at home."),
    "CTX_0724": (
        "EXCLUDE",
        "Substantive purpose is treatment/management and investigation planning.",
    ),
    "CTX_0737": ("KEEP", "Substantive history: shortness-of-breath characterization."),
    "CTX_0827": (
        "EXCLUDE",
        "Substantive purpose is conversational transition/advice without history elicitation.",
    ),
    "CTX_0850": ("KEEP", "Substantive history: home context and caregiving."),
    "CTX_0856": (
        "EXCLUDE",
        "Substantive purpose is closing and transition to physical examination.",
    ),
    "CTX_0892": ("KEEP", "Substantive history: associated upper-respiratory symptoms."),
    "CTX_0896": (
        "EXCLUDE",
        "Substantive purpose is treatment/management (bronchodilator/steroid offer).",
    ),
    "CTX_0920": ("KEEP", "Substantive history: associated symptom review (headache)."),
    "CTX_0947": ("KEEP", "Substantive history: influenza vaccination status."),
    "CTX_0951": ("KEEP", "Substantive history: antipyretic use for fever."),
    "CTX_1024": ("KEEP", "Substantive history: hemoptysis quantity."),
    "CTX_1033": (
        "EXCLUDE",
        "Substantive purpose is closing/any-questions rather than history elicitation.",
    ),
    "CTX_1085": (
        "EXCLUDE",
        "Substantive purpose is treatment/management (monitoring, discharge, physiotherapy).",
    ),
    "CTX_1096": (
        "EXCLUDE",
        "Substantive purpose is treatment/management (symptomatic care advice).",
    ),
    "CTX_1109": (
        "EXCLUDE",
        "Substantive purpose is conversational summary without history elicitation.",
    ),
    "CTX_1111": ("KEEP", "Substantive history: childhood infections and birth history."),
    "CTX_1131": ("KEEP", "Substantive history: recent trauma to affected area."),
    "CTX_1138": ("KEEP", "Substantive history: growth and development concerns."),
    "CTX_1151": ("KEEP", "Substantive history: tuberculosis exposure before travel."),
    "CTX_1238": ("KEEP", "Substantive history: recent school attendance."),
    "CTX_1274": ("KEEP", "Substantive history: palpitations outside fainting episode."),
    "CTX_1368": (
        "EXCLUDE",
        "Substantive purpose is investigation planning and treatment/management discussion.",
    ),
    "CTX_1406": ("KEEP", "Substantive history: recreational drug use."),
    "CTX_1420": ("KEEP", "Substantive history: cough character and sputum/blood."),
    "CTX_1430": ("KEEP", "Substantive history: dry vs productive cough."),
    "CTX_1437": ("KEEP", "Substantive history: functional impact of COPD symptoms."),
    "CTX_1466": (
        "EXCLUDE",
        "Substantive purpose is visit summary/closing rather than history elicitation.",
    ),
    "CTX_1477": ("KEEP", "Substantive history: stress coping behaviors."),
    "CTX_1494": (
        "EXCLUDE",
        "Substantive purpose is conversational summary without history elicitation.",
    ),
}


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def update_reviewed_audit(audit_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    updated: list[dict[str, str]] = []
    seen: set[str] = set()

    for row in audit_rows:
        context_id = row["context_id"]
        seen.add(context_id)
        if context_id not in MANUAL_DECISIONS:
            raise ValueError(f"Missing manual decision for audit row {context_id}.")
        decision, reason = MANUAL_DECISIONS[context_id]
        enriched = dict(row)
        enriched["manual_decision"] = decision
        enriched["manual_reason"] = reason
        updated.append(enriched)

    missing = set(MANUAL_DECISIONS) - seen
    if missing:
        raise ValueError(f"Manual decisions exist for unknown audit rows: {sorted(missing)}")

    write_csv(
        REVIEWED_AUDIT,
        list(updated[0].keys()),
        updated,
    )
    return updated


def build_eligible_pool(
    characteristics: list[dict[str, str]],
) -> dict[tuple[str, str], list[dict[str, str]]]:
    pool: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in qc.eligible_rows(characteristics):
        pool.setdefault((row["dialogue_id"], row["position_category"]), []).append(row)
    return pool


def choose_replacement(
    rng: random.Random,
    dialogue_id: str,
    position_category: str,
    selected_target_ids: set[str],
    excluded_target_ids: set[str],
    eligible_pool: dict[tuple[str, str], list[dict[str, str]]],
) -> dict[str, str]:
    candidates = [
        row
        for row in eligible_pool.get((dialogue_id, position_category), [])
        if row["target_turn_id"] not in selected_target_ids
        and row["target_turn_id"] not in excluded_target_ids
    ]
    if not candidates:
        raise ValueError(
            f"No replacement available for dialogue={dialogue_id}, position={position_category}."
        )

    clean = [row for row in candidates if not qc.audit_flags_for_row(row)]
    pool = clean if clean else candidates
    rng.shuffle(pool)
    return pool[0]


def build_sample_row(candidate: dict[str, str], context_id: str) -> dict[str, str]:
    row = {column: candidate.get(column, "") for column in SAMPLE_COLUMNS}
    row["context_id"] = context_id
    return row


def finalize_sample() -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    audit_rows = update_reviewed_audit(load_csv(REVIEWED_AUDIT))
    frozen_rows = load_csv(INPUT_FROZEN)
    characteristics = load_csv(INPUT_CHARACTERISTICS)
    eligible_pool = build_eligible_pool(characteristics)

    exclude_rows = [
        row for row in audit_rows if row["manual_decision"].strip().upper() == "EXCLUDE"
    ]
    exclude_context_ids = {row["context_id"] for row in exclude_rows}
    excluded_target_ids = {
        row["target_turn_id"]
        for row in frozen_rows
        if row["context_id"] in exclude_context_ids
    }

    sample_by_context = {row["context_id"]: dict(row) for row in frozen_rows}
    selected_target_ids = {
        row["target_turn_id"]
        for context_id, row in sample_by_context.items()
        if context_id not in exclude_context_ids
    }

    rng = random.Random(RANDOM_SEED)
    replacement_audit: list[dict[str, str]] = []

    for exclude_row in sorted(exclude_rows, key=lambda row: row["context_id"]):
        context_id = exclude_row["context_id"]
        original = sample_by_context[context_id]
        replacement = choose_replacement(
            rng,
            original["dialogue_id"],
            original["position_category"],
            selected_target_ids,
            excluded_target_ids,
            eligible_pool,
        )
        selected_target_ids.add(replacement["target_turn_id"])
        sample_by_context[context_id] = build_sample_row(replacement, context_id)
        replacement_audit.append(
            {
                "original_context_id": context_id,
                "removed_target_turn_id": original["target_turn_id"],
                "manual_exclusion_reason": exclude_row["manual_reason"],
                "replacement_target_turn_id": replacement["target_turn_id"],
                "dialogue_id": original["dialogue_id"],
                "position_category": original["position_category"],
            }
        )

    truly_frozen = [sample_by_context[row["context_id"]] for row in frozen_rows]
    return truly_frozen, replacement_audit, exclude_rows


def verify_truly_frozen(
    rows: list[dict[str, str]], excluded_context_ids: set[str]
) -> dict[str, object]:
    checks: dict[str, object] = {}

    checks["row_count"] = len(rows)
    if len(rows) != 1500:
        raise ValueError(f"Expected 1500 rows, found {len(rows)}.")

    target_ids = [row["target_turn_id"] for row in rows]
    checks["unique_target_ids"] = len(set(target_ids))
    if len(set(target_ids)) != 1500:
        raise ValueError("Duplicate target_turn_id values found.")

    per_dialogue = Counter(row["dialogue_id"] for row in rows)
    checks["dialogue_count"] = len(per_dialogue)
    if len(per_dialogue) != 272:
        raise ValueError(f"Expected 272 dialogues, found {len(per_dialogue)}.")

    per_dialogue_values = set(per_dialogue.values())
    checks["targets_per_dialogue_values"] = sorted(per_dialogue_values)
    if per_dialogue_values - {5, 6}:
        raise ValueError(f"Unexpected targets-per-dialogue values: {sorted(per_dialogue_values)}")

    position_counts = Counter(row["position_category"] for row in rows)
    expected_position = Counter({"early": 544, "middle": 544, "late": 412})
    checks["position_counts"] = dict(position_counts)
    if position_counts != expected_position:
        raise ValueError(
            "Position allocation mismatch. "
            f"Expected {dict(expected_position)}, found {dict(position_counts)}."
        )

    excluded_targets_still_present = []
    with INPUT_FROZEN.open(encoding="utf-8-sig", newline="") as handle:
        original_by_context = {row["context_id"]: row for row in csv.DictReader(handle)}
    for context_id in excluded_context_ids:
        current = next(row for row in rows if row["context_id"] == context_id)
        original = original_by_context[context_id]
        if current["target_turn_id"] == original["target_turn_id"]:
            excluded_targets_still_present.append(context_id)
    checks["excluded_targets_replaced"] = len(excluded_context_ids) - len(
        excluded_targets_still_present
    )
    if excluded_targets_still_present:
        raise ValueError(
            "Manually excluded targets were not replaced: "
            f"{excluded_targets_still_present}"
        )

    removed_targets = {original_by_context[cid]["target_turn_id"] for cid in excluded_context_ids}
    present_removed_targets = removed_targets.intersection(target_ids)
    checks["removed_targets_absent"] = not present_removed_targets
    if present_removed_targets:
        raise ValueError(
            "Manually excluded target_turn_id values still present in final sample: "
            f"{sorted(present_removed_targets)}"
        )

    return checks


def write_report(
    rows: list[dict[str, str]],
    replacement_audit: list[dict[str, str]],
    exclude_rows: list[dict[str, str]],
    validation: dict[str, object],
) -> None:
    keep_count = 61 - len(exclude_rows)
    per_dialogue = Counter(row["dialogue_id"] for row in rows)
    per_dialogue_values = list(per_dialogue.values())
    position_counts = Counter(row["position_category"] for row in rows)
    specialty_counts = Counter(row["specialty"] for row in rows)
    context_words = [int(row["context_word_count"]) for row in rows]
    target_words = [int(row["target_word_count"]) for row in rows]
    exchanges = [int(row["completed_exchanges_before_target"]) for row in rows]

    lines = [
        "# Final Frozen Validation Report",
        "",
        "Dataset correction applied after GPT pilot review. "
        "Manual eligibility decisions were applied to flagged audit rows, "
        "excluded non-history targets were replaced within dialogue/position strata, "
        "and the corrected sample was frozen as `final_1500_TRULY_FROZEN.csv`.",
        "",
        "## Manual Review",
        "",
        f"- Flagged audit rows reviewed: **61**",
        f"- Rows marked KEEP: **{keep_count}**",
        f"- Rows marked EXCLUDE: **{len(exclude_rows)}**",
        f"- Replacements made: **{len(replacement_audit)}**",
        f"- Required exclusions confirmed: **CTX_0154**, **CTX_1368**",
        "",
        "## Automatic Validation",
        "",
        f"- Row count: **{validation['row_count']:,}**",
        f"- Unique target IDs: **{validation['unique_target_ids']:,}**",
        f"- Unique dialogues: **{validation['dialogue_count']:,}**",
        f"- Targets per dialogue values: **{validation['targets_per_dialogue_values']}**",
        f"- Position counts: **{validation['position_counts']}**",
        f"- Excluded targets replaced: **{validation['excluded_targets_replaced']}**",
        "",
        "## Final Sample Checks",
        "",
        f"- Targets per dialogue mean / median / SD: "
        f"**{mean(per_dialogue_values):.2f} / {median(per_dialogue_values):.1f} / "
        f"{pstdev(per_dialogue_values):.2f}**",
        f"- Targets per dialogue min / max: **{min(per_dialogue_values)} / {max(per_dialogue_values)}**",
        "",
        "### Position Category Distribution",
        "",
        "| Category | Count | Percentage |",
        "| --- | ---: | ---: |",
    ]

    for category in ("early", "middle", "late"):
        count = position_counts[category]
        lines.append(f"| {category} | {count:,} | {count / len(rows) * 100:.2f}% |")

    lines.extend(
        [
            "",
            "### Specialty Distribution",
            "",
            "| Specialty | Count | Percentage |",
            "| --- | ---: | ---: |",
        ]
    )
    for specialty in sorted(specialty_counts):
        count = specialty_counts[specialty]
        lines.append(f"| {specialty} | {count:,} | {count / len(rows) * 100:.2f}% |")

    lines.extend(
        [
            "",
            "### Context and Target Length",
            "",
            f"- Context words mean / median / SD: **{mean(context_words):.2f} / {median(context_words):.1f} / {pstdev(context_words):.2f}**",
            f"- Target words mean / median / SD: **{mean(target_words):.2f} / {median(target_words):.1f} / {pstdev(target_words):.2f}**",
            "",
            "### Completed Exchanges Before Target",
            "",
            f"- Mean: **{mean(exchanges):.2f}**",
            f"- Min / max: **{min(exchanges)} / {max(exchanges)}**",
            "",
            "## Replacement Policy",
            "",
            "- Source sample: `final_1500_FROZEN.csv`",
            "- For each manually excluded target, replacement selected from same "
            "`dialogue_id` and `position_category`.",
            "- Replacement candidates required `primary_turn_function == history_information_seeking` "
            "and `completed_exchanges_before_target >= 2`.",
            "- Unflagged eligible replacements preferred; deterministic fallback otherwise.",
            f"- Random seed: **{RANDOM_SEED}**",
            "",
            "## Output Files",
            "",
            f"- `{OUTPUT_FROZEN.name}`",
            f"- `{OUTPUT_REPLACEMENT_AUDIT.name}`",
            f"- `{REVIEWED_AUDIT.name}` (updated with manual decisions)",
            f"- `{OUTPUT_REPORT.name}`",
            f"- `{Path(__file__).name}`",
        ]
    )

    OUTPUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    truly_frozen, replacement_audit, exclude_rows = finalize_sample()
    excluded_context_ids = {row["context_id"] for row in exclude_rows}
    validation = verify_truly_frozen(truly_frozen, excluded_context_ids)

    write_csv(OUTPUT_FROZEN, SAMPLE_COLUMNS, truly_frozen)
    write_csv(OUTPUT_REPLACEMENT_AUDIT, REPLACEMENT_AUDIT_COLUMNS, replacement_audit)
    write_report(truly_frozen, replacement_audit, exclude_rows, validation)

    print(f"Manual EXCLUDE rows: {len(exclude_rows)}")
    print(f"Replacements made: {len(replacement_audit)}")
    print(f"Wrote corrected sample to {OUTPUT_FROZEN}")
    print(f"Wrote replacement audit to {OUTPUT_REPLACEMENT_AUDIT}")
    print(f"Updated reviewed audit at {REVIEWED_AUDIT}")
    print(f"Wrote validation report to {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()
