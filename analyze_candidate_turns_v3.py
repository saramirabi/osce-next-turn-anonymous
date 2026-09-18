#!/usr/bin/env python3
"""Primary turn-function classification with sentence-level precedence (v3)."""

from __future__ import annotations

import csv
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median, pstdev

import analyze_candidate_turns_v2 as v2

ROOT = Path(__file__).resolve().parent
INPUT_CSV = ROOT / "all_candidate_turns.csv"
OUTPUT_CHARACTERISTICS = ROOT / "candidate_characteristics_v3.csv"
OUTPUT_REPORT = ROOT / "candidate_analysis_report_v3.md"
OUTPUT_EXAMPLES = ROOT / "candidate_examples_for_review_v3.csv"

RANDOM_SEED = 42
EXAMPLE_TARGET_COUNT = 175
HISTORY_MIN_EXAMPLES = 30

PRIMARY_FUNCTIONS = (
    "history_information_seeking",
    "acknowledgement_only",
    "greeting_opening",
    "closing_farewell",
    "conversational_transition",
    "examination_procedure",
    "diagnosis_explanation",
    "treatment_management",
    "administrative",
    "other_unclassified",
)

# Lower number = higher precedence when choosing dominant substantive function.
FUNCTION_PRECEDENCE = {
    "examination_procedure": 1,
    "diagnosis_explanation": 2,
    "treatment_management": 3,
    "administrative": 4,
    "closing_farewell": 5,
    "conversational_transition": 6,
    "greeting_opening": 7,
    "history_information_seeking": 8,
    "other_unclassified": 9,
    "acknowledgement_only": 10,
}

CLINICIAN_RECAP_RE = re.compile(
    r"\b(?:based on|from) what you(?:'ve| have) (?:told|said)\b", re.IGNORECASE
)
INCIDENTAL_CLOSING_Q_RE = re.compile(
    r"^\s*(?:and\s+)?(?:do you have any (?:other )?questions(?: for me)?|"
    r"any questions(?: for me)?|"
    r"anything else(?: you(?:'d| would) like to (?:ask|know|discuss|add))?|"
    r"does that (?:make )?sense|is that (?:ok|okay|clear|alright|fine)|"
    r"do you understand|are you (?:ok|okay) with that|how does that sound|"
    r"did you have any questions)\??\s*$",
    re.IGNORECASE,
)
HISTORY_MEDICATION_RE = re.compile(
    r"\b(?:do you take|did you take|are you (?:on|taking)|were you (?:on|taking)|"
    r"what medications|which medications|any medications|any meds|"
    r"medications?(?:\s+regularly|\s+at home|\s+currently|\s+are you on)?|"
    r"prescribed or over the counter|allergies to medications|any allergies|"
    r"over[- ]the[- ]counter|supplements? you take)\b",
    re.IGNORECASE,
)
HISTORY_SOCIAL_RE = re.compile(
    r"\b(?:do you smoke|how much do you smoke|do you drink|how much alcohol|"
    r"who are you sexually active with|any new partners|occupation|where do you live|"
    r"family history|past medical history|prior surgeries|hospitalizations|"
    r"immunizations?|living situation|support yourself financially)\b",
    re.IGNORECASE,
)
TREATMENT_DIRECTIVE_RE = re.compile(
    r"\b(?:i recommend|we(?:'ll| will) (?:prescribe|give|start|order|refer|"
    r"arrange|book|send you for)|you should (?:take|try|use|start|stop|avoid|"
    r"rest|drink|eat)|start you on|put you on|try this|take this|prescribe|"
    r"prescription for|make sure you (?:take|use|continue|keep)|"
    r"come back if|return if|follow up in|book a follow|go to (?:the )?(?:er|emergency)|"
    r"apply (?:this|a)|over the counter for relief)\b",
    re.IGNORECASE,
)
DIAGNOSIS_STATEMENT_RE = re.compile(
    r"\b(?:i think (?:you|this|it)|it (?:sounds|looks|seems) like|"
    r"this (?:is|could be|may be|might be)|appears to be|"
    r"likely (?:that )?(?:you have|this is)|probably (?:that )?(?:you have|this is)|"
    r"consistent with|indicative of|working diagnosis|differential (?:includes|is)|"
    r"based on what you(?:'ve| have) told me,?\s+(?:i think|this|it|you|likely)|"
    r"from what you(?:'ve| have) said,?\s+(?:i think|this|it|you|likely)|"
    r"what(?:'s| is) (?:most )?likely going on)\b",
    re.IGNORECASE,
)
EXAMINATION_SENTENCE_RE = re.compile(
    r"\b(?:let me (?:look|check|listen|feel|palpate|have a look|take a look|inspect)|"
    r"i(?:'ll| will) (?:look|check|listen|examine|palpate|need to examine|"
    r"do an exam|have a look|take a look)|"
    r"listen to your|auscult|palpate|inspect your|"
    r"going to (?:examine|check|look at|inspect|palpate)|"
    r"move on to the physical|start the physical|during the exam|"
    r"on physical exam|perform an exam|get the patient'?s vitals|"
    r"just going to get (?:the )?(?:patient'?s )?(?:vitals|nurse)|"
    r"can you (?:lie down|sit up|stand up|remove|take off|lift|bend|move|"
    r"show me|pull up|pull down))\b",
    re.IGNORECASE,
)
CLOSING_STATEMENT_RE = re.compile(
    r"\b(?:goodbye|bye|see you|take care|have a (?:good|nice)(?: day| one)|"
    r"that(?:'s| is) (?:all|everything)(?: my questions(?: for now)?)?|"
    r"those are (?:all|all of) my questions|all my questions(?: for now)?|"
    r"we(?:'re| are) (?:done|finished|all set)|"
    r"thank you for (?:coming|your time)|"
    r"nice (?:meeting|talking to) you|"
    r"we will (?:definitely )?(?:do that|take care of that) for you)\b",
    re.IGNORECASE,
)


def pct(part: int, whole: int) -> float:
    return (part / whole * 100) if whole else 0.0


def is_brief_acknowledgement_sentence(sentence: str) -> bool:
    return v2.is_acknowledgement_only(sentence)


def sentence_history_information_seeking(sentence: str) -> tuple[bool, str]:
    normalized = v2.normalize_for_rules(sentence)
    core = v2.strip_discourse_prefix(normalized)

    if INCIDENTAL_CLOSING_Q_RE.match(normalized):
        return False, ""

    if CLINICIAN_RECAP_RE.search(normalized):
        return False, ""

    if HISTORY_MEDICATION_RE.search(normalized) or HISTORY_MEDICATION_RE.search(core):
        return True, "history_medication"

    if HISTORY_SOCIAL_RE.search(normalized) or HISTORY_SOCIAL_RE.search(core):
        return True, "history_social"

    regex_hit, regex_rules = v2.regex_information_seeking(normalized)
    if regex_hit:
        return True, "regex:" + ",".join(regex_rules)

    spacy_hit, spacy_rules = v2.spacy_information_seeking(normalized)
    if spacy_hit:
        return True, "spacy:" + ",".join(spacy_rules)

    return False, ""


def sentence_treatment_management(sentence: str) -> tuple[bool, str]:
    normalized = v2.normalize_for_rules(sentence)
    if HISTORY_MEDICATION_RE.search(normalized):
        return False, ""
    if TREATMENT_DIRECTIVE_RE.search(normalized):
        return True, "treatment_directive"
    if v2.TREATMENT_RE.search(normalized) and not sentence_history_information_seeking(
        normalized
    )[0]:
        if re.search(
            r"\b(recommend|prescribe|refer|follow up|follow-up|management plan|"
            r"treatment plan|you should|we(?:'ll| will))\b",
            normalized,
            re.IGNORECASE,
        ):
            return True, "treatment_keyword_with_directive_context"
    return False, ""


def sentence_diagnosis_explanation(sentence: str) -> tuple[bool, str]:
    normalized = v2.normalize_for_rules(sentence)
    if DIAGNOSIS_STATEMENT_RE.search(normalized):
        return True, "diagnosis_statement"
    if CLINICIAN_RECAP_RE.search(normalized) and re.search(
        r"\b(i think|likely|probably|this (?:is|could be|may be|might be))\b",
        normalized,
        re.IGNORECASE,
    ):
        return True, "diagnosis_recap_statement"
    if v2.DIAGNOSIS_RE.search(normalized) and not re.search(
        r"^\s*(?:do|does|did|have|has|had|are|is|was|were|can|could|would|will)\b",
        normalized,
        re.IGNORECASE,
    ):
        return True, "diagnosis_keyword_non_interrogative"
    return False, ""


def classify_sentence(sentence: str, index: int) -> tuple[str, str]:
    normalized = v2.normalize_for_rules(sentence)
    if not normalized:
        return "other_unclassified", f"sentence_{index + 1}:empty"

    if is_brief_acknowledgement_sentence(normalized):
        return "acknowledgement_only", f"sentence_{index + 1}:acknowledgement_only"

    if INCIDENTAL_CLOSING_Q_RE.match(normalized):
        return "closing_farewell", f"sentence_{index + 1}:incidental_closing_question"

    if v2.ADMINISTRATIVE_RE.search(normalized):
        return "administrative", f"sentence_{index + 1}:administrative"

    if EXAMINATION_SENTENCE_RE.search(normalized) or (
        v2.EXAMINATION_RE.search(normalized)
        and not sentence_history_information_seeking(normalized)[0]
    ):
        return "examination_procedure", f"sentence_{index + 1}:examination_procedure"

    diagnosis_hit, diagnosis_reason = sentence_diagnosis_explanation(normalized)
    if diagnosis_hit:
        return "diagnosis_explanation", f"sentence_{index + 1}:{diagnosis_reason}"

    treatment_hit, treatment_reason = sentence_treatment_management(normalized)
    if treatment_hit:
        return "treatment_management", f"sentence_{index + 1}:{treatment_reason}"

    if v2.TRANSITION_RE.search(normalized):
        return "conversational_transition", f"sentence_{index + 1}:transition"

    if CLOSING_STATEMENT_RE.search(normalized) or v2.CLOSING_RE.search(normalized):
        return "closing_farewell", f"sentence_{index + 1}:closing_statement"

    if v2.GREETING_RE.search(normalized):
        return "greeting_opening", f"sentence_{index + 1}:greeting"

    history_hit, history_reason = sentence_history_information_seeking(normalized)
    if history_hit:
        return "history_information_seeking", f"sentence_{index + 1}:{history_reason}"

    if "?" in normalized or v2.AUX_LED_SHORT_RE.search(normalized):
        return "other_unclassified", f"sentence_{index + 1}:unresolved_interrogative"

    return "other_unclassified", f"sentence_{index + 1}:no_rule_match"


def classify_primary_turn_function(text: str) -> tuple[str, str, str]:
    sentences = v2.split_sentences(text)
    if not sentences:
        return "other_unclassified", "no_sentences", ""

    sentence_results: list[tuple[str, str, str]] = []
    for index, sentence in enumerate(sentences):
        function, reason = classify_sentence(sentence, index)
        sentence_results.append((function, reason, sentence))

    sentence_summary = " | ".join(
        f"{idx + 1}:{function}[{reason.split(':', 1)[-1]}]"
        for idx, (function, reason, _) in enumerate(sentence_results)
    )

    if all(function == "acknowledgement_only" for function, _, _ in sentence_results):
        return (
            "acknowledgement_only",
            "all_sentences_acknowledgement_only",
            sentence_summary,
        )

    substantive = [
        (function, reason, sentence)
        for function, reason, sentence in sentence_results
        if function != "acknowledgement_only"
    ]

    if not substantive:
        return (
            "acknowledgement_only",
            "only_brief_acknowledgements",
            sentence_summary,
        )

    dominant_function, dominant_reason, _ = min(
        substantive,
        key=lambda item: FUNCTION_PRECEDENCE[item[0]],
    )

    incidental = [
        reason
        for function, reason, _ in substantive
        if function != dominant_function
        and function in {"closing_farewell", "other_unclassified"}
    ]
    reason = f"dominant={dominant_reason}"
    if incidental:
        reason += "; incidental=" + ",".join(incidental)

    return dominant_function, reason, sentence_summary


def enrich_candidate(row: dict[str, str]) -> dict[str, object]:
    enriched = v2.enrich_candidate(row)
    primary_function, primary_reason, sentence_summary = classify_primary_turn_function(
        row["human_doctor_target"]
    )
    enriched["primary_turn_function"] = primary_function
    enriched["primary_function_reason"] = primary_reason
    enriched["sentence_level_classifications"] = sentence_summary
    return enriched


def stratified_examples(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rng = random.Random(RANDOM_SEED)
    selected: list[dict[str, object]] = []
    selected_ids: set[str] = set()

    per_category_quota = {
        "history_information_seeking": HISTORY_MIN_EXAMPLES,
        "acknowledgement_only": 12,
        "greeting_opening": 10,
        "closing_farewell": 12,
        "conversational_transition": 12,
        "examination_procedure": 12,
        "diagnosis_explanation": 12,
        "treatment_management": 12,
        "administrative": 10,
        "other_unclassified": 12,
    }

    for function in PRIMARY_FUNCTIONS:
        pool = [
            row
            for row in rows
            if row["primary_turn_function"] == function
            and row["target_turn_id"] not in selected_ids
        ]
        rng.shuffle(pool)
        quota = per_category_quota[function]
        for row in pool[:quota]:
            selected.append(row)
            selected_ids.add(str(row["target_turn_id"]))

    for specialty in sorted(set(str(row["specialty"]) for row in rows)):
        pool = [
            row
            for row in rows
            if row["specialty"] == specialty and row["target_turn_id"] not in selected_ids
        ]
        rng.shuffle(pool)
        if pool:
            selected.append(pool[0])
            selected_ids.add(str(pool[0]["target_turn_id"]))

    remaining = [row for row in rows if row["target_turn_id"] not in selected_ids]
    rng.shuffle(remaining)
    for row in remaining:
        selected.append(row)
        selected_ids.add(str(row["target_turn_id"]))
        if len(selected) >= EXAMPLE_TARGET_COUNT:
            break

    return selected[:EXAMPLE_TARGET_COUNT]


def example_fieldnames(rows: list[dict[str, object]]) -> list[str]:
    preferred = [
        "dialogue_id",
        "specialty",
        "target_turn_id",
        "normalized_target_position",
        "position_category",
        "primary_turn_function",
        "primary_function_reason",
        "sentence_level_classifications",
        "context",
        "latest_patient_turn",
        "human_doctor_target",
        "contains_question_mark",
        "flag_question_containing",
        "flag_information_seeking",
        "information_seeking_rules_matched",
        "flag_acknowledgement_only",
        "flag_greeting_opening",
        "flag_closing_farewell",
        "flag_conversational_transition",
        "flag_examination_procedure",
        "flag_diagnosis_explanation",
        "flag_treatment_management",
        "flag_administrative",
    ]
    available = set(rows[0].keys()) if rows else set()
    return [name for name in preferred if name in available]


def write_report(rows: list[dict[str, object]]) -> None:
    total = len(rows)
    primary_counts = Counter(str(row["primary_turn_function"]) for row in rows)
    per_dialogue = candidates_per_dialogue_stats(rows)

    lines = [
        "# Candidate Analysis Report v3",
        "",
        "Deterministic primary turn-function classification with sentence-level precedence.",
        "All v2 descriptive flags are preserved. No final sampling or LLM calls were performed.",
        "",
        "## Primary Turn Function Distribution",
        "",
        "| Primary function | Count | Percentage |",
        "| --- | ---: | ---: |",
    ]

    for function in PRIMARY_FUNCTIONS:
        count = primary_counts.get(function, 0)
        lines.append(f"| `{function}` | {count:,} | {pct(count, total):.2f}% |")

    lines.extend(
        [
            "",
            "## Corpus Summary",
            "",
            f"- Total candidates: **{total:,}**",
            f"- Unique dialogues: **{len(set(row['dialogue_id'] for row in rows)):,}**",
            f"- Candidates per dialogue (mean / median / SD): "
            f"**{per_dialogue['mean']:.2f} / {per_dialogue['median']:.1f} / {per_dialogue['sd']:.2f}**",
            f"- History-information-seeking primary function: **{primary_counts['history_information_seeking']:,}** "
            f"({pct(primary_counts['history_information_seeking'], total):.2f}%)",
            "",
            "## Precedence Logic",
            "",
            "Each turn is split into sentences. Each sentence receives an independent function label.",
            "Brief acknowledgement-only sentences are ignored when selecting the dominant substantive function.",
            "The turn-level primary function is the highest-precedence substantive sentence function.",
            "",
            "Precedence order (highest to lowest):",
            "1. `examination_procedure`",
            "2. `diagnosis_explanation`",
            "3. `treatment_management`",
            "4. `administrative`",
            "5. `closing_farewell`",
            "6. `conversational_transition`",
            "7. `greeting_opening`",
            "8. `history_information_seeking`",
            "9. `other_unclassified`",
            "",
            "If every sentence is acknowledgement-only, primary function = `acknowledgement_only`.",
            "",
            "## Sentence-Level Rules",
            "",
            "### History information seeking",
            "- Assigned when the substantive sentence elicits patient history.",
            "- Includes medication-history questions such as `do you take`, `any medications`, and `allergies`.",
            "- Includes social/family/history patterns via `HISTORY_SOCIAL_RE`.",
            "- Uses v2 hybrid interrogative detection only after ruling out incidental closing, treatment, and diagnosis patterns.",
            "- Brief acknowledgement prefixes do not prevent history classification of the substantive sentence.",
            "",
            "### Treatment/management",
            "- Requires directive language via `TREATMENT_DIRECTIVE_RE`.",
            "- Medication-history questions are explicitly routed to history first.",
            "",
            "### Diagnosis/explanation",
            "- Requires declarative diagnosis/explanation patterns via `DIAGNOSIS_STATEMENT_RE`.",
            "- Interrogative history-taking sentences are excluded.",
            "",
            "### Incidental closing interrogatives",
            "- Sentences such as `Do you have any questions?` are classified as `closing_farewell`, not history.",
            f"- Regex: `{INCIDENTAL_CLOSING_Q_RE.pattern}`",
            "",
            "## Descriptive Flags Preserved",
            "",
            "All v2 flags remain available in the CSV, including `flag_information_seeking` and the exploratory exclusion flags.",
            "These are not identical to `primary_turn_function`; use primary function for inclusion decisions.",
            "",
            "## Review Sample",
            "",
            f"- Output file: `{OUTPUT_EXAMPLES.name}`",
            f"- Random seed: **{RANDOM_SEED}**",
            f"- Target size: **{EXAMPLE_TARGET_COUNT}**",
            f"- Minimum history-information-seeking examples: **{HISTORY_MIN_EXAMPLES}**",
            "- Coverage includes every primary function category and all specialties where available.",
            "",
            "## Output Files",
            "",
            f"- `{OUTPUT_CHARACTERISTICS.name}`",
            f"- `{OUTPUT_REPORT.name}`",
            f"- `{OUTPUT_EXAMPLES.name}`",
            f"- `{Path(__file__).name}`",
            f"- Source input (unchanged): `{INPUT_CSV.name}`",
        ]
    )

    OUTPUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def candidates_per_dialogue_stats(rows: list[dict[str, object]]) -> dict[str, float]:
    per_dialogue = Counter(row["dialogue_id"] for row in rows)
    values = list(per_dialogue.values())
    return {
        "mean": mean(values),
        "median": median(values),
        "sd": pstdev(values) if len(values) > 1 else 0.0,
    }


def main() -> None:
    raw_rows = v2.load_candidates()
    enriched_rows = [enrich_candidate(row) for row in raw_rows]
    examples = stratified_examples(enriched_rows)

    fieldnames = list(enriched_rows[0].keys())
    with OUTPUT_CHARACTERISTICS.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(enriched_rows)

    example_fields = example_fieldnames(examples)
    with OUTPUT_EXAMPLES.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=example_fields)
        writer.writeheader()
        for row in examples:
            writer.writerow({name: row.get(name, "") for name in example_fields})

    write_report(enriched_rows)

    print(f"Wrote {len(enriched_rows):,} rows to {OUTPUT_CHARACTERISTICS}")
    print(f"Wrote report to {OUTPUT_REPORT}")
    print(f"Wrote {len(examples)} examples to {OUTPUT_EXAMPLES}")


if __name__ == "__main__":
    main()
