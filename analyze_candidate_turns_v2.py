#!/usr/bin/env python3
"""Refined descriptive characterization of next-doctor-turn candidates (v2)."""

from __future__ import annotations

import csv
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median, pstdev

import spacy

ROOT = Path(__file__).resolve().parent
INPUT_CSV = ROOT / "all_candidate_turns.csv"
OUTPUT_CHARACTERISTICS = ROOT / "candidate_characteristics_v2.csv"
OUTPUT_REPORT = ROOT / "candidate_analysis_report_v2.md"
OUTPUT_EXAMPLES = ROOT / "candidate_examples_for_review_v2.csv"

RANDOM_SEED = 42
EXAMPLE_TARGET_COUNT = 150

NLP = spacy.load("en_core_web_sm")
WORD_RE = re.compile(r"[A-Za-z0-9']+")

# ---------------------------------------------------------------------------
# Shared text helpers
# ---------------------------------------------------------------------------

WH_WORDS = {"what", "when", "where", "why", "who", "which", "how"}
AUX_LEMMAS = {
    "do", "does", "did", "have", "has", "had", "be", "am", "is", "are", "was",
    "were", "can", "could", "will", "would", "shall", "should", "may", "might",
    "must",
}

DISCOURSE_PREFIX_RE = re.compile(
    r"^(?:(?:ok(?:ay)?|alright|right|sure|and|so|well|now|um+|uh+|hmm+|"
    r"great|good|perfect|awesome|excellent|wonderful|thanks|thank you|"
    r"all right)[,.\s-]*)+",
    re.IGNORECASE,
)

WH_AFTER_PREFIX_RE = re.compile(
    r"(?:^|\b)(what|when|where|why|who|which|how)\b", re.IGNORECASE
)
AUX_LED_RE = re.compile(
    r"(?:^|\b)(?:do|does|did|have|has|had|is|are|was|were|can|could|will|"
    r"would|shall|should|may|might|must|am)\b(?:\s+\w+){0,4}\s+"
    r"(?:you|your|there|it|this|that|he|she|they|we|the|any|anything|anyone|"
    r"ever|either|one|she's|he's|it's|they're|we're|i|u)\b",
    re.IGNORECASE,
)
AUX_LED_SHORT_RE = re.compile(
    r"^\s*(?:do|does|did|have|has|had|is|are|was|were|can|could|will|"
    r"would|shall|should|may|might|must|am)\b",
    re.IGNORECASE,
)
ELLIPTICAL_REQUEST_RE = re.compile(
    r"(?:^|\b)(?:any(?:thing|one|body)?|ever|either)\b", re.IGNORECASE
)
REQUEST_VERB_RE = re.compile(
    r"\b(tell me|let me know|describe|explain|clarify|walk me through|"
    r"can you tell me|could you tell me|can you describe|could you describe|"
    r"can you explain|could you explain|would you mind telling me)\b",
    re.IGNORECASE,
)
TAG_QUESTION_RE = re.compile(
    r"(?:,|\b)(?:right|correct|yes|no|or not)\s*[.?!]?\s*$", re.IGNORECASE
)

ACK_VOCAB = {
    "ok", "okay", "alright", "all", "right", "sure", "yes", "yeah", "yep",
    "no", "problem", "of", "course", "thanks", "thank", "you", "awesome",
    "great", "good", "perfect", "excellent", "wonderful", "sounds", "got",
    "it", "understood", "see", "i", "mm", "mhm", "hmm", "uh", "huh", "mhm",
    "worries", "my", "pleasure",
}


def is_acknowledgement_only(text: str) -> bool:
    normalized = normalize_for_rules(text).lower()
    normalized = re.sub(r"[.!?]+$", "", normalized).strip()
    if not normalized or "?" in normalized:
        return False
    if any(
        keyword in normalized
        for keyword in (
            "tell me", "describe", "explain", "when", "what", "where", "why",
            "how", "which", "who", "any", "anything", "ever", "have you",
            "do you", "does", "did", "is there", "are there",
        )
    ):
        return False
    tokens = [token for token in re.split(r"[,\s]+", normalized) if token]
    if not tokens:
        return False
    allowed_phrases = (
        {"thank you", "no problem", "i see", "got it", "sounds good", "all right"}
    )
    joined = normalized
    if joined in allowed_phrases:
        return True
    return all(token in ACK_VOCAB for token in tokens)


GREETING_RE = re.compile(
    r"\b(hi|hello|hey|good morning|good afternoon|good evening|nice to meet you|"
    r"welcome(?: to)?|my name is|i(?:'m| am)\s+(?:dr|doctor))\b",
    re.IGNORECASE,
)
CLOSING_RE = re.compile(
    r"\b(goodbye|bye|see you|take care|have a (?:good|nice)(?: day| one)|"
    r"that(?:'s| is) (?:all|everything)(?: my questions(?: for now)?)?|"
    r"those are (?:all|all of) my questions|all my questions(?: for now)?|"
    r"we(?:'re| are) (?:done|finished|all set)|"
    r"thank you for (?:coming|your time)|"
    r"do you have any (?:other )?questions(?: for me)?|"
    r"anything else(?: you(?:'d| would) like to (?:ask|know|discuss))?|"
    r"nice (?:meeting|talking to) you|"
    r"we will (?:definitely )?(?:do that|take care of that) for you)\b",
    re.IGNORECASE,
)
TRANSITION_RE = re.compile(
    r"\b(i(?:'ll| will)|let me|give me a (?:moment|minute|sec)|bear with me|"
    r"be right back|step out(?:side)?|go talk to|talk to the doctor|"
    r"discuss this with|grab the doctor|get the doctor|consult with|"
    r"come back (?:in|with)|back with you in a moment|"
    r"move on to|moving on to|let's move on|let us move on|"
    r"just going to get (?:the )?(?:patient'?s )?(?:vitals|nurse|doctor)|"
    r"get the (?:patient'?s )?vitals|"
    r"before we move forward to the physical|"
    r"we(?:'ll| will) (?:go|move) (?:ahead )?(?:with|to)|"
    r"that sounds good\??$)\b",
    re.IGNORECASE,
)
EXAMINATION_RE = re.compile(
    r"\b(examine|examination|physical exam|let me (?:look|check|listen|feel|"
    r"palpate|have a look|take a look|inspect)|"
    r"i(?:'ll| will) (?:look|check|listen|examine|palpate|need to examine|"
    r"do an exam|have a look|take a look)|"
    r"listen to your|auscult|palpate|inspect|"
    r"check your (?:blood pressure|vitals|pulse|heart|lungs|abdomen|back|leg|"
    r"skin|reflexes|throat|ears|eyes)|"
    r"going to (?:examine|check|look at|inspect|palpate)|"
    r"move on to the physical|start the physical|during the exam|"
    r"on physical exam|perform an exam|"
    r"can you (?:lie down|sit up|stand up|remove|take off|lift|bend|move|"
    r"show me|pull up|pull down))\b",
    re.IGNORECASE,
)
DIAGNOSIS_RE = re.compile(
    r"\b(i think|it (?:sounds|looks|seems) like|likely|probably|possibly|"
    r"this (?:is|could be|may be|might be)|appears to be|diagnosis|"
    r"based on what you(?:'ve| have) told me|from what you(?:'ve| have) said|"
    r"suggest(?:s|ing)? (?:that )?you (?:have|may have|might have)|"
    r"consistent with|indicative of|concerned about|worried about|"
    r"what(?:'s| is) (?:most )?likely going on|"
    r"working diagnosis|differential (?:includes|is)|"
    r"doesn't sound like|does not sound like)\b",
    re.IGNORECASE,
)
TREATMENT_RE = re.compile(
    r"\b(recommend|prescribe|prescription|medication|treatment|manage(?:ment)?|"
    r"you should (?:take|try|use|start|stop|avoid|rest|drink|eat)|"
    r"we(?:'ll| will) (?:start|give|prescribe|order|refer|arrange|book|"
    r"send you for)|follow up|follow-up|refer you|send you for|"
    r"take (?:this|these|it|them)|over the counter|"
    r"apply (?:this|a)|come back if|return if|go to (?:the )?(?:er|emergency)|"
    r"start you on|put you on|trial of|"
    r"make sure you (?:take|use|continue|keep))\b",
    re.IGNORECASE,
)
ADMINISTRATIVE_RE = re.compile(
    r"\b(appointment|schedule|paperwork|form|insurance|billing|registration|"
    r"wait here|waiting room|front desk|referral letter|fill out|sign this|"
    r"contact (?:the )?clinic|book (?:you )?(?:in|an appointment)|"
    r"medical record number|health card|identification)\b",
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


def strip_discourse_prefix(text: str) -> str:
    stripped = text.strip()
    previous = None
    while previous != stripped:
        previous = stripped
        stripped = DISCOURSE_PREFIX_RE.sub("", stripped).strip()
    return stripped


def normalize_for_rules(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def spacy_information_seeking(text: str) -> tuple[bool, list[str]]:
    doc = NLP(normalize_for_rules(text))
    matched: list[str] = []

    for sent in doc.sents:
        sent_text = sent.text.strip()
        if not sent_text:
            continue

        if any(token.text.lower() in WH_WORDS for token in sent):
            if any(
                token.text.lower() in WH_WORDS
                and token.dep_ in {"advmod", "attr", "dobj", "pobj", "nsubj", "ROOT"}
                for token in sent
            ):
                matched.append("spacy_wh")
                break

        first_token = sent[0] if len(sent) else None
        if first_token is not None:
            if first_token.dep_ in {"aux", "auxpass"} or first_token.lemma_.lower() in AUX_LEMMAS:
                if any(child.dep_ in {"nsubj", "nsubjpass", "expl"} for child in sent):
                    matched.append("spacy_aux_inversion")
                    break

        if sent.root.tag_ == "VB" and sent.root.lemma_.lower() in {
            "tell", "describe", "explain", "clarify", "list", "mention", "note",
        }:
            matched.append("spacy_request_verb")
            break

        if any(token.dep_ == "mark" and token.text.lower() == "if" for token in sent):
            if any(token.dep_ == "aux" for token in sent):
                matched.append("spacy_conditional_request")
                break

    return bool(matched), matched


def regex_information_seeking(text: str) -> tuple[bool, list[str]]:
    matched: list[str] = []
    normalized = normalize_for_rules(text)
    core = strip_discourse_prefix(normalized)

    if "?" in normalized:
        matched.append("question_mark")

    for label, candidate in (
        ("wh_construction", normalized),
        ("wh_construction", core),
    ):
        if WH_AFTER_PREFIX_RE.search(candidate):
            matched.append(label)
            break

    for label, candidate in (
        ("auxiliary_led", normalized),
        ("auxiliary_led", core),
    ):
        if AUX_LED_RE.search(candidate) or AUX_LED_SHORT_RE.search(candidate):
            matched.append(label)
            break

    for label, candidate in (
        ("elliptical_request", normalized),
        ("elliptical_request", core),
    ):
        if ELLIPTICAL_REQUEST_RE.search(candidate):
            matched.append(label)
            break

    if REQUEST_VERB_RE.search(normalized):
        matched.append("request_verb")

    if TAG_QUESTION_RE.search(normalized):
        matched.append("tag_question")

    return bool(matched), sorted(set(matched))


def classify_information_seeking(text: str) -> tuple[bool, str]:
    regex_hit, regex_rules = regex_information_seeking(text)
    spacy_hit, spacy_rules = spacy_information_seeking(text)
    rules = sorted(set(regex_rules + spacy_rules))
    return regex_hit or spacy_hit, "|".join(rules)


def classify_exclusion_flags(text: str) -> dict[str, bool]:
    normalized = normalize_for_rules(text)
    return {
        "flag_acknowledgement_only": is_acknowledgement_only(normalized),
        "flag_greeting_opening": bool(GREETING_RE.search(normalized)),
        "flag_closing_farewell": bool(CLOSING_RE.search(normalized)),
        "flag_conversational_transition": bool(TRANSITION_RE.search(normalized)),
        "flag_examination_procedure": bool(EXAMINATION_RE.search(normalized)),
        "flag_diagnosis_explanation": bool(DIAGNOSIS_RE.search(normalized)),
        "flag_treatment_management": bool(TREATMENT_RE.search(normalized)),
        "flag_administrative": bool(ADMINISTRATIVE_RE.search(normalized)),
    }


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
    sentence_count = len(split_sentences(target))

    preceding_doctor, preceding_patient, preceding_total = count_context_turns(context)
    info_seeking, info_rules = classify_information_seeking(target)
    exclusion_flags = classify_exclusion_flags(target)

    enriched: dict[str, object] = {
        **row,
        "target_word_count": target_words,
        "context_word_count": context_words,
        "latest_patient_word_count": latest_patient_words,
        "context_turn_count": preceding_total,
        "preceding_doctor_turns": preceding_doctor,
        "preceding_patient_turns": preceding_patient,
        "preceding_total_turns": preceding_total,
        "completed_exchanges_before_target": preceding_patient,
        "absolute_target_doctor_turn_index": int(row["doctor_turn_number"]),
        "normalized_target_position": float(row["target_position"]),
        "position_category": row["target_position_third"],
        "contains_question_mark": question_marks > 0,
        "question_mark_count": question_marks,
        "sentence_count": sentence_count,
        "flag_question_containing": question_marks > 0,
        "flag_information_seeking": info_seeking,
        "information_seeking_rules_matched": info_rules,
        "flag_question_information_disagreement": (question_marks > 0) != info_seeking,
        "flag_very_short_le2": target_words <= 2,
        "flag_very_short_le3": target_words <= 3,
        "flag_very_short_le5": target_words <= 5,
        **exclusion_flags,
    }
    return enriched


def count_where(rows: list[dict[str, object]], predicate) -> int:
    return sum(1 for row in rows if predicate(row))


def combo_counts(rows: list[dict[str, object]]) -> dict[str, int]:
    def info(row):
        return row["flag_information_seeking"]

    def ge2(row):
        return row["completed_exchanges_before_target"] >= 2

    def ge3(row):
        return row["completed_exchanges_before_target"] >= 3

    def not_excluded(row):
        return not (
            row["flag_acknowledgement_only"]
            or row["flag_greeting_opening"]
            or row["flag_closing_farewell"]
            or row["flag_conversational_transition"]
            or row["flag_examination_procedure"]
            or row["flag_diagnosis_explanation"]
            or row["flag_treatment_management"]
            or row["flag_administrative"]
        )

    return {
        "info_seeking_and_ge2_exchanges": count_where(rows, lambda r: info(r) and ge2(r)),
        "info_seeking_and_ge3_exchanges": count_where(rows, lambda r: info(r) and ge3(r)),
        "info_seeking_and_not_excluded": count_where(rows, lambda r: info(r) and not_excluded(r)),
        "info_seeking_and_ge2_and_not_excluded": count_where(
            rows, lambda r: info(r) and ge2(r) and not_excluded(r)
        ),
        "question_only_and_ge2_exchanges": count_where(
            rows, lambda r: r["flag_question_containing"] and ge2(r)
        ),
        "info_seeking_not_question_mark": count_where(
            rows, lambda r: info(r) and not r["flag_question_containing"]
        ),
    }


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


def breakdown_table(
    rows: list[dict[str, object]], key_field: str, combo_names: list[str]
) -> list[str]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key_field])].append(row)

    lines = [
        f"| {key_field} | " + " | ".join(combo_names) + " |",
        "| --- | " + " | ".join(["---:"] * len(combo_names)) + " |",
    ]
    for key in sorted(grouped):
        combos = combo_counts(grouped[key])
        lines.append(
            f"| {key} | " + " | ".join(str(combos[name]) for name in combo_names) + " |"
        )
    return lines


def coverage_tags(row: dict[str, object]) -> list[str]:
    tags = [
        "info_true" if row["flag_information_seeking"] else "info_false",
        "question_true" if row["flag_question_containing"] else "question_false",
    ]
    if row["flag_information_seeking"] != row["flag_question_containing"]:
        tags.append("disagreement")
    tags.append(f"pos_{row['position_category']}")
    tags.append(f"spec_{row['specialty']}")
    for field in (
        "flag_acknowledgement_only",
        "flag_greeting_opening",
        "flag_closing_farewell",
        "flag_conversational_transition",
        "flag_examination_procedure",
        "flag_diagnosis_explanation",
        "flag_treatment_management",
        "flag_administrative",
    ):
        if row[field]:
            tags.append(field.replace("flag_", "excl_"))
    return tags


def stratified_examples(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rng = random.Random(RANDOM_SEED)
    selected: list[dict[str, object]] = []
    selected_ids: set[str] = set()

    priority_groups: list[tuple[str, callable]] = [
        ("disagreement", lambda r: r["flag_question_information_disagreement"]),
        ("info_true", lambda r: r["flag_information_seeking"]),
        ("info_false", lambda r: not r["flag_information_seeking"]),
        ("excl_ack", lambda r: r["flag_acknowledgement_only"]),
        ("excl_greeting", lambda r: r["flag_greeting_opening"]),
        ("excl_closing", lambda r: r["flag_closing_farewell"]),
        ("excl_transition", lambda r: r["flag_conversational_transition"]),
        ("excl_exam", lambda r: r["flag_examination_procedure"]),
        ("excl_diagnosis", lambda r: r["flag_diagnosis_explanation"]),
        ("excl_treatment", lambda r: r["flag_treatment_management"]),
        ("excl_admin", lambda r: r["flag_administrative"]),
    ]

    for specialty in sorted(set(str(r["specialty"]) for r in rows)):
        specialty_rows = [r for r in rows if r["specialty"] == specialty]
        rng.shuffle(specialty_rows)
        if specialty_rows and specialty_rows[0]["target_turn_id"] not in selected_ids:
            selected.append(specialty_rows[0])
            selected_ids.add(str(specialty_rows[0]["target_turn_id"]))

    for position in ("early", "middle", "late"):
        pool = [r for r in rows if r["position_category"] == position]
        rng.shuffle(pool)
        for row in pool:
            if row["target_turn_id"] not in selected_ids:
                selected.append(row)
                selected_ids.add(str(row["target_turn_id"]))
                break

    for _, predicate in priority_groups:
        pool = [r for r in rows if predicate(r) and r["target_turn_id"] not in selected_ids]
        rng.shuffle(pool)
        quota = 8 if _ == "disagreement" else 5
        for row in pool[:quota]:
            selected.append(row)
            selected_ids.add(str(row["target_turn_id"]))

    remaining = [r for r in rows if r["target_turn_id"] not in selected_ids]
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
        "flag_question_containing",
        "flag_information_seeking",
        "information_seeking_rules_matched",
        "flag_question_information_disagreement",
        "flag_acknowledgement_only",
        "flag_greeting_opening",
        "flag_closing_farewell",
        "flag_conversational_transition",
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


def write_report(rows: list[dict[str, object]], combos: dict[str, int]) -> None:
    total = len(rows)
    per_dialogue_stats = candidates_per_dialogue_stats(rows)
    specialty_counts = Counter(row["specialty"] for row in rows)
    position_counts = Counter(row["position_category"] for row in rows)
    combo_names = list(combos.keys())

    flag_fields = [
        ("flag_question_containing", "Question-containing (`?` present)"),
        ("flag_information_seeking", "Information-seeking (hybrid rules)"),
        ("flag_question_information_disagreement", "Disagreement: question mark vs information-seeking"),
        ("flag_acknowledgement_only", "Acknowledgement-only"),
        ("flag_greeting_opening", "Greeting/opening"),
        ("flag_closing_farewell", "Closing/farewell"),
        ("flag_conversational_transition", "Conversational transition"),
        ("flag_examination_procedure", "Examination/procedure instruction"),
        ("flag_diagnosis_explanation", "Diagnosis/explanation"),
        ("flag_treatment_management", "Treatment/management"),
        ("flag_administrative", "Administrative"),
        ("flag_very_short_le2", "Very short (<=2 words)"),
        ("flag_very_short_le3", "Very short (<=3 words)"),
        ("flag_very_short_le5", "Very short (<=5 words)"),
    ]

    rule_counts = Counter()
    for row in rows:
        if row["information_seeking_rules_matched"]:
            for rule in str(row["information_seeking_rules_matched"]).split("|"):
                rule_counts[rule] += 1

    lines = [
        "# Candidate Analysis Report v2",
        "",
        "Refined descriptive characterization with hybrid information-seeking detection.",
        "No final sampling, no LLM classification, and no candidate deletion were performed.",
        "",
        "## 1. Basic Corpus Statistics",
        "",
        f"- Total candidate instances: **{total:,}**",
        f"- Unique dialogues: **{len(set(row['dialogue_id'] for row in rows)):,}**",
        f"- Candidate turns per dialogue (mean / median / SD / min / max): "
        f"**{per_dialogue_stats['mean']:.2f} / {per_dialogue_stats['median']:.1f} / "
        f"{per_dialogue_stats['sd']:.2f} / {per_dialogue_stats['min']} / {per_dialogue_stats['max']}**",
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
            "### Target Position Distribution",
            "",
            *position_histogram(rows),
            "",
            "## 2. Question vs Information-Seeking",
            "",
            f"- Question mark present: **{count_where(rows, lambda r: r['flag_question_containing']):,}** "
            f"({pct(count_where(rows, lambda r: r['flag_question_containing']), total):.2f}%)",
            f"- Information-seeking by hybrid rules: **{count_where(rows, lambda r: r['flag_information_seeking']):,}** "
            f"({pct(count_where(rows, lambda r: r['flag_information_seeking']), total):.2f}%)",
            f"- Information-seeking without question mark: **{combos['info_seeking_not_question_mark']:,}** "
            f"({pct(combos['info_seeking_not_question_mark'], total):.2f}%)",
            f"- Question mark but not information-seeking: **{count_where(rows, lambda r: r['flag_question_containing'] and not r['flag_information_seeking']):,}**",
            f"- Disagreement between old and new flags: **{count_where(rows, lambda r: r['flag_question_information_disagreement']):,}**",
            "",
            "### Information-Seeking Rule Hits",
            "",
            "| Rule | Count |",
            "| --- | ---: |",
        ]
    )

    for rule, count in rule_counts.most_common():
        lines.append(f"| `{rule}` | {count:,} |")

    lines.extend(
        [
            "",
            "## 3. Exploratory Exclusion Flags",
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
            "## 4. Rule Documentation",
            "",
            "### `flag_information_seeking`",
            "True when any hybrid rule matches on `human_doctor_target`.",
            "",
            "1. **Question punctuation**: at least one `?`.",
            "2. **WH interrogatives**: regex "
            f"`{WH_AFTER_PREFIX_RE.pattern}` on full text or discourse-stripped text.",
            "3. **Auxiliary-led interrogatives**: regex "
            f"`{AUX_LED_RE.pattern}` or `{AUX_LED_SHORT_RE.pattern}`.",
            "4. **Elliptical clinical requests**: regex "
            f"`{ELLIPTICAL_REQUEST_RE.pattern}` on full text or discourse-stripped text.",
            "5. **Explicit request verbs**: regex "
            f"`{REQUEST_VERB_RE.pattern}`.",
            "6. **Tag-like prompts**: regex "
            f"`{TAG_QUESTION_RE.pattern}`.",
            "7. **spaCy syntactic cues** on any sentence:",
            "   - WH token with dependency in `{advmod, attr, dobj, pobj, nsubj, ROOT}`",
            "   - sentence-initial auxiliary with subject (`nsubj`, `nsubjpass`, or `expl`)",
            "   - root lemma in `{tell, describe, explain, clarify, list, mention, note}`",
            "   - conditional request with `if` mark plus auxiliary",
            "",
            "Discourse prefixes stripped before several regex checks:",
            f"`{DISCOURSE_PREFIX_RE.pattern}`",
            "",
            "Word count is **not** used as an exclusion criterion.",
            "",
            "### Exclusion flags",
            "",
            "#### Acknowledgement-only",
            "- Entire turn must contain only acknowledgement vocabulary tokens after lowercasing and removing final punctuation.",
            "- Allowed token set: "
            f"`{sorted(ACK_VOCAB)}`",
            "- Allowed multiword phrases: `thank you`, `no problem`, `i see`, `got it`, `sounds good`, `all right`.",
            "- Rejected if `?` is present or if information-request keywords such as `any`, `what`, `how`, or `do you` appear.",
            "",
            "#### Greeting/opening",
            f"- `{GREETING_RE.pattern}`",
            "",
            "#### Closing/farewell",
            f"- `{CLOSING_RE.pattern}`",
            "",
            "#### Conversational transition",
            f"- `{TRANSITION_RE.pattern}`",
            "",
            "#### Examination/procedure instruction",
            f"- `{EXAMINATION_RE.pattern}`",
            "",
            "#### Diagnosis/explanation",
            f"- `{DIAGNOSIS_RE.pattern}`",
            "",
            "#### Treatment/management",
            f"- `{TREATMENT_RE.pattern}`",
            "",
            "#### Administrative",
            f"- `{ADMINISTRATIVE_RE.pattern}`",
            "",
            "## 5. Alternative Inclusion-Rule Counts",
            "",
            "| Rule | Count | Percentage |",
            "| --- | ---: | ---: |",
        ]
    )

    for name, count in combos.items():
        lines.append(f"| {name.replace('_', ' ')} | {count:,} | {pct(count, total):.2f}% |")

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
            "## 6. Review Sample",
            "",
            f"- Output file: `{OUTPUT_EXAMPLES.name}`",
            f"- Random seed: **{RANDOM_SEED}**",
            f"- Target sample size: **{EXAMPLE_TARGET_COUNT}**",
            "- Coverage targets: information-seeking true/false, old/new disagreement, each exclusion category, all specialties, and all position categories.",
            "",
            "## 7. Output Files",
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

    write_report(enriched_rows, combos)

    print(f"Wrote {len(enriched_rows):,} rows to {OUTPUT_CHARACTERISTICS}")
    print(f"Wrote report to {OUTPUT_REPORT}")
    print(f"Wrote {len(examples)} examples to {OUTPUT_EXAMPLES}")


if __name__ == "__main__":
    main()
