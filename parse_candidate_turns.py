#!/usr/bin/env python3
"""Parse OSCE transcripts into position-matched next-doctor-turn instances."""

from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parent
TRANSCRIPT_DIR = (
    ROOT / "doctor-patient-consultation-dialogues" / "Data" / "Clean Transcripts"
)
OUTPUT_CSV = ROOT / "all_candidate_turns.csv"
OUTPUT_REPORT = ROOT / "candidate_turn_report.md"

SPECIALTY_BY_PREFIX = {
    "CAR": "Cardiovascular",
    "DER": "Dermatological",
    "GAS": "Gastrointestinal",
    "GEN": "General",
    "MSK": "Musculoskeletal",
    "RES": "Respiratory",
}

SPEAKER_LINE = re.compile(
    r"^(?P<speaker>DL|[DP])[\s;:.\u201c\"\u201d\u2018\u2019-]*\s*(?P<text>.*)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Turn:
    speaker: str
    text: str
    source_line: int


def read_transcript(path: Path) -> str:
    raw = path.read_bytes()
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        return raw.decode("utf-16")
    if b"\x00" in raw[:100]:
        for encoding in ("utf-16-le", "utf-16-be"):
            try:
                return raw.decode(encoding)
            except UnicodeDecodeError:
                continue
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("unknown", b"", 0, 1, f"Could not decode {path.name}")


def normalize_speaker(raw_speaker: str) -> str:
    speaker = raw_speaker.upper()
    if speaker == "DL":
        return "D"
    return speaker


def parse_transcript(text: str) -> list[Turn]:
    turns: list[Turn] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue

        match = SPEAKER_LINE.match(stripped)
        if match:
            speaker = normalize_speaker(match.group("speaker"))
            utterance = match.group("text").strip()
            turns.append(Turn(speaker=speaker, text=utterance, source_line=line_number))
            continue

        if not turns:
            continue
        previous = turns[-1]
        merged_text = f"{previous.text} {stripped}".strip()
        turns[-1] = Turn(
            speaker=previous.speaker,
            text=merged_text,
            source_line=previous.source_line,
        )
    return turns


def format_turn(speaker: str, text: str) -> str:
    label = "Doctor" if speaker == "D" else "Patient"
    return f"{label}: {text}"


def format_context(turns: list[Turn]) -> str:
    return "\n\n".join(format_turn(turn.speaker, turn.text) for turn in turns)


def specialty_for_dialogue(dialogue_id: str) -> str:
    prefix = re.match(r"^[A-Z]+", dialogue_id)
    if not prefix:
        return "Unknown"
    return SPECIALTY_BY_PREFIX.get(prefix.group(0), "Unknown")


def position_third(normalized_position: float) -> str:
    if normalized_position <= 1 / 3:
        return "early"
    if normalized_position <= 2 / 3:
        return "middle"
    return "late"


def build_instances() -> tuple[list[dict[str, object]], dict[str, object]]:
    transcript_files = sorted(TRANSCRIPT_DIR.glob("*.txt"))
    rows: list[dict[str, object]] = []

    total_doctor_turns = 0
    candidates_per_dialogue: list[int] = []
    specialty_candidate_counts: Counter[str] = Counter()
    specialty_dialogue_counts: Counter[str] = Counter()
    position_third_counts: Counter[str] = Counter()

    for transcript_path in transcript_files:
        dialogue_id = transcript_path.stem
        specialty = specialty_for_dialogue(dialogue_id)
        specialty_dialogue_counts[specialty] += 1

        turns = parse_transcript(read_transcript(transcript_path))
        doctor_turn_indices = [
            index for index, turn in enumerate(turns) if turn.speaker == "D"
        ]
        total_doctor_turns += len(doctor_turn_indices)

        candidate_indices = [
            index
            for index in doctor_turn_indices
            if index > 0 and turns[index - 1].speaker == "P"
        ]
        candidate_count = len(candidate_indices)
        candidates_per_dialogue.append(candidate_count)
        specialty_candidate_counts[specialty] += candidate_count

        for candidate_number, turn_index in enumerate(candidate_indices, start=1):
            doctor_turn = turns[turn_index]
            patient_turn = turns[turn_index - 1]
            context_turns = turns[:turn_index]
            normalized_position = (
                candidate_number / candidate_count if candidate_count else 0.0
            )
            third = position_third(normalized_position)
            position_third_counts[third] += 1

            rows.append(
                {
                    "dialogue_id": dialogue_id,
                    "specialty": specialty,
                    "target_turn_id": f"{dialogue_id}_turn_{turn_index + 1:04d}",
                    "turn_index": turn_index + 1,
                    "source_line": doctor_turn.source_line,
                    "context": format_context(context_turns),
                    "latest_patient_turn": patient_turn.text,
                    "human_doctor_target": doctor_turn.text,
                    "doctor_turn_number": doctor_turn_indices.index(turn_index) + 1,
                    "candidate_turn_number": candidate_number,
                    "candidate_turns_in_dialogue": candidate_count,
                    "target_position": round(normalized_position, 6),
                    "target_position_third": third,
                }
            )

    summary = {
        "dialogue_count": len(transcript_files),
        "total_doctor_turns": total_doctor_turns,
        "candidate_turn_count": len(rows),
        "specialty_dialogue_counts": specialty_dialogue_counts,
        "specialty_candidate_counts": specialty_candidate_counts,
        "candidates_per_dialogue": candidates_per_dialogue,
        "position_third_counts": position_third_counts,
    }
    return rows, summary


def write_csv(rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "dialogue_id",
        "specialty",
        "target_turn_id",
        "turn_index",
        "source_line",
        "context",
        "latest_patient_turn",
        "human_doctor_target",
        "doctor_turn_number",
        "candidate_turn_number",
        "candidate_turns_in_dialogue",
        "target_position",
        "target_position_third",
    ]
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def distribution_lines(values: list[int]) -> list[str]:
    counter = Counter(values)
    lines = []
    for key in sorted(counter):
        lines.append(f"| {key} | {counter[key]} |")
    return lines


def write_report(summary: dict[str, object]) -> None:
    candidates_per_dialogue = summary["candidates_per_dialogue"]
    specialty_dialogue_counts: Counter[str] = summary["specialty_dialogue_counts"]
    specialty_candidate_counts: Counter[str] = summary["specialty_candidate_counts"]
    position_third_counts: Counter[str] = summary["position_third_counts"]

    lines = [
        "# Candidate Turn Report",
        "",
        "Generated from `parse_candidate_turns.py` using all 272 OSCE clean transcripts.",
        "",
        "## Overview",
        "",
        f"- Dialogues parsed: **{summary['dialogue_count']}**",
        f"- Total doctor turns: **{summary['total_doctor_turns']}**",
        f"- Candidate next-doctor-turn instances: **{summary['candidate_turn_count']}**",
        "",
        "Candidate rule: include every doctor turn immediately preceded by a patient turn.",
        "No LLM generation, no sampling, and no subjective turn removal were applied.",
        "",
        "## Specialty Distribution",
        "",
        "| Specialty | Dialogues | Candidate turns | Avg candidates / dialogue |",
        "| --- | ---: | ---: | ---: |",
    ]

    for specialty in sorted(specialty_dialogue_counts):
        dialogue_count = specialty_dialogue_counts[specialty]
        candidate_count = specialty_candidate_counts[specialty]
        avg_candidates = candidate_count / dialogue_count if dialogue_count else 0
        lines.append(
            f"| {specialty} | {dialogue_count} | {candidate_count} | {avg_candidates:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Candidate Turns Per Dialogue",
            "",
            f"- Minimum: **{min(candidates_per_dialogue)}**",
            f"- Median: **{median(candidates_per_dialogue):.1f}**",
            f"- Mean: **{mean(candidates_per_dialogue):.2f}**",
            f"- Maximum: **{max(candidates_per_dialogue)}**",
            "",
            "| Candidate turns / dialogue | Dialogue count |",
            "| ---: | ---: |",
            *distribution_lines(candidates_per_dialogue),
            "",
            "## Target Position Distribution",
            "",
            "Normalized target position is computed within each dialogue as "
            "`candidate_turn_number / candidate_turns_in_dialogue`.",
            "",
            "| Position third | Candidate turns | Share |",
            "| --- | ---: | ---: |",
        ]
    )

    total_candidates = summary["candidate_turn_count"]
    for third in ("early", "middle", "late"):
        count = position_third_counts[third]
        share = (count / total_candidates * 100) if total_candidates else 0
        lines.append(f"| {third} | {count} | {share:.2f}% |")

    lines.extend(
        [
            "",
            "## Output Files",
            "",
            f"- CSV: `{OUTPUT_CSV.name}`",
            f"- Report: `{OUTPUT_REPORT.name}`",
            f"- Parser: `parse_candidate_turns.py`",
        ]
    )

    OUTPUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows, summary = build_instances()
    write_csv(rows)
    write_report(summary)
    print(f"Wrote {len(rows)} candidate turns to {OUTPUT_CSV}")
    print(f"Wrote report to {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()
