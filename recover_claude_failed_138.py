#!/usr/bin/env python3
"""Recover ONLY failed Claude NEXT_TURN_V1 generations. Does not modify original outputs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anthropic

from generate_claude_next_turn_v1 import (
    EXPERIMENT_MODEL_ID,
    INPUT_SAMPLE,
    MAX_OUTPUT_TOKENS,
    MODEL_PROVIDER,
    OUTPUT_COLUMNS,
    OUTPUT_CSV,
    PROMPT_VERSION,
    REASONING_SETTING,
    REQUESTED_TEMPERATURE,
    SAMPLE_SIZE,
    build_user_input,
    call_model_with_retries,
    extract_generated_text,
    get_api_key,
    response_to_dict,
)

PROJECT_DIR = Path(__file__).resolve().parent
EXPECTED_FAILED_COUNT = 138
MAX_ATTEMPTS = 3  # initial call + 2 retries

RECOVERY_CSV = PROJECT_DIR / "claude_failed_138_rerun.csv"
RECOVERY_JSONL = PROJECT_DIR / "claude_failed_138_rerun_raw.jsonl"
MERGED_CSV = PROJECT_DIR / "final_Claude_1500_NEXT_TURN_V1.csv"
REPORT_MD = PROJECT_DIR / "claude_failed_138_recovery_report.md"

RECOVERY_COLUMNS = [
    "context_id",
    "consultation_id",
    "dialogue_id",
    "specialty",
    "target_turn_id",
    "dialogue_position",
    "position_category",
    "normalized_target_position",
    "context",
    "latest_patient_turn",
    "human_doctor_target",
    "model_provider",
    "model_id",
    "prompt_version",
    "run_timestamp_utc",
    "requested_temperature",
    "actual_temperature_if_known",
    "reasoning_setting",
    "max_output_tokens",
    "original_failure_status",
    "original_error_message",
    "rerun_status",
    "number_of_attempts",
    "raw_claude_output",
    "generated_doctor_turn",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "error_message",
]


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def is_failed_or_empty(row: dict[str, str]) -> bool:
    status = (row.get("api_request_status") or "").strip()
    text = (row.get("generated_doctor_turn") or "").strip()
    return status != "success" or not text


def identify_failures(
    original_rows: list[dict[str, str]], frozen_rows: list[dict[str, str]]
) -> list[dict[str, str]]:
    if len(frozen_rows) != SAMPLE_SIZE:
        raise RuntimeError(f"Frozen sample has {len(frozen_rows)} rows, expected {SAMPLE_SIZE}.")
    if len(original_rows) != SAMPLE_SIZE:
        raise RuntimeError(f"Original Claude file has {len(original_rows)} rows, expected {SAMPLE_SIZE}.")

    frozen_ids = [row["context_id"] for row in frozen_rows]
    original_ids = [row["context_id"] for row in original_rows]
    if frozen_ids != original_ids:
        raise RuntimeError("Original Claude rows are not in frozen sample order/IDs.")

    frozen_by_id = {row["context_id"]: row for row in frozen_rows}
    failures: list[dict[str, str]] = []
    for row in original_rows:
        if not is_failed_or_empty(row):
            continue
        frozen = frozen_by_id[row["context_id"]]
        if row.get("context") != frozen.get("context"):
            raise RuntimeError(f"Context mismatch vs frozen sample for {row['context_id']}")
        failures.append(row)

    if len(failures) != EXPECTED_FAILED_COUNT:
        raise RuntimeError(
            f"Detected {len(failures)} failed/empty generations, expected {EXPECTED_FAILED_COUNT}. Stopping."
        )
    return failures


def append_recovery_row(row: dict[str, Any]) -> None:
    write_header = not RECOVERY_CSV.exists()
    with RECOVERY_CSV.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RECOVERY_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({col: row.get(col, "") for col in RECOVERY_COLUMNS})


def append_recovery_jsonl(record: dict[str, Any]) -> None:
    with RECOVERY_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in fieldnames})


def already_recovered_ids() -> set[str]:
    if not RECOVERY_CSV.exists():
        return set()
    recovered: set[str] = set()
    for row in load_csv(RECOVERY_CSV):
        if (row.get("rerun_status") or "").strip() == "success" and (
            row.get("raw_claude_output") or ""
        ).strip():
            recovered.add(row["context_id"])
    return recovered


def latest_recovery_by_id() -> dict[str, dict[str, str]]:
    if not RECOVERY_CSV.exists():
        return {}
    latest: dict[str, dict[str, str]] = {}
    for row in load_csv(RECOVERY_CSV):
        latest[row["context_id"]] = row
    return latest


def run_recovery(failures: list[dict[str, str]], frozen_by_id: dict[str, dict[str, str]]) -> None:
    client = anthropic.Anthropic(api_key=get_api_key())
    done = already_recovered_ids()
    pending = [row for row in failures if row["context_id"] not in done]
    print(
        f"Recovery resume: {len(done)} already recovered, {len(pending)} pending of {len(failures)}."
    )
    if not pending:
        return

    consecutive_credit_failures = 0
    parameter_notes: list[str] = []
    for index, original in enumerate(pending, start=1):
        context_id = original["context_id"]
        frozen = frozen_by_id[context_id]
        user_input = build_user_input(frozen["context"])
        timestamp = datetime.now(timezone.utc).isoformat()

        response, retry_count, error_message, call_metadata = call_model_with_retries(
            client,
            user_input,
            parameter_notes,
            max_retries=MAX_ATTEMPTS - 1,
        )
        attempts = retry_count + 1
        generated_text = ""
        rerun_status = "failed"
        input_tokens = ""
        output_tokens = ""
        total_tokens = ""
        model_id = EXPERIMENT_MODEL_ID

        raw_record: dict[str, Any] = {
            "context_id": context_id,
            "timestamp_utc": timestamp,
            "model_id": EXPERIMENT_MODEL_ID,
            "prompt_version": PROMPT_VERSION,
            "generation_settings": {
                "requested_temperature": REQUESTED_TEMPERATURE,
                "reasoning_setting": REASONING_SETTING,
                "max_output_tokens": MAX_OUTPUT_TOKENS,
                "parameter_notes": call_metadata.get("parameter_notes", []),
            },
            "status": "failed",
            "retry_count": retry_count,
            "number_of_attempts": attempts,
            "error_message": error_message,
            "raw_generated_text": "",
            "token_usage": {},
        }

        if response is not None:
            generated_text = extract_generated_text(response)
            if generated_text.strip():
                usage = response.usage
                rerun_status = "success"
                model_id = response.model
                input_tokens = usage.input_tokens
                output_tokens = usage.output_tokens
                total_tokens = usage.input_tokens + usage.output_tokens
                error_message = ""
                raw_record["status"] = "success"
                raw_record["raw_generated_text"] = generated_text
                raw_record["token_usage"] = response_to_dict(usage)
                raw_record["response_object"] = response_to_dict(response)
                consecutive_credit_failures = 0
            else:
                error_message = "Model returned empty generated text."
                raw_record["error_message"] = error_message

        recovery_row = {
            "context_id": context_id,
            "consultation_id": original["dialogue_id"],
            "dialogue_id": original["dialogue_id"],
            "specialty": original["specialty"],
            "target_turn_id": original["target_turn_id"],
            "dialogue_position": original["position_category"],
            "position_category": original["position_category"],
            "normalized_target_position": original["normalized_target_position"],
            "context": frozen["context"],
            "latest_patient_turn": frozen["latest_patient_turn"],
            "human_doctor_target": frozen["human_doctor_target"],
            "model_provider": MODEL_PROVIDER,
            "model_id": model_id,
            "prompt_version": PROMPT_VERSION,
            "run_timestamp_utc": timestamp,
            "requested_temperature": REQUESTED_TEMPERATURE,
            "actual_temperature_if_known": str(REQUESTED_TEMPERATURE),
            "reasoning_setting": REASONING_SETTING,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "original_failure_status": original.get("api_request_status", "failed"),
            "original_error_message": original.get("error_message", ""),
            "rerun_status": rerun_status,
            "number_of_attempts": attempts,
            "raw_claude_output": generated_text,
            "generated_doctor_turn": generated_text,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "error_message": error_message or "",
        }
        append_recovery_row(recovery_row)
        append_recovery_jsonl(raw_record)
        print(f"[{index}/{len(pending)}] {context_id}: {rerun_status} (attempts={attempts})")

        err_l = (error_message or "").lower()
        if rerun_status != "success" and "credit balance is too low" in err_l:
            consecutive_credit_failures += 1
            if consecutive_credit_failures >= 3:
                raise RuntimeError(
                    "Stopped after 3 consecutive Anthropic credit-balance failures. "
                    "Add credits and re-run this script; already-recovered rows will be skipped."
                )


def validate_and_merge(
    original_rows: list[dict[str, str]],
    frozen_rows: list[dict[str, str]],
    failed_ids: set[str],
) -> dict[str, Any]:
    recovery_latest = latest_recovery_by_id()
    recovery_ids = set(recovery_latest)
    extra_ids = sorted(recovery_ids - failed_ids)
    missing_ids = sorted(failed_ids - recovery_ids)
    accidental = extra_ids[:]

    recovered_success = [
        row
        for cid, row in recovery_latest.items()
        if cid in failed_ids
        and (row.get("rerun_status") or "").strip() == "success"
        and (row.get("raw_claude_output") or "").strip()
    ]
    still_failed = [
        recovery_latest[cid]
        if cid in recovery_latest
        else {
            "context_id": cid,
            "rerun_status": "missing_from_recovery_file",
            "error_message": "No recovery row written",
        }
        for cid in sorted(failed_ids)
        if cid not in {r["context_id"] for r in recovered_success}
    ]

    original_success = [row for row in original_rows if not is_failed_or_empty(row)]
    original_success_by_id = {row["context_id"]: row for row in original_success}

    stats = {
        "expected_failed_cases": EXPECTED_FAILED_COUNT,
        "successfully_recovered": len(recovered_success),
        "still_failed": len(still_failed),
        "duplicate_ids": 0,
        "missing_ids": len(missing_ids),
        "accidentally_regenerated_successful_cases": len(accidental),
        "extra_ids": extra_ids,
        "missing_id_list": missing_ids,
    }

    print("\nRecovery validation:")
    print(f"  Expected failed cases: {stats['expected_failed_cases']}")
    print(f"  Successfully recovered: {stats['successfully_recovered']}")
    print(f"  Still failed: {stats['still_failed']}")
    print(f"  Duplicate IDs: {stats['duplicate_ids']}")
    print(f"  Missing IDs: {stats['missing_ids']}")
    print(f"  Accidentally regenerated successful cases: {stats['accidentally_regenerated_successful_cases']}")

    if extra_ids:
        print(f"  Extra/non-failed IDs in recovery file: {extra_ids}")
    if missing_ids:
        print(f"  Failed IDs missing from recovery file: {missing_ids}")

    if extra_ids or missing_ids:
        raise RuntimeError("Recovery ID validation failed; merge not written.")

    frozen_by_id = {row["context_id"]: row for row in frozen_rows}
    recovered_by_id = {row["context_id"]: row for row in recovered_success}
    merged: list[dict[str, Any]] = []

    for frozen in frozen_rows:
        cid = frozen["context_id"]
        if cid in original_success_by_id:
            merged.append(dict(original_success_by_id[cid]))
            continue
        if cid in recovered_by_id:
            rec = recovered_by_id[cid]
            orig = next(row for row in original_rows if row["context_id"] == cid)
            merged.append(
                {
                    "context_id": cid,
                    "dialogue_id": orig["dialogue_id"],
                    "specialty": orig["specialty"],
                    "target_turn_id": orig["target_turn_id"],
                    "position_category": orig["position_category"],
                    "normalized_target_position": orig["normalized_target_position"],
                    "context": frozen["context"],
                    "latest_patient_turn": frozen["latest_patient_turn"],
                    "human_doctor_target": frozen["human_doctor_target"],
                    "model_provider": rec.get("model_provider", MODEL_PROVIDER),
                    "model_id": rec.get("model_id", EXPERIMENT_MODEL_ID),
                    "prompt_version": PROMPT_VERSION,
                    "run_timestamp_utc": rec.get("run_timestamp_utc", ""),
                    "requested_temperature": REQUESTED_TEMPERATURE,
                    "actual_temperature_if_known": str(REQUESTED_TEMPERATURE),
                    "reasoning_setting": REASONING_SETTING,
                    "max_output_tokens": MAX_OUTPUT_TOKENS,
                    "generated_doctor_turn": rec.get("raw_claude_output", ""),
                    "input_tokens": rec.get("input_tokens", ""),
                    "output_tokens": rec.get("output_tokens", ""),
                    "total_tokens": rec.get("total_tokens", ""),
                    "api_request_status": "success",
                    "retry_count": rec.get("number_of_attempts", ""),
                    "error_message": "",
                }
            )
            continue
        orig = next(row for row in original_rows if row["context_id"] == cid)
        rec = recovery_latest.get(cid, {})
        merged.append(
            {
                "context_id": cid,
                "dialogue_id": orig["dialogue_id"],
                "specialty": orig["specialty"],
                "target_turn_id": orig["target_turn_id"],
                "position_category": orig["position_category"],
                "normalized_target_position": orig["normalized_target_position"],
                "context": frozen["context"],
                "latest_patient_turn": frozen["latest_patient_turn"],
                "human_doctor_target": frozen["human_doctor_target"],
                "model_provider": MODEL_PROVIDER,
                "model_id": rec.get("model_id", orig.get("model_id", EXPERIMENT_MODEL_ID)),
                "prompt_version": PROMPT_VERSION,
                "run_timestamp_utc": rec.get("run_timestamp_utc", orig.get("run_timestamp_utc", "")),
                "requested_temperature": REQUESTED_TEMPERATURE,
                "actual_temperature_if_known": str(REQUESTED_TEMPERATURE),
                "reasoning_setting": REASONING_SETTING,
                "max_output_tokens": MAX_OUTPUT_TOKENS,
                "generated_doctor_turn": "",
                "input_tokens": "",
                "output_tokens": "",
                "total_tokens": "",
                "api_request_status": "failed",
                "retry_count": rec.get("number_of_attempts", orig.get("retry_count", "")),
                "error_message": rec.get("error_message", orig.get("error_message", "")),
            }
        )

    merged_ids = [row["context_id"] for row in merged]
    if len(merged) != SAMPLE_SIZE or len(set(merged_ids)) != SAMPLE_SIZE:
        raise RuntimeError("Merged file does not contain 1500 unique context IDs.")
    if merged_ids != [row["context_id"] for row in frozen_rows]:
        raise RuntimeError("Merged file is not in frozen sample order.")

    unchanged = 0
    for row in merged:
        cid = row["context_id"]
        if cid in original_success_by_id:
            orig = original_success_by_id[cid]
            if orig.get("generated_doctor_turn") != row.get("generated_doctor_turn"):
                raise RuntimeError(f"Original successful output changed for {cid}")
            if orig.get("context") != row.get("context"):
                raise RuntimeError(f"Context changed for original success {cid}")
            unchanged += 1
        else:
            if cid not in failed_ids:
                raise RuntimeError(f"Merged non-success ID {cid} was not in original failed set.")
        if row.get("context") != frozen_by_id[cid].get("context"):
            raise RuntimeError(f"Dialogue context modified for {cid}")

    if unchanged != 1362:
        raise RuntimeError(f"Expected 1362 unchanged original successes, found {unchanged}.")

    write_csv(MERGED_CSV, merged, OUTPUT_COLUMNS)

    usable = sum(
        1
        for row in merged
        if row.get("api_request_status") == "success" and str(row.get("generated_doctor_turn", "")).strip()
    )
    stats["final_usable"] = usable
    stats["unchanged_original_successes"] = unchanged
    return stats


def write_report(
    failures: list[dict[str, str]],
    stats: dict[str, Any],
) -> None:
    dialogues = sorted({row["dialogue_id"] for row in failures})
    positions = Counter(row["position_category"] for row in failures)
    specialties = Counter(row["specialty"] for row in failures)
    lines = [
        "# Claude failed-138 recovery report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Identification",
        "",
        "- Failure rule: `api_request_status != success` OR empty `generated_doctor_turn`",
        "- Failures were **not** selected by inspecting utterance content",
        f"- Expected failed cases: {EXPECTED_FAILED_COUNT}",
        f"- Detected failed cases: {len(failures)}",
        f"- Unique consultations among failures: {len(dialogues)}",
        f"- Affected consultations: {', '.join(dialogues)}",
        f"- Position distribution: {dict(positions)}",
        f"- Specialty distribution: {dict(specialties)}",
        "",
        "## Recovery result",
        "",
        f"- original successful = 1,362",
        f"- original failed = 138",
        f"- recovered = {stats['successfully_recovered']}",
        f"- remaining failed = {stats['still_failed']}",
        f"- final usable Claude generations = {stats.get('final_usable', 'n/a')}/1,500",
        "",
        "## Integrity",
        "",
        f"- Duplicate IDs: {stats['duplicate_ids']}",
        f"- Missing IDs: {stats['missing_ids']}",
        f"- Accidentally regenerated successful cases: {stats['accidentally_regenerated_successful_cases']}",
        f"- Unchanged original successes: {stats.get('unchanged_original_successes', 'n/a')}",
        "",
        "## Files",
        "",
        f"- Original results (untouched): `{OUTPUT_CSV.name}`",
        f"- Recovery-only file: `{RECOVERY_CSV.name}`",
        f"- Merged recovered dataset: `{MERGED_CSV.name}`",
        "",
        "## Experimental settings reused",
        "",
        f"- Model: `{EXPERIMENT_MODEL_ID}`",
        f"- Prompt version: `{PROMPT_VERSION}`",
        f"- Temperature: `{REQUESTED_TEMPERATURE}`",
        f"- Max tokens: `{MAX_OUTPUT_TOKENS}`",
        f"- Reasoning: `{REASONING_SETTING}`",
        f"- Max attempts per case: {MAX_ATTEMPTS}",
        "",
    ]
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Recover only the 138 failed Claude generations.")
    parser.add_argument(
        "--run",
        action="store_true",
        help="Call Anthropic for failed cases only. Default is identify/report without API calls.",
    )
    args = parser.parse_args()

    frozen_rows = load_csv(INPUT_SAMPLE)
    original_rows = load_csv(OUTPUT_CSV)
    failures = identify_failures(original_rows, frozen_rows)
    frozen_by_id = {row["context_id"]: row for row in frozen_rows}
    failed_ids = {row["context_id"] for row in failures}

    success_n = sum(1 for row in original_rows if not is_failed_or_empty(row))
    dialogues = sorted({row["dialogue_id"] for row in failures})
    print("Identification (no original files modified):")
    print(f"  total expected contexts: {SAMPLE_SIZE}")
    print(f"  successful Claude generations: {success_n}")
    print(f"  failed/empty generations: {len(failures)}")
    print(f"  unique consultations among failures: {len(dialogues)}")
    print(f"  affected consultations: {', '.join(dialogues)}")
    print(f"  position distribution: {dict(Counter(row['position_category'] for row in failures))}")
    print(f"  specialty distribution: {dict(Counter(row['specialty'] for row in failures))}")

    if not args.run:
        print("Re-run with --run to execute recovery API calls.")
        return

    if OUTPUT_CSV.resolve() == MERGED_CSV.resolve() or OUTPUT_CSV.resolve() == RECOVERY_CSV.resolve():
        raise RuntimeError("Refusing to write onto the original Claude results path.")

    run_recovery(failures, frozen_by_id)
    stats = validate_and_merge(original_rows, frozen_rows, failed_ids)
    write_report(failures, stats)
    print(f"\nWrote {RECOVERY_CSV.name}, {MERGED_CSV.name}, {REPORT_MD.name}")
    print("Original Claude results file was not modified.")


if __name__ == "__main__":
    main()
