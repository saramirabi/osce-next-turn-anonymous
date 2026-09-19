#!/usr/bin/env python3
"""Official MedGemma (LM Studio) full 1500-sample generation (NEXT_TURN_V1)."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any

from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)

PROJECT_DIR = Path(__file__).resolve().parent

PROMPT_VERSION = "NEXT_TURN_V1"
EXPERIMENT_MODEL_ID = "medgemma-27b-text-it"
MODEL_PROVIDER = "lmstudio"
SAMPLE_SIZE = 1500
PREVIEW_SEED = 42
MAX_OUTPUT_TOKENS = 300
REQUESTED_TEMPERATURE = 0.0
REASONING_SETTING = "n/a"

DEFAULT_BASE_URL = "http://127.0.0.1:1234/v1"
DEFAULT_API_KEY = "lm-studio"

INSTRUCTIONS = (
    "You are acting as the clinician in a simulated clinical consultation. "
    "Given the complete conversation history provided below, generate the "
    "single next clinician turn that would naturally follow from the current "
    "dialogue state. Produce only the clinician's next utterance, without "
    "additional explanation, commentary, role labels, or continuation of the "
    "dialogue."
)

CONVERSATION_HEADER = "CONVERSATION:{context}"
PROMPT_TEMPLATE = f"{INSTRUCTIONS}\n{CONVERSATION_HEADER}"

INPUT_SAMPLE = PROJECT_DIR / "final_1500_TRULY_FROZEN.csv"
OUTPUT_CSV = PROJECT_DIR / "final_MedGemma_1500_NEXT_TURN_V1.csv"
RAW_JSONL = PROJECT_DIR / "final_MedGemma_1500_NEXT_TURN_V1_raw.jsonl"
REPORT_MD = PROJECT_DIR / "final_MedGemma_1500_NEXT_TURN_V1_report.md"
CONFIG_JSON = PROJECT_DIR / "MedGemma_1500_NEXT_TURN_V1_config.json"

FORBIDDEN_PROMPT_MARKERS = [
    "human_doctor_target",
    "target_turn_id",
    "position_category",
    "normalized_target_position",
    "dialogue_id",
    "specialty",
    "primary_turn_function",
    "latest_patient_turn",
    "context_id",
    "completed_exchanges_before_target",
    "context_word_count",
    "target_word_count",
]

OUTPUT_COLUMNS = [
    "context_id",
    "dialogue_id",
    "specialty",
    "target_turn_id",
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
    "generated_doctor_turn",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "api_request_status",
    "retry_count",
    "error_message",
]


def load_sample_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != SAMPLE_SIZE:
        raise ValueError(f"Expected {SAMPLE_SIZE} pilot rows, found {len(rows)}.")
    context_ids = [row["context_id"] for row in rows]
    target_ids = [row["target_turn_id"] for row in rows]
    if len(set(context_ids)) != SAMPLE_SIZE:
        raise ValueError("Sample file does not contain 1500 unique context IDs.")
    if len(set(target_ids)) != SAMPLE_SIZE:
        raise ValueError("Sample file does not contain 1500 unique target-turn IDs.")
    if any(not row.get("context", "").strip() for row in rows):
        raise ValueError("Sample file contains missing/empty context values.")
    return rows


def build_user_input(context: str) -> str:
    return CONVERSATION_HEADER.format(context=context)


def build_full_prompt_preview(context: str) -> str:
    return PROMPT_TEMPLATE.format(context=context)


def validate_prompts(rows: list[dict[str, str]]) -> None:
    for row in rows:
        user_input = build_user_input(row["context"])
        combined = f"{INSTRUCTIONS}\n\n{user_input}"
        if row["context"] not in user_input:
            raise RuntimeError(f"API input missing context for {row['context_id']}")
        lowered = combined.lower()
        for marker in FORBIDDEN_PROMPT_MARKERS:
            if marker in lowered:
                raise RuntimeError(
                    f"Forbidden metadata marker `{marker}` found in prompt for {row['context_id']}"
                )
        if row["human_doctor_target"] and row["human_doctor_target"] in combined:
            if row["human_doctor_target"] not in row["context"]:
                raise RuntimeError(
                    f"human_doctor_target text leaked into API prompt for {row['context_id']}"
                )


def print_prompt_previews(rows: list[dict[str, str]], seed: int = PREVIEW_SEED) -> None:
    rng = random.Random(seed)
    preview_rows = rng.sample(rows, 3)
    print("\nPrompt preview (3 randomly selected examples):\n")
    for index, row in enumerate(preview_rows, start=1):
        print(f"--- Preview {index}: {row['context_id']} ({row['position_category']}) ---")
        print(build_full_prompt_preview(row["context"]))
        print()


def save_config(
    model_id: str,
    base_url: str,
    parameter_notes: list[str] | None = None,
) -> None:
    config = {
        "model_id": model_id,
        "provider": MODEL_PROVIDER,
        "base_url": base_url,
        "prompt_version": PROMPT_VERSION,
        "instructions": INSTRUCTIONS,
        "prompt_template": PROMPT_TEMPLATE,
        "generation_settings": {
            "temperature": REQUESTED_TEMPERATURE,
            "reasoning_setting": REASONING_SETTING,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "tools": None,
            "web_search": False,
            "retrieval": False,
            "function_calling": False,
            "text_output_only": True,
        },
        "parameter_notes": parameter_notes or [],
        "output_token_limit": MAX_OUTPUT_TOKENS,
        "run_date_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "input_dataset_filename": INPUT_SAMPLE.name,
        "context_count": SAMPLE_SIZE,
    }
    CONFIG_JSON.write_text(json.dumps(config, indent=2), encoding="utf-8")


def response_to_dict(response: Any) -> dict[str, Any]:
    if hasattr(response, "model_dump"):
        return response.model_dump(mode="json")
    if hasattr(response, "to_dict"):
        return response.to_dict()
    return {"repr": repr(response)}


def extract_generated_text(response: Any) -> str:
    choice = response.choices[0]
    content = choice.message.content
    if content is None:
        return ""
    return str(content)


def make_client(base_url: str) -> OpenAI:
    api_key = os.environ.get("LMSTUDIO_API_KEY", DEFAULT_API_KEY)
    return OpenAI(base_url=base_url, api_key=api_key)


def call_model_with_retries(
    client: OpenAI,
    user_input: str,
    parameter_notes: list[str],
    max_retries: int = 6,
) -> tuple[Any | None, int, str | None, dict[str, Any]]:
    retry_count = 0
    last_error: str | None = None
    use_single_user_message = False
    request_kwargs: dict[str, Any] = {
        "model": EXPERIMENT_MODEL_ID,
        "messages": [
            {"role": "system", "content": INSTRUCTIONS},
            {"role": "user", "content": user_input},
        ],
        "temperature": REQUESTED_TEMPERATURE,
        "max_tokens": MAX_OUTPUT_TOKENS,
    }

    while retry_count <= max_retries:
        try:
            if use_single_user_message:
                request_kwargs["messages"] = [
                    {
                        "role": "user",
                        "content": f"{INSTRUCTIONS}\n\n{user_input}",
                    }
                ]
            response = client.chat.completions.create(**request_kwargs)
            generated_text = extract_generated_text(response)
            if not generated_text.strip():
                raise RuntimeError("Model returned empty generated text.")
            return response, retry_count, None, {"parameter_notes": list(parameter_notes)}
        except TypeError as exc:
            message = str(exc)
            if "temperature" in message and "temperature" in request_kwargs:
                parameter_notes.append("temperature unsupported; using API default")
                request_kwargs.pop("temperature", None)
                continue
            return None, retry_count, message, {"parameter_notes": parameter_notes}
        except (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError) as exc:
            last_error = str(exc)
            if retry_count >= max_retries:
                break
            time.sleep(min(60, 2**retry_count))
            retry_count += 1
        except RuntimeError as exc:
            last_error = str(exc)
            if "empty generated text" in last_error and retry_count < max_retries:
                if not use_single_user_message:
                    use_single_user_message = True
                    parameter_notes.append(
                        "empty response with system+user; retried with single user message"
                    )
                time.sleep(min(10, 2**retry_count))
                retry_count += 1
                continue
            return None, retry_count, last_error, {"parameter_notes": parameter_notes}
        except Exception as exc:  # noqa: BLE001
            return None, retry_count, str(exc), {"parameter_notes": parameter_notes}

    return None, retry_count, last_error, {"parameter_notes": parameter_notes}


def load_completed_context_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed: set[str] = set()
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("api_request_status") == "success":
                completed.add(row["context_id"])
    return completed


def append_csv_row(path: Path, row: dict[str, Any]) -> None:
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({col: row.get(col, "") for col in OUTPUT_COLUMNS})


def append_raw_jsonl(record: dict[str, Any]) -> None:
    with RAW_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def verify_model_access(client: OpenAI, parameter_notes: list[str]) -> str:
    probe_input = build_user_input(
        "Patient: Hello.\n\nDoctor: Hi, what brings you in today?\n\nPatient: Headache."
    )
    response, _, error, _ = call_model_with_retries(
        client, probe_input, parameter_notes, max_retries=2
    )
    if response is None:
        raise RuntimeError(
            f"Could not access experiment model `{EXPERIMENT_MODEL_ID}`: {error}"
        )
    actual_model = str(response.model)
    print(f"Experiment model ID: {actual_model}")
    if actual_model != EXPERIMENT_MODEL_ID:
        raise RuntimeError(
            f"API returned model `{actual_model}` but experiment model is `{EXPERIMENT_MODEL_ID}`."
        )
    return actual_model


def run_generation(
    rows: list[dict[str, str]],
    model_id: str,
    parameter_notes: list[str],
    base_url: str,
) -> None:
    client = make_client(base_url)
    completed = load_completed_context_ids(OUTPUT_CSV)
    all_rows: list[dict[str, Any]] = []
    if OUTPUT_CSV.exists():
        with OUTPUT_CSV.open(encoding="utf-8-sig", newline="") as handle:
            all_rows = list(csv.DictReader(handle))

    pending = [row for row in rows if row["context_id"] not in completed]
    print(f"Resume state: {len(completed)} completed, {len(pending)} pending.")

    total_retries = 0
    for index, row in enumerate(pending, start=1):
        context_id = row["context_id"]
        user_input = build_user_input(row["context"])
        timestamp = datetime.now(timezone.utc).isoformat()

        response, retry_count, error_message, call_metadata = call_model_with_retries(
            client, user_input, parameter_notes
        )
        total_retries += retry_count

        output_row: dict[str, Any] = {
            "context_id": context_id,
            "dialogue_id": row["dialogue_id"],
            "specialty": row["specialty"],
            "target_turn_id": row["target_turn_id"],
            "position_category": row["position_category"],
            "normalized_target_position": row["normalized_target_position"],
            "context": row["context"],
            "latest_patient_turn": row["latest_patient_turn"],
            "human_doctor_target": row["human_doctor_target"],
            "model_provider": MODEL_PROVIDER,
            "model_id": model_id,
            "prompt_version": PROMPT_VERSION,
            "run_timestamp_utc": timestamp,
            "requested_temperature": REQUESTED_TEMPERATURE,
            "actual_temperature_if_known": str(REQUESTED_TEMPERATURE),
            "reasoning_setting": REASONING_SETTING,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "generated_doctor_turn": "",
            "input_tokens": "",
            "output_tokens": "",
            "total_tokens": "",
            "api_request_status": "failed",
            "retry_count": retry_count,
            "error_message": error_message or "",
        }

        raw_record: dict[str, Any] = {
            "context_id": context_id,
            "timestamp_utc": timestamp,
            "model_id": model_id,
            "prompt_version": PROMPT_VERSION,
            "base_url": base_url,
            "generation_settings": {
                "requested_temperature": REQUESTED_TEMPERATURE,
                "reasoning_setting": REASONING_SETTING,
                "max_output_tokens": MAX_OUTPUT_TOKENS,
                "parameter_notes": call_metadata.get("parameter_notes", []),
            },
            "status": "failed",
            "retry_count": retry_count,
            "error_message": error_message,
            "raw_generated_text": "",
            "token_usage": {},
        }

        if response is not None:
            generated_text = extract_generated_text(response)
            usage = getattr(response, "usage", None)
            output_row["generated_doctor_turn"] = generated_text
            output_row["model_id"] = str(response.model)
            if usage is not None:
                output_row["input_tokens"] = usage.prompt_tokens
                output_row["output_tokens"] = usage.completion_tokens
                output_row["total_tokens"] = usage.total_tokens
            output_row["api_request_status"] = "success"
            output_row["error_message"] = ""

            raw_record["status"] = "success"
            raw_record["raw_generated_text"] = generated_text
            raw_record["token_usage"] = (
                usage.model_dump(mode="json") if usage is not None and hasattr(usage, "model_dump") else {}
            )
            raw_record["response_object"] = response_to_dict(response)

        append_csv_row(OUTPUT_CSV, output_row)
        append_raw_jsonl(raw_record)
        all_rows = [existing for existing in all_rows if existing.get("context_id") != context_id]
        all_rows.append(output_row)
        print(f"[{index}/{len(pending)}] {context_id}: {output_row['api_request_status']}")

    write_report(all_rows, model_id, parameter_notes, total_retries, base_url)
    final_validation(rows, all_rows)


def write_report(
    rows: list[dict[str, Any]],
    model_id: str,
    parameter_notes: list[str],
    total_retries: int,
    base_url: str,
) -> None:
    successes = [row for row in rows if row.get("api_request_status") == "success"]
    failures = [row for row in rows if row.get("api_request_status") != "success"]
    input_tokens = [int(row["input_tokens"]) for row in successes if row.get("input_tokens")]
    output_tokens = [int(row["output_tokens"]) for row in successes if row.get("output_tokens")]
    empty_responses = sum(
        1 for row in successes if not str(row.get("generated_doctor_turn", "")).strip()
    )

    lines = [
        "# MedGemma Full 1500 Report (NEXT_TURN_V1)",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Operational summary",
        "",
        f"- Requested contexts: {SAMPLE_SIZE}",
        f"- Successful generations: {len(successes)}",
        f"- Failed generations: {len(failures)}",
        f"- Retries: {total_retries}",
        f"- Empty responses: {empty_responses}",
        "",
        "## Model and prompt",
        "",
        f"- Model ID: `{model_id}`",
        f"- Provider: `{MODEL_PROVIDER}`",
        f"- Base URL: `{base_url}`",
        f"- Prompt version: `{PROMPT_VERSION}`",
        "",
        "## Generation settings",
        "",
        f"- Requested temperature: `{REQUESTED_TEMPERATURE}`",
        f"- Reasoning setting: `{REASONING_SETTING}`",
        f"- Max output tokens: `{MAX_OUTPUT_TOKENS}`",
        f"- Tools / web search / retrieval / function calling: disabled",
        "",
    ]
    if parameter_notes:
        lines.extend(["### Parameter notes", ""])
        for note in parameter_notes:
            lines.append(f"- {note}")
        lines.append("")

    if input_tokens:
        lines.extend(
            [
                "## Token usage",
                "",
                f"- Mean input tokens: {mean(input_tokens):.1f}",
                f"- Median input tokens: {median(input_tokens):.1f}",
                f"- Mean output tokens: {mean(output_tokens):.1f}",
                f"- Median output tokens: {median(output_tokens):.1f}",
                f"- Total input tokens: {sum(input_tokens)}",
                f"- Total output tokens: {sum(output_tokens)}",
                "",
            ]
        )

    if failures:
        lines.extend(["## Failures", ""])
        for row in failures:
            lines.append(
                f"- `{row.get('context_id')}`: {row.get('error_message', 'unknown error')}"
            )
        lines.append("")

    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def final_validation(
    pilot_rows: list[dict[str, str]], output_rows: list[dict[str, Any]]
) -> None:
    pilot_context_ids = {row["context_id"] for row in pilot_rows}
    successes = [row for row in output_rows if row.get("api_request_status") == "success"]
    generated_context_ids = [row["context_id"] for row in successes]

    checks = {
        "successful_generations_eq_1500": len(successes) == SAMPLE_SIZE,
        "unique_context_ids_eq_1500": len(set(generated_context_ids)) == SAMPLE_SIZE,
        "one_generation_per_context": len(generated_context_ids) == len(set(generated_context_ids)),
        "all_contexts_from_sample": set(generated_context_ids) == pilot_context_ids,
        "zero_missing_generated_text": all(
            str(row.get("generated_doctor_turn", "")).strip() for row in successes
        ),
    }

    print("\nFinal validation:")
    all_pass = True
    for name, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        all_pass = all_pass and passed

    print(f"\nOverall: {'PASS' if all_pass else 'FAIL'}")
    if not all_pass:
        raise RuntimeError("Final validation failed.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Official MedGemma (LM Studio) full 1500-sample generation using NEXT_TURN_V1."
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Execute LM Studio API calls after local validation.",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("LMSTUDIO_BASE_URL", DEFAULT_BASE_URL),
        help=f"LM Studio OpenAI-compatible base URL (default: {DEFAULT_BASE_URL})",
    )
    args = parser.parse_args()

    rows = load_sample_rows(INPUT_SAMPLE)
    validate_prompts(rows)

    print("Input validation passed:")
    print(f"  - {len(rows)} rows")
    print(f"  - {len(set(r['context_id'] for r in rows))} unique context IDs")
    print(f"  - {len(set(r['target_turn_id'] for r in rows))} unique target-turn IDs")
    print("  - no missing contexts")
    print("  - zero prompts contain forbidden target metadata")
    print("  - human_doctor_target not inserted into API prompts")
    print(f"  - LM Studio base URL: {args.base_url}")
    print(f"  - Model: {EXPERIMENT_MODEL_ID}")

    print_prompt_previews(rows)
    parameter_notes: list[str] = [
        f"LM Studio OpenAI-compatible chat.completions at {args.base_url}",
    ]
    save_config(EXPERIMENT_MODEL_ID, args.base_url, parameter_notes)

    if not args.run:
        print("Local pre-run checks complete. No API calls made.")
        print(f"Saved config to {CONFIG_JSON.name}")
        print("Re-run with --run to execute generation.")
        return

    client = make_client(args.base_url)
    model_id = verify_model_access(client, parameter_notes)
    save_config(model_id, args.base_url, parameter_notes)
    run_generation(rows, model_id, parameter_notes, args.base_url)
    print(f"Full 1500 run complete. Outputs written to {OUTPUT_CSV.name}")


if __name__ == "__main__":
    main()


