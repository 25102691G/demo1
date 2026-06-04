from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator


def validate_jsonl(jsonl_path: str | Path, schema_path: str | Path) -> int:
    schema = _read_json(schema_path)
    validator = Draft202012Validator(schema)

    error_count = 0
    record_count = 0
    for line_number, line in enumerate(Path(jsonl_path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record_count += 1
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            print(f"line {line_number}: invalid JSON: {exc}")
            error_count += 1
            continue

        errors = sorted(validator.iter_errors(record), key=lambda error: list(error.path))
        for error in errors:
            path = ".".join(str(part) for part in error.path) or "<root>"
            print(f"line {line_number}: {path}: {error.message}")
            error_count += 1

    if error_count:
        print(f"FAILED: {error_count} validation error(s) across {record_count} record(s)")
        return 1

    print(f"OK: {record_count} record(s) match schema")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate recommendation card JSONL records against a JSON Schema.")
    parser.add_argument("--jsonl", required=True, help="Path to result.jsonl.")
    parser.add_argument(
        "--schema",
        default="schema/recommendation_card.schema.json",
        help="Path to recommendation_card.schema.json.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    return validate_jsonl(args.jsonl, args.schema)


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Schema must be a JSON object: {path}")
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
