from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "ontology" / "hp-zh.babelon.tsv"
DEFAULT_OUTPUT = ROOT / "data" / "ontology" / "definition2id.json"


def build_definition2id(input_path: Path) -> dict[str, list[str]]:
    definition2id: dict[str, list[str]] = {}

    with input_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file, delimiter="\t")
        required_columns = {"subject_id", "translation_value"}
        missing_columns = required_columns.difference(reader.fieldnames or [])
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Missing required TSV column(s): {missing}")

        for line_no, row in enumerate(reader, start=2):
            subject_id = (row.get("subject_id") or "").strip()
            translation_value = (row.get("translation_value") or "").strip()
            if not subject_id or not translation_value:
                continue

            subject_ids = definition2id.setdefault(translation_value, [])
            if subject_id not in subject_ids:
                subject_ids.append(subject_id)

    return definition2id


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Build definition2id.json from HPO Babelon Chinese translations."
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Input Babelon TSV path.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output JSON path.")
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    output_path = Path(args.output)
    definition2id = build_definition2id(input_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(definition2id, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(definition2id)} definitions to {output_path}")
    multi_id_count = sum(1 for subject_ids in definition2id.values() if len(subject_ids) > 1)
    print(f"Found {multi_id_count} definitions with multiple IDs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
