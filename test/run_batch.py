from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUN_SKILL_ENGINE_PATH = ROOT / "scripts" / "run_skill_engine.py"
CASE_TEXT_FIELDS = (
    "clinical_presentation",
    "lab_tests",
    "imaging_tests",
    "endoscopy",
    "pathology",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run SkillEngine for patient JSON files.")
    parser.add_argument("--patients-dir", default="test/patients", help="Directory containing patient JSON files.")
    parser.add_argument("--output-dir", default="test/outputs", help="Directory for workflow output JSON files.")
    parser.add_argument("--skills-dir", default="data/skills")
    parser.add_argument("--skill-schema", default="schema/skill_pack.schema.json")
    parser.add_argument("--case-schema", default="schema/canonical_case.schema.json")
    parser.add_argument("--output-schema", default="schema/workflow_output.schema.json")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-score", type=float)
    parser.add_argument("--similarity-threshold", type=_similarity_threshold)
    parser.add_argument("--model-path", default=None)
    feature_group = parser.add_mutually_exclusive_group(required=True)
    feature_group.add_argument("--hpo", action="store_true", help="Use HPO feature extraction.")
    feature_group.add_argument("--icd10", action="store_true", help="Use ICD10 feature extraction.")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--print-canonical-case", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop after the first failed patient.")
    args = parser.parse_args(argv)

    patients_dir = _resolve(args.patients_dir)
    output_dir = _resolve(args.output_dir)
    patient_paths = sorted(patients_dir.glob("*.json"))
    if not patient_paths:
        print(f"run_batch: error: no patient JSON files found in {patients_dir}", file=sys.stderr)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    runner = _load_run_skill_engine()
    feature_mode = "hpo" if args.hpo else "icd10"
    failures: list[tuple[Path, int]] = []

    for patient_path in patient_paths:
        try:
            patient = _read_patient(patient_path)
            output_path = output_dir / _output_filename(patient, patient_path, feature_mode)
            run_args = _build_run_args(args, patient, output_path)
        except ValueError as exc:
            print(f"run_batch: error: {patient_path}: {exc}", file=sys.stderr)
            failures.append((patient_path, 1))
            if args.stop_on_error:
                break
            continue

        print(f"running {patient_path} -> {output_path}")
        exit_code = runner.main(run_args)
        if exit_code:
            failures.append((patient_path, exit_code))
            if args.stop_on_error:
                break

    if failures:
        print(f"run_batch: finished with {len(failures)} failed patient(s)", file=sys.stderr)
        return 1

    print(f"run_batch: finished {len(patient_paths)} patient(s)")
    return 0


def _load_run_skill_engine() -> Any:
    spec = importlib.util.spec_from_file_location("run_skill_engine", RUN_SKILL_ENGINE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {RUN_SKILL_ENGINE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_patient(path: Path) -> dict[str, Any]:
    try:
        patient = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {exc}") from exc
    if not isinstance(patient, dict):
        raise ValueError("patient file must contain a JSON object")

    for field in ("name", "sex", "age"):
        if patient.get(field) in (None, ""):
            raise ValueError(f"missing required field: {field}")
    if patient["sex"] not in {"male", "female", "unknown"}:
        raise ValueError("sex must be one of: male, female, unknown")
    try:
        patient["age"] = int(patient["age"])
    except (TypeError, ValueError) as exc:
        raise ValueError("age must be an integer") from exc
    if not any(_clean_text(patient.get(field)) for field in CASE_TEXT_FIELDS):
        raise ValueError("at least one case text field is required")
    return patient


def _build_run_args(args: argparse.Namespace, patient: dict[str, Any], output_path: Path) -> list[str]:
    run_args = [
        "--skills-dir",
        args.skills_dir,
        "--skill-schema",
        args.skill_schema,
        "--case-schema",
        args.case_schema,
        "--output-schema",
        args.output_schema,
        "--name",
        _clean_text(patient["name"]),
        "--sex",
        patient["sex"],
        "--age",
        str(patient["age"]),
        "--top-k",
        str(args.top_k),
        "--output",
        str(output_path),
    ]
    for field in CASE_TEXT_FIELDS:
        value = _clean_text(patient.get(field))
        if value:
            run_args.extend((f"--{field.replace('_', '-')}", value))
    if args.min_score is not None:
        run_args.extend(("--min-score", str(args.min_score)))
    if args.similarity_threshold is not None:
        run_args.extend(("--similarity-threshold", str(args.similarity_threshold)))
    if args.model_path:
        run_args.extend(("--model-path", args.model_path))
    if args.debug:
        run_args.append("--debug")
    if args.print_canonical_case:
        run_args.append("--print-canonical-case")
    run_args.append("--hpo" if args.hpo else "--icd10")
    return run_args


def _output_filename(patient: dict[str, Any], patient_path: Path, feature_mode: str) -> str:
    case_id = _clean_text(patient.get("case_id")) or patient_path.stem
    name = _clean_text(patient.get("name")) or patient_path.stem
    return f"{_slug(case_id)}_{_slug(name)}_{feature_mode}.json"


def _slug(value: str) -> str:
    slug = re.sub(r"[^\w\u4e00-\u9fff.-]+", "_", value.strip(), flags=re.UNICODE)
    return slug.strip("._") or "patient"


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _similarity_threshold(value: str) -> float:
    try:
        threshold = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number between 0 and 1") from exc
    if not 0 <= threshold <= 1:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return threshold


if __name__ == "__main__":
    raise SystemExit(main())
