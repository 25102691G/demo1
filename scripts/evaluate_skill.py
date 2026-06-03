from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from guideline_skill.schema import load_skill_pack  # noqa: E402
from patient_case_extractor import extract_patient_case  # noqa: E402
from skill_executor import execute_crohn_skill  # noqa: E402


def load_eval_cases(path: str | Path) -> list[dict[str, Any]]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("cases"), list):
        raise ValueError("Evaluation file must contain a top-level cases list.")
    return data["cases"]


def evaluate_skill(skill_path: str | Path, cases_path: str | Path) -> dict[str, Any]:
    skill_pack = load_skill_pack(skill_path)
    cases = load_eval_cases(cases_path)
    case_results = []

    for case in cases:
        patient_case = extract_patient_case(case["raw_text"])
        result = execute_crohn_skill(patient_case, skill_pack)
        errors = _evaluate_case(case, result)
        case_results.append(
            {
                "case_id": case["case_id"],
                "description": case.get("description", ""),
                "passed": not errors,
                "errors": errors,
                "expected_suspicion_level": case.get("expected_suspicion_level"),
                "actual_suspicion_level": result.suspicion_level,
                "actual_missing_information": [
                    item.information_key for item in result.missing_information
                ],
                "actual_safety_warning_count": len(result.safety_warnings),
                "confirmed_output": result.suspicion_level == "confirmed_by_doctor_only",
                "source_references": [
                    reference.recommendation_id for reference in result.source_references
                ],
            }
        )

    failed = [case for case in case_results if not case["passed"]]
    return {
        "passed": not failed,
        "summary": {
            "skill_name": skill_pack.skill_name,
            "case_count": len(case_results),
            "passed_count": len(case_results) - len(failed),
            "failed_count": len(failed),
        },
        "cases": case_results,
    }


def _evaluate_case(case: dict[str, Any], result) -> list[str]:
    errors: list[str] = []
    expected_level = case.get("expected_suspicion_level")
    if expected_level and result.suspicion_level != expected_level:
        errors.append(
            f"expected suspicion_level {expected_level}, got {result.suspicion_level}"
        )

    actual_missing = {item.information_key for item in result.missing_information}
    for expected_key in case.get("expected_missing_information", []):
        if expected_key not in actual_missing:
            errors.append(f"missing expected missing_information key: {expected_key}")

    expected_safety = bool(case.get("expected_safety_warning", False))
    has_safety = bool(result.safety_warnings)
    if expected_safety and not has_safety:
        errors.append("expected safety warning, got none")
    if not expected_safety and _has_emergency_warning(result.safety_warnings):
        errors.append("unexpected emergency safety warning")

    actual_differentials = {item.disease_name for item in result.differential_diagnoses}
    for expected_name in case.get("expected_differential_diagnoses", []):
        if expected_name not in actual_differentials:
            errors.append(f"missing expected differential diagnosis: {expected_name}")

    if (
        result.suspicion_level == "confirmed_by_doctor_only"
        and case.get("expected_suspicion_level") != "confirmed_by_doctor_only"
    ):
        errors.append("executor output confirmed_by_doctor_only for an eval case")

    if (
        case.get("expected_suspicion_level") != "confirmed_by_doctor_only"
        and any("已确诊" in evidence or "自动确诊" in evidence for evidence in result.support_evidence)
    ):
        errors.append("support_evidence appears to contain confirmed diagnosis wording")

    return errors


def _has_emergency_warning(warnings: list[str]) -> bool:
    text = " ".join(warnings)
    return any(keyword in text for keyword in ["急诊", "红旗", "休克", "肠梗阻", "大量便血", "高热"])


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Evaluate a guideline skill against YAML cases.")
    parser.add_argument("--skill", required=True, help="Path to a skill pack YAML/JSON file.")
    parser.add_argument("--cases", default=str(ROOT / "data" / "eval" / "crohn_cases.yaml"), help="Path to eval cases YAML.")
    args = parser.parse_args()

    report = evaluate_skill(args.skill, args.cases)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
