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

from agent_orchestrator import run_agent  # noqa: E402


def load_eval_cases(path: str | Path) -> list[dict[str, Any]]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("cases"), list):
        raise ValueError("Evaluation file must contain a top-level cases list.")
    return payload["cases"]


def evaluate_cases(
    *,
    skills_path: str | Path,
    cases_path: str | Path,
    top_k: int = 5,
) -> dict[str, Any]:
    case_reports = []
    for case in load_eval_cases(cases_path):
        response = run_agent(case["raw_text"], skills_path=skills_path, top_k=top_k)
        errors = evaluate_one_case(case, response)
        top_result = response.skill_results[0] if response.skill_results else None
        case_reports.append(
            {
                "case_id": case["case_id"],
                "description": case.get("description", ""),
                "passed": not errors,
                "errors": errors,
                "expected_stages": case.get("expected_stages", []),
                "actual_stages": response.clinical_stage.stages,
                "expected_suspicion_level": case.get("expected_suspicion_level"),
                "actual_suspicion_level": top_result.suspicion_level if top_result else None,
                "actual_top_missing_information": [
                    item.information_key for item in response.top_missing_information
                ],
                "actual_next_steps": response.recommended_next_steps,
                "actual_differential_diagnoses": [
                    item.disease_name
                    for item in (top_result.differential_diagnoses if top_result else [])
                ],
                "source_reference_pages": [
                    reference.page
                    for reference in (top_result.source_references if top_result else [])
                ],
                "final_assessment": response.final_assessment,
            }
        )

    failed = [report for report in case_reports if not report["passed"]]
    return {
        "passed": not failed,
        "summary": {
            "case_count": len(case_reports),
            "passed_count": len(case_reports) - len(failed),
            "failed_count": len(failed),
        },
        "cases": case_reports,
    }


def evaluate_one_case(case: dict[str, Any], response) -> list[str]:
    errors: list[str] = []
    top_result = response.skill_results[0] if response.skill_results else None
    if top_result is None:
        return ["no skill result returned"]

    expected_stages = set(case.get("expected_stages", []))
    actual_stages = set(response.clinical_stage.stages)
    missing_stages = sorted(expected_stages - actual_stages)
    if missing_stages:
        errors.append(f"missing expected stages: {', '.join(missing_stages)}")

    expected_level = case.get("expected_suspicion_level")
    if expected_level and top_result.suspicion_level != expected_level:
        errors.append(
            f"expected suspicion_level {expected_level}, got {top_result.suspicion_level}"
        )

    visible_missing_keys = {
        item.information_key
        for item in [
            *response.full_missing_information,
            *response.top_missing_information,
            *top_result.full_missing_information,
            *top_result.top_missing_information,
            *top_result.missing_information,
        ]
    }
    for key in case.get("must_include_missing_information", []):
        if key not in visible_missing_keys:
            errors.append(f"missing expected missing_information key: {key}")

    joined_steps = "\n".join(response.recommended_next_steps + top_result.recommended_next_steps)
    for expected in case.get("must_include_next_steps", []):
        if expected not in joined_steps:
            errors.append(f"missing expected next step text: {expected}")

    actual_differentials = {item.disease_name for item in top_result.differential_diagnoses}
    for expected in case.get("must_include_differential_diagnoses", []):
        if expected not in actual_differentials:
            errors.append(f"missing expected differential diagnosis: {expected}")

    visible_payload = json.dumps(
        {
            "final_assessment": response.final_assessment,
            "readable_summary": response.readable_summary,
            "recommended_next_steps": response.recommended_next_steps,
            "top_missing_information": [
                item.model_dump(mode="json") for item in response.top_missing_information
            ],
            "skill_result_recommended_next_steps": top_result.recommended_next_steps,
            "skill_result_missing_information": [
                item.model_dump(mode="json") for item in top_result.missing_information
            ],
        },
        ensure_ascii=False,
    )
    for forbidden in case.get("must_not_include", []):
        if forbidden and forbidden in visible_payload:
            errors.append(f"forbidden text found in visible output: {forbidden}")

    if any(reference.page is None for reference in top_result.source_references):
        errors.append("source_references contain null page")

    if len(response.top_missing_information) > 7:
        errors.append("top_missing_information has more than 7 items")

    return errors


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Evaluate Agent responses against YAML cases.")
    parser.add_argument("--skills", default=str(ROOT / "data" / "skills"), help="Skill pack file or directory.")
    parser.add_argument("--cases", default=str(ROOT / "data" / "eval" / "crohn_cases.yaml"), help="Evaluation cases YAML.")
    parser.add_argument("--top-k", type=int, default=5, help="Router top-k.")
    args = parser.parse_args()

    report = evaluate_cases(skills_path=args.skills, cases_path=args.cases, top_k=args.top_k)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
