from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SKILLS_DIR = ROOT / "data" / "skills"
CROHN_CASES_PATH = ROOT / "data" / "eval" / "crohn_cases.yaml"


def test_crohn_eval_cases_file_contains_expanded_scenarios() -> None:
    payload = yaml.safe_load(CROHN_CASES_PATH.read_text(encoding="utf-8"))
    cases = payload["cases"]
    case_ids = {case["case_id"] for case in cases}

    assert len(cases) >= 20
    assert {
        "symptoms_only_abdominal_pain_diarrhea",
        "ileocecal_ulcers_stricture",
        "igra_positive_tb_differential",
        "doctor_confirmed_asks_treatment",
        "near_probable_complete_workup",
    }.issubset(case_ids)


def test_evaluate_cases_cli_passes_all_crohn_cases() -> None:
    script = ROOT / "scripts" / "evaluate_cases.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--skills",
            str(SKILLS_DIR),
            "--cases",
            str(CROHN_CASES_PATH),
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    payload = json.loads(completed.stdout)

    assert payload["passed"] is True
    assert payload["summary"]["case_count"] >= 20
    assert payload["summary"]["failed_count"] == 0
    assert all(case["passed"] for case in payload["cases"])
