from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from guideline_skill.schema import EvidenceReference, SkillExecutionResult, load_skill_pack
from skill_quality_check import check_skill_quality, validate_source_references


ROOT = Path(__file__).resolve().parents[1]
CROHN_SEED_PATH = ROOT / "data" / "skills" / "crohn_disease_2023_guangzhou.yaml"


def test_seed_skill_quality_passes() -> None:
    skill_pack = load_skill_pack(CROHN_SEED_PATH)

    report = check_skill_quality(skill_pack)

    assert report.passed
    assert report.errors == []
    assert report.summary["subskill_count"] == 6
    assert report.summary["recommendation_card_count"] == 12
    assert report.summary["differential_diagnosis_count"] >= 6


def test_quality_check_detects_duplicate_recommendation_ids() -> None:
    skill_pack = load_skill_pack(CROHN_SEED_PATH).model_copy(deep=True)
    skill_pack.recommendation_cards.append(skill_pack.recommendation_cards[0].model_copy(deep=True))

    report = check_skill_quality(skill_pack)

    assert not report.passed
    assert any(error.code == "duplicate_recommendation_id" for error in report.errors)


def test_quality_check_detects_unknown_source_references() -> None:
    skill_pack = load_skill_pack(CROHN_SEED_PATH)
    result = SkillExecutionResult(
        skill_name=skill_pack.skill_name,
        disease_name=skill_pack.disease_name,
        suspicion_level="possible",
        source_references=[
            EvidenceReference(
                source_name="bad",
                recommendation_id="UNKNOWN-REC",
            )
        ],
    )

    report = check_skill_quality(skill_pack, skill_results=[result])

    assert not report.passed
    assert any(error.code == "unknown_source_reference" for error in report.errors)


def test_validate_source_references_helper() -> None:
    skill_pack = load_skill_pack(CROHN_SEED_PATH)
    issues = validate_source_references(
        skill_pack,
        [
            EvidenceReference(source_name="ok", recommendation_id="CD-REC-001"),
            EvidenceReference(source_name="bad", recommendation_id="NOPE"),
        ],
    )

    assert len(issues) == 1
    assert issues[0].code == "unknown_source_reference"


def test_check_skill_quality_cli_outputs_json() -> None:
    script = ROOT / "scripts" / "check_skill_quality.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--skill", str(CROHN_SEED_PATH)],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    payload = json.loads(completed.stdout)

    assert payload["passed"] is True
    assert payload["errors"] == []
    assert payload["summary"]["skill_name"] == "crohn_disease_2023_guangzhou"
