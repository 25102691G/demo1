from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from guideline_skill.schema import load_skill_pack, validate_skill_pack


ROOT = Path(__file__).resolve().parents[1]
SEED_PATH = ROOT / "data" / "skills" / "crohn_disease_2023_guangzhou.yaml"


def test_crohn_seed_skill_pack_loads_and_validates() -> None:
    skill_pack = load_skill_pack(SEED_PATH)
    validated = validate_skill_pack(skill_pack)

    assert validated.skill_name == "crohn_disease_2023_guangzhou"
    assert validated.disease_name == "Crohn disease"
    assert len(validated.subskills) == 6
    assert len(validated.recommendation_cards) == 12


def test_crohn_seed_skill_pack_contains_required_subskills() -> None:
    skill_pack = load_skill_pack(SEED_PATH)

    subskill_ids = {subskill.subskill_id for subskill in skill_pack.subskills}

    assert subskill_ids == {
        "cd_initial_screening",
        "cd_diagnostic_workup",
        "cd_differential_diagnosis",
        "cd_extent_and_activity_assessment",
        "cd_treatment_readiness_check",
        "cd_followup_monitoring",
    }


def test_crohn_seed_recommendations_cover_required_topics() -> None:
    skill_pack = load_skill_pack(SEED_PATH)
    cards_text = "\n".join(
        " ".join(
            [
                card.recommendation_id,
                card.source_section,
                card.clinical_task,
                card.condition,
                card.action,
                card.rationale,
                " ".join(card.safety_notes),
            ]
        )
        for card in skill_pack.recommendation_cards
    )

    assert "gold standard" in cards_text
    assert "fecal calprotectin" in cards_text
    assert "negative result" in cards_text
    assert "terminal ileum" in cards_text
    assert "gastroduodenoscopy" in cards_text
    assert "capsule endoscopy" in cards_text
    assert "retention risk" in cards_text
    assert "MRE or CTE" in cards_text
    assert "perianal MRI" in cards_text
    assert "intestinal tuberculosis" in cards_text
    assert "drug-induced enteritis" in cards_text
    assert "final diagnosis" in cards_text
    assert "Treatment" in cards_text or "treatment" in cards_text


def test_validate_skill_pack_script_reports_summary() -> None:
    script = ROOT / "scripts" / "validate_skill_pack.py"

    completed = subprocess.run(
        [sys.executable, str(script), "--input", str(SEED_PATH)],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    assert "skill_name: crohn_disease_2023_guangzhou" in completed.stdout
    assert "disease_name: Crohn disease" in completed.stdout
    assert "subskill_count: 6" in completed.stdout
    assert "recommendation_card_count: 12" in completed.stdout
    assert "routing_keywords_count:" in completed.stdout
    assert "validation: pass" in completed.stdout
