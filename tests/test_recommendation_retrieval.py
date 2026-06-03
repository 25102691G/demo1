from __future__ import annotations

from pathlib import Path

from guideline_skill.schema import load_skill_pack
from patient_case_extractor import extract_patient_case
from skill_executor import (
    execute_crohn_skill,
    retrieve_recommendation_cards,
    score_recommendation_card,
)


ROOT = Path(__file__).resolve().parents[1]
CROHN_SEED_PATH = ROOT / "data" / "skills" / "crohn_disease_2023_guangzhou.yaml"


def _retrieve(text: str):
    patient_case = extract_patient_case(text)
    skill_pack = load_skill_pack(CROHN_SEED_PATH)
    return retrieve_recommendation_cards(patient_case, skill_pack, top_k=5)


def test_colonoscopy_terminal_ileum_and_biopsy_hits_colonoscopy_card() -> None:
    cards = _retrieve("肠镜提示回肠末段病变，并已进行多肠段活检。")

    assert cards[0].recommendation_id == "CD-REC-003"
    assert "colonoscopy" in cards[0].clinical_task


def test_capsule_endoscopy_and_stricture_hits_retention_risk_card() -> None:
    cards = _retrieve("拟行胶囊内镜，但既往提示小肠狭窄，需要评估胶囊滞留风险。")

    assert cards[0].recommendation_id == "CD-REC-005"
    assert "retention risk" in cards[0].rationale


def test_cte_mre_hits_extent_and_complication_card() -> None:
    cards = _retrieve("已完成 CTE/MRE，用于评估小肠病变范围和并发症。")

    assert cards[0].recommendation_id == "CD-REC-006"
    assert "extent and complication" in cards[0].clinical_task


def test_perianal_fistula_or_abscess_hits_perianal_mri_card() -> None:
    cards = _retrieve("患者有肛瘘和肛周脓肿，肛周疼痛明显。")

    assert cards[0].recommendation_id == "CD-REC-007"
    assert "perianal" in cards[0].clinical_task


def test_score_recommendation_card_prefers_contextual_card() -> None:
    patient_case = extract_patient_case("结肠镜提示回肠末段纵行溃疡，并完成活检。")
    skill_pack = load_skill_pack(CROHN_SEED_PATH)
    card_by_id = {card.recommendation_id: card for card in skill_pack.recommendation_cards}

    colonoscopy_score = score_recommendation_card(patient_case, card_by_id["CD-REC-003"])
    capsule_score = score_recommendation_card(patient_case, card_by_id["CD-REC-005"])

    assert colonoscopy_score > capsule_score


def test_executor_uses_retrieved_cards_for_sources_and_next_steps() -> None:
    patient_case = extract_patient_case("拟行胶囊内镜，但既往提示小肠狭窄，需要评估胶囊滞留风险。")
    skill_pack = load_skill_pack(CROHN_SEED_PATH)

    result = execute_crohn_skill(patient_case, skill_pack)
    source_ids = [reference.recommendation_id for reference in result.source_references]

    assert "CD-REC-005" in source_ids
    assert any(step.startswith("[CD-REC-005]") for step in result.recommended_next_steps)
    assert result.source_references
