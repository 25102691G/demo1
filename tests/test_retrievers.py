from __future__ import annotations

from pathlib import Path

import pytest

from guideline_skill.schema import (
    DifferentialDiagnosisItem,
    DiseaseSkillPack,
    PatientCase,
    RecommendationCard,
    load_skill_pack,
)
from patient_case_extractor import extract_patient_case
from retrievers.base import GuidelineRetriever
from retrievers.graph_retriever import GraphRetriever
from retrievers.local_recommendation_retriever import LocalRecommendationRetriever
from skill_executor import CrohnDiseaseSkillExecutor


ROOT = Path(__file__).resolve().parents[1]
CROHN_SEED_PATH = ROOT / "data" / "skills" / "crohn_disease_2023_guangzhou.yaml"


def test_local_retriever_retrieves_by_patient_case() -> None:
    skill_pack = load_skill_pack(CROHN_SEED_PATH)
    patient_case = extract_patient_case("肠镜提示回肠末段病变，并已进行多肠段活检。")

    cards = LocalRecommendationRetriever().retrieve_by_patient_case(
        patient_case,
        skill_pack,
        top_k=3,
    )

    assert cards
    assert cards[0].recommendation_id == "CD-REC-003"


def test_local_retriever_retrieves_by_query() -> None:
    skill_pack = load_skill_pack(CROHN_SEED_PATH)

    cards = LocalRecommendationRetriever().retrieve_by_query(
        "肛瘘 肛周脓肿 肛周 MRI",
        skill_pack,
        top_k=3,
    )

    assert cards[0].recommendation_id == "CD-REC-007"


def test_local_retriever_retrieves_differential_diagnoses() -> None:
    skill_pack = load_skill_pack(CROHN_SEED_PATH)
    patient_case = extract_patient_case("腹痛腹泻，回盲部溃疡。")

    diagnoses = LocalRecommendationRetriever().retrieve_differential_diagnoses(
        patient_case,
        skill_pack,
        top_k=6,
    )

    names = {item.disease_name for item in diagnoses}
    assert {"肠结核", "溃疡性结肠炎", "感染性肠炎"}.issubset(names)


def test_graph_retriever_interface_is_reserved() -> None:
    skill_pack = load_skill_pack(CROHN_SEED_PATH)
    patient_case = extract_patient_case("腹痛腹泻。")
    retriever = GraphRetriever()

    with pytest.raises(NotImplementedError):
        retriever.retrieve_by_patient_case(patient_case, skill_pack)
    with pytest.raises(NotImplementedError):
        retriever.retrieve_by_query("腹痛", skill_pack)
    with pytest.raises(NotImplementedError):
        retriever.retrieve_differential_diagnoses(patient_case, skill_pack)


def test_executor_can_work_through_guideline_retriever_interface() -> None:
    skill_pack = load_skill_pack(CROHN_SEED_PATH)
    patient_case = extract_patient_case("腹痛腹泻。")
    retriever = StaticRetriever(skill_pack.recommendation_cards[0])

    result = CrohnDiseaseSkillExecutor(retriever=retriever).execute(
        patient_case,
        skill_pack,
    )

    assert result.source_references[0].recommendation_id == "CD-REC-001"
    assert result.recommended_next_steps[0].startswith("[CD-REC-001]")
    assert result.differential_diagnoses[0].disease_name == "接口测试鉴别诊断"


class StaticRetriever(GuidelineRetriever):
    def __init__(self, card: RecommendationCard) -> None:
        self.card = card

    def retrieve_by_patient_case(
        self,
        patient_case: PatientCase,
        skill_pack: DiseaseSkillPack,
        top_k: int = 5,
    ) -> list[RecommendationCard]:
        return [self.card]

    def retrieve_by_query(
        self,
        query: str,
        skill_pack: DiseaseSkillPack,
        top_k: int = 5,
    ) -> list[RecommendationCard]:
        return [self.card]

    def retrieve_differential_diagnoses(
        self,
        patient_case: PatientCase,
        skill_pack: DiseaseSkillPack,
        top_k: int = 5,
    ) -> list[DifferentialDiagnosisItem]:
        return [
            DifferentialDiagnosisItem(
                disease_name="接口测试鉴别诊断",
                rationale="Used to prove executor works through GuidelineRetriever.",
            )
        ]
