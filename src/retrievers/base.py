from __future__ import annotations

from abc import ABC, abstractmethod

from guideline_skill.schema import (
    DifferentialDiagnosisItem,
    DiseaseSkillPack,
    PatientCase,
    RecommendationCard,
)


class GuidelineRetriever(ABC):
    """Common evidence retrieval interface for local, GraphRAG, or KG backends."""

    @abstractmethod
    def retrieve_by_patient_case(
        self,
        patient_case: PatientCase,
        skill_pack: DiseaseSkillPack,
        top_k: int = 5,
    ) -> list[RecommendationCard]:
        raise NotImplementedError

    @abstractmethod
    def retrieve_by_query(
        self,
        query: str,
        skill_pack: DiseaseSkillPack,
        top_k: int = 5,
    ) -> list[RecommendationCard]:
        raise NotImplementedError

    @abstractmethod
    def retrieve_differential_diagnoses(
        self,
        patient_case: PatientCase,
        skill_pack: DiseaseSkillPack,
        top_k: int = 5,
    ) -> list[DifferentialDiagnosisItem]:
        raise NotImplementedError
