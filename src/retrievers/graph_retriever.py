from __future__ import annotations

from guideline_skill.schema import (
    DifferentialDiagnosisItem,
    DiseaseSkillPack,
    PatientCase,
    RecommendationCard,
)

from .base import GuidelineRetriever


class GraphRetriever(GuidelineRetriever):
    """Placeholder for a future Neo4j / GraphRAG-backed guideline retriever."""

    def __init__(self, graph_client: object | None = None) -> None:
        self.graph_client = graph_client

    def retrieve_by_patient_case(
        self,
        patient_case: PatientCase,
        skill_pack: DiseaseSkillPack,
        top_k: int = 5,
    ) -> list[RecommendationCard]:
        raise NotImplementedError("GraphRetriever.retrieve_by_patient_case is reserved for GraphRAG/Neo4j integration.")

    def retrieve_by_query(
        self,
        query: str,
        skill_pack: DiseaseSkillPack,
        top_k: int = 5,
    ) -> list[RecommendationCard]:
        raise NotImplementedError("GraphRetriever.retrieve_by_query is reserved for GraphRAG/Neo4j integration.")

    def retrieve_differential_diagnoses(
        self,
        patient_case: PatientCase,
        skill_pack: DiseaseSkillPack,
        top_k: int = 5,
    ) -> list[DifferentialDiagnosisItem]:
        raise NotImplementedError("GraphRetriever.retrieve_differential_diagnoses is reserved for GraphRAG/Neo4j integration.")
