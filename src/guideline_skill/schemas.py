from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


EvidenceQualityNormalized = Literal["high", "moderate", "low", "very_low", "unknown"]
StrengthNormalized = Literal[
    "strong",
    "weak",
    "best_practice_statement",
    "consensus_statement",
    "unknown",
]
ClinicalInfoUnitType = Literal[
    "definition",
    "classification",
    "clinical_manifestation",
    "diagnostic_criteria",
    "test_order",
    "instrumental_exam",
    "imaging_exam",
    "endoscopy_exam",
    "drug_regimen",
    "indication",
    "contraindication",
    "differential_diagnosis",
    "medical_record_writing",
    "surgery",
    "prognosis",
    "knowledge",
    "other",
]
ClinicalTopic = Literal[
    "background",
    "diagnosis",
    "treatment",
    "surgery",
    "follow_up",
    "documentation",
    "prognosis",
    "other",
]


class GuidelineMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str | None = None
    source_file: str | None = None
    doc_type: str


class SourceLocation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    page_start: int | None = None
    page_end: int | None = None
    section: str | None = None


class StatementEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_quality_raw: str | None = None
    evidence_quality_normalized: EvidenceQualityNormalized | None = None
    recommendation_strength_raw: str | None = None
    recommendation_strength_normalized: StrengthNormalized | None = None
    consensus_level: str | None = None


class StatementUnitBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    original_label: str
    statement_type: str
    statement_text: str
    clinical_question: str | None = None
    evidence_quality_raw: str | None = None
    evidence_quality_normalized: EvidenceQualityNormalized | None = None
    # TODO: 需要处理下BPS的情况
    strength_raw: str | None = None
    strength_normalized: StrengthNormalized | None = None
    consensus_level: str | None = None
    implementation_advice: str | None = None
    rationale: str | None = None
    source_location: SourceLocation
    # TODO：BPS情况下为0，后续需要处理（可以考虑拆分为多个置信度）
    confidence: float = Field(ge=0.0, le=1.0)
    needs_human_review: bool
    review_reasons: list[str] = Field(default_factory=list)


class StatementUnit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    record_type: Literal["recommendation_card"] = "recommendation_card"
    guideline: GuidelineMeta
    card_id: str
    source_statement_id: str
    disease: str
    statement_type: str
    statement_text: str
    clinical_question: str | None = None
    clinical_stage: str
    clinical_task: str = ""
    population: str | None = None
    condition: str | None = None
    action: str
    do_not: list[str] = Field(default_factory=list)
    required_inputs: list[str] = Field(default_factory=list)
    supporting_features: list[str] = Field(default_factory=list)
    recommended_tests: list[str] = Field(default_factory=list)
    evidence: StatementEvidence
    implementation_advice: str | None = None
    rationale: str | None = None
    source_location: SourceLocation
    confidence: float = Field(ge=0.0, le=1.0)
    needs_human_review: bool
    review_reasons: list[str] = Field(default_factory=list)

    def to_json(self, **kwargs: object) -> str:
        return json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            **kwargs,
        )


class ClinicalInfoUnitBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    section_path: list[str] = Field(default_factory=list)
    title: str | None = None
    raw_text: str
    unit_type: ClinicalInfoUnitType
    clinical_topic: ClinicalTopic | None = None
    action: str | None = None
    condition: str | None = None
    indication: list[str] = Field(default_factory=list)
    contraindication: list[str] = Field(default_factory=list)
    diagnostic_criteria: list[str] = Field(default_factory=list)
    differential_diagnosis: list[str] = Field(default_factory=list)
    drug: str | None = None
    dose: str | None = None
    route: str | None = None
    frequency: str | None = None
    duration: str | None = None
    source_location: SourceLocation
    confidence: float = Field(ge=0.0, le=1.0)
    needs_human_review: bool
    review_reasons: list[str] = Field(default_factory=list)


class ClinicalInfoUnit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    record_type: Literal["clinical_info_unit"] = "clinical_info_unit"
    guideline_meta: GuidelineMeta
    unit: ClinicalInfoUnitBody

    def to_json(self, **kwargs: object) -> str:
        return json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            **kwargs,
        )
