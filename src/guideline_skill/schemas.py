from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


EvidenceQualityNormalized = float
StrengthNormalized = float
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


class StatementEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_quality_raw: str | None = None
    evidence_quality_normalized: EvidenceQualityNormalized = 0.5
    recommendation_strength_raw: str | None = None
    recommendation_strength_normalized: StrengthNormalized = 0.5
    consensus_level: str | None = None


class StatementUnitBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    original_label: str
    statement_text: str
    evidence_quality_raw: str | None = None
    evidence_quality_normalized: EvidenceQualityNormalized = 0.5
    strength_raw: str | None = None
    strength_normalized: StrengthNormalized = 0.5
    consensus_level: str | None = None
    source_location: SourceLocation


class StatementUnit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    record_type: Literal["recommendation_card"] = "recommendation_card"
    guideline: GuidelineMeta
    card_id: str
    source_statement_id: str
    disease: str
    statement_text: str
    raw_chunk_text: str
    clinical_stage: str
    clinical_task: str = ""
    population: str | None = None
    condition: str | None = None
    action: str
    required_inputs: list[str] = Field(default_factory=list)
    evidence: StatementEvidence
    source_location: SourceLocation

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
