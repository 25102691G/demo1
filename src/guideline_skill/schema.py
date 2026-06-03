from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, Mapping

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


SuspicionLevel = Literal[
    "unlikely",
    "possible",
    "suspected",
    "probable",
    "confirmed_by_doctor_only",
]


class SchemaModel(BaseModel):
    """Strict base model so schema typos fail early."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class EvidenceReference(SchemaModel):
    source_name: str = Field(..., min_length=1)
    source_type: str = Field(default="guideline", min_length=1)
    recommendation_id: str | None = None
    source_section: str | None = None
    source_section_cn: str | None = None
    source_span: str | None = None
    source_quote: str | None = None
    evidence_level: str | None = None
    recommendation_strength: str | None = None
    page: int | None = Field(default=None, ge=1)
    url: str | None = None


class MissingInformationItem(SchemaModel):
    information_key: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1)
    reason: str = Field(..., min_length=1)
    priority: Literal["low", "medium", "high", "critical"] = "medium"


class DifferentialDiagnosisItem(SchemaModel):
    disease_name: str = Field(..., min_length=1)
    rationale: str = Field(..., min_length=1)
    supporting_features: list[str] = Field(default_factory=list)
    against_features: list[str] = Field(default_factory=list)
    distinguishing_tests: list[str] = Field(default_factory=list)
    missing_tests: list[str] = Field(default_factory=list)
    urgency: Literal["routine", "soon", "urgent", "emergency"] = "routine"


class RoutingProfile(SchemaModel):
    body_system: str = Field(..., min_length=1)
    key_symptoms: list[str] = Field(default_factory=list)
    key_tests: list[str] = Field(default_factory=list)
    key_findings: list[str] = Field(default_factory=list)
    red_flags: list[str] = Field(default_factory=list)
    must_differentiate: list[str] = Field(default_factory=list)
    disease_aliases: list[str] = Field(default_factory=list)


class RecommendationCard(SchemaModel):
    recommendation_id: str = Field(..., min_length=1)
    source_section: str = Field(..., min_length=1)
    source_section_cn: str | None = None
    clinical_stage: str | None = None
    clinical_task: str = Field(..., min_length=1)
    population: str = Field(..., min_length=1)
    condition: str = Field(..., min_length=1)
    action: str = Field(..., min_length=1)
    evidence_level: str = Field(..., min_length=1)
    recommendation_strength: str = Field(..., min_length=1)
    rationale: str = Field(..., min_length=1)
    required_inputs: list[str] = Field(default_factory=list)
    safety_notes: list[str] = Field(default_factory=list)
    source_span: str = Field(..., min_length=1)
    source_quote: str | None = None
    page: int | None = Field(default=None, ge=1)
    review_status: Literal["needs_human_review", "reviewed", "approved"] | None = None


class SubSkill(SchemaModel):
    subskill_id: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    clinical_tasks: list[str] = Field(default_factory=list)
    required_inputs: list[str] = Field(default_factory=list)
    recommendation_ids: list[str] = Field(default_factory=list)
    output_fields: list[str] = Field(default_factory=list)


class PatientCase(SchemaModel):
    raw_text: str = Field(..., min_length=1)
    demographics: dict[str, Any] = Field(default_factory=dict)
    symptoms: list[str] = Field(default_factory=list)
    labs: dict[str, Any] = Field(default_factory=dict)
    imaging: list[str] = Field(default_factory=list)
    endoscopy: list[str] = Field(default_factory=list)
    pathology: list[str] = Field(default_factory=list)
    medication_history: list[str] = Field(default_factory=list)
    past_history: list[str] = Field(default_factory=list)
    red_flags: list[str] = Field(default_factory=list)
    unknowns: list[str] = Field(default_factory=list)


class SkillExecutionResult(SchemaModel):
    skill_name: str = Field(..., min_length=1)
    disease_name: str = Field(..., min_length=1)
    suspicion_level: SuspicionLevel
    support_evidence: list[str] = Field(default_factory=list)
    against_evidence: list[str] = Field(default_factory=list)
    raw_missing_information: list[MissingInformationItem] = Field(default_factory=list)
    full_missing_information: list[MissingInformationItem] = Field(default_factory=list)
    top_missing_information: list[MissingInformationItem] = Field(default_factory=list)
    missing_information: list[MissingInformationItem] = Field(default_factory=list)
    raw_recommended_next_steps: list[str] = Field(default_factory=list)
    recommended_next_steps: list[str] = Field(default_factory=list)
    differential_diagnoses: list[DifferentialDiagnosisItem] = Field(default_factory=list)
    safety_warnings: list[str] = Field(default_factory=list)
    source_references: list[EvidenceReference] = Field(default_factory=list)


class DiseaseSkillPack(SchemaModel):
    skill_name: str = Field(..., min_length=1)
    disease_name: str = Field(..., min_length=1)
    disease_aliases: list[str] = Field(default_factory=list)
    guideline_name: str = Field(..., min_length=1)
    guideline_version: str = Field(..., min_length=1)
    source_pdf: str = Field(..., min_length=1)
    target_users: list[str] = Field(default_factory=list)
    scope: str = Field(..., min_length=1)
    routing_profile: RoutingProfile
    subskills: list[SubSkill] = Field(default_factory=list)
    recommendation_cards: list[RecommendationCard] = Field(default_factory=list)
    safety_constraints: list[str] = Field(default_factory=list)

    @field_validator(
        "disease_aliases",
        "target_users",
        "subskills",
        "recommendation_cards",
        "safety_constraints",
    )
    @classmethod
    def _require_non_empty_lists(cls, value: list[Any]) -> list[Any]:
        if not value:
            raise ValueError("field must contain at least one item")
        return value


def _read_serialized(path: Path) -> Mapping[str, Any]:
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    if suffix == ".json":
        data = json.loads(text)
    elif suffix in {".yaml", ".yml"}:
        data = yaml.safe_load(text)
    else:
        raise ValueError(f"Unsupported skill pack file extension: {path.suffix}")

    if not isinstance(data, Mapping):
        raise ValueError("Skill pack file must contain a mapping/object at the top level")
    return data


def _to_jsonable(skill_pack: DiseaseSkillPack) -> dict[str, Any]:
    return skill_pack.model_dump(mode="json", exclude_none=True)


def load_skill_pack(path: str | Path) -> DiseaseSkillPack:
    loaded = DiseaseSkillPack.model_validate(_read_serialized(Path(path)))
    return validate_skill_pack(loaded)


def save_skill_pack(skill_pack: DiseaseSkillPack, path: str | Path) -> None:
    validated = validate_skill_pack(skill_pack)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = _to_jsonable(validated)

    suffix = output_path.suffix.lower()
    if suffix == ".json":
        output_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    elif suffix in {".yaml", ".yml"}:
        output_path.write_text(
            yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
    else:
        raise ValueError(f"Unsupported skill pack file extension: {output_path.suffix}")


def validate_skill_pack(skill_pack: DiseaseSkillPack | Mapping[str, Any]) -> DiseaseSkillPack:
    pack = (
        skill_pack
        if isinstance(skill_pack, DiseaseSkillPack)
        else DiseaseSkillPack.model_validate(skill_pack)
    )

    _ensure_unique("subskill_id", [subskill.subskill_id for subskill in pack.subskills])
    recommendation_ids = [card.recommendation_id for card in pack.recommendation_cards]
    _ensure_unique("recommendation_id", recommendation_ids)

    known_recommendations = set(recommendation_ids)
    for subskill in pack.subskills:
        unknown = sorted(set(subskill.recommendation_ids) - known_recommendations)
        if unknown:
            raise ValueError(
                f"SubSkill {subskill.subskill_id!r} references unknown recommendations: "
                + ", ".join(unknown)
            )

    routing_aliases = {alias.casefold() for alias in pack.routing_profile.disease_aliases}
    pack_aliases = {alias.casefold() for alias in pack.disease_aliases}
    if not pack_aliases.issubset(routing_aliases):
        missing = sorted(pack_aliases - routing_aliases)
        raise ValueError(
            "routing_profile.disease_aliases must include disease_aliases: "
            + ", ".join(missing)
        )

    return pack


def _ensure_unique(field_name: str, values: list[str]) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    if duplicates:
        raise ValueError(f"Duplicate {field_name} values: {', '.join(sorted(duplicates))}")
