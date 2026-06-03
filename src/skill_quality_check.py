from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from guideline_skill.schema import (
    DiseaseSkillPack,
    EvidenceReference,
    SkillExecutionResult,
    load_skill_pack,
)


class SkillQualityIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    message: str
    location: str | None = None


class SkillQualityReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    passed: bool
    errors: list[SkillQualityIssue] = Field(default_factory=list)
    warnings: list[SkillQualityIssue] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)


def check_skill_quality(
    skill_pack: DiseaseSkillPack,
    skill_results: Iterable[SkillExecutionResult] | None = None,
) -> SkillQualityReport:
    errors: list[SkillQualityIssue] = []
    warnings: list[SkillQualityIssue] = []

    _check_core_sections(skill_pack, errors)
    _check_recommendation_cards(skill_pack, errors, warnings)
    _check_safety_constraints(skill_pack, errors, warnings)
    _check_differential_diagnosis(skill_pack, errors)
    _check_forbidden_items(skill_pack, errors, warnings)
    _check_unique_recommendation_ids(skill_pack, errors)
    checked_source_references = _check_source_references(
        skill_pack,
        skill_results or [],
        errors,
    )

    return SkillQualityReport(
        passed=not errors,
        errors=errors,
        warnings=warnings,
        summary={
            "skill_name": skill_pack.skill_name,
            "disease_name": skill_pack.disease_name,
            "subskill_count": len(skill_pack.subskills),
            "recommendation_card_count": len(skill_pack.recommendation_cards),
            "safety_constraint_count": len(skill_pack.safety_constraints),
            "differential_diagnosis_count": len(skill_pack.routing_profile.must_differentiate),
            "source_reference_count": checked_source_references,
            "error_count": len(errors),
            "warning_count": len(warnings),
        },
    )


def check_skill_quality_from_path(path: str | Path) -> SkillQualityReport:
    return check_skill_quality(load_skill_pack(path))


def _check_core_sections(skill_pack: DiseaseSkillPack, errors: list[SkillQualityIssue]) -> None:
    if not skill_pack.routing_profile:
        errors.append(_issue("missing_routing_profile", "Skill pack must contain routing_profile.", "routing_profile"))
    if not skill_pack.subskills:
        errors.append(_issue("missing_subskills", "Skill pack must contain subskills.", "subskills"))
    if not skill_pack.recommendation_cards:
        errors.append(_issue("missing_recommendation_cards", "Skill pack must contain recommendation_cards.", "recommendation_cards"))


def _check_recommendation_cards(
    skill_pack: DiseaseSkillPack,
    errors: list[SkillQualityIssue],
    warnings: list[SkillQualityIssue],
) -> None:
    for card in skill_pack.recommendation_cards:
        location = f"recommendation_cards[{card.recommendation_id}]"
        required_scalar_fields = {
            "source_section": card.source_section,
            "clinical_task": card.clinical_task,
            "action": card.action,
        }
        for field_name, value in required_scalar_fields.items():
            if not value.strip():
                errors.append(_issue("missing_recommendation_field", f"Recommendation {card.recommendation_id} must include {field_name}.", location))
        if not card.required_inputs:
            errors.append(_issue("missing_required_inputs", f"Recommendation {card.recommendation_id} must include required_inputs.", location))
        if not card.safety_notes:
            warnings.append(_issue("missing_safety_notes", f"Recommendation {card.recommendation_id} has no safety_notes.", location))


def _check_safety_constraints(
    skill_pack: DiseaseSkillPack,
    errors: list[SkillQualityIssue],
    warnings: list[SkillQualityIssue],
) -> None:
    if not skill_pack.safety_constraints:
        errors.append(_issue("missing_safety_constraints", "Skill pack must contain safety_constraints.", "safety_constraints"))

    output_fields = {field for subskill in skill_pack.subskills for field in subskill.output_fields}
    if not output_fields.intersection({"safety_warnings", "safety_warning", "安全提醒", "安全警示"}):
        errors.append(_issue("missing_safety_warning_output", "At least one subskill should expose safety_warnings in output_fields.", "subskills"))

    safety_text = " ".join(skill_pack.safety_constraints).casefold()
    if not re.search(
        r"must not output a final diagnosis|never output a final diagnosis|not output a final diagnosis|"
        r"不得.{0,12}(最终诊断|确诊)|不能.{0,12}(最终诊断|确诊)|禁止.{0,12}(自动确诊|最终诊断)|不.{0,8}最终诊断",
        safety_text,
    ):
        errors.append(_issue("missing_no_auto_diagnosis_constraint", "Safety constraints must explicitly prohibit automatic final diagnosis.", "safety_constraints"))

    if not re.search(r"red flags?|emergency|urgent|红旗|急诊|紧急|及时就医|急腹症", safety_text):
        warnings.append(_issue("missing_red_flag_constraint", "Safety constraints should mention red flags or urgent care handling.", "safety_constraints"))


def _check_differential_diagnosis(
    skill_pack: DiseaseSkillPack,
    errors: list[SkillQualityIssue],
) -> None:
    if not skill_pack.routing_profile.must_differentiate:
        errors.append(_issue("missing_differential_diagnosis", "routing_profile.must_differentiate must contain differential diagnoses.", "routing_profile.must_differentiate"))


def _check_forbidden_items(
    skill_pack: DiseaseSkillPack,
    errors: list[SkillQualityIssue],
    warnings: list[SkillQualityIssue],
) -> None:
    for card in skill_pack.recommendation_cards:
        location = f"recommendation_cards[{card.recommendation_id}]"
        action = card.action.casefold()
        condition = card.condition.casefold()
        safety_notes = " ".join(card.safety_notes).casefold()

        if _looks_like_auto_diagnosis_instruction(action):
            errors.append(_issue("forbidden_auto_diagnosis", f"Recommendation {card.recommendation_id} appears to instruct automatic diagnosis.", location))

        if "incomplete" in condition and _looks_like_direct_treatment_instruction(action):
            errors.append(_issue("forbidden_treatment_when_incomplete", f"Recommendation {card.recommendation_id} appears to treat despite incomplete information.", location))

        if _is_treatment_selection_card(card.clinical_task, card.clinical_stage):
            required_text = " ".join(card.required_inputs).casefold()
            expected_groups = (
                ("diagnosis_status", "确诊状态", "诊断状态"),
                ("disease_extent", "病变范围"),
                ("disease_activity", "活动度"),
                ("complications", "并发症"),
                ("contraindications", "禁忌证"),
                ("infection_risk", "感染风险"),
            )
            if not all(any(term.casefold() in required_text for term in group) for group in expected_groups):
                warnings.append(_issue("treatment_readiness_inputs_incomplete", f"Treatment recommendation {card.recommendation_id} should include diagnosis, extent, activity, complication, contraindication, and infection risk inputs.", location))
            if not re.search(r"clinician|doctor|医生|医师|专科|review|评估", safety_notes):
                warnings.append(_issue("treatment_safety_notes_weak", f"Treatment recommendation {card.recommendation_id} should include clinician review safety notes.", location))


def _check_unique_recommendation_ids(
    skill_pack: DiseaseSkillPack,
    errors: list[SkillQualityIssue],
) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for card in skill_pack.recommendation_cards:
        if card.recommendation_id in seen:
            duplicates.add(card.recommendation_id)
        seen.add(card.recommendation_id)
    for duplicate in sorted(duplicates):
        errors.append(_issue("duplicate_recommendation_id", f"Duplicate recommendation_id: {duplicate}.", "recommendation_cards"))


def _check_source_references(
    skill_pack: DiseaseSkillPack,
    skill_results: Iterable[SkillExecutionResult],
    errors: list[SkillQualityIssue],
) -> int:
    known_ids = {card.recommendation_id for card in skill_pack.recommendation_cards}
    checked = 0
    for result in skill_results:
        for reference in result.source_references:
            checked += 1
            if reference.recommendation_id and reference.recommendation_id not in known_ids:
                errors.append(
                    _issue(
                        "unknown_source_reference",
                        f"source_reference {reference.recommendation_id} does not map to a recommendation_card.",
                        f"skill_results[{result.skill_name}].source_references",
                    )
                )
    return checked


def validate_source_references(
    skill_pack: DiseaseSkillPack,
    source_references: Iterable[EvidenceReference],
) -> list[SkillQualityIssue]:
    known_ids = {card.recommendation_id for card in skill_pack.recommendation_cards}
    return [
        _issue(
            "unknown_source_reference",
            f"source_reference {reference.recommendation_id} does not map to a recommendation_card.",
            "source_references",
        )
        for reference in source_references
        if reference.recommendation_id and reference.recommendation_id not in known_ids
    ]


def _looks_like_auto_diagnosis_instruction(text: str) -> bool:
    positive_patterns = (
        r"\b(output|declare|make|return)\b.{0,24}\b(final|confirmed)\s+diagnosis\b",
        r"自动确诊",
        r"直接确诊",
        r"输出.{0,12}最终诊断",
        r"作出.{0,12}确诊",
        r"自动确诊",
        r"直接确诊",
        r"无需医生.{0,8}确诊",
    )
    negating_patterns = (
        r"do not",
        r"never",
        r"rather than",
        r"must not",
        r"不得",
        r"不能",
        r"不要",
        r"禁止",
        r"不可",
        r"不能",
        r"不要",
        r"不得",
    )
    if not any(re.search(pattern, text, re.IGNORECASE) for pattern in positive_patterns):
        return False
    return not any(re.search(pattern, text, re.IGNORECASE) for pattern in negating_patterns)


def _is_treatment_selection_card(clinical_task: str, clinical_stage: str | None) -> bool:
    task = clinical_task.casefold()
    stage = (clinical_stage or "").casefold()
    if "treatment" in task and "monitor" not in task:
        return True
    if "药物与营养治疗选择" in stage:
        return True
    return any(
        keyword in clinical_task
        for keyword in (
            "治疗选择",
            "治疗方案",
            "诱导缓解",
            "维持缓解",
            "营养治疗",
            "用药",
        )
    )


def _looks_like_direct_treatment_instruction(text: str) -> bool:
    return any(
        re.search(pattern, text, re.IGNORECASE)
        for pattern in (
            r"\btreat\b",
            r"\bstart\b.{0,20}\btherapy\b",
            r"立即治疗",
            r"直接治疗",
            r"使用.{0,8}(生物制剂|糖皮质激素|抗TNF|JAK|免疫抑制剂)",
            r"立即治疗",
            r"直接治疗",
            r"使用.{0,8}(生物制剂|糖皮质激素|抗TNF|JAK)",
        )
    )


def _issue(code: str, message: str, location: str | None = None) -> SkillQualityIssue:
    return SkillQualityIssue(code=code, message=message, location=location)
