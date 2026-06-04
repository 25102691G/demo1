from __future__ import annotations

from .schemas import ClinicalInfoUnit, StatementUnit


CLINICAL_INFO_UNIT_TYPES = {
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
}
CLINICAL_TOPICS = {
    "background",
    "diagnosis",
    "treatment",
    "surgery",
    "follow_up",
    "documentation",
    "prognosis",
    "other",
}


def validate_statement_unit(
    statement_unit: StatementUnit,
    *,
    primary_anchor_count: int = 1,
) -> StatementUnit:
    reasons = list(statement_unit.review_reasons)

    if not statement_unit.statement_text.strip():
        reasons.append("empty_statement_text")
    if not statement_unit.evidence.evidence_quality_raw and not _is_bps(
        statement_unit.evidence.recommendation_strength_raw
    ):
        reasons.append("missing_evidence_quality_raw")
    if not statement_unit.evidence.recommendation_strength_raw and statement_unit.statement_type != "consensus":
        reasons.append("missing_strength_raw")
    if len(statement_unit.statement_text) > 1000:
        reasons.append("statement_text_too_long")
    if primary_anchor_count > 1:
        reasons.append("multiple_primary_unit_anchors_in_segment")
    if statement_unit.confidence < 0.75:
        reasons.append("low_confidence")
    if statement_unit.needs_human_review:
        reasons.append("normalizer_needs_human_review")

    deduped_reasons = _dedupe(reasons)
    return statement_unit.model_copy(
        update={
            "needs_human_review": bool(deduped_reasons),
            "review_reasons": deduped_reasons,
        }
    )


def validate_clinical_info_unit(clinical_info_unit: ClinicalInfoUnit) -> ClinicalInfoUnit:
    unit = clinical_info_unit.unit
    reasons = list(unit.review_reasons)

    if not unit.raw_text.strip():
        reasons.append("empty_raw_text")
    if not unit.unit_type or unit.unit_type not in CLINICAL_INFO_UNIT_TYPES:
        reasons.append("invalid_unit_type")
    if not unit.clinical_topic or unit.clinical_topic not in CLINICAL_TOPICS:
        reasons.append("invalid_clinical_topic")
    if unit.unit_type == "drug_regimen" and not any([unit.drug, unit.dose, unit.frequency]):
        reasons.append("drug_regimen_missing_drug_dose_frequency")
    if unit.unit_type == "indication" and not unit.indication:
        reasons.append("indication_unit_missing_indication")
    if unit.unit_type == "diagnostic_criteria" and not unit.diagnostic_criteria:
        reasons.append("diagnostic_criteria_unit_missing_criteria")
    if len(unit.raw_text) > 1500:
        reasons.append("raw_text_too_long")
    if unit.confidence < 0.70:
        reasons.append("low_confidence")
    if unit.needs_human_review:
        reasons.append("llm_needs_human_review")

    deduped_reasons = _dedupe(reasons)
    updated_unit = unit.model_copy(
        update={
            "needs_human_review": bool(deduped_reasons),
            "review_reasons": deduped_reasons,
        }
    )
    return clinical_info_unit.model_copy(update={"unit": updated_unit})


def _is_bps(value: str | None) -> bool:
    return bool(value and ("BPS" in value.upper() or "最佳临床实践" in value))


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped
