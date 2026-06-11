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
    return statement_unit


def validate_clinical_info_unit(clinical_info_unit: ClinicalInfoUnit) -> ClinicalInfoUnit:
    return clinical_info_unit


def _is_bps(value: str | None) -> bool:
    return bool(value and ("BPS" in value.upper() or "最佳临床实践" in value))
