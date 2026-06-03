from __future__ import annotations

from clinical_stage_classifier import classify_clinical_stage
from patient_case_extractor import extract_patient_case


def test_symptoms_only_is_initial_screening() -> None:
    result = classify_clinical_stage(extract_patient_case("腹痛腹泻三个月，体重下降。"))

    assert result.primary_stage == "initial_screening"
    assert result.stages == ["initial_screening"]


def test_endoscopy_with_stricture_enters_diagnostic_differential_and_extent_stages() -> None:
    result = classify_clinical_stage(
        extract_patient_case("腹痛腹泻三个月，肠镜提示回盲部及回肠末端多发溃疡并狭窄。")
    )

    assert result.primary_stage == "diagnostic_workup"
    assert "diagnostic_workup" in result.stages
    assert "differential_diagnosis" in result.stages
    assert "extent_and_complication_assessment" in result.stages


def test_red_flags_enter_emergency_triage_first() -> None:
    result = classify_clinical_stage(extract_patient_case("腹痛腹泻，出现肠梗阻和大量便血。"))

    assert result.primary_stage == "emergency_triage"
    assert "emergency_triage" in result.stages


def test_doctor_confirmed_and_asking_medication_enters_treatment_selection() -> None:
    result = classify_clinical_stage(extract_patient_case("医生已确诊克罗恩病，现在应该怎么用药治疗？"))

    assert "treatment_selection" in result.stages


def test_treated_patient_asking_review_enters_followup_monitoring() -> None:
    result = classify_clinical_stage(extract_patient_case("已治疗半年，想问复查肠镜和复发监测怎么安排。"))

    assert "followup_monitoring" in result.stages
