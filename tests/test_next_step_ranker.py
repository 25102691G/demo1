from __future__ import annotations

from clinical_stage_classifier import classify_clinical_stage
from guideline_skill.schema import MissingInformationItem, SkillExecutionResult
from next_step_ranker import rank_next_steps
from patient_case_extractor import extract_patient_case


def _missing(key: str, priority: str = "high") -> MissingInformationItem:
    return MissingInformationItem(
        information_key=key,
        question=f"{key}?",
        reason=f"{key} reason",
        priority=priority,  # type: ignore[arg-type]
    )


def _result() -> SkillExecutionResult:
    return SkillExecutionResult(
        skill_name="crohn_disease_2023_guangzhou",
        disease_name="Crohn disease",
        suspicion_level="suspected",
        support_evidence=["内镜/肠腔表现支持：狭窄"],
        recommended_next_steps=[
            "[CD-REC-011] Base treatment suggestions on diagnostic status and contraindications.",
            "[CD-REC-005] Consider capsule endoscopy, and assess intestinal stricture and capsule retention risk before the examination.",
        ],
    )


def test_diagnostic_next_steps_are_prioritized_and_no_direct_treatment_is_shown() -> None:
    patient_case = extract_patient_case("腹痛腹泻三个月，肠镜提示回盲部及回肠末端多发溃疡并狭窄。")
    clinical_stage = classify_clinical_stage(patient_case)
    steps = rank_next_steps(
        patient_case,
        _result(),
        clinical_stage,
        [
            _missing("pathology", "critical"),
            _missing("biopsy_sites", "high"),
            _missing("cross_sectional_imaging", "high"),
            _missing("inflammatory_labs", "high"),
            _missing("intestinal_tuberculosis_exclusion", "critical"),
            _missing("infectious_enteritis_exclusion", "high"),
        ],
    )

    joined = "\n".join(steps)
    assert "活检" in steps[0] and "病理" in steps[0]
    assert any("CTE 或 MRE" in step for step in steps)
    assert any("CRP、ESR" in step for step in steps)
    assert any("肠结核" in step for step in steps)
    assert any("感染性肠炎" in step for step in steps)
    assert "Base treatment suggestions" not in joined


def test_capsule_endoscopy_does_not_precede_cte_mre_when_stricture_is_present() -> None:
    patient_case = extract_patient_case("拟行胶囊内镜，但肠镜提示小肠狭窄。")
    clinical_stage = classify_clinical_stage(patient_case)
    steps = rank_next_steps(
        patient_case,
        _result(),
        clinical_stage,
        [
            _missing("cross_sectional_imaging", "high"),
            _missing("stricture_risk_assessment", "critical"),
        ],
    )

    cte_index = next(i for i, step in enumerate(steps) if "CTE 或 MRE" in step)
    capsule_index = next(i for i, step in enumerate(steps) if "胶囊" in step)

    assert cte_index < capsule_index
    assert "滞留风险" in steps[capsule_index]


def test_red_flag_warning_is_first_step() -> None:
    patient_case = extract_patient_case("腹痛腹泻，出现大量便血和高热。")
    clinical_stage = classify_clinical_stage(patient_case)

    steps = rank_next_steps(patient_case, _result(), clinical_stage, [])

    assert steps[0].startswith("优先处理红旗征象")
