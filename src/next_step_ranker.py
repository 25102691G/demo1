from __future__ import annotations

from clinical_stage_classifier import ClinicalStageResult
from guideline_skill.schema import MissingInformationItem, PatientCase, SkillExecutionResult


def rank_next_steps(
    patient_case: PatientCase,
    skill_result: SkillExecutionResult | None,
    clinical_stage: ClinicalStageResult,
    top_missing_information: list[MissingInformationItem],
    top_k: int = 8,
) -> list[str]:
    """Create stage-aware, user-facing next steps from missing info and evidence."""

    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    steps: list[str] = []
    if patient_case.red_flags:
        steps.append("优先处理红旗征象：建议及时线下就医或急诊评估，不要先等待常规检查排期。")

    missing_keys = {item.information_key for item in top_missing_information}
    stages = set(clinical_stage.stages)

    if stages.intersection({"diagnostic_workup", "differential_diagnosis", "extent_and_complication_assessment", "initial_screening"}):
        steps.extend(_diagnostic_steps(missing_keys, patient_case))

    if "extent_and_complication_assessment" in stages:
        steps.extend(_complication_steps(patient_case, missing_keys))

    if "followup_monitoring" in stages:
        steps.append("按随访监测阶段处理：复查症状变化、粪便钙卫蛋白/CRP 等炎症指标，并由医生决定是否复查结肠镜或 CTE/MRE。")

    if _allow_treatment_steps(skill_result, clinical_stage):
        steps.extend(_safe_treatment_readiness_steps(skill_result))

    steps.extend(_fallback_steps(skill_result, patient_case, clinical_stage))
    return _dedupe(steps)[:top_k]


def _diagnostic_steps(missing_keys: set[str], patient_case: PatientCase) -> list[str]:
    steps: list[str] = []
    if {"pathology", "biopsy_sites", "ileocolonoscopy", "terminal_ileum_assessment"}.intersection(missing_keys):
        steps.append("完善或复核结肠镜多肠段活检和病理结果，重点看慢性炎症、肉芽肿、透壁性炎症，以及感染或肿瘤线索。")
    if "cross_sectional_imaging" in missing_keys or _has_stricture(patient_case):
        steps.append("优先完善 CTE 或 MRE，评估小肠受累范围，以及狭窄、瘘管、脓肿等并发症。")
    if "inflammatory_labs" in missing_keys:
        steps.append("补充 CRP、ESR、血常规、白蛋白、粪便钙卫蛋白等炎症和营养相关检查。")
    if "intestinal_tuberculosis_exclusion" in missing_keys:
        steps.append("在消化专科医生指导下评估肠结核鉴别，包括结核感染证据、胸部影像、病理或病原学线索。")
    if "infectious_enteritis_exclusion" in missing_keys:
        steps.append("结合病程、粪便病原学或培养等检查，评估并排除感染性肠炎。")
    if "drug_induced_enteritis_review" in missing_keys:
        steps.append("回顾 NSAID、抗生素、免疫治疗等用药史，评估药物性肠炎或药物性肠损伤可能。")
    return steps


def _complication_steps(patient_case: PatientCase, missing_keys: set[str]) -> list[str]:
    steps: list[str] = []
    if _has_stricture(patient_case):
        steps.append("因已提到狭窄，若后续考虑胶囊内镜，应先完成狭窄和胶囊滞留风险评估，通常不应排在 CTE/MRE 前面。")
    if {"perianal_mri", "abscess_assessment"}.intersection(missing_keys) or _has_perianal_signal(patient_case):
        steps.append("若存在肛瘘、肛周脓肿或肛周疼痛流脓，建议完善肛周 MRI 或肛周专科评估。")
    return steps


def _safe_treatment_readiness_steps(skill_result: SkillExecutionResult | None) -> list[str]:
    if skill_result is None:
        return []
    return [
        _translate_known_seed_step(step)
        for step in skill_result.recommended_next_steps
        if _is_treatment_readiness_step(step) and not _is_direct_drug_treatment(step)
    ][:2]


def _fallback_steps(
    skill_result: SkillExecutionResult | None,
    patient_case: PatientCase,
    clinical_stage: ClinicalStageResult,
) -> list[str]:
    if skill_result is None:
        return ["补充症状持续时间、体温、体重变化、便血情况，以及已有实验室、内镜、影像和病理资料后再评估。"]

    filtered: list[str] = []
    for step in skill_result.recommended_next_steps:
        if _is_direct_drug_treatment(step):
            continue
        if not _allow_treatment_steps(skill_result, clinical_stage) and _is_treatment_stage_step(step):
            continue
        if _is_followup_or_mental_health_step(step) and "followup_monitoring" not in clinical_stage.stages:
            continue
        if "胶囊内镜" in step and _has_stricture(patient_case):
            continue
        filtered.append(_translate_known_seed_step(step))
    return filtered[:3]


def _allow_treatment_steps(
    skill_result: SkillExecutionResult | None,
    clinical_stage: ClinicalStageResult,
) -> bool:
    if skill_result is None:
        return False
    return (
        skill_result.suspicion_level == "confirmed_by_doctor_only"
        or "treatment_selection" in clinical_stage.stages
        or "treatment_readiness" in clinical_stage.stages
    )


def _has_stricture(patient_case: PatientCase) -> bool:
    text = " ".join([patient_case.raw_text, *patient_case.endoscopy, *patient_case.red_flags])
    return any(term in text for term in ("狭窄", "肠梗阻", "梗阻", "stricture", "stenosis"))


def _has_perianal_signal(patient_case: PatientCase) -> bool:
    text = " ".join([patient_case.raw_text, *patient_case.symptoms, *patient_case.imaging])
    return any(term in text for term in ("肛瘘", "肛周", "脓肿", "瘘管", "perianal", "abscess"))


def _is_treatment_readiness_step(step: str) -> bool:
    return any(term in step for term in ("diagnostic status", "disease extent", "disease activity", "并发症", "禁忌", "感染风险"))


def _is_treatment_stage_step(step: str) -> bool:
    lower = step.casefold()
    return any(
        term in lower
        for term in (
            "treatment",
            "therapy",
            "contraindication",
            "medication availability",
            "prior treatment",
            "用药",
            "治疗建议",
            "生物制剂",
            "激素",
            "免疫抑制",
            "抗tnf",
        )
    )


def _is_direct_drug_treatment(step: str) -> bool:
    lower = step.casefold()
    return any(
        term in lower
        for term in (
            "start ",
            "initiate ",
            "dose",
            "prescribe",
            "anti-tnf",
            "ustekinumab",
            "vedolizumab",
            "infliximab",
            "adalimumab",
            "corticosteroid",
            "开始使用",
            "处方",
            "剂量",
            "英夫利昔",
            "阿达木",
            "乌司奴",
            "维得利珠",
        )
    )


def _is_followup_or_mental_health_step(step: str) -> bool:
    lower = step.casefold()
    return any(term in lower for term in ("follow-up", "followup", "monitor", "mental health", "nutrition", "随访", "心理", "营养"))


def _translate_known_seed_step(step: str) -> str:
    translations = {
        "[CD-REC-001]": "综合临床表现、实验室、影像、内镜和病理信息判断疑似程度，不依赖单一检查作最终诊断。",
        "[CD-REC-002]": "可用粪便钙卫蛋白评估肠道炎症水平，但阴性结果需结合症状和小肠受累情况谨慎解读。",
        "[CD-REC-003]": "完善结肠镜评估，尽量进入回肠末端，并在疑诊时进行多肠段活检。",
        "[CD-REC-004]": "必要时评估上消化道受累，并结合胃十二指肠镜和病理活检判断。",
        "[CD-REC-005]": "若考虑胶囊内镜，应先确认结肠镜和小肠影像未明确诊断，并评估狭窄和胶囊滞留风险。",
        "[CD-REC-006]": "完善 CTE 或 MRE 评估病变范围和并发症。",
        "[CD-REC-007]": "肛瘘或肛周脓肿线索明显时，优先完善肛周 MRI 评估瘘管、脓肿和复杂程度。",
        "[CD-REC-008]": "同步进行鉴别诊断，尤其关注肠结核、感染性肠炎、肠白塞病、淋巴瘤和药物性肠炎。",
        "[CD-REC-009]": "信息不足时只输出疑似或拟诊倾向，不能由系统直接确诊。",
        "[CD-REC-010]": "进一步描述病变范围、疾病活动度和并发症，尤其关注狭窄、穿透性病变和肛周病变等高风险因素。",
        "[CD-REC-011]": "在讨论治疗前，先确认医生诊断状态、病变范围、活动度、并发症、禁忌证、感染风险和既往治疗反应。",
        "[CD-REC-012]": "随访时监测症状、炎症指标和治疗反应，必要时结合结肠镜或 CTE/MRE 复评病变状态。",
    }
    for prefix, translated in translations.items():
        if step.startswith(prefix):
            return translated
    return step


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(value)
    return deduped
