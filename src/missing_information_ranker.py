from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from clinical_stage_classifier import ClinicalStageResult
from guideline_skill.schema import MissingInformationItem


class MissingInformationRanking(BaseModel):
    model_config = ConfigDict(extra="forbid")

    full_missing_information: list[MissingInformationItem] = Field(default_factory=list)
    top_missing_information: list[MissingInformationItem] = Field(default_factory=list)


def rank_missing_information(
    missing_information: list[MissingInformationItem],
    clinical_stage: ClinicalStageResult,
    top_k: int = 7,
) -> MissingInformationRanking:
    """Deduplicate, merge synonyms, stage-filter, and rank missing information."""

    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    merged = _merge_synonyms(missing_information)
    stage_filtered = [
        item for item in merged if _is_relevant_to_stage(item.information_key, clinical_stage)
    ]
    if not stage_filtered:
        stage_filtered = merged

    ranked = sorted(
        stage_filtered,
        key=lambda item: (
            _stage_rank(item.information_key, clinical_stage),
            _priority_rank(item.priority),
            item.information_key,
        ),
    )
    return MissingInformationRanking(
        full_missing_information=ranked,
        top_missing_information=ranked[:top_k],
    )


def _merge_synonyms(items: list[MissingInformationItem]) -> list[MissingInformationItem]:
    chosen: dict[str, MissingInformationItem] = {}
    for item in items:
        canonical_key = _canonical_key(item.information_key)
        canonical_item = _canonical_item(canonical_key, item)
        existing = chosen.get(canonical_key)
        if existing is None or _priority_rank(canonical_item.priority) < _priority_rank(existing.priority):
            chosen[canonical_key] = canonical_item
    return list(chosen.values())


def _canonical_item(
    canonical_key: str,
    fallback: MissingInformationItem,
) -> MissingInformationItem:
    if canonical_key not in CANONICAL_MESSAGES:
        display = field_display_name(canonical_key)
        return MissingInformationItem(
            information_key=canonical_key,
            question=f"是否已有{display}相关信息？",
            reason=f"{display}有助于当前临床阶段的完整判断。",
            priority=fallback.priority,
        )
    question, reason, priority = CANONICAL_MESSAGES[canonical_key]
    effective_priority = priority
    if _priority_rank(fallback.priority) < _priority_rank(priority):
        effective_priority = fallback.priority
    return MissingInformationItem(
        information_key=canonical_key,
        question=question,
        reason=reason,
        priority=effective_priority,  # type: ignore[arg-type]
    )


def _canonical_key(key: str) -> str:
    normalized = key.strip().casefold().replace("-", "_")
    return SYNONYM_KEYS.get(normalized, normalized)


def field_display_name(key: str) -> str:
    normalized = _canonical_key(key)
    return FIELD_DISPLAY_NAMES.get(normalized, _humanize_key(normalized))


def _humanize_key(key: str) -> str:
    return key.replace("_", " ")


def _is_relevant_to_stage(key: str, clinical_stage: ClinicalStageResult) -> bool:
    stages = set(clinical_stage.stages)
    if "emergency_triage" in stages and key in EMERGENCY_DEFERRED_KEYS:
        return False
    if stages.intersection({"diagnostic_workup", "differential_diagnosis", "extent_and_complication_assessment"}):
        return key not in TREATMENT_OR_FOLLOWUP_KEYS
    if "initial_screening" in stages:
        return key not in TREATMENT_OR_FOLLOWUP_KEYS
    if "followup_monitoring" in stages:
        return key not in DIAGNOSTIC_ONLY_LOW_VALUE_KEYS
    return True


def _stage_rank(key: str, clinical_stage: ClinicalStageResult) -> int:
    stages = set(clinical_stage.stages)
    if "emergency_triage" in stages:
        return EMERGENCY_PRIORITY.get(key, 50)
    if stages.intersection({"diagnostic_workup", "differential_diagnosis", "extent_and_complication_assessment"}):
        return DIAGNOSTIC_PRIORITY.get(key, 50)
    if "initial_screening" in stages:
        return INITIAL_PRIORITY.get(key, 50)
    if "followup_monitoring" in stages:
        return FOLLOWUP_PRIORITY.get(key, 50)
    if stages.intersection({"treatment_readiness", "treatment_selection"}):
        return TREATMENT_PRIORITY.get(key, 50)
    return 50


def _priority_rank(priority: str) -> int:
    return {"critical": 0, "high": 1, "medium": 2, "low": 3}.get(priority, 4)


SYNONYM_KEYS = {
    "pathology": "pathology",
    "pathology_report": "pathology",
    "upper_gi_biopsy_pathology": "pathology",
    "inflammatory_labs": "inflammatory_labs",
    "inflammatory_markers": "inflammatory_labs",
    "current_labs": "inflammatory_labs",
    "labs": "inflammatory_labs",
    "fecal_calprotectin": "inflammatory_labs",
    "imaging": "cross_sectional_imaging",
    "prior_imaging": "cross_sectional_imaging",
    "small_bowel_imaging": "cross_sectional_imaging",
    "prior_small_bowel_imaging": "cross_sectional_imaging",
    "cte_report": "cross_sectional_imaging",
    "mre_report": "cross_sectional_imaging",
    "colonoscopy_report": "ileocolonoscopy",
    "prior_colonoscopy": "ileocolonoscopy",
    "endoscopy": "ileocolonoscopy",
    "terminal_ileum_assessment": "terminal_ileum_assessment",
    "biopsy_sites": "biopsy_sites",
    "gastroduodenoscopy_report": "gastroduodenoscopy_report",
    "tuberculosis_workup": "intestinal_tuberculosis_exclusion",
    "intestinal_tuberculosis_exclusion": "intestinal_tuberculosis_exclusion",
    "infection_workup": "infectious_enteritis_exclusion",
    "infectious_enteritis_exclusion": "infectious_enteritis_exclusion",
    "medication_history": "drug_induced_enteritis_review",
    "drug_induced_enteritis_review": "drug_induced_enteritis_review",
    "stricture_risk_assessment": "stricture_risk_assessment",
    "abscess_assessment": "abscess_assessment",
    "perianal_mri": "perianal_mri",
    "perianal_exam": "perianal_exam",
    "perianal_symptoms": "perianal_symptoms",
    "perianal_disease": "perianal_disease",
    "disease_extent": "disease_extent",
    "disease_activity": "disease_activity",
    "complications": "complications",
    "phenotype": "phenotype",
    "age_at_onset": "age_at_onset",
    "smoking_status": "smoking_status",
    "renal_function_if_contrast_needed": "renal_function_if_contrast_needed",
    "pregnancy_status_if_relevant": "pregnancy_status_if_relevant",
    "contraindications": "contraindications",
    "infection_risk": "infection_risk",
    "prior_medication_response": "prior_medication_response",
    "comorbidities": "comorbidities",
    "diagnosis_status": "diagnosis_status",
    "treatment_history": "treatment_history",
    "current_symptoms": "current_symptoms",
    "prior_endoscopy": "prior_endoscopy",
    "nutrition_status": "nutrition_status",
    "mental_health_status": "mental_health_status",
}

FIELD_DISPLAY_NAMES = {
    "available_case_information": "病例资料完整性",
    "missing_information_review": "缺失信息复核",
    "symptoms": "主要症状、持续时间和严重程度",
    "current_symptoms": "当前症状变化",
    "systemic_features": "全身表现",
    "pathology": "活检病理结果",
    "biopsy_sites": "多肠段活检取材部位",
    "terminal_ileum_assessment": "回肠末端评估",
    "gastroduodenoscopy_report": "胃十二指肠镜报告",
    "ileocolonoscopy": "结肠镜和回肠末端评估",
    "cross_sectional_imaging": "CTE 或 MRE 等横断面影像",
    "inflammatory_labs": "炎症和营养相关实验室检查",
    "intestinal_tuberculosis_exclusion": "肠结核鉴别信息",
    "infectious_enteritis_exclusion": "感染性肠炎鉴别信息",
    "drug_induced_enteritis_review": "药物性肠炎相关用药史",
    "stricture_risk_assessment": "狭窄和胶囊滞留风险评估",
    "perianal_mri": "肛周 MRI",
    "perianal_exam": "肛周体格检查或外科评估",
    "perianal_symptoms": "肛周症状",
    "perianal_disease": "肛周病变情况",
    "abscess_assessment": "脓肿评估",
    "disease_extent": "病变范围",
    "disease_activity": "疾病活动度",
    "complications": "并发症情况",
    "phenotype": "疾病表型",
    "age_at_onset": "发病年龄",
    "smoking_status": "吸烟情况",
    "renal_function_if_contrast_needed": "使用增强造影前的肾功能信息",
    "pregnancy_status_if_relevant": "妊娠或备孕情况",
    "contraindications": "治疗或检查禁忌证",
    "infection_risk": "感染风险评估",
    "prior_medication_response": "既往治疗反应",
    "comorbidities": "合并疾病",
    "diagnosis_status": "医生诊断状态",
    "treatment_history": "既往治疗史",
    "nutrition_status": "营养状态",
    "mental_health_status": "心理健康状态",
}

CANONICAL_MESSAGES = {
    "pathology": (
        "是否已有活检病理结果，例如慢性炎症、肉芽肿、透壁性炎症，或感染/肿瘤线索？",
        "病理结果是把疑似 IBD/CD 往前推进的关键证据，也能帮助排除感染、肿瘤等相似疾病。",
        "critical",
    ),
    "biopsy_sites": (
        "是否已进行多肠段活检，并记录每个取材部位？",
        "疑似克罗恩病时，多肠段活检比单点取材更有助于综合诊断和鉴别诊断。",
        "high",
    ),
    "terminal_ileum_assessment": (
        "结肠镜是否已尽量进入并描述回肠末端？",
        "回肠末端和回盲部是克罗恩病常见受累部位，是否进入和描述会影响诊断判断。",
        "high",
    ),
    "ileocolonoscopy": (
        "是否已有完整结肠镜报告，并说明回肠末端、回盲部、溃疡形态和是否活检？",
        "内镜报告是当前阶段判断疑似程度和安排后续检查的核心资料。",
        "critical",
    ),
    "gastroduodenoscopy_report": (
        "是否已有胃十二指肠镜报告，用于评估上消化道受累？",
        "上消化道受累会影响病变范围判断，也需要结合病理和鉴别诊断解释。",
        "medium",
    ),
    "cross_sectional_imaging": (
        "是否已有 CTE 或 MRE 来评估小肠受累范围，以及狭窄、瘘管、脓肿等并发症？",
        "CTE/MRE 能补充内镜看不到的肠壁和肠外病变，狭窄病例尤其需要先评估。",
        "high",
    ),
    "inflammatory_labs": (
        "是否已有 CRP、ESR、血常规、白蛋白、粪便钙卫蛋白等炎症和营养相关检查？",
        "这些检查可帮助判断是否存在炎症、贫血或营养风险，但不能单独确诊或排除。",
        "high",
    ),
    "intestinal_tuberculosis_exclusion": (
        "是否已评估或排除肠结核，例如结核感染证据、胸部影像、病理或病原学线索？",
        "肠结核可模拟回盲部溃疡和狭窄，是进入治疗前必须重视的鉴别诊断。",
        "critical",
    ),
    "infectious_enteritis_exclusion": (
        "是否已结合病程、粪便病原学或培养评估感染性肠炎？",
        "感染性肠炎也可出现腹泻、发热、炎症指标升高和黏膜炎症。",
        "high",
    ),
    "drug_induced_enteritis_review": (
        "是否已回顾 NSAID、抗生素、免疫治疗等可能造成肠道损伤的用药史？",
        "药物性肠炎可能模拟炎症性肠病，需要结合用药时间线判断。",
        "medium",
    ),
    "stricture_risk_assessment": (
        "是否已评估狭窄程度、梗阻风险，以及胶囊内镜滞留风险？",
        "存在狭窄时，应先评估并发症和胶囊滞留风险，不能直接把胶囊内镜作为优先检查。",
        "critical",
    ),
    "perianal_mri": (
        "若有肛瘘或肛周脓肿，是否已完善肛周 MRI？",
        "肛周 MRI 有助于判断瘘管、脓肿和复杂程度。",
        "high",
    ),
    "perianal_disease": (
        "是否已明确有无肛瘘、肛周脓肿、肛周疼痛或流脓等肛周病变？",
        "肛周病变会影响疾病表型、并发症评估和后续处理优先级。",
        "medium",
    ),
    "abscess_assessment": (
        "是否已评估有无脓肿或需要及时处理的感染灶？",
        "脓肿会改变处置优先级，发热或疼痛加重时应及时线下评估。",
        "high",
    ),
    "renal_function_if_contrast_needed": (
        "如果考虑增强 CTE，是否已有肾功能和造影剂禁忌评估？",
        "增强影像检查前需要评估肾功能、过敏史和造影剂相关风险。",
        "medium",
    ),
    "pregnancy_status_if_relevant": (
        "是否存在妊娠或备孕情况，以便选择更合适的影像检查方式？",
        "妊娠或备孕会影响 CTE/MRE 等检查方式选择和辐射暴露权衡。",
        "medium",
    ),
    "diagnosis_status": (
        "是否已有医生给出的明确诊断状态？",
        "治疗选择应建立在医生诊断、病变范围、活动度和禁忌信息充分的基础上。",
        "critical",
    ),
    "contraindications": (
        "是否已评估用药或检查相关禁忌证？",
        "禁忌证会影响后续检查和治疗选择，不能在缺失时直接给出个体化治疗方案。",
        "high",
    ),
    "infection_risk": (
        "是否已评估潜在感染风险，尤其是结核、乙肝等治疗前需要排查的问题？",
        "感染风险会影响免疫相关治疗是否安全。",
        "high",
    ),
}

DIAGNOSTIC_PRIORITY = {
    "pathology": 0,
    "biopsy_sites": 1,
    "cross_sectional_imaging": 2,
    "stricture_risk_assessment": 3,
    "inflammatory_labs": 4,
    "intestinal_tuberculosis_exclusion": 5,
    "infectious_enteritis_exclusion": 6,
    "drug_induced_enteritis_review": 7,
    "terminal_ileum_assessment": 8,
    "ileocolonoscopy": 9,
    "perianal_mri": 10,
    "abscess_assessment": 11,
}

INITIAL_PRIORITY = {
    "ileocolonoscopy": 0,
    "inflammatory_labs": 1,
    "cross_sectional_imaging": 2,
    "pathology": 3,
    "intestinal_tuberculosis_exclusion": 4,
    "infectious_enteritis_exclusion": 5,
}

EMERGENCY_PRIORITY = {
    "abscess_assessment": 0,
    "cross_sectional_imaging": 1,
    "inflammatory_labs": 2,
    "pathology": 3,
}

TREATMENT_PRIORITY = {
    "diagnosis_status": 0,
    "disease_activity": 1,
    "disease_extent": 2,
    "complications": 3,
    "contraindications": 4,
    "infection_risk": 5,
}

FOLLOWUP_PRIORITY = {
    "current_symptoms": 0,
    "inflammatory_labs": 1,
    "cross_sectional_imaging": 2,
    "prior_endoscopy": 3,
    "nutrition_status": 4,
}

TREATMENT_OR_FOLLOWUP_KEYS = {
    "diagnosis_status",
    "contraindications",
    "infection_risk",
    "prior_medication_response",
    "comorbidities",
    "treatment_history",
    "nutrition_status",
    "mental_health_status",
    "age_at_onset",
    "smoking_status",
}

DIAGNOSTIC_ONLY_LOW_VALUE_KEYS = {
    "available_case_information",
    "missing_information_review",
}

EMERGENCY_DEFERRED_KEYS = TREATMENT_OR_FOLLOWUP_KEYS | {
    "nutrition_status",
    "mental_health_status",
}
