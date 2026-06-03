from __future__ import annotations

import re

from differential_diagnosis_enricher import enrich_differential_diagnoses
from guideline_skill.schema import (
    DifferentialDiagnosisItem,
    DiseaseSkillPack,
    EvidenceReference,
    MissingInformationItem,
    PatientCase,
    RecommendationCard,
    SkillExecutionResult,
)
from retrievers.base import GuidelineRetriever
from retrievers.local_recommendation_retriever import LocalRecommendationRetriever


class CrohnDiseaseSkillExecutor:
    """First-pass rule executor for the Crohn disease seed skill pack."""

    nonspecific_symptoms = {
        "腹痛",
        "腹泻",
        "体重下降",
        "发热",
        "肛瘘",
        "肛周脓肿",
        "口腔溃疡",
        "乏力",
    }
    endoscopy_support_terms = {
        "回盲部",
        "回肠末端",
        "溃疡",
        "纵行溃疡",
        "铺路石样改变",
        "狭窄",
        "瘘管",
    }
    pathology_support_terms = {"慢性炎症", "肉芽肿", "透壁性炎症"}
    inflammatory_labs = {"CRP", "ESR", "粪便钙卫蛋白"}
    red_flag_messages = {
        "剧烈腹痛": "出现剧烈腹痛，需警惕急腹症、穿孔、梗阻或重症炎症，建议及时就医或急诊评估。",
        "肠梗阻": "出现肠梗阻相关信息，建议及时就医或急诊评估，避免自行等待。",
        "大量便血": "出现大量便血，建议及时就医或急诊评估出血和循环状态。",
        "高热": "出现高热，需警惕严重感染、脓肿或重症炎症，建议及时就医。",
        "休克": "出现休克相关信息，应立即急诊处理。",
        "严重脱水": "出现严重脱水相关信息，建议及时就医评估补液和电解质紊乱。",
        "意识障碍": "出现意识障碍，应立即急诊处理。",
    }

    def __init__(self, retriever: GuidelineRetriever | None = None) -> None:
        self.retriever = retriever or LocalRecommendationRetriever()

    def execute(
        self,
        patient_case: PatientCase,
        skill_pack: DiseaseSkillPack,
    ) -> SkillExecutionResult:
        matched_cards = self.retriever.retrieve_by_patient_case(
            patient_case,
            skill_pack,
            top_k=6,
        )
        symptom_evidence = self._symptom_evidence(patient_case)
        lab_evidence = self._lab_evidence(patient_case)
        endoscopy_evidence = self._endoscopy_evidence(patient_case)
        imaging_evidence = self._imaging_evidence(patient_case)
        pathology_evidence = self._pathology_evidence(patient_case)
        doctor_confirmed = _doctor_confirmed(patient_case.raw_text)

        support_evidence = _dedupe(
            [
                *symptom_evidence,
                *lab_evidence,
                *endoscopy_evidence,
                *imaging_evidence,
                *pathology_evidence,
            ]
        )
        against_evidence = self._against_evidence(patient_case, support_evidence)
        suspicion_level = self._suspicion_level(
            doctor_confirmed=doctor_confirmed,
            symptom_evidence=symptom_evidence,
            lab_evidence=lab_evidence,
            endoscopy_evidence=endoscopy_evidence,
            imaging_evidence=imaging_evidence,
            pathology_evidence=pathology_evidence,
        )
        missing_information = self._missing_information(
            patient_case=patient_case,
            has_endoscopy_support=bool(endoscopy_evidence),
            has_imaging_support=bool(imaging_evidence),
            has_pathology_support=bool(pathology_evidence),
            matched_cards=matched_cards,
        )
        safety_warnings = self._safety_warnings(patient_case)
        recommended_next_steps = self._recommended_next_steps(
            patient_case,
            missing_information,
            matched_cards,
        )
        differential_diagnoses = self.retriever.retrieve_differential_diagnoses(
            patient_case,
            skill_pack,
            top_k=6,
        )
        if not differential_diagnoses:
            differential_diagnoses = self._differential_diagnoses()
        differential_diagnoses = enrich_differential_diagnoses(
            patient_case,
            differential_diagnoses,
        )
        source_references = self._source_references(
            matched_cards,
        )

        if not support_evidence:
            support_evidence = ["当前输入未提供克罗恩病相关的典型症状、检查或病理支持信息。"]

        if suspicion_level in {"possible", "suspected", "probable"}:
            safety_warnings.append("本执行器不能自动确诊克罗恩病；需由医生结合临床、实验室、影像、内镜、病理及鉴别诊断综合判断。")
        if suspicion_level == "confirmed_by_doctor_only":
            support_evidence.append("用户文本明确提到医生已确诊克罗恩病；执行器仅记录该前提，不独立作出确诊。")

        return SkillExecutionResult(
            skill_name=skill_pack.skill_name,
            disease_name=skill_pack.disease_name,
            suspicion_level=suspicion_level,
            support_evidence=_dedupe(support_evidence),
            against_evidence=against_evidence,
            missing_information=missing_information,
            recommended_next_steps=recommended_next_steps,
            differential_diagnoses=differential_diagnoses,
            safety_warnings=_dedupe(safety_warnings),
            source_references=source_references,
        )

    def _symptom_evidence(self, patient_case: PatientCase) -> list[str]:
        return [
            f"出现症状：{symptom}"
            for symptom in patient_case.symptoms
            if symptom in self.nonspecific_symptoms
        ]

    def _lab_evidence(self, patient_case: PatientCase) -> list[str]:
        evidence: list[str] = []
        for lab_name, lab_detail in patient_case.labs.items():
            if lab_name not in self.inflammatory_labs:
                continue
            value_texts = lab_detail.get("value_texts", []) if isinstance(lab_detail, dict) else []
            if _lab_suggests_inflammation(value_texts):
                evidence.append(f"炎症指标支持：{lab_name} {', '.join(value_texts) if value_texts else '已提及'}")
            elif not value_texts:
                evidence.append(f"炎症相关检查已提及：{lab_name}，但未提供结果方向或数值。")
        return evidence

    def _endoscopy_evidence(self, patient_case: PatientCase) -> list[str]:
        evidence = [
            f"内镜/肠腔表现支持：{term}"
            for term in patient_case.endoscopy
            if term in self.endoscopy_support_terms
        ]
        if re.search(r"多发.{0,4}溃疡|多处.{0,4}溃疡", patient_case.raw_text):
            evidence.append("内镜/肠腔表现支持：多发溃疡")
        return evidence

    def _imaging_evidence(self, patient_case: PatientCase) -> list[str]:
        evidence = []
        for modality in patient_case.imaging:
            if modality in {"CTE", "MRE", "肛周 MRI", "肠道超声"}:
                evidence.append(f"影像学检查已提供：{modality}")
        return evidence

    def _pathology_evidence(self, patient_case: PatientCase) -> list[str]:
        return [
            f"病理支持信息：{term}"
            for term in patient_case.pathology
            if term in self.pathology_support_terms
        ]

    def _against_evidence(self, patient_case: PatientCase, support_evidence: list[str]) -> list[str]:
        against = []
        raw_text = patient_case.raw_text
        if re.search(r"FC.{0,8}(正常|阴性)|粪.*钙卫蛋白.{0,8}(正常|阴性)", raw_text, re.IGNORECASE):
            against.append("粪便钙卫蛋白正常或阴性，对肠道炎症支持较弱；小肠受累时仍需谨慎解读。")
        if re.search(r"CRP.{0,8}(正常|阴性)|ESR.{0,8}(正常|阴性)|血沉.{0,8}(正常|阴性)", raw_text, re.IGNORECASE):
            against.append("CRP/ESR正常或阴性，对活动性炎症支持较弱，但不能单独排除克罗恩病。")
        if not support_evidence:
            against.append("未提供支持克罗恩病的症状、实验室、影像、内镜或病理信息。")
        return against

    def _suspicion_level(
        self,
        *,
        doctor_confirmed: bool,
        symptom_evidence: list[str],
        lab_evidence: list[str],
        endoscopy_evidence: list[str],
        imaging_evidence: list[str],
        pathology_evidence: list[str],
    ):
        if doctor_confirmed:
            return "confirmed_by_doctor_only"

        has_possible_features = bool(symptom_evidence)
        has_supportive_tests = bool(lab_evidence or endoscopy_evidence or imaging_evidence)
        has_endoscopy_or_imaging = bool(endoscopy_evidence or imaging_evidence)

        if has_supportive_tests and has_endoscopy_or_imaging and pathology_evidence:
            return "probable"
        if has_possible_features and has_supportive_tests:
            return "suspected"
        if has_possible_features:
            return "possible"
        return "unlikely"

    def _missing_information(
        self,
        *,
        patient_case: PatientCase,
        has_endoscopy_support: bool,
        has_imaging_support: bool,
        has_pathology_support: bool,
        matched_cards: list[RecommendationCard],
    ) -> list[MissingInformationItem]:
        missing: list[MissingInformationItem] = []
        if not patient_case.labs:
            missing.append(
                _missing(
                    "inflammatory_labs",
                    "是否已有 CRP、ESR、血常规、白蛋白、粪便钙卫蛋白等实验室检查？",
                    "实验室与粪便炎症指标可帮助评估是否存在肠道炎症及严重程度。",
                    "high",
                )
            )
        if not has_endoscopy_support:
            missing.append(
                _missing(
                    "ileocolonoscopy",
                    "是否已完成结肠镜并尽量进入回肠末端，且进行多肠段活检？",
                    "指南建议结肠镜用于诊断、疗效评估和监测，疑诊患者应多肠段活检。",
                    "critical",
                )
            )
        if not has_imaging_support:
            missing.append(
                _missing(
                    "cross_sectional_imaging",
                    "是否已有 CTE 或 MRE 来评估小肠受累范围、狭窄、瘘管、脓肿等并发症？",
                    "CTE/MRE有助于评估病变范围和并发症，是疑诊或新诊断阶段的重要信息。",
                    "high",
                )
            )
        if not has_pathology_support:
            missing.append(
                _missing(
                    "pathology",
                    "是否已有活检病理结果，例如慢性炎症、肉芽肿或透壁性炎症等描述？",
                    "病理支持有助于提高疑似程度，但仍需综合判断并排除其他疾病。",
                    "critical",
                )
            )
        if not _has_exclusion_info(patient_case.raw_text, ("肠结核", "结核", "TB", "T-SPOT", "PPD", "IGRA")):
            missing.append(
                _missing(
                    "intestinal_tuberculosis_exclusion",
                    "是否已评估或排除肠结核及相关结核感染证据？",
                    "肠结核可模拟回盲部炎症、溃疡和狭窄，是克罗恩病诊断前必须鉴别的疾病。",
                    "critical",
                )
            )
        if not _has_exclusion_info(patient_case.raw_text, ("感染性肠炎", "粪便培养", "病原", "艰难梭菌", "寄生虫")):
            missing.append(
                _missing(
                    "infectious_enteritis_exclusion",
                    "是否已结合粪便病原学、培养或临床过程评估感染性肠炎？",
                    "感染性肠炎可导致腹泻、发热、炎症指标升高和肠黏膜炎症。",
                    "high",
                )
            )
        if not _has_exclusion_info(patient_case.raw_text, ("NSAID", "非甾体", "药物性肠炎", "止痛药")):
            missing.append(
                _missing(
                    "drug_induced_enteritis_review",
                    "是否有 NSAID、免疫治疗药物、抗生素等用药史可解释肠道损伤？",
                    "药物性肠炎可能模拟炎症性肠病表现，需要在鉴别诊断中回顾。",
                    "medium",
                )
            )
        missing.extend(_missing_from_recommendation_requirements(patient_case, matched_cards))
        return _dedupe_missing_information(missing)

    def _safety_warnings(self, patient_case: PatientCase) -> list[str]:
        warnings = [
            self.red_flag_messages[red_flag]
            for red_flag in patient_case.red_flags
            if red_flag in self.red_flag_messages
        ]
        if patient_case.red_flags:
            warnings.append("红旗征象优先于常规指南推理，应先进行安全分诊和及时线下医疗评估。")
        return warnings

    def _recommended_next_steps(
        self,
        patient_case: PatientCase,
        missing_information: list[MissingInformationItem],
        matched_cards: list[RecommendationCard],
    ) -> list[str]:
        steps: list[str] = []
        missing_keys = {item.information_key for item in missing_information}
        steps.extend(_steps_from_recommendation_cards(matched_cards))
        if "inflammatory_labs" in missing_keys:
            steps.append("补充 CRP、ESR、血常规、白蛋白、粪便钙卫蛋白等检查，用于评估炎症和营养状态。")
        if "ileocolonoscopy" in missing_keys:
            steps.append("完善结肠镜评估，尽量进入回肠末端，并进行多肠段活检及病理评估。")
        if "cross_sectional_imaging" in missing_keys:
            steps.append("完善 CTE 或 MRE，评估小肠病变范围以及狭窄、瘘管、脓肿等并发症。")
        if any(symptom in patient_case.symptoms for symptom in ["肛瘘", "肛周脓肿"]):
            steps.append("存在肛周表现时，考虑肛周 MRI 评估瘘管、脓肿和复杂程度。")
        if "pathology" in missing_keys:
            steps.append("获取或复核活检病理，关注慢性炎症、肉芽肿、透壁性炎症及感染/肿瘤线索。")
        if "intestinal_tuberculosis_exclusion" in missing_keys:
            steps.append("在医生指导下评估肠结核，包括结核感染证据、影像/内镜特征和病理结果。")
        if "infectious_enteritis_exclusion" in missing_keys:
            steps.append("结合病程和粪便病原学检查排除感染性肠炎。")
        if "drug_induced_enteritis_review" in missing_keys:
            steps.append("回顾 NSAID、抗生素、免疫治疗等用药史，评估药物性肠炎可能。")
        if patient_case.red_flags:
            steps.insert(0, "因出现红旗征象，建议优先及时就医或急诊评估。")
        if not steps:
            steps.append("将现有资料交由消化专科医生综合判断，并继续完成鉴别诊断和疾病活动度评估。")
        return _dedupe(steps)

    def _differential_diagnoses(self) -> list[DifferentialDiagnosisItem]:
        return [
            DifferentialDiagnosisItem(
                disease_name="肠结核",
                rationale="可出现回盲部受累、溃疡、狭窄和全身症状，需结合结核感染证据、影像、内镜和病理鉴别。",
                distinguishing_tests=["结核感染相关检查", "胸部影像", "病理抗酸染色/分枝杆菌检测", "治疗反应由医生判断"],
                urgency="soon",
            ),
            DifferentialDiagnosisItem(
                disease_name="溃疡性结肠炎",
                rationale="同属炎症性肠病，可表现为腹泻、便血和结肠炎症；病变连续性、直肠受累和小肠表现有助鉴别。",
                distinguishing_tests=["结肠镜病变分布", "病理评估", "小肠影像"],
            ),
            DifferentialDiagnosisItem(
                disease_name="肠白塞病",
                rationale="可出现肠道溃疡并伴口腔溃疡、外阴溃疡、眼/皮肤表现等系统性线索。",
                distinguishing_tests=["白塞病系统症状评估", "眼科/皮肤评估", "内镜溃疡形态与病理"],
            ),
            DifferentialDiagnosisItem(
                disease_name="肠道淋巴瘤",
                rationale="可造成肠壁增厚、溃疡、狭窄、肿块或全身消耗症状，需病理和免疫组化鉴别。",
                distinguishing_tests=["深取材或重复活检", "免疫组化", "影像评估肿块/淋巴结"],
                urgency="soon",
            ),
            DifferentialDiagnosisItem(
                disease_name="感染性肠炎",
                rationale="可导致腹痛、腹泻、发热、炎症指标升高和肠黏膜炎症，需结合病程和病原学检查。",
                distinguishing_tests=["粪便培养/病原学", "艰难梭菌检测", "寄生虫或特殊感染评估"],
            ),
            DifferentialDiagnosisItem(
                disease_name="药物性肠炎",
                rationale="NSAID等药物可导致肠道溃疡、出血或炎症，需回顾用药时间线。",
                distinguishing_tests=["详细用药史", "停药后变化由医生评估", "内镜和病理排除其他病因"],
            ),
        ]

    def _source_references(
        self,
        matched_cards: list[RecommendationCard],
    ) -> list[EvidenceReference]:
        references = []
        seen: set[str] = set()
        for card in matched_cards:
            if card.recommendation_id in seen:
                continue
            seen.add(card.recommendation_id)
            references.append(
                EvidenceReference(
                    source_name=card.source_section_cn or card.source_section,
                    recommendation_id=card.recommendation_id,
                    source_section=card.source_section,
                    source_section_cn=card.source_section_cn,
                    source_span=card.source_span,
                    source_quote=card.source_quote or card.action,
                    evidence_level=card.evidence_level,
                    recommendation_strength=card.recommendation_strength,
                    page=card.page or _page_from_source_span(card.source_span),
                )
            )
        return references


def execute_crohn_skill(
    patient_case: PatientCase,
    skill_pack: DiseaseSkillPack,
) -> SkillExecutionResult:
    return CrohnDiseaseSkillExecutor().execute(patient_case, skill_pack)


def retrieve_recommendation_cards(
    patient_case: PatientCase,
    skill_pack: DiseaseSkillPack,
    top_k: int = 5,
) -> list[RecommendationCard]:
    return LocalRecommendationRetriever().retrieve_by_patient_case(
        patient_case,
        skill_pack,
        top_k=top_k,
    )


def score_recommendation_card(
    patient_case: PatientCase,
    recommendation_card: RecommendationCard,
) -> int:
    return LocalRecommendationRetriever().score_recommendation_card(
        patient_case,
        recommendation_card,
    )


def _steps_from_recommendation_cards(cards: list[RecommendationCard]) -> list[str]:
    return [
        f"[{card.recommendation_id}] {card.action}"
        for card in cards
        if card.action
    ]


def _missing_from_recommendation_requirements(
    patient_case: PatientCase,
    cards: list[RecommendationCard],
) -> list[MissingInformationItem]:
    missing: list[MissingInformationItem] = []
    for card in cards:
        for required_input in card.required_inputs:
            if _case_has_required_input(patient_case, required_input):
                continue
            missing.append(
                _missing_for_required_input(required_input, card.recommendation_id)
            )
    return missing


def _missing_for_required_input(
    required_input: str,
    recommendation_id: str,
) -> MissingInformationItem:
    key = _required_input_key(required_input)
    display = required_input.replace("_", " ")
    specific = REQUIRED_INPUT_MISSING_MESSAGES.get(
        key,
        (
            f"是否已有 {display} 相关信息？",
            f"{recommendation_id} 的 recommendation card 将 {display} 列为 required input。",
            "medium",
        ),
    )
    return _missing(
        key,
        specific[0],
        f"{specific[1]} 来源：{recommendation_id}。",
        specific[2],
    )


def _patient_concepts(patient_case: PatientCase) -> set[str]:
    searchable_parts = [
        patient_case.raw_text,
        *patient_case.symptoms,
        *patient_case.labs.keys(),
        *patient_case.imaging,
        *patient_case.endoscopy,
        *patient_case.pathology,
        *patient_case.red_flags,
        *patient_case.unknowns,
    ]
    text = " ".join(str(part) for part in searchable_parts)
    return _concepts_from_text(text)


def _concepts_from_text(text: str) -> set[str]:
    normalized_text = text.replace("_", " ").replace("-", " ")
    concepts: set[str] = set()
    for concept, surface_forms in CONCEPT_SURFACES.items():
        if any(_contains_surface_form(normalized_text, surface_form) for surface_form in surface_forms):
            concepts.add(concept)
    return concepts


def _case_has_required_input(patient_case: PatientCase, required_input: str) -> bool:
    key = _required_input_key(required_input)
    raw_text = patient_case.raw_text

    if key in {"available_case_information", "missing_information_review"}:
        return True
    if key in {"symptoms", "current_symptoms", "systemic_features"}:
        return bool(patient_case.symptoms)
    if key in {"labs", "current_labs", "inflammatory_markers"}:
        return bool(patient_case.labs)
    if key in {"fecal_calprotectin"}:
        return "粪便钙卫蛋白" in patient_case.labs or _contains_any(raw_text, ("粪便钙卫蛋白", "粪钙卫蛋白", "FC", "fecal calprotectin"))
    if key in {"imaging", "prior_imaging", "small_bowel_imaging", "prior_small_bowel_imaging"}:
        return bool(patient_case.imaging)
    if key in {"endoscopy", "prior_endoscopy"}:
        return bool(patient_case.endoscopy)
    if key in {"pathology", "pathology_report", "upper_gi_biopsy_pathology"}:
        return bool(patient_case.pathology)
    if key in {"colonoscopy_report", "prior_colonoscopy"}:
        return "结肠镜" in patient_case.endoscopy or _contains_any(raw_text, ("结肠镜", "肠镜", "colonoscopy"))
    if key == "terminal_ileum_assessment":
        return "回肠末端" in patient_case.endoscopy or _contains_any(raw_text, ("回肠末端", "回肠末段", "末端回肠", "terminal ileum"))
    if key == "biopsy_sites":
        return "活检" in patient_case.endoscopy or _contains_any(raw_text, ("活检", "biopsy"))
    if key == "cte_report":
        return "CTE" in patient_case.imaging or _contains_any(raw_text, ("CTE", "CT小肠成像", "CT enterography"))
    if key == "mre_report":
        return "MRE" in patient_case.imaging or _contains_any(raw_text, ("MRE", "磁共振小肠成像", "MR enterography"))
    if key == "perianal_symptoms":
        return any(symptom in patient_case.symptoms for symptom in ["肛瘘", "肛周脓肿"]) or _contains_any(raw_text, ("肛周", "肛瘘", "肛周脓肿"))
    if key == "perianal_exam":
        return _contains_any(raw_text, ("肛周检查", "肛门检查", "肛周", "肛瘘", "肛周脓肿"))
    if key == "perianal_mri":
        return "肛周 MRI" in patient_case.imaging or _contains_any(raw_text, ("肛周 MRI", "肛周MRI", "肛周磁共振"))
    if key == "abscess_assessment":
        return "肛周脓肿" in patient_case.symptoms or _contains_any(raw_text, ("脓肿", "abscess"))
    if key == "stricture_risk_assessment":
        return "狭窄" in patient_case.endoscopy or _contains_any(raw_text, ("狭窄", "肠梗阻", "梗阻", "stricture", "stenosis"))
    if key == "tuberculosis_workup":
        return _contains_any(raw_text, ("肠结核", "结核", "TB", "T-SPOT", "PPD", "IGRA"))
    if key == "infection_workup":
        return _contains_any(raw_text, ("感染性肠炎", "粪便培养", "病原", "艰难梭菌", "寄生虫"))
    if key == "medication_history":
        return bool(patient_case.medication_history) or _contains_any(raw_text, ("NSAID", "非甾体", "抗生素", "用药", "药物"))
    if key in {"disease_extent", "disease_activity", "complications", "phenotype"}:
        return bool(patient_case.imaging or patient_case.endoscopy)
    if key in {"age_at_onset", "smoking_status", "renal_function_if_contrast_needed", "pregnancy_status_if_relevant", "contraindications", "infection_risk", "prior_medication_response", "comorbidities", "nutrition_status", "mental_health_status", "diagnosis_status", "treatment_history"}:
        return _contains_any(raw_text, (key.replace("_", " "),))
    return _contains_any(raw_text, (required_input, required_input.replace("_", " ")))


def _required_input_key(required_input: str) -> str:
    return required_input.strip().casefold().replace("-", "_")


def _contains_any(text: str, surface_forms: tuple[str, ...]) -> bool:
    return any(_contains_surface_form(text, surface_form) for surface_form in surface_forms)


def _contains_surface_form(text: str, surface_form: str) -> bool:
    if not surface_form:
        return False
    if re.fullmatch(r"[A-Za-z0-9 .'\-/]+", surface_form):
        pattern = re.escape(surface_form).replace(r"\ ", r"\s+")
        return re.search(
            rf"(?<![A-Za-z0-9]){pattern}(?![A-Za-z0-9])",
            text,
            re.IGNORECASE,
        ) is not None
    return surface_form.casefold() in text.casefold()


def _lab_suggests_inflammation(value_texts: object) -> bool:
    if not isinstance(value_texts, list):
        return False
    if not value_texts:
        return False
    text = " ".join(str(value) for value in value_texts)
    if re.search(r"升高|阳性|>", text):
        return True
    return re.search(r"\d", text) is not None and not re.search(r"正常|阴性|降低", text)


def _doctor_confirmed(raw_text: str) -> bool:
    patterns = (
        r"医生.{0,8}(已确诊|确诊|诊断).{0,8}(克罗恩病|Crohn|CD)",
        r"(克罗恩病|Crohn|CD).{0,8}(医生.{0,8}(已确诊|确诊|诊断))",
        r"病理明确支持.{0,12}医生.{0,8}(已确诊|确诊)",
    )
    return any(re.search(pattern, raw_text, re.IGNORECASE) for pattern in patterns)


def _has_exclusion_info(raw_text: str, terms: tuple[str, ...]) -> bool:
    return any(re.search(re.escape(term), raw_text, re.IGNORECASE) for term in terms)


def _page_from_source_span(source_span: str) -> int | None:
    match = re.search(r"PDF page\s*(\d+)", source_span, re.IGNORECASE)
    if match:
        return int(match.group(1))
    match = re.search(r"PDF pages\s*(\d+)", source_span, re.IGNORECASE)
    if match:
        return int(match.group(1))
    match = re.search(r"pages?\s*(\d+)", source_span, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def _missing(
    information_key: str,
    question: str,
    reason: str,
    priority: str,
) -> MissingInformationItem:
    return MissingInformationItem(
        information_key=information_key,
        question=question,
        reason=reason,
        priority=priority,  # type: ignore[arg-type]
    )


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


def _dedupe_missing_information(
    items: list[MissingInformationItem],
) -> list[MissingInformationItem]:
    seen: set[str] = set()
    deduped: list[MissingInformationItem] = []
    for item in items:
        if item.information_key in seen:
            continue
        seen.add(item.information_key)
        deduped.append(item)
    return deduped


CONCEPT_SURFACES = {
    "symptom": ("symptoms", "clinical manifestations", "症状", "临床表现"),
    "abdominal_pain": ("abdominal pain", "腹痛", "肚子痛", "腹部疼痛"),
    "diarrhea": ("diarrhea", "chronic diarrhea", "腹泻", "拉肚子", "稀便"),
    "weight_loss": ("weight loss", "体重下降", "体重减轻", "消瘦"),
    "fever": ("fever", "发热", "高热", "发烧"),
    "fecal_calprotectin": ("fecal calprotectin", "FC", "粪便钙卫蛋白", "粪钙卫蛋白", "钙卫蛋白"),
    "crp": ("C-reactive protein", "CRP", "C反应蛋白", "C-反应蛋白"),
    "esr": ("erythrocyte sedimentation rate", "ESR", "血沉", "红细胞沉降率"),
    "lab": ("laboratory", "laboratory tests", "labs", "实验室", "炎症指标"),
    "colonoscopy": ("colonoscopy", "ileocolonoscopy", "结肠镜", "肠镜"),
    "terminal_ileum": ("terminal ileum", "ileoscopy", "回肠末端", "回肠末段", "末端回肠"),
    "ileocecal": ("ileocecal", "ileocecal region", "回盲部", "回盲瓣"),
    "gastroduodenoscopy": ("gastroduodenoscopy", "upper gastrointestinal endoscopy", "胃十二指肠镜", "胃镜"),
    "capsule_endoscopy": ("capsule endoscopy", "胶囊内镜"),
    "biopsy": ("biopsy", "biopsies", "活检", "病理活检"),
    "pathology": ("pathology", "histopathology", "pathological", "病理", "组织学"),
    "cte": ("CTE", "CT enterography", "CT小肠成像"),
    "mre": ("MRE", "MR enterography", "磁共振小肠成像"),
    "mri": ("MRI", "磁共振"),
    "perianal_mri": ("perianal MRI", "肛周 MRI", "肛周MRI", "肛周磁共振"),
    "imaging": ("imaging", "radiologic imaging", "cross-sectional imaging", "影像", "成像"),
    "ulcer": ("ulcer", "ulcers", "溃疡"),
    "multiple_ulcers": ("multiple ulcers", "多发溃疡", "多处溃疡"),
    "longitudinal_ulcer": ("longitudinal ulcer", "纵行溃疡", "纵形溃疡"),
    "cobblestone": ("cobblestone", "cobblestone appearance", "铺路石样", "铺路石样改变", "鹅卵石样"),
    "stricture": ("stricture", "stenosis", "stricture risk", "狭窄"),
    "retention_risk": ("retention risk", "capsule retention", "capsule retention risk", "胶囊滞留", "滞留风险"),
    "fistula": ("fistula", "anal fistula", "perianal fistula", "瘘管", "肛瘘", "肛周瘘"),
    "perianal_abscess": ("abscess", "perianal abscess", "肛周脓肿", "肛旁脓肿", "脓肿"),
    "granuloma": ("granuloma", "肉芽肿"),
    "chronic_inflammation": ("chronic inflammation", "慢性炎症", "慢性活动性炎症"),
    "transmural_inflammation": ("transmural inflammation", "透壁性炎症", "全层炎症"),
    "complication": ("complication", "complications", "并发症"),
    "extent": ("extent", "lesion extent", "disease extent", "范围", "病变范围"),
    "intestinal_tuberculosis": ("intestinal tuberculosis", "肠结核", "结核"),
    "intestinal_behcet": ("intestinal Behcet", "Behcet", "肠白塞病", "肠型贝赫切特"),
    "lymphoma": ("lymphoma", "淋巴瘤"),
    "infectious_enteritis": ("infectious enteritis", "感染性肠炎"),
    "drug_induced_enteritis": ("drug-induced enteritis", "drug injury", "药物性肠炎", "药物损伤"),
}


REQUIRED_INPUT_MISSING_MESSAGES = {
    "symptoms": ("是否已补充主要症状、持续时间和严重程度？", "该推荐需要症状信息作为判断基础", "high"),
    "labs": ("是否已有实验室检查信息？", "该推荐需要实验室信息参与综合判断", "high"),
    "fecal_calprotectin": ("是否已有粪便钙卫蛋白/FC结果？", "该推荐需要粪便钙卫蛋白评估肠道炎症水平", "medium"),
    "imaging": ("是否已有影像学检查信息？", "该推荐需要影像学信息参与综合判断", "high"),
    "endoscopy": ("是否已有内镜检查信息？", "该推荐需要内镜信息参与综合判断", "high"),
    "pathology": ("是否已有病理检查信息？", "该推荐需要病理信息参与综合判断", "critical"),
    "colonoscopy_report": ("是否已有结肠镜报告？", "结肠镜相关推荐需要结肠镜报告", "critical"),
    "terminal_ileum_assessment": ("结肠镜是否已尽量进入并描述回肠末端？", "结肠镜推荐要求尽量进入回肠末端", "high"),
    "biopsy_sites": ("是否已进行多肠段活检并记录取材部位？", "疑诊患者推荐多肠段活检", "high"),
    "pathology_report": ("是否已有活检病理报告？", "内镜活检推荐需要病理结果闭环", "critical"),
    "prior_colonoscopy": ("是否已有既往结肠镜结果？", "胶囊内镜选择需先了解结肠镜是否未能明确诊断", "high"),
    "prior_small_bowel_imaging": ("是否已有小肠放射影像学检查结果？", "胶囊内镜选择需结合小肠影像是否未能明确诊断", "high"),
    "stricture_risk_assessment": ("胶囊内镜前是否已评估肠道狭窄和胶囊滞留风险？", "胶囊内镜推荐要求先评估狭窄和滞留风险", "critical"),
    "cte_report": ("是否已有 CTE 报告？", "CTE/MRE 推荐需要影像报告评估病变范围和并发症", "high"),
    "mre_report": ("是否已有 MRE 报告？", "CTE/MRE 推荐需要影像报告评估病变范围和并发症", "high"),
    "perianal_symptoms": ("是否有肛周疼痛、流脓、肛瘘或脓肿等肛周表现？", "肛周 MRI 推荐需要肛周症状线索", "medium"),
    "perianal_exam": ("是否已有肛周体格检查或外科评估？", "肛周病变评估需要体格检查/专科评估", "medium"),
    "perianal_mri": ("是否已完善肛周 MRI？", "肛瘘诊断首选肛周 MRI", "high"),
    "abscess_assessment": ("是否评估有无肛周脓肿？", "肛周复杂病变需要明确脓肿情况", "high"),
    "tuberculosis_workup": ("是否已评估肠结核或结核感染证据？", "鉴别诊断推荐要求排除肠结核", "critical"),
    "infection_workup": ("是否已评估感染性肠炎？", "鉴别诊断推荐要求排除感染性肠炎", "high"),
    "medication_history": ("是否已补充 NSAID、抗生素、免疫治疗等用药史？", "鉴别诊断推荐需要回顾药物性肠炎可能", "medium"),
}
