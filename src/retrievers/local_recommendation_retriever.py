from __future__ import annotations

import re

from clinical_stage_classifier import ClinicalStageResult, classify_clinical_stage
from guideline_skill.schema import (
    DifferentialDiagnosisItem,
    DiseaseSkillPack,
    PatientCase,
    RecommendationCard,
)

from .base import GuidelineRetriever


class LocalRecommendationRetriever(GuidelineRetriever):
    """YAML/JSON-backed mock retriever over DiseaseSkillPack recommendation cards."""

    def retrieve_by_patient_case(
        self,
        patient_case: PatientCase,
        skill_pack: DiseaseSkillPack,
        top_k: int = 5,
    ) -> list[RecommendationCard]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        clinical_stage = classify_clinical_stage(patient_case)
        scored_cards = [
            (self.score_recommendation_card(patient_case, card, clinical_stage), index, card)
            for index, card in enumerate(skill_pack.recommendation_cards)
        ]
        scored_cards.sort(key=lambda item: (-item[0], item[1]))
        return [card for score, _, card in scored_cards if score > 0][:top_k]

    def retrieve_by_query(
        self,
        query: str,
        skill_pack: DiseaseSkillPack,
        top_k: int = 5,
    ) -> list[RecommendationCard]:
        patient_case = PatientCase(raw_text=query)
        return self.retrieve_by_patient_case(patient_case, skill_pack, top_k=top_k)

    def retrieve_differential_diagnoses(
        self,
        patient_case: PatientCase,
        skill_pack: DiseaseSkillPack,
        top_k: int = 5,
    ) -> list[DifferentialDiagnosisItem]:
        diagnoses = [
            _differential_item(disease_name)
            for disease_name in skill_pack.routing_profile.must_differentiate
        ]
        return diagnoses[:top_k]

    def score_recommendation_card(
        self,
        patient_case: PatientCase,
        recommendation_card: RecommendationCard,
        clinical_stage: ClinicalStageResult | None = None,
    ) -> int:
        patient_concepts = patient_concepts_from_case(patient_case)
        score = 0
        stage_result = clinical_stage or classify_clinical_stage(patient_case)

        score += 4 * len(
            patient_concepts.intersection(concepts_from_text(recommendation_card.clinical_task))
        )
        score += 3 * len(
            patient_concepts.intersection(
                concepts_from_text(
                    " ".join(
                        [
                            recommendation_card.condition,
                            recommendation_card.action,
                        ]
                    )
                )
            )
        )
        score += 2 * len(
            patient_concepts.intersection(
                concepts_from_text(
                    " ".join(
                        [
                            recommendation_card.source_section,
                            recommendation_card.rationale,
                            " ".join(recommendation_card.safety_notes),
                        ]
                    )
                )
            )
        )

        for required_input in recommendation_card.required_inputs:
            if case_has_required_input(patient_case, required_input):
                score += 3
            elif patient_concepts.intersection(concepts_from_text(required_input)):
                score += 1

        score += clinical_stage_score(recommendation_card, stage_result)
        score += clinical_context_bonus(patient_concepts, recommendation_card, stage_result)
        return max(score, 0)


def clinical_stage_score(
    recommendation_card: RecommendationCard,
    clinical_stage: ClinicalStageResult,
) -> int:
    card_stage = recommendation_card.clinical_stage
    if not card_stage:
        return 0

    active_stages = set(clinical_stage.stages)
    if card_stage in active_stages:
        bonus = 10
        if card_stage == clinical_stage.primary_stage:
            bonus += 2
        return bonus

    if "treatment_selection" in active_stages and card_stage == "treatment_readiness":
        return 10
    if "followup_monitoring" in active_stages and card_stage in {"diagnostic_workup", "extent_and_complication_assessment"}:
        return 2

    if card_stage in {"treatment_selection", "treatment_readiness"}:
        return -12
    if card_stage in {"followup_monitoring", "mental_health_monitoring"}:
        return -12

    if active_stages.intersection({"diagnostic_workup", "differential_diagnosis"}):
        if card_stage in {"extent_and_complication_assessment", "initial_screening"}:
            return 2
        return -4

    if "extent_and_complication_assessment" in active_stages and card_stage == "diagnostic_workup":
        return 2

    return -2


def clinical_context_bonus(
    patient_concepts: set[str],
    recommendation_card: RecommendationCard,
    clinical_stage: ClinicalStageResult,
) -> int:
    card_stage = recommendation_card.clinical_stage
    if not card_stage:
        return 0

    bonus = 0
    card_concepts = concepts_from_card(recommendation_card)
    if (
        card_stage == "extent_and_complication_assessment"
        and "extent_and_complication_assessment" in clinical_stage.stages
    ):
        if "stricture" in patient_concepts and card_concepts.intersection({"stricture", "cte", "mre", "extent", "complication"}):
            bonus += 12
        if patient_concepts.intersection({"fistula", "perianal_abscess"}) and card_concepts.intersection({"fistula", "perianal_abscess", "perianal_mri"}):
            bonus += 12
    if (
        card_stage == "differential_diagnosis"
        and "differential_diagnosis" in clinical_stage.stages
    ):
        bonus += 8
    return bonus


def concepts_from_card(recommendation_card: RecommendationCard) -> set[str]:
    return concepts_from_text(
        " ".join(
            [
                recommendation_card.source_section,
                recommendation_card.clinical_task,
                recommendation_card.condition,
                recommendation_card.action,
                recommendation_card.rationale,
                " ".join(recommendation_card.required_inputs),
                " ".join(recommendation_card.safety_notes),
            ]
        )
    )


def patient_concepts_from_case(patient_case: PatientCase) -> set[str]:
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
    return concepts_from_text(text)


def concepts_from_text(text: str) -> set[str]:
    normalized_text = text.replace("_", " ").replace("-", " ")
    concepts: set[str] = set()
    for concept, surface_forms in CONCEPT_SURFACES.items():
        if any(contains_surface_form(normalized_text, surface_form) for surface_form in surface_forms):
            concepts.add(concept)
    return concepts


def case_has_required_input(patient_case: PatientCase, required_input: str) -> bool:
    key = required_input_key(required_input)
    raw_text = patient_case.raw_text

    if key in {"available_case_information", "missing_information_review"}:
        return True
    if key in {"symptoms", "current_symptoms", "systemic_features"}:
        return bool(patient_case.symptoms)
    if key in {"labs", "current_labs", "inflammatory_markers"}:
        return bool(patient_case.labs)
    if key in {"fecal_calprotectin"}:
        return "粪便钙卫蛋白" in patient_case.labs or contains_any(raw_text, ("粪便钙卫蛋白", "粪钙卫蛋白", "FC", "fecal calprotectin"))
    if key in {"imaging", "prior_imaging", "small_bowel_imaging", "prior_small_bowel_imaging"}:
        return bool(patient_case.imaging)
    if key in {"endoscopy", "prior_endoscopy"}:
        return bool(patient_case.endoscopy)
    if key in {"pathology", "pathology_report", "upper_gi_biopsy_pathology"}:
        return bool(patient_case.pathology)
    if key in {"colonoscopy_report", "prior_colonoscopy"}:
        return "结肠镜" in patient_case.endoscopy or contains_any(raw_text, ("结肠镜", "肠镜", "colonoscopy"))
    if key == "terminal_ileum_assessment":
        return "回肠末端" in patient_case.endoscopy or contains_any(raw_text, ("回肠末端", "回肠末段", "末端回肠", "terminal ileum"))
    if key == "biopsy_sites":
        return "活检" in patient_case.endoscopy or contains_any(raw_text, ("活检", "biopsy"))
    if key == "cte_report":
        return "CTE" in patient_case.imaging or contains_any(raw_text, ("CTE", "CT小肠成像", "CT enterography"))
    if key == "mre_report":
        return "MRE" in patient_case.imaging or contains_any(raw_text, ("MRE", "磁共振小肠成像", "MR enterography"))
    if key == "perianal_symptoms":
        return any(symptom in patient_case.symptoms for symptom in ["肛瘘", "肛周脓肿"]) or contains_any(raw_text, ("肛周", "肛瘘", "肛周脓肿"))
    if key == "perianal_exam":
        return contains_any(raw_text, ("肛周检查", "肛门检查", "肛周", "肛瘘", "肛周脓肿"))
    if key == "perianal_mri":
        return "肛周 MRI" in patient_case.imaging or contains_any(raw_text, ("肛周 MRI", "肛周MRI", "肛周磁共振"))
    if key == "abscess_assessment":
        return "肛周脓肿" in patient_case.symptoms or contains_any(raw_text, ("脓肿", "abscess"))
    if key == "stricture_risk_assessment":
        return "狭窄" in patient_case.endoscopy or contains_any(raw_text, ("狭窄", "肠梗阻", "梗阻", "stricture", "stenosis"))
    if key == "tuberculosis_workup":
        return contains_any(raw_text, ("肠结核", "结核", "TB", "T-SPOT", "PPD", "IGRA"))
    if key == "infection_workup":
        return contains_any(raw_text, ("感染性肠炎", "粪便培养", "病原", "艰难梭菌", "寄生虫"))
    if key == "medication_history":
        return bool(patient_case.medication_history) or contains_any(raw_text, ("NSAID", "非甾体", "抗生素", "用药", "药物"))
    if key in {"disease_extent", "disease_activity", "complications", "phenotype"}:
        return bool(patient_case.imaging or patient_case.endoscopy)
    if key in {"age_at_onset", "smoking_status", "renal_function_if_contrast_needed", "pregnancy_status_if_relevant", "contraindications", "infection_risk", "prior_medication_response", "comorbidities", "nutrition_status", "mental_health_status", "diagnosis_status", "treatment_history"}:
        return contains_any(raw_text, (key.replace("_", " "),))
    return contains_any(raw_text, (required_input, required_input.replace("_", " ")))


def required_input_key(required_input: str) -> str:
    return required_input.strip().casefold().replace("-", "_")


def contains_any(text: str, surface_forms: tuple[str, ...]) -> bool:
    return any(contains_surface_form(text, surface_form) for surface_form in surface_forms)


def contains_surface_form(text: str, surface_form: str) -> bool:
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


def _differential_item(disease_name: str) -> DifferentialDiagnosisItem:
    return DIFFERENTIAL_DIAGNOSIS_MAP.get(
        disease_name,
        DifferentialDiagnosisItem(
            disease_name=disease_name,
            rationale="Listed in the skill pack routing profile as a disease that must be differentiated.",
            distinguishing_tests=[],
        ),
    )


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


DIFFERENTIAL_DIAGNOSIS_MAP = {
    "intestinal tuberculosis": DifferentialDiagnosisItem(
        disease_name="肠结核",
        rationale="可出现回盲部受累、溃疡、狭窄和全身症状，需结合结核感染证据、影像、内镜和病理鉴别。",
        distinguishing_tests=["结核感染相关检查", "胸部影像", "病理抗酸染色/分枝杆菌检测", "治疗反应由医生判断"],
        urgency="soon",
    ),
    "intestinal Behcet disease": DifferentialDiagnosisItem(
        disease_name="肠白塞病",
        rationale="可出现肠道溃疡并伴口腔溃疡、外阴溃疡、眼/皮肤表现等系统性线索。",
        distinguishing_tests=["白塞病系统症状评估", "眼科/皮肤评估", "内镜溃疡形态与病理"],
    ),
    "lymphoma": DifferentialDiagnosisItem(
        disease_name="肠道淋巴瘤",
        rationale="可造成肠壁增厚、溃疡、狭窄、肿块或全身消耗症状，需病理和免疫组化鉴别。",
        distinguishing_tests=["深取材或重复活检", "免疫组化", "影像评估肿块/淋巴结"],
        urgency="soon",
    ),
    "infectious enteritis": DifferentialDiagnosisItem(
        disease_name="感染性肠炎",
        rationale="可导致腹痛、腹泻、发热、炎症指标升高和肠黏膜炎症，需结合病程和病原学检查。",
        distinguishing_tests=["粪便培养/病原学", "艰难梭菌检测", "寄生虫或特殊感染评估"],
    ),
    "drug-induced enteritis": DifferentialDiagnosisItem(
        disease_name="药物性肠炎",
        rationale="NSAID等药物可导致肠道溃疡、出血或炎症，需回顾用药时间线。",
        distinguishing_tests=["详细用药史", "停药后变化由医生评估", "内镜和病理排除其他病因"],
    ),
    "ulcerative colitis": DifferentialDiagnosisItem(
        disease_name="溃疡性结肠炎",
        rationale="同属炎症性肠病，可表现为腹泻、便血和结肠炎症；病变连续性、直肠受累和小肠表现有助鉴别。",
        distinguishing_tests=["结肠镜病变分布", "病理评估", "小肠影像"],
    ),
}
