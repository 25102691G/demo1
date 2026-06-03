from __future__ import annotations

import re
from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict, Field

from guideline_skill.schema import DiseaseSkillPack, PatientCase


class RoutingResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    skill_name: str
    disease_name: str
    score: int = Field(ge=0)
    matched_symptoms: list[str] = Field(default_factory=list)
    matched_tests: list[str] = Field(default_factory=list)
    matched_findings: list[str] = Field(default_factory=list)
    matched_red_flags: list[str] = Field(default_factory=list)
    matched_aliases: list[str] = Field(default_factory=list)
    reason: str


class DiseaseSkillRouter:
    """Rule-based disease skill recall. Scores are not diagnostic probabilities."""

    alias_weight = 5
    symptom_weight = 2
    test_weight = 2
    finding_weight = 3
    red_flag_weight = 4

    def route(
        self,
        patient_case: PatientCase,
        skill_packs: list[DiseaseSkillPack],
        top_k: int = 5,
    ) -> list[RoutingResult]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        results = [
            self._score_skill_pack(patient_case, skill_pack)
            for skill_pack in skill_packs
        ]
        results.sort(key=lambda result: result.score, reverse=True)
        return results[:top_k]

    def _score_skill_pack(
        self,
        patient_case: PatientCase,
        skill_pack: DiseaseSkillPack,
    ) -> RoutingResult:
        routing = skill_pack.routing_profile

        matched_aliases = _match_terms(
            routing.disease_aliases,
            patient_terms=[],
            raw_text=patient_case.raw_text,
        )
        matched_symptoms = _match_terms(
            routing.key_symptoms,
            patient_terms=patient_case.symptoms,
            raw_text=patient_case.raw_text,
        )
        matched_tests = _match_terms(
            routing.key_tests,
            patient_terms=_case_test_terms(patient_case),
            raw_text=patient_case.raw_text,
        )
        matched_findings = _match_terms(
            routing.key_findings,
            patient_terms=_case_finding_terms(patient_case),
            raw_text=patient_case.raw_text,
        )
        matched_red_flags = _match_terms(
            routing.red_flags,
            patient_terms=patient_case.red_flags,
            raw_text=patient_case.raw_text,
        )

        score = (
            len(matched_aliases) * self.alias_weight
            + len(matched_symptoms) * self.symptom_weight
            + len(matched_tests) * self.test_weight
            + len(matched_findings) * self.finding_weight
            + len(matched_red_flags) * self.red_flag_weight
        )

        return RoutingResult(
            skill_name=skill_pack.skill_name,
            disease_name=skill_pack.disease_name,
            score=score,
            matched_symptoms=matched_symptoms,
            matched_tests=matched_tests,
            matched_findings=matched_findings,
            matched_red_flags=matched_red_flags,
            matched_aliases=matched_aliases,
            reason=_build_reason(
                score=score,
                matched_aliases=matched_aliases,
                matched_symptoms=matched_symptoms,
                matched_tests=matched_tests,
                matched_findings=matched_findings,
                matched_red_flags=matched_red_flags,
            ),
        )


def route_disease_skills(
    patient_case: PatientCase,
    skill_packs: list[DiseaseSkillPack],
    top_k: int = 5,
) -> list[RoutingResult]:
    return DiseaseSkillRouter().route(patient_case, skill_packs, top_k=top_k)


def _case_test_terms(patient_case: PatientCase) -> list[str]:
    return _dedupe(
        [
            *patient_case.labs.keys(),
            *patient_case.imaging,
            *patient_case.endoscopy,
            *patient_case.pathology,
        ]
    )


def _case_finding_terms(patient_case: PatientCase) -> list[str]:
    return _dedupe(
        [
            *patient_case.symptoms,
            *patient_case.imaging,
            *patient_case.endoscopy,
            *patient_case.pathology,
        ]
    )


def _match_terms(
    profile_terms: Iterable[str],
    patient_terms: Iterable[str],
    raw_text: str,
) -> list[str]:
    patient_keys = set()
    for term in patient_terms:
        patient_keys.update(_expanded_keys(term))

    matched: list[str] = []
    for profile_term in profile_terms:
        profile_keys = _expanded_keys(profile_term)
        if patient_keys.intersection(profile_keys) or _raw_text_contains_any(raw_text, profile_keys):
            matched.append(profile_term)
    return matched


def _expanded_keys(term: str) -> set[str]:
    key = _term_key(term)
    return SYNONYM_INDEX.get(key, {key})


def _raw_text_contains_any(raw_text: str, term_keys: set[str]) -> bool:
    for term_key in term_keys:
        for surface_form in SURFACE_FORMS.get(term_key, {term_key}):
            if _contains_surface_form(raw_text, surface_form):
                return True
    return False


def _contains_surface_form(raw_text: str, surface_form: str) -> bool:
    if not surface_form:
        return False

    if re.fullmatch(r"[A-Za-z0-9 .'\-/]+", surface_form):
        pattern = re.escape(surface_form).replace(r"\ ", r"\s+")
        return re.search(
            rf"(?<![A-Za-z0-9]){pattern}(?![A-Za-z0-9])",
            raw_text,
            re.IGNORECASE,
        ) is not None

    return surface_form.casefold() in raw_text.casefold()


def _term_key(term: str) -> str:
    return re.sub(r"[\s\-_'/()]+", "", term.casefold())


def _build_reason(
    *,
    score: int,
    matched_aliases: list[str],
    matched_symptoms: list[str],
    matched_tests: list[str],
    matched_findings: list[str],
    matched_red_flags: list[str],
) -> str:
    if score == 0:
        return "No routing profile terms matched; score is 0 and is not a diagnostic probability."

    parts = [
        _format_reason_part("aliases", matched_aliases),
        _format_reason_part("symptoms", matched_symptoms),
        _format_reason_part("tests", matched_tests),
        _format_reason_part("findings", matched_findings),
        _format_reason_part("red_flags", matched_red_flags),
    ]
    matched_text = "; ".join(part for part in parts if part)
    return f"Rule-based recall score only, not diagnostic probability. Matched {matched_text}."


def _format_reason_part(label: str, values: list[str]) -> str:
    if not values:
        return ""
    return f"{label}: {', '.join(values)}"


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


SYNONYM_GROUPS = (
    ("Crohn disease", "Crohn's disease", "CD", "克罗恩病", "克隆病"),
    ("abdominal pain", "腹痛", "肚子痛", "腹部疼痛", "right lower abdominal pain", "右下腹痛"),
    ("chronic diarrhea", "diarrhea", "腹泻", "拉肚子", "稀便", "大便次数增多"),
    ("hematochezia", "便血", "血便", "大便带血", "black stool", "黑便"),
    ("weight loss", "体重下降", "体重减轻", "消瘦", "明显变瘦"),
    ("fever", "发热", "发烧", "低热", "高热"),
    ("perianal fistula", "fistula", "肛瘘", "肛周瘘", "瘘管"),
    ("perianal abscess", "abscess", "肛周脓肿", "肛旁脓肿"),
    ("recurrent oral ulcer", "oral ulcer", "口腔溃疡", "口疮"),
    ("fatigue", "乏力", "疲乏", "疲劳"),
    ("fecal calprotectin", "FC", "粪便钙卫蛋白", "粪钙卫蛋白", "钙卫蛋白"),
    ("C-reactive protein", "CRP", "C反应蛋白", "C-反应蛋白"),
    ("erythrocyte sedimentation rate", "ESR", "血沉", "红细胞沉降率"),
    ("albumin", "ALB", "白蛋白"),
    ("blood routine", "complete blood count", "CBC", "血常规", "白细胞", "血红蛋白", "贫血"),
    ("colonoscopy", "结肠镜", "肠镜"),
    ("ileoscopy", "terminal ileum", "terminal ileum involvement", "回肠末端", "末端回肠"),
    ("ileocecal", "ileocecal region", "回盲部", "回盲瓣"),
    ("gastroduodenoscopy", "upper gastrointestinal endoscopy", "胃十二指肠镜", "胃镜"),
    ("capsule endoscopy", "胶囊内镜"),
    ("biopsy pathology", "biopsy", "pathology", "活检", "病理", "病理活检"),
    ("CTE", "CT enterography", "CT小肠成像"),
    ("MRE", "MR enterography", "磁共振小肠成像"),
    ("perianal MRI", "肛周 MRI", "肛周MRI", "肛周磁共振", "盆腔 MRI"),
    ("MRI", "磁共振"),
    ("intestinal ultrasound", "肠道超声", "肠超"),
    ("ulcer", "溃疡"),
    ("longitudinal ulcer", "纵行溃疡", "纵形溃疡"),
    ("cobblestone appearance", "铺路石样改变", "铺路石样", "鹅卵石样"),
    ("stricture", "stenosis", "狭窄"),
    ("granuloma", "肉芽肿"),
    ("chronic inflammation", "慢性炎症", "慢性活动性炎症"),
    ("transmural inflammation", "透壁性炎症", "全层炎症"),
    ("acute abdomen", "急腹症"),
    ("suspected perforation", "肠穿孔", "疑似穿孔"),
    ("intestinal obstruction", "肠梗阻", "停止排气排便", "无法排气排便"),
    ("massive gastrointestinal bleeding", "大量便血", "大量血便"),
    ("high fever with severe abdominal pain", "高热", "剧烈腹痛", "严重腹痛"),
    ("sepsis concern", "脓毒症", "感染性休克"),
    ("severe dehydration", "严重脱水", "明显脱水", "尿少口干"),
    ("consciousness disturbance", "意识障碍", "意识不清", "昏迷", "嗜睡"),
    ("intestinal tuberculosis", "肠结核"),
    ("intestinal Behcet disease", "肠白塞病", "肠型贝赫切特综合征"),
    ("lymphoma", "淋巴瘤"),
    ("infectious enteritis", "感染性肠炎"),
    ("drug-induced enteritis", "药物性肠炎"),
    ("ulcerative colitis", "溃疡性结肠炎", "UC"),
    ("ischemic enteritis", "缺血性肠炎"),
)


def _build_synonym_index() -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    synonym_index: dict[str, set[str]] = {}
    surface_forms: dict[str, set[str]] = {}
    for group in SYNONYM_GROUPS:
        keys = {_term_key(term) for term in group}
        for term in group:
            key = _term_key(term)
            synonym_index[key] = keys
            for group_key in keys:
                surface_forms.setdefault(group_key, set()).add(term)
    return synonym_index, surface_forms


SYNONYM_INDEX, SURFACE_FORMS = _build_synonym_index()
