from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Pattern

from guideline_skill.schema import PatientCase


@dataclass(frozen=True)
class KeywordRule:
    canonical: str
    patterns: tuple[str, ...]

    def compiled(self) -> tuple[Pattern[str], ...]:
        return tuple(re.compile(pattern, re.IGNORECASE) for pattern in self.patterns)


SYMPTOM_RULES = (
    KeywordRule("腹痛", (r"腹痛", r"肚子痛", r"腹部疼痛", r"右下腹痛")),
    KeywordRule("腹泻", (r"腹泻", r"拉肚子", r"稀便", r"大便次数增多")),
    KeywordRule("便血", (r"便血", r"血便", r"大便带血", r"黑便")),
    KeywordRule("体重下降", (r"体重下降", r"体重减轻", r"消瘦", r"明显变瘦")),
    KeywordRule("发热", (r"发热", r"发烧", r"低热", r"高热", r"体温升高")),
    KeywordRule("肛瘘", (r"肛瘘", r"肛周瘘", r"肛门瘘")),
    KeywordRule("肛周脓肿", (r"肛周脓肿", r"肛旁脓肿", r"肛门周围脓肿")),
    KeywordRule("口腔溃疡", (r"口腔溃疡", r"口疮", r"复发性口腔溃疡")),
    KeywordRule("乏力", (r"乏力", r"疲乏", r"疲劳", r"没力气")),
)

LAB_RULES = (
    KeywordRule("CRP", (r"(?<![A-Za-z])CRP(?![A-Za-z])", r"C反应蛋白", r"C-反应蛋白")),
    KeywordRule("ESR", (r"(?<![A-Za-z])ESR(?![A-Za-z])", r"血沉", r"红细胞沉降率")),
    KeywordRule("白蛋白", (r"白蛋白", r"(?<![A-Za-z])ALB(?![A-Za-z])", r"(?<![A-Za-z])albumin(?![A-Za-z])")),
    KeywordRule("血常规", (r"血常规", r"白细胞", r"血红蛋白", r"贫血", r"(?<![A-Za-z])WBC(?![A-Za-z])", r"(?<![A-Za-z])Hb(?![A-Za-z])", r"(?<![A-Za-z])PLT(?![A-Za-z])", r"血小板")),
    KeywordRule("粪便钙卫蛋白", (r"粪便钙卫蛋白", r"粪钙卫蛋白", r"钙卫蛋白", r"(?<![A-Za-z])FC(?![A-Za-z])", r"fecal calprotectin")),
)

ENDOSCOPY_RULES = (
    KeywordRule("结肠镜", (r"结肠镜", r"肠镜", r"colonoscopy")),
    KeywordRule("回肠末端", (r"回肠末端", r"末端回肠", r"terminal ileum")),
    KeywordRule("回盲部", (r"回盲部", r"回盲瓣", r"ileocecal")),
    KeywordRule("纵行溃疡", (r"纵行溃疡", r"纵形溃疡", r"longitudinal ulcer")),
    KeywordRule("溃疡", (r"溃疡", r"ulcer")),
    KeywordRule("铺路石样改变", (r"铺路石样", r"鹅卵石样", r"cobblestone")),
    KeywordRule("狭窄", (r"狭窄", r"stricture", r"stenosis")),
    KeywordRule("瘘管", (r"瘘管", r"fistula")),
    KeywordRule("活检", (r"活检", r"病理活检", r"biopsy")),
)

IMAGING_RULES = (
    KeywordRule("CTE", (r"(?<![A-Za-z])CTE(?![A-Za-z])", r"CT小肠成像", r"CTE小肠成像")),
    KeywordRule("MRE", (r"(?<![A-Za-z])MRE(?![A-Za-z])", r"磁共振小肠成像", r"MRE小肠成像")),
    KeywordRule("肛周 MRI", (r"肛周\s*MRI", r"肛周磁共振", r"盆腔\s*MRI")),
    KeywordRule("MRI", (r"(?<![A-Za-z])MRI(?![A-Za-z])", r"磁共振")),
    KeywordRule("肠道超声", (r"肠道超声", r"肠超", r"intestinal ultrasound")),
)

PATHOLOGY_RULES = (
    KeywordRule("肉芽肿", (r"肉芽肿", r"granuloma")),
    KeywordRule("慢性炎症", (r"慢性炎症", r"慢性活动性炎症")),
    KeywordRule("透壁性炎症", (r"透壁性炎症", r"全层炎症", r"transmural")),
)

RED_FLAG_RULES = (
    KeywordRule("剧烈腹痛", (r"剧烈腹痛", r"严重腹痛", r"腹痛剧烈")),
    KeywordRule("肠梗阻", (r"肠梗阻", r"停止排气排便", r"无法排气排便", r"腹胀.*呕吐", r"呕吐.*腹胀")),
    KeywordRule("大量便血", (r"大量便血", r"便血很多", r"大量血便")),
    KeywordRule("高热", (r"高热", r"高烧", r"体温\s*(?:≥|>=|>|超过)?\s*39")),
    KeywordRule("休克", (r"休克", r"血压很低", r"低血压休克")),
    KeywordRule("严重脱水", (r"严重脱水", r"明显脱水", r"尿少口干")),
    KeywordRule("意识障碍", (r"意识障碍", r"意识不清", r"昏迷", r"嗜睡")),
)


class PatientCaseExtractor:
    """Rule-based first-pass extractor for patient-entered clinical text."""

    def extract(self, raw_text: str) -> PatientCase:
        cleaned = raw_text.strip()
        if not cleaned:
            raise ValueError("raw_text must not be empty")

        symptoms = self._extract_terms(cleaned, SYMPTOM_RULES)
        labs = self._extract_labs(cleaned)
        endoscopy = self._extract_terms(cleaned, ENDOSCOPY_RULES)
        imaging = self._extract_terms(cleaned, IMAGING_RULES)
        pathology = self._extract_terms(cleaned, PATHOLOGY_RULES)
        red_flags = self._extract_terms(cleaned, RED_FLAG_RULES)
        unknowns = self._extract_unknown_clauses(cleaned)

        return PatientCase(
            raw_text=cleaned,
            symptoms=symptoms,
            labs=labs,
            imaging=imaging,
            endoscopy=endoscopy,
            pathology=pathology,
            red_flags=red_flags,
            unknowns=unknowns,
        )

    @staticmethod
    def _extract_terms(text: str, rules: tuple[KeywordRule, ...]) -> list[str]:
        terms: list[str] = []
        for rule in rules:
            if any(pattern.search(text) for pattern in rule.compiled()):
                terms.append(rule.canonical)
        return terms

    @staticmethod
    def _extract_labs(text: str) -> dict[str, dict[str, object]]:
        labs: dict[str, dict[str, object]] = {}
        for rule in LAB_RULES:
            matches: list[str] = []
            value_texts: list[str] = []
            for pattern in rule.compiled():
                for match in pattern.finditer(text):
                    matches.append(match.group(0))
                    value_text = _extract_value_near(text, match.start(), match.end())
                    if value_text and value_text not in value_texts:
                        value_texts.append(value_text)
            if matches:
                labs[rule.canonical] = {
                    "mentioned": True,
                    "matches": _dedupe(matches),
                }
                if value_texts:
                    labs[rule.canonical]["value_texts"] = value_texts
        return labs

    @staticmethod
    def _extract_unknown_clauses(text: str) -> list[str]:
        all_rules = (
            SYMPTOM_RULES
            + LAB_RULES
            + ENDOSCOPY_RULES
            + IMAGING_RULES
            + PATHOLOGY_RULES
            + RED_FLAG_RULES
        )
        unknowns: list[str] = []
        for clause in re.split(r"[，,。；;\n]+", text):
            cleaned = clause.strip()
            if len(cleaned) < 2:
                continue
            if not any(pattern.search(cleaned) for rule in all_rules for pattern in rule.compiled()):
                unknowns.append(cleaned)
        return unknowns


def extract_patient_case(raw_text: str) -> PatientCase:
    return PatientCaseExtractor().extract(raw_text)


def _extract_value_near(text: str, start: int, end: int) -> str | None:
    window = text[start : min(len(text), end + 28)]
    match = re.search(
        r"[:：=]?\s*(升高|降低|正常|阳性|阴性|[<>]?\s*\d+(?:\.\d+)?\s*(?:mg/L|mm/h|g/L|g/dL|μg/g|ug/g|ng/mL|×10\^9/L)?)",
        window,
        re.IGNORECASE,
    )
    if not match:
        return None
    return match.group(1).replace(" ", "")


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
