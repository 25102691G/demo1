from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from guideline_skill.schema import PatientCase


ClinicalStage = Literal[
    "emergency_triage",
    "initial_screening",
    "diagnostic_workup",
    "differential_diagnosis",
    "extent_and_complication_assessment",
    "treatment_readiness",
    "treatment_selection",
    "followup_monitoring",
]


class ClinicalStageResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    primary_stage: ClinicalStage
    stages: list[ClinicalStage] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)

    def has_stage(self, stage: str) -> bool:
        return stage in self.stages


def classify_clinical_stage(patient_case: PatientCase) -> ClinicalStageResult:
    """Classify the current clinical workflow stage from extracted case facts."""

    raw_text = patient_case.raw_text
    stages: list[ClinicalStage] = []
    reasons: list[str] = []

    if patient_case.red_flags:
        _add_stage(
            stages,
            reasons,
            "emergency_triage",
            "输入包含红旗征象，应优先安全分诊。",
        )

    if _asks_followup_or_relapse(raw_text) and _has_treatment_context(raw_text):
        _add_stage(
            stages,
            reasons,
            "followup_monitoring",
            "文本提示已治疗或复查/复发/监测诉求。",
        )

    if _doctor_confirmed(raw_text) and _asks_treatment(raw_text):
        _add_stage(
            stages,
            reasons,
            "treatment_selection",
            "文本提示医生已确诊且正在询问治疗或用药选择。",
        )
    elif _doctor_confirmed(raw_text):
        _add_stage(
            stages,
            reasons,
            "treatment_readiness",
            "文本提示医生已确诊，可进入治疗前资料完整性检查。",
        )

    if _has_ibd_supporting_test(patient_case):
        _add_stage(
            stages,
            reasons,
            "diagnostic_workup",
            "已有内镜或影像提示疑似 IBD/CD，需要在诊断路径中综合判断。",
        )
        _add_stage(
            stages,
            reasons,
            "differential_diagnosis",
            "疑似 IBD/CD 阶段需同步考虑肠结核、感染性肠炎等相似疾病。",
        )

    if _has_complication_or_perianal_signal(patient_case):
        _add_stage(
            stages,
            reasons,
            "extent_and_complication_assessment",
            "文本包含狭窄、瘘管、脓肿或肛周病变线索，需要评估病变范围和并发症。",
        )

    if _symptoms_only(patient_case):
        _add_stage(
            stages,
            reasons,
            "initial_screening",
            "目前只有症状信息，适合先做初筛和补充检查建议。",
        )

    if not stages:
        _add_stage(
            stages,
            reasons,
            "initial_screening",
            "当前信息不足以进入更具体阶段，默认作为初筛处理。",
        )

    return ClinicalStageResult(
        primary_stage=stages[0],
        stages=stages,
        reasons=reasons,
    )


def _add_stage(
    stages: list[ClinicalStage],
    reasons: list[str],
    stage: ClinicalStage,
    reason: str,
) -> None:
    if stage in stages:
        return
    stages.append(stage)
    reasons.append(reason)


def _symptoms_only(patient_case: PatientCase) -> bool:
    return bool(patient_case.symptoms) and not any(
        [
            patient_case.labs,
            patient_case.imaging,
            patient_case.endoscopy,
            patient_case.pathology,
        ]
    )


def _has_ibd_supporting_test(patient_case: PatientCase) -> bool:
    supportive_endoscopy = {
        "结肠镜",
        "回肠末端",
        "回盲部",
        "溃疡",
        "纵行溃疡",
        "铺路石样改变",
        "狭窄",
        "瘘管",
        "活检",
    }
    supportive_imaging = {"CTE", "MRE", "肛周 MRI", "MRI", "肠道超声"}
    return bool(
        set(patient_case.endoscopy).intersection(supportive_endoscopy)
        or set(patient_case.imaging).intersection(supportive_imaging)
    )


def _needs_diagnostic_completion(patient_case: PatientCase) -> bool:
    if not patient_case.pathology:
        return True
    return not (
        _has_any(patient_case.raw_text, ("肠结核", "结核", "TB", "T-SPOT", "PPD"))
        and _has_any(patient_case.raw_text, ("感染性肠炎", "粪便培养", "病原", "艰难梭菌", "寄生虫"))
    )


def _has_complication_or_perianal_signal(patient_case: PatientCase) -> bool:
    text = " ".join(
        [
            patient_case.raw_text,
            *patient_case.symptoms,
            *patient_case.endoscopy,
            *patient_case.imaging,
        ]
    )
    return _has_any(
        text,
        (
            "狭窄",
            "肠梗阻",
            "梗阻",
            "瘘管",
            "肛瘘",
            "肛周",
            "脓肿",
            "stricture",
            "fistula",
            "abscess",
        ),
    )


def _doctor_confirmed(raw_text: str) -> bool:
    return bool(
        re.search(r"医生.{0,8}(已确诊|确诊|诊断).{0,8}(克罗恩病|Crohn|CD)", raw_text, re.I)
        or re.search(r"(克罗恩病|Crohn|CD).{0,8}医生.{0,8}(已确诊|确诊|诊断)", raw_text, re.I)
        or re.search(r"病理明确支持.{0,12}医生.{0,8}(已确诊|确诊)", raw_text, re.I)
    )


def _asks_treatment(raw_text: str) -> bool:
    return _has_any(
        raw_text,
        ("用药", "吃什么药", "治疗", "药物", "生物制剂", "激素", "免疫抑制剂", "抗TNF", "JAK"),
    )


def _asks_followup_or_relapse(raw_text: str) -> bool:
    return _has_any(raw_text, ("复查", "复发", "随访", "监测", "复诊", "疗效", "复燃"))


def _has_treatment_context(raw_text: str) -> bool:
    return _has_any(
        raw_text,
        ("已治疗", "治疗后", "正在用", "用药后", "术后", "生物制剂", "激素", "维持治疗"),
    )


def _has_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term.casefold() in text.casefold() for term in terms)
