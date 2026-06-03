from __future__ import annotations

from clinical_stage_classifier import ClinicalStageResult
from guideline_skill.schema import MissingInformationItem, PatientCase, SkillExecutionResult
from skill_router import RoutingResult


def compose_readable_summary(
    *,
    patient_case: PatientCase,
    clinical_stage: ClinicalStageResult,
    candidate_diseases: list[RoutingResult],
    skill_results: list[SkillExecutionResult],
    final_assessment: str,
    top_missing_information: list[MissingInformationItem],
    recommended_next_steps: list[str],
    safety_warnings: list[str],
    disclaimer: str,
) -> str:
    """Compose a Chinese, stage-aware summary for users and clinicians."""

    lines: list[str] = ["Agent Summary（中文摘要）"]

    lines.append("")
    lines.append("当前判断")
    lines.append(f"- {final_assessment}")
    lines.append(f"- 当前临床阶段：{_stage_text(clinical_stage.stages)}。")
    if candidate_diseases:
        top = candidate_diseases[0]
        lines.append(f"- 召回最相关的指南 skill：{top.disease_name}（召回分 {top.score}，不是确诊概率）。")

    top_result = skill_results[0] if skill_results else None
    if top_result:
        lines.append("")
        lines.append("支持依据")
        for evidence in top_result.support_evidence[:6]:
            lines.append(f"- {evidence}")

        lines.append("")
        lines.append("为什么还不能确诊")
        if top_result.suspicion_level == "confirmed_by_doctor_only":
            lines.append("- 文本提示医生已确诊；系统只记录这个前提，不独立作出确诊。")
        elif top_missing_information:
            lines.append("- 目前仍缺少关键检查或鉴别诊断信息，系统只能给出疑似程度和下一步建议。")
            for item in top_missing_information[:3]:
                lines.append(f"- {item.question}")
        else:
            lines.append("- 即使资料较完整，最终诊断仍需要医生结合原始报告、体格检查和鉴别诊断综合判断。")

    if top_missing_information:
        lines.append("")
        lines.append("最优先补充信息")
        for index, item in enumerate(top_missing_information, start=1):
            lines.append(f"{index}. {item.question}")

    if recommended_next_steps:
        lines.append("")
        lines.append("建议下一步")
        for index, step in enumerate(recommended_next_steps, start=1):
            lines.append(f"{index}. {step}")

    if top_result and top_result.differential_diagnoses:
        lines.append("")
        lines.append("需要鉴别的疾病")
        for item in top_result.differential_diagnoses[:6]:
            lines.append(f"- {item.disease_name}：{item.rationale}")

    if safety_warnings:
        lines.append("")
        lines.append("安全提示")
        for warning in safety_warnings[:6]:
            lines.append(f"- {warning}")

    lines.append("")
    lines.append(disclaimer)
    return "\n".join(lines)


def _stage_text(stages: list[str]) -> str:
    labels = {
        "emergency_triage": "安全分诊",
        "initial_screening": "初步筛查",
        "diagnostic_workup": "诊断资料补全",
        "differential_diagnosis": "鉴别诊断",
        "extent_and_complication_assessment": "病变范围和并发症评估",
        "treatment_readiness": "治疗前资料完整性检查",
        "treatment_selection": "治疗选择讨论",
        "followup_monitoring": "随访监测",
    }
    return "、".join(labels.get(stage, stage) for stage in stages)
