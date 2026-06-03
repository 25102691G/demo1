from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from clinical_stage_classifier import ClinicalStageResult, classify_clinical_stage
from guideline_skill.schema import (
    DiseaseSkillPack,
    MissingInformationItem,
    PatientCase,
    SkillExecutionResult,
    load_skill_pack,
)
from missing_information_ranker import rank_missing_information
from next_step_ranker import rank_next_steps
from patient_case_extractor import extract_patient_case
from response_composer import compose_readable_summary
from skill_executor import execute_crohn_skill
from skill_router import RoutingResult, route_disease_skills
from support_evidence_ranker import rank_support_evidence


DISCLAIMER = (
    "本系统仅基于已输入信息和指南 seed pack 做候选疾病召回与规则化分析，"
    "不能替代医生面诊、检查解读或最终诊断。"
)


class AgentResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    patient_case_summary: dict[str, Any]
    clinical_stage: ClinicalStageResult
    candidate_diseases: list[RoutingResult] = Field(default_factory=list)
    skill_results: list[SkillExecutionResult] = Field(default_factory=list)
    raw_missing_information: list[MissingInformationItem] = Field(default_factory=list)
    full_missing_information: list[MissingInformationItem] = Field(default_factory=list)
    top_missing_information: list[MissingInformationItem] = Field(default_factory=list)
    final_assessment: str
    recommended_next_steps: list[str] = Field(default_factory=list)
    safety_warnings: list[str] = Field(default_factory=list)
    disclaimer: str = DISCLAIMER
    readable_summary: str


class MedicalGuidelineAgentOrchestrator:
    """Minimal no-LLM orchestration pipeline for guideline skill packs."""

    def __init__(self, skills_path: str | Path = "data/skills", top_k: int = 5) -> None:
        self.skills_path = Path(skills_path)
        self.top_k = top_k

    def run(self, raw_text: str) -> AgentResponse:
        patient_case = extract_patient_case(raw_text)
        clinical_stage = classify_clinical_stage(patient_case)
        skill_packs = load_skill_packs(self.skills_path)
        candidates = route_disease_skills(patient_case, skill_packs, top_k=self.top_k)
        pack_by_name = {pack.skill_name: pack for pack in skill_packs}

        raw_skill_results: list[SkillExecutionResult] = []
        for candidate in candidates:
            if candidate.score <= 0:
                continue
            skill_pack = pack_by_name[candidate.skill_name]
            result = execute_skill_if_supported(patient_case, skill_pack)
            if result is not None:
                raw_skill_results.append(result)

        skill_results = refine_skill_results_for_response(
            patient_case,
            raw_skill_results,
            clinical_stage,
        )

        safety_warnings = aggregate_safety_warnings(patient_case, skill_results)
        top_result = skill_results[0] if skill_results else None
        raw_missing_information = top_result.raw_missing_information if top_result else []
        full_missing_information = top_result.full_missing_information if top_result else []
        top_missing_information = top_result.top_missing_information if top_result else []
        recommended_next_steps = top_result.recommended_next_steps if top_result else rank_next_steps(
            patient_case,
            None,
            clinical_stage,
            [],
            top_k=8,
        )
        final_assessment = build_final_assessment(
            patient_case=patient_case,
            candidates=candidates,
            skill_results=skill_results,
            safety_warnings=safety_warnings,
            top_missing_information=top_missing_information,
        )
        response = AgentResponse(
            patient_case_summary=summarize_patient_case(patient_case),
            clinical_stage=clinical_stage,
            candidate_diseases=candidates,
            skill_results=skill_results,
            raw_missing_information=raw_missing_information,
            full_missing_information=full_missing_information,
            top_missing_information=top_missing_information,
            final_assessment=final_assessment,
            recommended_next_steps=recommended_next_steps,
            safety_warnings=safety_warnings,
            disclaimer=DISCLAIMER,
            readable_summary="",
        )
        response.readable_summary = compose_readable_summary(
            patient_case=patient_case,
            clinical_stage=clinical_stage,
            candidate_diseases=candidates,
            skill_results=skill_results,
            final_assessment=final_assessment,
            top_missing_information=top_missing_information,
            recommended_next_steps=recommended_next_steps,
            safety_warnings=safety_warnings,
            disclaimer=DISCLAIMER,
        )
        return response


def run_agent(
    raw_text: str,
    skills_path: str | Path = "data/skills",
    top_k: int = 5,
) -> AgentResponse:
    return MedicalGuidelineAgentOrchestrator(skills_path=skills_path, top_k=top_k).run(raw_text)


def load_skill_packs(path: str | Path) -> list[DiseaseSkillPack]:
    skill_path = Path(path)
    if skill_path.is_file():
        return [load_skill_pack(skill_path)]
    if not skill_path.is_dir():
        raise ValueError(f"Skill path does not exist: {skill_path}")

    candidates = sorted(
        [
            *skill_path.rglob("*.yaml"),
            *skill_path.rglob("*.yml"),
            *skill_path.rglob("*.json"),
        ]
    )
    if not candidates:
        raise ValueError(f"No skill pack YAML/JSON files found under: {skill_path}")
    return [load_skill_pack(candidate) for candidate in candidates]


def refine_skill_results_for_response(
    patient_case: PatientCase,
    skill_results: list[SkillExecutionResult],
    clinical_stage: ClinicalStageResult,
) -> list[SkillExecutionResult]:
    refined: list[SkillExecutionResult] = []
    for result in skill_results:
        ranking = rank_missing_information(result.missing_information, clinical_stage, top_k=7)
        ranked_steps = rank_next_steps(
            patient_case,
            result,
            clinical_stage,
            ranking.top_missing_information,
            top_k=8,
        )
        refined.append(
            result.model_copy(
                update={
                    "support_evidence": rank_support_evidence(
                        patient_case,
                        result.support_evidence,
                    ),
                    "raw_missing_information": result.missing_information,
                    "full_missing_information": ranking.full_missing_information,
                    "top_missing_information": ranking.top_missing_information,
                    "missing_information": ranking.top_missing_information,
                    "raw_recommended_next_steps": result.recommended_next_steps,
                    "recommended_next_steps": ranked_steps,
                }
            )
        )
    return refined


def execute_skill_if_supported(
    patient_case: PatientCase,
    skill_pack: DiseaseSkillPack,
) -> SkillExecutionResult | None:
    if _is_crohn_skill(skill_pack):
        return execute_crohn_skill(patient_case, skill_pack)
    return None


def summarize_patient_case(patient_case: PatientCase) -> dict[str, Any]:
    return {
        "raw_text": patient_case.raw_text,
        "symptoms": patient_case.symptoms,
        "labs": patient_case.labs,
        "imaging": patient_case.imaging,
        "endoscopy": patient_case.endoscopy,
        "pathology": patient_case.pathology,
        "red_flags": patient_case.red_flags,
        "unknowns": patient_case.unknowns,
    }


def aggregate_safety_warnings(
    patient_case: PatientCase,
    skill_results: list[SkillExecutionResult],
) -> list[str]:
    warnings: list[str] = []
    if patient_case.red_flags:
        warnings.append("检测到红旗征象，需优先进行安全分诊；如症状严重或进展，应及时就医或急诊评估。")
    for result in skill_results:
        warnings.extend(result.safety_warnings)
    return _dedupe(warnings)


def aggregate_next_steps(
    skill_results: list[SkillExecutionResult],
    patient_case: PatientCase,
) -> list[str]:
    steps: list[str] = []
    if patient_case.red_flags:
        steps.append("优先处理红旗征象，必要时急诊评估。")
    for result in skill_results:
        steps.extend(result.recommended_next_steps)
    if not steps:
        steps.append("当前信息不足以触发已实现 disease skill；建议补充症状持续时间、实验室检查、影像、内镜和病理信息。")
    return _dedupe(steps)


def build_final_assessment(
    *,
    patient_case: PatientCase,
    candidates: list[RoutingResult],
    skill_results: list[SkillExecutionResult],
    safety_warnings: list[str],
    top_missing_information: list[MissingInformationItem] | None = None,
) -> str:
    if patient_case.red_flags:
        safety_prefix = "因输入包含红旗或潜在安全风险，应先关注及时线下医疗评估。"
    else:
        safety_prefix = ""

    if not candidates or candidates[0].score == 0:
        assessment = "当前信息尚未明显匹配已加载的疾病指南 skill，不能支持特定疾病倾向。建议补充检查和病史后重新评估。"
    elif not skill_results:
        top = candidates[0]
        assessment = (
            f"当前信息召回了 {top.disease_name} 相关 skill，但该 skill 的执行器尚未实现；"
            "因此只提供候选召回，不给出疾病倾向判断。"
        )
    else:
        top_result = skill_results[0]
        level_text = _suspicion_level_text(top_result.suspicion_level)
        missing_count = len(top_result.missing_information)
        assessment = (
            f"当前信息{level_text}{top_result.disease_name} 的可能性，"
            f"但这不是最终诊断。"
        )
        top_missing_count = len(top_missing_information or [])
        if top_missing_count:
            assessment += f" 当前最优先补充 {top_missing_count} 项关键信息，再进入下一步判断。"
        elif missing_count:
            assessment += f" 仍有 {missing_count} 项关键信息缺失，需优先补充检查或鉴别诊断信息。"
        if top_result.differential_diagnoses:
            names = "、".join(item.disease_name for item in top_result.differential_diagnoses[:6])
            assessment += f" 需要同时鉴别：{names}。"

    return " ".join(part for part in [safety_prefix, assessment] if part)


def build_readable_summary(response: AgentResponse) -> str:
    top_candidate = response.candidate_diseases[0] if response.candidate_diseases else None
    lines = ["Agent Summary", response.final_assessment]
    if top_candidate:
        lines.append(
            f"Top candidate: {top_candidate.disease_name} "
            f"({top_candidate.skill_name}), recall score {top_candidate.score}."
        )
    if response.recommended_next_steps:
        lines.append("Recommended next steps:")
        lines.extend(f"- {step}" for step in response.recommended_next_steps[:8])
    if response.safety_warnings:
        lines.append("Safety warnings:")
        lines.extend(f"- {warning}" for warning in response.safety_warnings[:8])
    lines.append(response.disclaimer)
    return "\n".join(lines)


def _is_crohn_skill(skill_pack: DiseaseSkillPack) -> bool:
    aliases = {alias.casefold() for alias in skill_pack.disease_aliases}
    return (
        skill_pack.skill_name == "crohn_disease_2023_guangzhou"
        or skill_pack.disease_name.casefold() == "crohn disease"
        or "克罗恩病" in aliases
    )


def _suspicion_level_text(suspicion_level: str) -> str:
    mapping = {
        "unlikely": "暂不支持",
        "possible": "提示存在",
        "suspected": "支持疑似",
        "probable": "较支持",
        "confirmed_by_doctor_only": "记录到医生已确认过",
    }
    return mapping.get(suspicion_level, "提示")


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
