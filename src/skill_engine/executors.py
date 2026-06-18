from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .card_store import card_to_evidence_snippet, card_to_recommendation
from .skill_loader import SkillPack
from .utils import clean_text, is_present, resolve_case_path, text_contains_term


@dataclass
class StepResult:
    step_id: str
    type: str
    status: str
    data: dict[str, Any]
    result_summary: str
    next_step: str | None = None

    def as_executed_step(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "type": self.type,
            "status": self.status,
            "result_summary": self.result_summary,
            "next_step": self.next_step,
        }


@dataclass
class ExecutionContext:
    skill_pack: SkillPack
    canonical_case: dict[str, Any]
    candidate: dict[str, Any]
    state: dict[str, Any]
    errors: list[str]


def execute_step(step: dict[str, Any], context: ExecutionContext) -> StepResult:
    step_type = clean_text(step.get("type"))
    executor = {
        "safety_triage": _execute_safety_triage,
        "evidence_check": _execute_evidence_check,
        "differential_check": _execute_differential_check,
        "management_gate": _execute_management_gate,
        "plan_generation": _execute_plan_generation,
        "terminal_output": _execute_terminal_output,
        "card_retrieval": _execute_card_retrieval,
        "llm_generation": _execute_llm_generation,
    }.get(step_type)
    if executor is None:
        return StepResult(
            step_id=clean_text(step.get("step_id")),
            type=step_type,
            status="error",
            data={"error": f"unsupported workflow step type {step_type!r}"},
            result_summary=f"Unsupported workflow step type {step_type!r}.",
        )
    return executor(step, context)


def _execute_safety_triage(step: dict[str, Any], context: ExecutionContext) -> StepResult:
    red_flags: list[dict[str, Any]] = []
    for item in context.canonical_case.get("red_flags") or []:
        red_flags.append(
            {
                "name": clean_text(item.get("name")),
                "severity": clean_text(item.get("severity")) or "high",
                "source": clean_text(item.get("source")) or "rule_detected",
                "recommended_action": "建议立即进行临床安全评估或急诊处理。",
            }
        )

    raw_input = clean_text(context.canonical_case.get("raw_input"))
    for item in (context.skill_pack.skill.get("routing_profile") or {}).get("red_flags") or []:
        name = clean_text(item.get("name"))
        synonyms = [name, *(item.get("synonyms") or [])]
        if any(text_contains_term(raw_input, term) for term in synonyms):
            red_flags.append(
                {
                    "name": name,
                    "severity": clean_text(item.get("severity")) or "high",
                    "source": "rule_detected",
                    "recommended_action": clean_text(item.get("action"))
                    or "建议立即进行临床安全评估或急诊处理。",
                }
            )

    red_flags = _dedupe_red_flags(red_flags)
    workflow_stopped = any(flag.get("severity") in {"high", "critical"} for flag in red_flags)
    status = "stopped" if workflow_stopped else "completed"
    return StepResult(
        step_id=clean_text(step.get("step_id")),
        type="safety_triage",
        status=status,
        data={
            "red_flags": red_flags,
            "has_red_flags": bool(red_flags),
            "workflow_stopped": workflow_stopped,
        },
        result_summary="Detected red flags." if red_flags else "No red flags detected by rules.",
    )


def _execute_evidence_check(step: dict[str, Any], context: ExecutionContext) -> StepResult:
    subskill, error = _subskill_from_step(step, context.skill_pack)
    if error:
        return _error_result(step, error)
    assert subskill is not None

    missing = _missing_requirements(context.canonical_case, subskill)
    cards = context.skill_pack.card_store.select_for_subskill(subskill, limit=_card_limit(context.skill_pack))
    supporting = [card_to_evidence_snippet(card) for card in cards if card_to_evidence_snippet(card)]
    used_cards = [clean_text(card.get("card_id")) for card in cards]
    recommended_next_steps = [
        f"补充或核对：{item['label']}" for item in missing
    ]
    data = {
        "supporting_evidence": supporting,
        "opposing_evidence": [],
        "missing_information": [item["label"] for item in missing],
        "missing_required_evidence": [item["label"] for item in missing],
        "recommended_next_steps": recommended_next_steps,
        "used_cards": used_cards,
    }
    return StepResult(
        step_id=clean_text(step.get("step_id")),
        type="evidence_check",
        status="completed",
        data=data,
        result_summary=(
            f"Missing {len(missing)} required evidence item(s)."
            if missing
            else f"Retrieved {len(used_cards)} evidence card(s)."
        ),
    )


def _execute_differential_check(step: dict[str, Any], context: ExecutionContext) -> StepResult:
    subskill, error = _subskill_from_step(step, context.skill_pack)
    if error:
        return _error_result(step, error)
    differentials = [
        clean_text(item.get("disease_name"))
        for item in (context.skill_pack.skill.get("routing_profile") or {}).get("must_differentiate") or []
        if clean_text(item.get("disease_name"))
    ]
    cards = context.skill_pack.card_store.select_for_subskill(subskill, limit=_card_limit(context.skill_pack)) if subskill else []
    data = {
        "differentials_to_consider": differentials,
        "differential_warnings": [],
        "recommended_next_steps": [
            f"结合病史、检查和必要检验排除或鉴别：{name}" for name in differentials[:5]
        ],
        "used_cards": [clean_text(card.get("card_id")) for card in cards],
    }
    return StepResult(
        step_id=clean_text(step.get("step_id")),
        type="differential_check",
        status="completed",
        data=data,
        result_summary=f"Listed {len(differentials)} differential diagnosis item(s).",
    )


def _execute_management_gate(step: dict[str, Any], context: ExecutionContext) -> StepResult:
    if _state_has_high_red_flag(context.state):
        allow = False
        reason = "安全红旗未排除，暂不生成管理建议。"
    elif context.state.get("missing_required_evidence"):
        allow = False
        reason = "关键诊断证据不足，暂不生成管理建议。"
    elif _has_supported_diagnosis(context.canonical_case) or context.candidate.get("score", 0) >= context.candidate.get("strong_candidate_threshold", 1):
        allow = True
        reason = "已有诊断线索或路由分数达到强候选阈值。"
    else:
        allow = False
        reason = "当前证据仅支持候选提示，需要人工复核。"
    return StepResult(
        step_id=clean_text(step.get("step_id")),
        type="management_gate",
        status="completed",
        data={"allow_management_plan": allow, "reason": reason},
        result_summary=reason,
    )


def _execute_plan_generation(step: dict[str, Any], context: ExecutionContext) -> StepResult:
    subskill, error = _subskill_from_step(step, context.skill_pack)
    if error:
        return _error_result(step, error)
    assert subskill is not None
    cards = context.skill_pack.card_store.select_for_subskill(subskill, limit=_card_limit(context.skill_pack))
    recommendations = [card_to_recommendation(card) for card in cards if card_to_recommendation(card)]
    used_cards = [clean_text(card.get("card_id")) for card in cards]

    monitoring_cards = _cards_for_named_subskill(context.skill_pack, "monitoring_follow_up")
    monitoring = [card_to_recommendation(card) for card in monitoring_cards if card_to_recommendation(card)]
    used_cards.extend(clean_text(card.get("card_id")) for card in monitoring_cards)
    safety_notes = [
        "不生成个体化处方、剂量或急诊替代建议；需由有资质临床医生结合患者情况决策。"
    ]
    data = {
        "management_recommendations": recommendations,
        "monitoring_or_follow_up": monitoring,
        "safety_notes": safety_notes,
        "used_cards": [card_id for card_id in used_cards if card_id],
    }
    return StepResult(
        step_id=clean_text(step.get("step_id")),
        type="plan_generation",
        status="completed",
        data=data,
        result_summary=f"Generated {len(recommendations)} guideline-based management item(s).",
    )


def _execute_card_retrieval(step: dict[str, Any], context: ExecutionContext) -> StepResult:
    subskill, error = _subskill_from_step(step, context.skill_pack)
    if error:
        return _error_result(step, error)
    assert subskill is not None
    cards = context.skill_pack.card_store.select_for_subskill(subskill, limit=_card_limit(context.skill_pack))
    data = {
        "used_cards": [clean_text(card.get("card_id")) for card in cards],
        "evidence_snippets": [card_to_evidence_snippet(card) for card in cards],
    }
    return StepResult(
        step_id=clean_text(step.get("step_id")),
        type="card_retrieval",
        status="completed",
        data=data,
        result_summary=f"Retrieved {len(cards)} card(s).",
    )


def _execute_llm_generation(step: dict[str, Any], context: ExecutionContext) -> StepResult:
    return StepResult(
        step_id=clean_text(step.get("step_id")),
        type="llm_generation",
        status="skipped",
        data={"message": "LLM generation is not enabled in rule-based engine."},
        result_summary="LLM generation is not enabled in rule-based engine.",
    )


def _execute_terminal_output(step: dict[str, Any], context: ExecutionContext) -> StepResult:
    template = clean_text((step.get("config") or {}).get("output_template"))
    return StepResult(
        step_id=clean_text(step.get("step_id")),
        type="terminal_output",
        status="completed",
        data={"output_template": template},
        result_summary=f"Reached terminal output template {template or 'unknown'}.",
    )


def _subskill_from_step(
    step: dict[str, Any],
    skill_pack: SkillPack,
) -> tuple[dict[str, Any] | None, str | None]:
    subskill_ref = clean_text((step.get("config") or {}).get("subskill_ref"))
    if not subskill_ref:
        return None, None
    for subskill in skill_pack.skill.get("subskills") or []:
        if clean_text(subskill.get("subskill_id")) == subskill_ref:
            return subskill, None
    return None, f"step {clean_text(step.get('step_id'))!r} references missing subskill {subskill_ref!r}"


def _missing_requirements(canonical_case: dict[str, Any], subskill: dict[str, Any]) -> list[dict[str, str]]:
    missing: list[dict[str, str]] = []
    requirements = (subskill.get("input_requirements") or {}).get("required_for_high_confidence") or []
    for requirement in requirements:
        if not isinstance(requirement, dict):
            continue
        path = clean_text(requirement.get("path"))
        label = clean_text(requirement.get("label")) or path
        if path and not is_present(resolve_case_path(canonical_case, path)):
            missing.append({"path": path, "label": label})
    return missing


def _card_limit(skill_pack: SkillPack) -> int:
    retrieval = (skill_pack.skill.get("knowledge_base") or {}).get("retrieval") or {}
    try:
        return max(1, int(retrieval.get("default_top_k") or 8))
    except (TypeError, ValueError):
        return 8


def _cards_for_named_subskill(skill_pack: SkillPack, subskill_id: str) -> list[dict[str, Any]]:
    for subskill in skill_pack.skill.get("subskills") or []:
        if clean_text(subskill.get("subskill_id")) == subskill_id:
            return skill_pack.card_store.select_for_subskill(subskill, limit=_card_limit(skill_pack))
    return []


def _state_has_high_red_flag(state: dict[str, Any]) -> bool:
    return any(flag.get("severity") in {"high", "critical"} for flag in state.get("red_flags") or [])


def _has_supported_diagnosis(canonical_case: dict[str, Any]) -> bool:
    return any(
        diagnosis.get("status") in {"confirmed", "highly_suspected", "suspected"}
        for diagnosis in _case_diagnosis_features(canonical_case)
    )


def _case_diagnosis_features(canonical_case: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        item
        for item in [
            *(canonical_case.get("features") or []),
            *(canonical_case.get("diagnoses") or []),
        ]
        if isinstance(item, dict)
    ]


def _error_result(step: dict[str, Any], message: str) -> StepResult:
    return StepResult(
        step_id=clean_text(step.get("step_id")),
        type=clean_text(step.get("type")),
        status="error",
        data={"error": message},
        result_summary=message,
    )


def _dedupe_red_flags(red_flags: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for flag in red_flags:
        key = clean_text(flag.get("name")).casefold()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(flag)
    return deduped
