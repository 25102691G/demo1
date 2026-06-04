from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .executors import ExecutionContext, StepResult, execute_step
from .skill_loader import SkillPack
from .utils import clean_text, get_path, is_present, resolve_prefixed_path


class WorkflowEngine:
    def __init__(self, *, max_steps: int = 50) -> None:
        self.max_steps = max_steps

    def run(
        self,
        skill_pack: SkillPack,
        canonical_case: dict[str, Any],
        candidate: dict[str, Any],
    ) -> dict[str, Any]:
        workflow = skill_pack.skill.get("workflow") or {}
        steps = {
            clean_text(step.get("step_id")): step
            for step in workflow.get("steps") or []
            if isinstance(step, dict)
        }
        state: dict[str, Any] = {
            "supporting_evidence": [],
            "opposing_evidence": [],
            "missing_information": [],
            "missing_required_evidence": [],
            "recommended_next_steps": [],
            "differentials_to_consider": [],
            "management_recommendations": [],
            "monitoring_or_follow_up": [],
            "safety_notes": [],
            "red_flags": [],
            "used_cards": [],
        }
        errors: list[str] = []
        executed: list[StepResult] = []
        current_step_id = clean_text(workflow.get("entrypoint"))
        terminal_template: str | None = None
        visited = 0

        while current_step_id and visited < self.max_steps:
            visited += 1
            step = steps.get(current_step_id)
            if step is None:
                errors.append(f"workflow references missing step {current_step_id!r}")
                break
            context = ExecutionContext(
                skill_pack=skill_pack,
                canonical_case=canonical_case,
                candidate=candidate,
                state=state,
                errors=errors,
            )
            try:
                result = execute_step(step, context)
            except Exception as exc:  # pragma: no cover - defensive workflow isolation.
                result = StepResult(
                    step_id=current_step_id,
                    type=clean_text(step.get("type")),
                    status="error",
                    data={"error": str(exc)},
                    result_summary=f"Executor failed: {exc}",
                )
            _accumulate_state(state, result.data)
            if result.status == "error":
                errors.append(clean_text(result.data.get("error")) or result.result_summary)
                executed.append(result)
                break
            next_step_id = _select_next_step(
                step=step,
                result_data=result.data,
                state=state,
                canonical_case=canonical_case,
                candidate=candidate,
            )
            result.next_step = next_step_id
            executed.append(result)
            if clean_text(step.get("type")) == "terminal_output":
                terminal_template = clean_text(result.data.get("output_template"))
                break
            if not next_step_id:
                break
            current_step_id = next_step_id
        else:
            if visited >= self.max_steps:
                errors.append(f"workflow exceeded max_steps={self.max_steps}")

        workflow_status = _workflow_status(executed, state, terminal_template, errors)
        return {
            "skill_id": skill_pack.skill_id,
            "disease_name": skill_pack.disease_name,
            "workflow_status": workflow_status,
            "executed_steps": [step.as_executed_step() for step in executed],
            "result": {
                "current_status": workflow_status,
                "suspicion_level": _suspicion_level(candidate),
                "supporting_evidence": state["supporting_evidence"],
                "opposing_evidence": state["opposing_evidence"],
                "missing_information": state["missing_information"],
                "recommended_next_steps": state["recommended_next_steps"],
                "differentials_to_consider": state["differentials_to_consider"],
                "management_recommendations": state["management_recommendations"],
                "monitoring_or_follow_up": state["monitoring_or_follow_up"],
                "safety_notes": state["safety_notes"],
                "red_flags": state["red_flags"],
                "used_cards": state["used_cards"],
                "terminal_template": terminal_template,
                "errors": errors,
            },
        }


def evaluate_condition(
    condition: Mapping[str, Any],
    *,
    result_data: Mapping[str, Any],
    state: Mapping[str, Any],
    canonical_case: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> bool:
    op = clean_text(condition.get("op"))
    if op == "default":
        return True
    left = _condition_value(
        condition.get("left", condition.get("path")),
        result_data=result_data,
        state=state,
        canonical_case=canonical_case,
        candidate=candidate,
    )
    right_value = condition.get("right")
    if condition.get("right_ref"):
        right_value = _condition_value(
            condition.get("right_ref"),
            result_data=result_data,
            state=state,
            canonical_case=canonical_case,
            candidate=candidate,
        )

    if op == "exists":
        return is_present(left)
    if op == "missing":
        return not is_present(left)
    if op == "eq":
        return left == right_value
    if op == "neq":
        return left != right_value
    if op in {"gt", "gte", "lt", "lte"}:
        return _compare_numbers(left, right_value, op)
    if op == "contains":
        return _contains(left, right_value)
    if op == "any_match":
        return bool(set(_as_list(left)).intersection(_as_list(right_value)))
    if op == "all_present":
        return all(
            is_present(
                _condition_value(
                    item,
                    result_data=result_data,
                    state=state,
                    canonical_case=canonical_case,
                    candidate=candidate,
                )
            )
            for item in _as_list(left if isinstance(left, list) else condition.get("paths", []))
        )
    if op == "missing_any":
        values = _as_list(left if isinstance(left, list) else condition.get("paths", []))
        return any(
            not is_present(
                _condition_value(
                    item,
                    result_data=result_data,
                    state=state,
                    canonical_case=canonical_case,
                    candidate=candidate,
                )
            )
            for item in values
        )
    return False


def _select_next_step(
    *,
    step: Mapping[str, Any],
    result_data: Mapping[str, Any],
    state: Mapping[str, Any],
    canonical_case: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> str | None:
    for transition in step.get("transitions") or []:
        if not isinstance(transition, Mapping):
            continue
        condition = transition.get("when") or {"op": "default"}
        if evaluate_condition(
            condition,
            result_data=result_data,
            state=state,
            canonical_case=canonical_case,
            candidate=candidate,
        ):
            return clean_text(transition.get("to")) or None
    return None


def _condition_value(
    value: Any,
    *,
    result_data: Mapping[str, Any],
    state: Mapping[str, Any],
    canonical_case: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> Any:
    if isinstance(value, str) and (
        value.startswith("result.")
        or value.startswith("context.")
        or value.startswith("case.")
        or value.startswith("state.")
    ):
        return resolve_prefixed_path(
            path=value,
            result=result_data,
            context={"route_score": candidate.get("score"), "candidate": candidate},
            canonical_case=canonical_case,
            state=state,
        )
    return value


def _accumulate_state(state: dict[str, Any], data: Mapping[str, Any]) -> None:
    list_fields = (
        "supporting_evidence",
        "opposing_evidence",
        "missing_information",
        "missing_required_evidence",
        "recommended_next_steps",
        "differentials_to_consider",
        "management_recommendations",
        "monitoring_or_follow_up",
        "safety_notes",
        "red_flags",
        "used_cards",
    )
    for field in list_fields:
        values = data.get(field)
        if isinstance(values, list):
            _extend_unique(state.setdefault(field, []), values)
    if "has_red_flags" in data:
        state["has_red_flags"] = data["has_red_flags"]
    if "workflow_stopped" in data:
        state["workflow_stopped"] = data["workflow_stopped"]
    if "allow_management_plan" in data:
        state["allow_management_plan"] = data["allow_management_plan"]


def _extend_unique(target: list[Any], values: list[Any]) -> None:
    for value in values:
        key = repr(value)
        if all(repr(existing) != key for existing in target):
            target.append(value)


def _workflow_status(
    executed: list[StepResult],
    state: Mapping[str, Any],
    terminal_template: str | None,
    errors: list[str],
) -> str:
    if errors or any(step.status == "error" for step in executed):
        return "error"
    if any(step.status == "stopped" for step in executed) or state.get("workflow_stopped"):
        return "stopped_for_safety"
    if terminal_template == "missing_information" or is_present(state.get("missing_required_evidence")):
        return "missing_information"
    if terminal_template in {"low_probability", "candidate_summary", "differential_needed"}:
        return "candidate_only"
    return "completed"


def _suspicion_level(candidate: Mapping[str, Any]) -> str:
    score = float(candidate.get("score") or 0)
    if score >= float(candidate.get("strong_candidate_threshold") or 1):
        return "strong_candidate"
    if score >= float(candidate.get("candidate_threshold") or 0):
        return "candidate"
    return "low_confidence"


def _compare_numbers(left: Any, right: Any, op: str) -> bool:
    try:
        left_number = float(left)
        right_number = float(right)
    except (TypeError, ValueError):
        return False
    if op == "gt":
        return left_number > right_number
    if op == "gte":
        return left_number >= right_number
    if op == "lt":
        return left_number < right_number
    return left_number <= right_number


def _contains(left: Any, right: Any) -> bool:
    if isinstance(left, str):
        return clean_text(right) in left
    if isinstance(left, list | tuple | set):
        return right in left
    if isinstance(left, Mapping):
        return right in left
    return False


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple | set):
        return list(value)
    return [value]
