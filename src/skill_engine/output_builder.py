from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from .schemas import load_json_schema, validate_json
from .skill_loader import SkillPack
from .utils import clean_text, dedupe_texts, shorten


DISCLAIMER = "本结果仅用于指南知识组织和临床决策支持，不替代医生诊断、处方或急诊处理。"


def build_workflow_output(
    *,
    canonical_case: dict[str, Any],
    top_candidates: list[dict[str, Any]],
    selected_skill_outputs: list[dict[str, Any]],
    skill_packs: list[SkillPack],
    output_schema_path: Path,
    run_id: str | None = None,
    debug: bool = False,
    errors: list[str] | None = None,
) -> dict[str, Any]:
    run_id = run_id or f"run_{uuid.uuid4().hex}"
    status = _overall_status(top_candidates, selected_skill_outputs, errors or [])
    structured_result = _structured_result(
        status=status,
        canonical_case=canonical_case,
        top_candidates=top_candidates,
        selected_skill_outputs=selected_skill_outputs,
    )
    safety = _safety(selected_skill_outputs)
    citations = _citations(selected_skill_outputs, skill_packs)
    output = {
        "run_id": run_id,
        "case_id": canonical_case["case_id"],
        "canonical_case": canonical_case,
        "status": status,
        "top_candidates": top_candidates,
        "selected_skill_outputs": selected_skill_outputs,
        "final_response": {
            "audience": "clinician",
            "summary": _summary(status, top_candidates, structured_result),
            "structured_result": structured_result,
            "disclaimer": DISCLAIMER,
        },
        "safety": safety,
        "citations": citations,
        "debug_trace": {
            "enabled": debug,
            "route_scores": [
                {
                    "skill_id": candidate["skill_id"],
                    "score": candidate["score"],
                    "matched": [item["name"] for item in candidate.get("matched_features", [])],
                    "penalties": candidate.get("negative_features", []),
                }
                for candidate in top_candidates
            ],
            "executed_steps": [
                step
                for output_item in selected_skill_outputs
                for step in output_item.get("executed_steps", [])
            ],
            "errors": errors or _execution_errors(selected_skill_outputs),
        },
    }
    validate_json(output, load_json_schema(Path(output_schema_path)), label="workflow_output")
    return output


def build_error_output(
    *,
    canonical_case: dict[str, Any],
    output_schema_path: Path,
    errors: list[str],
    run_id: str | None = None,
    debug: bool = True,
) -> dict[str, Any]:
    return build_workflow_output(
        canonical_case=canonical_case,
        top_candidates=[],
        selected_skill_outputs=[],
        skill_packs=[],
        output_schema_path=output_schema_path,
        run_id=run_id,
        debug=debug,
        errors=errors,
    )


def _overall_status(
    top_candidates: list[dict[str, Any]],
    selected_skill_outputs: list[dict[str, Any]],
    errors: list[str],
) -> str:
    if errors and not selected_skill_outputs:
        return "error"
    statuses = [clean_text(item.get("workflow_status")) for item in selected_skill_outputs]
    if "stopped_for_safety" in statuses:
        return "stopped_for_safety"
    if top_candidates and all(
        float(candidate.get("score") or 0) < float(candidate.get("candidate_threshold") or 0)
        for candidate in top_candidates
    ):
        return "low_confidence"
    if "missing_information" in statuses:
        return "missing_information"
    if "error" in statuses and not any(status in {"completed", "candidate_only", "missing_information"} for status in statuses):
        return "error"
    if "candidate_only" in statuses and "completed" not in statuses:
        return "needs_human_review"
    return "completed"


def _structured_result(
    *,
    status: str,
    canonical_case: dict[str, Any],
    top_candidates: list[dict[str, Any]],
    selected_skill_outputs: list[dict[str, Any]],
) -> dict[str, Any]:
    result = {
        "diagnosis_status": _diagnosis_status(status, canonical_case, top_candidates, selected_skill_outputs),
        "supporting_evidence": [],
        "opposing_evidence": [],
        "missing_information": [],
        "recommended_next_steps": [],
        "differentials_to_consider": [],
        "management_recommendations": [],
        "monitoring_or_follow_up": [],
        "safety_notes": [],
    }
    for output_item in selected_skill_outputs:
        item_result = output_item.get("result") or {}
        for field in result:
            if field == "diagnosis_status":
                continue
            values = item_result.get(field)
            if isinstance(values, list):
                result[field] = dedupe_texts([*result[field], *(str(value) for value in values)])
    return result


def _diagnosis_status(
    status: str,
    canonical_case: dict[str, Any],
    top_candidates: list[dict[str, Any]],
    selected_skill_outputs: list[dict[str, Any]],
) -> str:
    if any(diagnosis.get("status") == "confirmed" for diagnosis in canonical_case.get("diagnoses") or []):
        return "confirmed"
    if status == "low_confidence":
        return "low_probability"
    if status == "missing_information":
        return "evidence_insufficient"
    if any(item.get("workflow_status") == "candidate_only" for item in selected_skill_outputs):
        return "candidate"
    if top_candidates:
        top = top_candidates[0]
        if float(top.get("score") or 0) >= float(top.get("strong_candidate_threshold") or 1):
            return "suspected"
    return "candidate"


def _safety(selected_skill_outputs: list[dict[str, Any]]) -> dict[str, Any]:
    red_flags: list[dict[str, Any]] = []
    for output_item in selected_skill_outputs:
        item_result = output_item.get("result") or {}
        for flag in item_result.get("red_flags") or []:
            if isinstance(flag, dict):
                red_flags.append(flag)
    workflow_stopped = any(item.get("workflow_status") == "stopped_for_safety" for item in selected_skill_outputs)
    return {
        "has_red_flags": bool(red_flags),
        "red_flags": red_flags,
        "workflow_stopped": workflow_stopped,
        "safety_message": "存在安全红旗，建议优先急诊或即时临床评估。" if workflow_stopped else None,
    }


def _citations(
    selected_skill_outputs: list[dict[str, Any]],
    skill_packs: list[SkillPack],
) -> list[dict[str, Any]]:
    cards_by_id = {
        card_id: card
        for pack in skill_packs
        for card_id, card in pack.cards_by_id.items()
    }
    used_cards = dedupe_texts(
        card_id
        for output_item in selected_skill_outputs
        for card_id in (output_item.get("result") or {}).get("used_cards", [])
    )
    citations: list[dict[str, Any]] = []
    for card_id in used_cards:
        card = cards_by_id.get(card_id)
        if not card:
            citations.append({"card_id": card_id, "recommendation_label": None, "source": {}})
            continue
        citations.append(
            {
                "card_id": card_id,
                "recommendation_label": card.get("recommendation_label"),
                "source": _citation_source(card),
            }
        )
    return citations


def _citation_source(card: dict[str, Any]) -> dict[str, Any]:
    location = card.get("source_location") if isinstance(card.get("source_location"), dict) else {}
    retrieval = card.get("retrieval") if isinstance(card.get("retrieval"), dict) else {}
    source = card.get("source") if isinstance(card.get("source"), dict) else {}
    guideline = card.get("guideline") if isinstance(card.get("guideline"), dict) else {}
    guideline_meta = card.get("guideline_meta") if isinstance(card.get("guideline_meta"), dict) else {}
    pdf = (
        clean_text(location.get("pdf"))
        or clean_text(retrieval.get("source_pdf"))
        or clean_text(source.get("pdf"))
        or clean_text(guideline.get("source_file"))
        or clean_text(guideline_meta.get("source_file"))
    )
    page_start = _positive_int(location.get("page_start") or retrieval.get("page_start") or source.get("page_start"))
    page_end = _positive_int(location.get("page_end") or retrieval.get("page_end") or source.get("page_end"))
    quote = clean_text(card.get("original_text") or card.get("statement_text") or card.get("action"))
    payload: dict[str, Any] = {}
    if pdf:
        payload["pdf"] = pdf
    if page_start is not None:
        payload["page_start"] = page_start
    if page_end is not None:
        payload["page_end"] = page_end
    if quote:
        payload["quote"] = shorten(quote, limit=500)
    return payload


def _positive_int(value: Any) -> int | None:
    try:
        integer = int(value)
    except (TypeError, ValueError):
        return None
    return integer if integer >= 1 else None


def _summary(
    status: str,
    top_candidates: list[dict[str, Any]],
    structured_result: dict[str, Any],
) -> str:
    disease = top_candidates[0]["disease_name"] if top_candidates else "未选择技能"
    if status == "stopped_for_safety":
        return "检测到安全红旗，流程已停止并建议优先安全评估。"
    if status == "error":
        return "SkillEngine 执行失败，请查看 debug_trace.errors。"
    if status == "low_confidence":
        return f"未达到候选阈值，当前仅返回最高分技能：{disease}。"
    if status == "missing_information":
        missing = structured_result.get("missing_information") or []
        return f"{disease} 工作流提示关键证据不足，需补充 {len(missing)} 项信息。"
    return f"{disease} 工作流已完成规则版指南信息组织。"


def _execution_errors(selected_skill_outputs: list[dict[str, Any]]) -> list[str]:
    return [
        error
        for output_item in selected_skill_outputs
        for error in (output_item.get("result") or {}).get("errors", [])
        if clean_text(error)
    ]
