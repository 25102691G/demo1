from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from guideline_skill.normalizer import JsonChatClient, LLMNormalizer
from guideline_skill.schemas import (
    ClinicalInfoUnitType,
    ClinicalTopic,
    GuidelineMeta,
    SourceLocation,
    StatementEvidence,
    StatementUnit,
)
from guideline_skill.segmenters.heading_segmenter import HeadingSegment


CLINICAL_INFO_SYSTEM_PROMPT = """你是医学指南结构化抽取助手。请从给定中文指南小节中抽取结构化临床信息。
只输出 JSON，不要输出解释。
不要编造原文没有的信息。
不能确定的字段填 null 或空数组。
保留 raw_text 原文。
unit_type 必须从以下枚举中选择：
- definition
- classification
- clinical_manifestation
- diagnostic_criteria
- test_order
- instrumental_exam
- imaging_exam
- endoscopy_exam
- drug_regimen
- indication
- contraindication
- differential_diagnosis
- medical_record_writing
- surgery
- prognosis
- knowledge
- other

clinical_topic 必须从以下枚举中选择：
- background
- diagnosis
- treatment
- surgery
- follow_up
- documentation
- prognosis
- other

需要输出字段：

{
  "unit_type": "",
  "clinical_topic": "",
  "action": null,
  "condition": null,
  "indication": [],
  "diagnostic_criteria": [],
  "drug": null,
  "dose": null,
  "route": null,
  "frequency": null,
  "duration": null
}"""


class ClinicalInfoPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    unit_type: ClinicalInfoUnitType
    clinical_topic: ClinicalTopic
    action: str | None = None
    condition: str | None = None
    indication: list[str] = Field(default_factory=list)
    diagnostic_criteria: list[str] = Field(default_factory=list)
    drug: str | None = None
    dose: str | None = None
    route: str | None = None
    frequency: str | None = None
    duration: str | None = None


class ClinicalInfoExtractor:
    def __init__(self, deepseek_client: JsonChatClient, normalizer: LLMNormalizer | None = None) -> None:
        self.deepseek_client = deepseek_client
        self.normalizer = normalizer

    def extract(
        self,
        segment: HeadingSegment,
        *,
        title: str | None = None,
        source_file: str | None = None,
    ) -> StatementUnit:
        guideline_meta = GuidelineMeta(
            title=title,
            source_file=source_file,
            doc_type="narrative_guideline",
        )
        try:
            payload = self.deepseek_client.chat_json(
                CLINICAL_INFO_SYSTEM_PROMPT,
                _build_user_prompt(segment),
            )
            parsed = ClinicalInfoPayload.model_validate(payload)
            return _build_statement_unit(
                segment=segment,
                guideline_meta=guideline_meta,
                payload=parsed,
                normalizer=self.normalizer,
            )
        except (Exception, ValidationError) as exc:
            return _fallback_unit(
                segment=segment,
                guideline_meta=guideline_meta,
                error=exc,
            )


def _build_user_prompt(segment: HeadingSegment) -> str:
    return json.dumps(
        {
            "section_path": segment.section_path,
            "title": segment.title or None,
            "raw_text": segment.raw_text,
            "output_json_schema": {
                "unit_type": "definition | classification | clinical_manifestation | diagnostic_criteria | test_order | instrumental_exam | imaging_exam | endoscopy_exam | drug_regimen | indication | contraindication | differential_diagnosis | medical_record_writing | surgery | prognosis | knowledge | other",
                "clinical_topic": "background | diagnosis | treatment | surgery | follow_up | documentation | prognosis | other",
                "action": "string | null",
                "condition": "string | null",
                "indication": "string[]",
                "diagnostic_criteria": "string[]",
                "drug": "string | null",
                "dose": "string | null",
                "route": "string | null",
                "frequency": "string | null",
                "duration": "string | null",
            },
        },
        ensure_ascii=False,
    )


def _fallback_unit(
    *,
    segment: HeadingSegment,
    guideline_meta: GuidelineMeta,
    error: Exception,
) -> StatementUnit:
    payload = ClinicalInfoPayload(
        unit_type="other",
        clinical_topic="other",
        action=None,
        condition=None,
        indication=[],
        diagnostic_criteria=[],
        drug=None,
        dose=None,
        route=None,
        frequency=None,
        duration=None,
    )
    return _build_statement_unit(
        segment=segment,
        guideline_meta=guideline_meta,
        payload=payload,
        normalizer=None,
    )


def _build_statement_unit(
    *,
    segment: HeadingSegment,
    guideline_meta: GuidelineMeta,
    payload: ClinicalInfoPayload,
    normalizer: LLMNormalizer | None,
) -> StatementUnit:
    source_location = SourceLocation(
        page_start=segment.page_start,
        page_end=segment.page_end,
    )
    unit_id = _unit_id(segment)
    action = _first_text(payload.action, segment.title, segment.raw_text, "See statement_text.")
    summary_payload = (
        normalizer.summarize_recommendation(
            raw_chunk_text=segment.raw_text,
            statement_text=action,
        )
        if normalizer is not None
        else {}
    )
    clinical_stage = " / ".join(segment.section_path) if segment.section_path else None
    return StatementUnit(
        guideline=guideline_meta,
        card_id=unit_id,
        source_statement_id=unit_id,
        disease=_infer_disease_name_from_title(guideline_meta.title),
        statement_text=segment.raw_text,
        raw_chunk_text=segment.raw_text,
        clinical_stage=clinical_stage or payload.clinical_topic or "general_guideline_support",
        clinical_task=_first_text(summary_payload.get("clinical_task"), payload.clinical_topic, payload.unit_type),
        population=summary_payload.get("population"),
        condition=_first_text(summary_payload.get("condition"), payload.condition) or None,
        action=_first_text(summary_payload.get("action"), action),
        required_inputs=list(summary_payload.get("required_inputs") or _required_inputs_from_payload(payload)),
        evidence=StatementEvidence(
            evidence_quality_raw=None,
            evidence_quality_normalized="unknown",
            recommendation_strength_raw=None,
            recommendation_strength_normalized="unknown",
            consensus_level=None,
        ),
        source_location=source_location,
    )


def _required_inputs_from_payload(payload: ClinicalInfoPayload) -> list[str]:
    if payload.unit_type == "diagnostic_criteria":
        return list(payload.diagnostic_criteria)
    return []


def _first_text(*values: str | None) -> str:
    for value in values:
        text = (value or "").strip()
        if text:
            return text
    return ""


def _dedupe_texts(values: list[str | None]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        text = (value or "").strip()
        key = " ".join(text.lower().split())
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(text)
    return deduped


def _infer_disease_name_from_title(title: str | None) -> str:
    text = (title or "").strip()
    if not text:
        return "unknown"
    for marker in ("诊治指南", "诊断和治疗共识意见", "专家共识意见", "筛查与早诊早治指南", "指南", "共识"):
        if marker in text:
            text = text.split(marker, 1)[0]
            break
    return text.strip(" ：:，,。.()（）") or title or "unknown"


def _unit_id(segment: HeadingSegment) -> str:
    seed = f"{segment.start}:{segment.end}:{segment.title}:{segment.raw_text[:120]}"
    digest = hashlib.md5(seed.encode("utf-8")).hexdigest()[:12]
    return f"clinical_info_unit_{digest}"
