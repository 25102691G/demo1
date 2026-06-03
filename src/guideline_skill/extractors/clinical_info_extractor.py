from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from guideline_skill.normalizer import JsonChatClient
from guideline_skill.schemas import (
    ClinicalInfoUnit,
    ClinicalInfoUnitBody,
    ClinicalInfoUnitType,
    ClinicalTopic,
    GuidelineMeta,
    SourceLocation,
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
  "contraindication": [],
  "diagnostic_criteria": [],
  "differential_diagnosis": [],
  "drug": null,
  "dose": null,
  "route": null,
  "frequency": null,
  "duration": null,
  "confidence": 0.0,
  "needs_human_review": false,
  "review_reasons": []
}"""


class ClinicalInfoPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    unit_type: ClinicalInfoUnitType
    clinical_topic: ClinicalTopic
    action: str | None = None
    condition: str | None = None
    indication: list[str] = Field(default_factory=list)
    contraindication: list[str] = Field(default_factory=list)
    diagnostic_criteria: list[str] = Field(default_factory=list)
    differential_diagnosis: list[str] = Field(default_factory=list)
    drug: str | None = None
    dose: str | None = None
    route: str | None = None
    frequency: str | None = None
    duration: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    needs_human_review: bool
    review_reasons: list[str] = Field(default_factory=list)


class ClinicalInfoExtractor:
    def __init__(self, deepseek_client: JsonChatClient) -> None:
        self.deepseek_client = deepseek_client

    def extract(
        self,
        segment: HeadingSegment,
        *,
        title: str | None = None,
        source_file: str | None = None,
    ) -> ClinicalInfoUnit:
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
            return ClinicalInfoUnit(
                guideline_meta=guideline_meta,
                unit=_build_unit_body(
                    segment=segment,
                    payload=parsed,
                    review_reasons=parsed.review_reasons,
                ),
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
                "contraindication": "string[]",
                "diagnostic_criteria": "string[]",
                "differential_diagnosis": "string[]",
                "drug": "string | null",
                "dose": "string | null",
                "route": "string | null",
                "frequency": "string | null",
                "duration": "string | null",
                "confidence": "number between 0 and 1",
                "needs_human_review": "boolean",
                "review_reasons": "string[]",
            },
        },
        ensure_ascii=False,
    )


def _build_unit_body(
    *,
    segment: HeadingSegment,
    payload: ClinicalInfoPayload,
    review_reasons: list[str],
) -> ClinicalInfoUnitBody:
    return ClinicalInfoUnitBody(
        id=_unit_id(segment),
        section_path=list(segment.section_path),
        title=segment.title or None,
        raw_text=segment.raw_text,
        unit_type=payload.unit_type,
        clinical_topic=payload.clinical_topic,
        action=payload.action,
        condition=payload.condition,
        indication=list(payload.indication),
        contraindication=list(payload.contraindication),
        diagnostic_criteria=list(payload.diagnostic_criteria),
        differential_diagnosis=list(payload.differential_diagnosis),
        drug=payload.drug,
        dose=payload.dose,
        route=payload.route,
        frequency=payload.frequency,
        duration=payload.duration,
        source_location=SourceLocation(
            page_start=segment.page_start,
            page_end=segment.page_end,
            section=" / ".join(segment.section_path) if segment.section_path else None,
        ),
        confidence=payload.confidence,
        needs_human_review=payload.needs_human_review,
        review_reasons=list(review_reasons),
    )


def _fallback_unit(
    *,
    segment: HeadingSegment,
    guideline_meta: GuidelineMeta,
    error: Exception,
) -> ClinicalInfoUnit:
    payload = ClinicalInfoPayload(
        unit_type="other",
        clinical_topic="other",
        action=None,
        condition=None,
        indication=[],
        contraindication=[],
        diagnostic_criteria=[],
        differential_diagnosis=[],
        drug=None,
        dose=None,
        route=None,
        frequency=None,
        duration=None,
        confidence=0.0,
        needs_human_review=True,
        review_reasons=[f"clinical_info_llm_failed: {error}"],
    )
    return ClinicalInfoUnit(
        guideline_meta=guideline_meta,
        unit=_build_unit_body(
            segment=segment,
            payload=payload,
            review_reasons=payload.review_reasons,
        ),
    )


def _unit_id(segment: HeadingSegment) -> str:
    seed = f"{segment.start}:{segment.end}:{segment.title}:{segment.raw_text[:120]}"
    digest = hashlib.md5(seed.encode("utf-8")).hexdigest()[:12]
    return f"clinical_info_unit_{digest}"
