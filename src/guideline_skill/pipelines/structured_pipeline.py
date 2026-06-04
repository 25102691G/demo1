from __future__ import annotations

import hashlib
import re
from typing import Any, Sequence

from guideline_skill.anchors import AnchorRegistry
from guideline_skill.classifier import ClassificationResult
from guideline_skill.extractors.statement_extractor import ExtractedStatementFields, StatementExtractor
from guideline_skill.normalizer import LLMNormalizer
from guideline_skill.schemas import GuidelineMeta, SourceLocation, StatementEvidence, StatementUnit
from guideline_skill.segmenters.statement_segmenter import StatementSegment, StatementSegmenter
from guideline_skill.validators import validate_statement_unit


class StructuredGuidelinePipeline:
    def __init__(
        self,
        *,
        anchor_registry: AnchorRegistry,
        normalizer: LLMNormalizer,
        statement_segmenter: StatementSegmenter | None = None,
        statement_extractor: StatementExtractor | None = None,
    ) -> None:
        self.anchor_registry = anchor_registry
        self.normalizer = normalizer
        self.statement_segmenter = statement_segmenter or StatementSegmenter(anchor_registry)
        self.statement_extractor = statement_extractor or StatementExtractor(anchor_registry.get_field_anchors())

    def run(
        self,
        pages_or_text: str | Sequence[Any],
        classification_result: ClassificationResult,
        *,
        title: str | None = None,
        source_file: str | None = None,
    ) -> list[StatementUnit]:
        if classification_result.doc_type != "structured_guideline":
            return []
        if not classification_result.primary_unit_anchor:
            return []

        primary_anchor = self.anchor_registry.get_unit_anchor(classification_result.primary_unit_anchor)
        if primary_anchor is None:
            raise ValueError(f"Unknown primary unit anchor: {classification_result.primary_unit_anchor}")

        text = _coerce_text(pages_or_text)
        segments = self.statement_segmenter.segment(text, classification_result.primary_unit_anchor)
        guideline_meta = GuidelineMeta(
            title=title,
            source_file=source_file,
            doc_type=classification_result.doc_type,
        )
        disease_payload = self.normalizer.extract_disease_from_filename(source_file)
        disease = str(disease_payload.get("disease") or "unknown")
        disease_review_reasons = list(disease_payload.get("review_reasons", []))

        extracted_items: list[tuple[int, StatementSegment, ExtractedStatementFields]] = []
        for index, segment in enumerate(segments, 1):
            extracted_items.append(
                (
                    index,
                    segment,
                    self.statement_extractor.extract(
                        segment,
                        surrounding_context=text[: segment.start],
                    ),
                )
            )

        evidence_payload = self.normalizer.normalize_evidence_quality_values(
            [item[2].evidence_quality_raw for item in extracted_items]
        )
        strength_payload = self.normalizer.normalize_strength_values(
            [item[2].strength_raw for item in extracted_items]
        )
        evidence_normalizations = dict(evidence_payload.get("normalizations", {}))
        strength_normalizations = dict(strength_payload.get("normalizations", {}))
        batch_review_reasons = [
            *list(evidence_payload.get("review_reasons", [])),
            *list(strength_payload.get("review_reasons", [])),
        ]

        units: list[StatementUnit] = []
        for index, segment, extracted in extracted_items:
            action_payload = self.normalizer.summarize_recommendation_action(
                statement_text=extracted.statement_text,
                implementation_advice=extracted.implementation_advice,
                rationale=extracted.rationale,
                clinical_stage=segment.section,
            )
            review_reasons = [
                *batch_review_reasons,
                *disease_review_reasons,
                *list(action_payload.get("review_reasons", [])),
            ]
            source_location = SourceLocation(
                page_start=segment.page_start,
                page_end=segment.page_end,
                section=segment.section,
            )
            unit = StatementUnit(
                guideline=guideline_meta,
                card_id=_statement_id(index, segment),
                source_statement_id=extracted.original_label,
                disease=disease,
                statement_type=extracted.statement_type,
                statement_text=extracted.statement_text,
                clinical_question=extracted.clinical_question,
                clinical_stage=source_location.section or "unknown",
                clinical_task=str(action_payload.get("clinical_task") or ""),
                population=action_payload.get("population"),
                condition=action_payload.get("condition"),
                action=str(action_payload.get("action") or extracted.statement_text),
                do_not=list(action_payload.get("do_not", [])),
                required_inputs=list(action_payload.get("required_inputs", [])),
                supporting_features=list(action_payload.get("supporting_features", [])),
                recommended_tests=list(action_payload.get("recommended_tests", [])),
                evidence=StatementEvidence(
                    evidence_quality_raw=extracted.evidence_quality_raw,
                    evidence_quality_normalized=evidence_normalizations.get(extracted.evidence_quality_raw, "unknown"),
                    recommendation_strength_raw=extracted.strength_raw,
                    recommendation_strength_normalized=strength_normalizations.get(extracted.strength_raw, "unknown"),
                    consensus_level=extracted.consensus_level,
                ),
                implementation_advice=extracted.implementation_advice,
                rationale=extracted.rationale,
                source_location=source_location,
                confidence=0.9,
                needs_human_review=bool(review_reasons),
                review_reasons=review_reasons,
            )
            units.append(
                validate_statement_unit(
                    unit,
                    primary_anchor_count=_count_primary_anchor_matches(segment.raw_text, primary_anchor),
                )
            )

        return units


def _coerce_text(pages_or_text: str | Sequence[Any]) -> str:
    if isinstance(pages_or_text, str):
        return pages_or_text
    parts: list[str] = []
    for index, page in enumerate(pages_or_text, 1):
        page_number = getattr(page, "page_number", None)
        page_text = getattr(page, "text", None)
        if isinstance(page, dict):
            page_number = page.get("page_number", page_number)
            page_text = page.get("text", page_text)
        parts.append(f"## Page {page_number or index}\n{page_text or ''}")
    return "\n".join(parts)


def _statement_id(index: int, segment: StatementSegment) -> str:
    digest = hashlib.md5(f"{segment.start}:{segment.original_label}".encode("utf-8")).hexdigest()[:8]
    return f"statement_unit_{index:03d}_{digest}"


def _count_primary_anchor_matches(raw_text: str, primary_anchor: dict[str, Any]) -> int:
    count = 0
    for pattern_text in primary_anchor.get("patterns", []):
        count += len(re.findall(str(pattern_text), raw_text, flags=re.IGNORECASE | re.MULTILINE))
    return count
