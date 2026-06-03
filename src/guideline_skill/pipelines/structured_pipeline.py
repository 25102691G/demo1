from __future__ import annotations

import hashlib
import re
from typing import Any, Sequence

from guideline_skill.anchors import AnchorRegistry
from guideline_skill.classifier import ClassificationResult
from guideline_skill.extractors.statement_extractor import StatementExtractor
from guideline_skill.normalizer import LLMNormalizer
from guideline_skill.schemas import GuidelineMeta, SourceLocation, StatementUnit, StatementUnitBody
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

        units: list[StatementUnit] = []
        for index, segment in enumerate(segments, 1):
            extracted = self.statement_extractor.extract(
                segment,
                surrounding_context=text[: segment.start],
            )
            normalized = self.normalizer.normalize_statement_fields(
                extracted.evidence_quality_raw,
                extracted.strength_raw,
                extracted.statement_type,
            )
            unit = StatementUnit(
                guideline_meta=guideline_meta,
                unit=StatementUnitBody(
                    id=_statement_id(index, segment),
                    original_label=extracted.original_label,
                    statement_type=extracted.statement_type,
                    statement_text=extracted.statement_text,
                    clinical_question=extracted.clinical_question,
                    evidence_quality_raw=extracted.evidence_quality_raw,
                    evidence_quality_normalized=normalized.get("evidence_quality_normalized"),
                    strength_raw=extracted.strength_raw,
                    strength_normalized=normalized.get("strength_normalized"),
                    consensus_level=extracted.consensus_level,
                    implementation_advice=extracted.implementation_advice,
                    rationale=extracted.rationale,
                    source_location=SourceLocation(
                        page_start=segment.page_start,
                        page_end=segment.page_end,
                        section=None,
                    ),
                    confidence=float(normalized.get("confidence", 0.0)),
                    needs_human_review=bool(normalized.get("needs_human_review", False)),
                    review_reasons=list(normalized.get("review_reasons", [])),
                ),
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
