from __future__ import annotations

from typing import Any, Sequence

from guideline_skill.extractors.clinical_info_extractor import ClinicalInfoExtractor
from guideline_skill.schemas import ClinicalInfoUnit
from guideline_skill.segmenters.heading_segmenter import HeadingSegment, HeadingSegmenter
from guideline_skill.validators import validate_clinical_info_unit


class NarrativeGuidelinePipeline:
    def __init__(
        self,
        *,
        clinical_info_extractor: ClinicalInfoExtractor,
        heading_segmenter: HeadingSegmenter | None = None,
        max_chunk_chars: int = 1200,
    ) -> None:
        self.clinical_info_extractor = clinical_info_extractor
        self.heading_segmenter = heading_segmenter or HeadingSegmenter()
        self.max_chunk_chars = max_chunk_chars

    def run(
        self,
        pages_or_text: str | Sequence[Any],
        *,
        title: str | None = None,
        source_file: str | None = None,
    ) -> list[ClinicalInfoUnit]:
        text = _coerce_text(pages_or_text)
        segments = self.heading_segmenter.segment(text)
        units: list[ClinicalInfoUnit] = []

        for segment in segments:
            for chunk in _chunk_segment(segment, max_chunk_chars=self.max_chunk_chars):
                extracted = self.clinical_info_extractor.extract(
                    chunk,
                    title=title,
                    source_file=source_file,
                )
                units.append(validate_clinical_info_unit(extracted))

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


def _chunk_segment(segment: HeadingSegment, *, max_chunk_chars: int) -> list[HeadingSegment]:
    if max_chunk_chars <= 0 or len(segment.raw_text) <= max_chunk_chars:
        return [segment]

    chunks: list[HeadingSegment] = []
    offset = 0
    current_parts: list[str] = []
    current_start: int | None = None
    current_length = 0

    for part in _paragraph_parts(segment.raw_text):
        if len(part) > max_chunk_chars:
            if current_parts:
                chunks.append(_make_chunk(segment, current_parts, current_start or 0))
                current_parts = []
                current_start = None
                current_length = 0
            for sliced, slice_start in _slice_long_part(part, max_chunk_chars):
                chunks.append(_make_chunk(segment, [sliced], offset + slice_start))
            offset += len(part)
            continue

        if current_parts and current_length + len(part) > max_chunk_chars:
            chunks.append(_make_chunk(segment, current_parts, current_start or 0))
            current_parts = []
            current_start = None
            current_length = 0

        if current_start is None:
            current_start = offset
        current_parts.append(part)
        current_length += len(part)
        offset += len(part)

    if current_parts:
        chunks.append(_make_chunk(segment, current_parts, current_start or 0))

    return chunks or [segment]


def _paragraph_parts(text: str) -> list[str]:
    parts: list[str] = []
    current = ""
    lines = text.splitlines(keepends=True)
    for line in lines:
        current += line
        if not line.strip():
            if current:
                parts.append(current)
                current = ""
    if current:
        parts.append(current)
    return parts


def _slice_long_part(text: str, max_chunk_chars: int) -> list[tuple[str, int]]:
    slices: list[tuple[str, int]] = []
    start = 0
    while start < len(text):
        end = min(start + max_chunk_chars, len(text))
        slices.append((text[start:end], start))
        start = end
    return slices


def _make_chunk(
    segment: HeadingSegment,
    parts: list[str],
    local_start: int,
) -> HeadingSegment:
    raw_text = "".join(parts).strip()
    start = segment.start + local_start
    return HeadingSegment(
        section_path=list(segment.section_path),
        title=segment.title,
        raw_text=raw_text,
        start=start,
        end=start + len(raw_text),
        page_start=segment.page_start,
        page_end=segment.page_end,
        heading_pattern_name=segment.heading_pattern_name,
        heading_rank=segment.heading_rank,
    )
