from __future__ import annotations

from typing import Any, Sequence

from guideline_skill.extractors.clinical_info_extractor import ClinicalInfoExtractor
from guideline_skill.schemas import StatementUnit
from guideline_skill.segmenters.heading_segmenter import HeadingSegment, HeadingSegmenter
from guideline_skill.validators import validate_statement_unit


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
        # Kept for backwards-compatible construction; narrative chunks are now paragraph-based.
        self.max_chunk_chars = max_chunk_chars

    def run(
        self,
        pages_or_text: str | Sequence[Any],
        *,
        title: str | None = None,
        source_file: str | None = None,
    ) -> list[StatementUnit]:
        text = _coerce_text(pages_or_text)
        segments = self.heading_segmenter.segment(text)
        units: list[StatementUnit] = []

        for segment in segments:
            for chunk in _paragraph_chunks(segment):
                extracted = self.clinical_info_extractor.extract(
                    chunk,
                    title=title,
                    source_file=source_file,
                )
                units.append(validate_statement_unit(extracted))

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


def _paragraph_chunks(segment: HeadingSegment) -> list[HeadingSegment]:
    chunks: list[HeadingSegment] = []
    offset = 0

    for part in _paragraph_parts(segment.raw_text):
        if part.strip():
            chunks.append(_make_chunk(segment, [part], offset))
        offset += len(part)

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
