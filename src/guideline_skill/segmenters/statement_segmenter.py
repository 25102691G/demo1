from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Pattern

from guideline_skill.anchors import AnchorRegistry


@dataclass(frozen=True)
class StatementSegment:
    original_label: str
    raw_text: str
    start: int
    end: int
    page_start: int | None
    page_end: int | None
    section: str | None


class StatementSegmenter:
    def __init__(self, anchor_registry: AnchorRegistry) -> None:
        self.anchor_registry = anchor_registry

    def segment(self, text: str, primary_unit_anchor_name: str) -> list[StatementSegment]:
        anchor = self.anchor_registry.get_unit_anchor(primary_unit_anchor_name)
        if anchor is None:
            raise ValueError(f"Unknown unit anchor: {primary_unit_anchor_name}")

        reference_start = _reference_section_start(text)
        search_text = text[:reference_start] if reference_start is not None else text
        patterns = _compile_patterns(anchor.get("patterns", []), primary_unit_anchor_name)
        matches = _find_non_overlapping_matches(search_text, patterns)
        if not matches:
            return []

        segments: list[StatementSegment] = []
        for index, match in enumerate(matches):
            start = match.start()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(search_text)
            raw_text = text[start:end].strip()
            if not raw_text:
                continue
            segments.append(
                StatementSegment(
                    original_label=match.group(0).strip(),
                    raw_text=raw_text,
                    start=start,
                    end=end,
                    page_start=_page_for_offset(text, start),
                    page_end=_page_for_offset(text, max(start, end - 1)),
                    section=_section_for_offset(text, start),
                )
            )
        return segments


def _compile_patterns(patterns: object, anchor_name: str) -> list[Pattern[str]]:
    if not isinstance(patterns, list):
        raise ValueError(f"Unit anchor {anchor_name!r} patterns must be a list.")
    compiled: list[Pattern[str]] = []
    for pattern in patterns:
        if not isinstance(pattern, str):
            raise ValueError(f"Unit anchor {anchor_name!r} patterns must contain strings.")
        compiled.append(re.compile(pattern, re.IGNORECASE | re.MULTILINE))
    return compiled


def _find_non_overlapping_matches(text: str, patterns: list[Pattern[str]]) -> list[re.Match[str]]:
    candidates: list[re.Match[str]] = []
    for pattern in patterns:
        candidates.extend(pattern.finditer(text))
    candidates.sort(key=lambda match: (match.start(), -(match.end() - match.start())))

    matches: list[re.Match[str]] = []
    last_end = -1
    for match in candidates:
        if match.start() < last_end:
            continue
        matches.append(match)
        last_end = match.end()
    return matches


def _reference_section_start(text: str) -> int | None:
    match = re.search(r"(?im)^\s*(参考文献|References)\s*$", text)
    return match.start() if match else None


def _page_for_offset(text: str, offset: int) -> int | None:
    page = 1
    found_marker = False
    for marker in re.finditer(r"(?m)^\s*##\s*Page\s+(\d+)\s*$", text):
        if marker.start() > offset:
            break
        found_marker = True
        page = int(marker.group(1))
    return page if found_marker or text else None


def _section_for_offset(text: str, offset: int) -> str | None:
    preceding_text = text[:offset]
    matches = list(
        re.finditer(
            r"(?m)^\s*([一二三四五六七八九十]+、[^\n]{2,80})\s*$",
            preceding_text,
        )
    )
    if not matches:
        return None
    section = matches[-1].group(1).strip()
    return _clean_section_title(section)


def _clean_section_title(section: str) -> str:
    section = re.sub(r"\s+", " ", section)
    section = re.sub(r"^([一二三四五六七八九十]+、)\s*", r"\1", section)
    return section.strip(" \t\r\n")
