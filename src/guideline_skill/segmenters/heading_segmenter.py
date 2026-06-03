from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass(frozen=True)
class HeadingMatch:
    name: str
    rank: int
    pattern: str
    text: str
    title: str
    line_start: int
    line_end: int


@dataclass(frozen=True)
class HeadingSegment:
    section_path: list[str]
    title: str
    raw_text: str
    start: int
    end: int
    page_start: int | None
    page_end: int | None
    heading_pattern_name: str | None
    heading_rank: int | None


class HeadingPatternRegistry:
    def __init__(self, rules_path: str | Path | None = None) -> None:
        self.rules_path = Path(rules_path) if rules_path is not None else _default_config_path("heading_rules.yaml")
        payload = _load_yaml_mapping(self.rules_path)
        self._rules = _normalize_heading_rules(payload.get("heading_patterns"))
        self._compiled = _compile_heading_rules(self._rules)

    def match_heading(
        self,
        line: str,
        *,
        line_start: int = 0,
        line_end: int | None = None,
    ) -> HeadingMatch | None:
        text = line.strip()
        if not text:
            return None
        actual_line_end = line_start + len(line) if line_end is None else line_end

        candidates: list[HeadingMatch] = []
        for rule, pattern_text, pattern in self._compiled:
            if pattern.match(text):
                candidates.append(
                    HeadingMatch(
                        name=str(rule["name"]),
                        rank=int(rule["rank"]),
                        pattern=pattern_text,
                        text=text,
                        title=_extract_title(text),
                        line_start=line_start,
                        line_end=actual_line_end,
                    )
                )

        if not candidates:
            return None
        return min(candidates, key=lambda item: (item.rank, item.line_start, item.name))


class HeadingSegmenter:
    def __init__(self, heading_registry: HeadingPatternRegistry | None = None) -> None:
        self.heading_registry = heading_registry or HeadingPatternRegistry()

    def segment(self, text: str) -> list[HeadingSegment]:
        if not text:
            return []

        page_index = _build_page_index(text)
        segments: list[HeadingSegment] = []
        stack: list[HeadingMatch] = []
        active_start: int | None = None
        active_path: list[str] = []
        active_heading: HeadingMatch | None = None

        for line, line_start, line_end in _iter_lines_with_offsets(text):
            if _is_page_marker(line):
                continue

            heading = self.heading_registry.match_heading(
                line,
                line_start=line_start,
                line_end=line_end,
            )
            if heading is None:
                continue

            if active_start is not None:
                _append_segment(
                    segments,
                    text=text,
                    start=active_start,
                    end=heading.line_start,
                    section_path=active_path,
                    heading=active_heading,
                    page_index=page_index,
                )

            while stack and stack[-1].rank >= heading.rank:
                stack.pop()
            stack.append(heading)

            active_start = heading.line_start
            active_path = [item.title for item in stack]
            active_heading = heading

        if active_start is not None:
            _append_segment(
                segments,
                text=text,
                start=active_start,
                end=len(text),
                section_path=active_path,
                heading=active_heading,
                page_index=page_index,
            )
        elif text.strip():
            _append_segment(
                segments,
                text=text,
                start=0,
                end=len(text),
                section_path=[],
                heading=None,
                page_index=page_index,
            )

        return segments


def _append_segment(
    segments: list[HeadingSegment],
    *,
    text: str,
    start: int,
    end: int,
    section_path: list[str],
    heading: HeadingMatch | None,
    page_index: list[tuple[int, int]],
) -> None:
    raw_text = text[start:end].strip()
    if not raw_text:
        return
    segments.append(
        HeadingSegment(
            section_path=list(section_path),
            title=section_path[-1] if section_path else "",
            raw_text=raw_text,
            start=start,
            end=end,
            page_start=_page_for_offset(page_index, start),
            page_end=_page_for_offset(page_index, max(start, end - 1)),
            heading_pattern_name=heading.name if heading else None,
            heading_rank=heading.rank if heading else None,
        )
    )


def _default_config_path(filename: str) -> Path:
    return Path(__file__).resolve().parents[3] / "configs" / filename


def _load_yaml_mapping(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} 顶层必须是映射对象。")
    return payload


def _normalize_heading_rules(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("heading_rules.yaml 的 heading_patterns 必须是列表。")

    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(value, 1):
        if not isinstance(item, Mapping):
            raise ValueError(f"heading_patterns[{index}] 必须是映射对象。")
        if not item.get("name"):
            raise ValueError(f"heading_patterns[{index}] 缺少 name。")
        if "rank" not in item:
            raise ValueError(f"heading_patterns[{index}] 缺少 rank。")
        patterns = item.get("patterns")
        if not isinstance(patterns, list) or not all(isinstance(pattern, str) for pattern in patterns):
            raise ValueError(f"heading_patterns[{index}].patterns 必须是字符串列表。")
        normalized.append(dict(item))
    return normalized


def _compile_heading_rules(
    rules: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], str, re.Pattern[str]]]:
    compiled: list[tuple[dict[str, Any], str, re.Pattern[str]]] = []
    for rule in rules:
        for pattern_text in rule["patterns"]:
            try:
                pattern = re.compile(pattern_text)
            except re.error as exc:
                raise ValueError(f"heading pattern {rule.get('name')!r} 的正则无效：{pattern_text!r}") from exc
            compiled.append((rule, pattern_text, pattern))
    return compiled


def _extract_title(text: str) -> str:
    title = text.strip()
    prefix_patterns = (
        r"^[一二三四五六七八九十百]+(?:[、．.]|\s+)\s*",
        r"^\d+(?!\.\d)(?:[、．.]|\s+)\s*",
        r"^[IVXLCDM]+(?:\.\d+)+\s+",
        r"^[IVXLCDM]+\s+",
        r"^\d+(?:\.\d+)+\s+",
        r"^[（(][一二三四五六七八九十百A-Za-z][）)]\s*",
        r"^[①②③④⑤⑥⑦⑧⑨⑩]\s*",
        r"^\[\d+\]\s*",
    )
    for pattern in prefix_patterns:
        cleaned = re.sub(pattern, "", title)
        if cleaned != title:
            return cleaned.strip()
    return title


def _iter_lines_with_offsets(text: str) -> list[tuple[str, int, int]]:
    lines: list[tuple[str, int, int]] = []
    offset = 0
    for raw_line in text.splitlines(keepends=True):
        line_end = offset + len(raw_line)
        line_without_newline = raw_line.rstrip("\r\n")
        lines.append((line_without_newline, offset, offset + len(line_without_newline)))
        offset = line_end
    if text and not text.endswith(("\n", "\r")) and (not lines or lines[-1][2] != len(text)):
        lines.append((text[offset:], offset, len(text)))
    return lines


def _is_page_marker(line: str) -> bool:
    return re.match(r"^\s*##\s*Page\s+\d+\s*$", line) is not None


def _build_page_index(text: str) -> list[tuple[int, int]]:
    page_index: list[tuple[int, int]] = []
    current_page = 1
    for line, line_start, line_end in _iter_lines_with_offsets(text):
        marker = re.match(r"^\s*##\s*Page\s+(\d+)\s*$", line)
        if marker:
            current_page = int(marker.group(1))
        page_index.append((line_start, current_page))
        if line_end > line_start:
            page_index.append((line_end, current_page))
    if not page_index:
        page_index.append((0, 1))
    return sorted(page_index, key=lambda item: item[0])


def _page_for_offset(page_index: list[tuple[int, int]], offset: int) -> int | None:
    if not page_index:
        return None
    page = page_index[0][1]
    for start, candidate_page in page_index:
        if start > offset:
            break
        page = candidate_page
    return page
