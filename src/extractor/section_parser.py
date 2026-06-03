from __future__ import annotations

try:
    import regex as re
except ImportError:  # pragma: no cover
    import re  # type: ignore

from .models import SectionRecord, TextPage, make_node_id, stable_token
from .text_cleaner import compact_text

LEVEL1_RE = re.compile(r"^\s*([一二三四五六七八九十]+)[、.]\s*(?P<title>[^。；;:]{2,40})\s*$")
LEVEL2_RE = re.compile(r"^\s*[(]([一二三四五六七八九十]+)[)]\s*(?P<title>[^。；;:]{2,45})\s*$")
NUMBERED_RE = re.compile(r"^\s*([0-9]+)[.、]\s*(?P<title>[^。；;:]{2,45})\s*$")


def _clean_title(title: str) -> str:
    title = re.sub(r"\s+", "", title)
    return title.strip(" .。:：")


def _looks_like_heading(line: str) -> tuple[int, str] | None:
    line = line.strip()
    if not line or len(line) > 55:
        return None

    for level, pattern in ((1, LEVEL1_RE), (2, LEVEL2_RE), (2, NUMBERED_RE)):
        match = pattern.match(line)
        if match:
            title = _clean_title(match.group("title"))
            if title and len(title) <= 45:
                return level, title
    return None


def parse_sections(pages: list[TextPage]) -> list[SectionRecord]:
    """Parse first- and second-level sections from cleaned PDF pages."""

    sections: list[SectionRecord] = []
    last_level1_id = ""
    seen_ids: set[str] = set()

    for page in pages:
        for line in page.text.splitlines():
            heading = _looks_like_heading(line)
            if heading is None:
                continue
            level, title = heading
            if level != 1:
                continue
            token = stable_token(f"{level}_{title}_{page.page_number}_{len(sections)}")
            section_id = make_node_id("Section", token)
            if section_id in seen_ids:
                continue
            record = SectionRecord(
                section_id=section_id,
                parent_section_id="",
                title=title,
                level=level,
                page_start=page.page_number,
                page_end=page.page_number,
                source_text=compact_text(line, max_len=500),
            )
            sections.append(record)
            seen_ids.add(section_id)
            if level == 1:
                last_level1_id = section_id

    if not sections:
        max_page = pages[-1].page_number if pages else None
        return [
            SectionRecord(
                section_id=make_node_id("Section", "full_text"),
                title="全文",
                level=1,
                page_start=pages[0].page_number if pages else None,
                page_end=max_page,
                source_text=compact_text(pages[0].text if pages else "", max_len=500),
            )
        ]

    last_page = pages[-1].page_number if pages else sections[-1].page_start
    for index, section in enumerate(sections):
        next_section = sections[index + 1] if index + 1 < len(sections) else None
        section.page_end = next_section.page_start if next_section else last_page

    return sections


def find_section_for_page(sections: list[SectionRecord], page_number: int | None) -> SectionRecord | None:
    """Return the nearest section covering a page."""

    if page_number is None or not sections:
        return None
    top_sections = [section for section in sections if section.level == 1]
    if not top_sections:
        top_sections = sections
    candidates = [
        section
        for section in top_sections
        if (section.page_start or 0) <= page_number <= (section.page_end or page_number)
    ]
    if candidates:
        return candidates[-1]
    before = [section for section in top_sections if (section.page_start or 0) <= page_number]
    return before[-1] if before else top_sections[0]
