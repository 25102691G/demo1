from __future__ import annotations

try:
    import regex as re
except ImportError:  # pragma: no cover
    import re  # type: ignore

from .models import RecommendationRecord, SectionRecord, TextPage, make_node_id
from .section_parser import find_section_for_page
from .text_cleaner import compact_text, normalize_text

HEADER_RE = re.compile(r"推荐意见\s*([0-9]+[a-zA-Z]?)\s*[:：]")
EVIDENCE_MARKER = r"证(?:\s*据|[^()（）]{1,180}?据)\s*等级"
STRICT_EVIDENCE_RE = re.compile(rf"[(]\s*{EVIDENCE_MARKER}\s*[:：]\s*([1-5])")
EVIDENCE_RE = re.compile(rf"{EVIDENCE_MARKER}\s*[:：]\s*([1-5])")
STRENGTH_RE = re.compile(r"推荐强度\s*[:：]\s*(强|弱)")
BPS_RE = re.compile(r"(?i)\bBPS\b|最佳临床实践声明")
REASON_RE = re.compile(r"推荐理由\s*[:：]?")
ADVICE_RE = re.compile(r"实施建议\s*[:：]?")
PAGE_MARKER_RE = re.compile(r"\s*<<PAGE:([0-9]+)>>\s*")
EVIDENCE_BLOCK_RE = re.compile(
    rf"[(]\s*(?:{EVIDENCE_MARKER}\s*[:：]\s*[1-5]\s*[,，;；]?\s*)?(?:推荐强度\s*[:：]\s*(强|弱)\s*)?(?:BPS|最佳临床实践声明)?\s*[)]"
)
TRAILING_CONTENT_RE = re.compile(r"\n\s*(?:[一二三四五六七八九十]+[、.．]\s*展望|利益冲突|参考文献|起草小组专家|专家组成员)")


def _build_document_text(pages: list[TextPage]) -> tuple[str, list[tuple[int, int, int]]]:
    parts: list[str] = []
    spans: list[tuple[int, int, int]] = []
    offset = 0
    for page in pages:
        marker = f"\n<<PAGE:{page.page_number}>>\n"
        parts.append(marker)
        offset += len(marker)
        start = offset
        parts.append(page.text)
        offset += len(page.text)
        spans.append((start, offset, page.page_number))
    return "".join(parts), spans


def _pages_for_span(spans: list[tuple[int, int, int]], start: int, end: int) -> tuple[int | None, int | None]:
    pages = [page for span_start, span_end, page in spans if span_start <= end and span_end >= start]
    if not pages:
        return None, None
    return min(pages), max(pages)


def _strip_page_markers(text: str) -> str:
    return PAGE_MARKER_RE.sub(" ", text).strip()


def _trim_trailing_content(document_text: str, start: int, end: int) -> int:
    match = TRAILING_CONTENT_RE.search(document_text[start:end])
    return start + match.start() if match else end


def _looks_like_paragraph_start(line: str) -> bool:
    return bool(
        re.match(
            r"^(?:一项|另一项|多项|目前|因此|此外|然而|对于|当|若|如果|建议|推荐|需|应|可|临床|在|接受|CD|IBD)",
            line,
        )
    )


def _paragraphs(text: str) -> list[str]:
    lines = [line.strip() for line in normalize_text(text).splitlines() if line.strip()]
    paragraphs: list[str] = []
    current = ""

    for line in lines:
        if current and current[-1] in "。！？!?；;" and _looks_like_paragraph_start(line):
            paragraphs.append(compact_text(current))
            current = line
        else:
            current = f"{current}{line}" if current else line

    if current:
        paragraphs.append(compact_text(current))

    return paragraphs


def _format_paragraph_text(text: str) -> str:
    return "\n".join(_paragraphs(text))


def _slice_between(body: str, start_match: re.Match[str], end_match: re.Match[str] | None) -> str:
    start = start_match.end()
    end = end_match.start() if end_match and end_match.start() > start else len(body)
    return _format_paragraph_text(body[start:end])


def _split_reason_and_advice(
    body: str,
    reason_match: re.Match[str] | None,
    advice_match: re.Match[str] | None,
) -> tuple[str, str]:
    if not reason_match:
        if advice_match:
            return "", _slice_between(body, advice_match, None)
        return "", ""

    if advice_match:
        return _slice_between(body, reason_match, advice_match), _slice_between(body, advice_match, None)

    reason_start = reason_match.end()
    paragraphs = _paragraphs(body[reason_start:])
    if not paragraphs:
        return "", ""
    return paragraphs[0], "\n".join(paragraphs[1:])


def _parse_chunk(
    number: str,
    raw_chunk: str,
    page_start: int | None,
    page_end: int | None,
    sections: list[SectionRecord],
) -> RecommendationRecord:
    chunk = normalize_text(_strip_page_markers(raw_chunk))
    header = HEADER_RE.search(chunk)
    body = chunk[header.end() :] if header else chunk

    evidence_match = STRICT_EVIDENCE_RE.search(chunk) or EVIDENCE_RE.search(chunk)
    strength_match = STRENGTH_RE.search(chunk)
    evidence_grade = evidence_match.group(1) if evidence_match else ""
    recommendation_strength = strength_match.group(1) if strength_match else ""
    is_bps = bool(BPS_RE.search(chunk))

    reason_match = REASON_RE.search(body)
    advice_match = ADVICE_RE.search(body)
    split_points = [m.start() for m in (reason_match, advice_match) if m]
    main_end = min(split_points) if split_points else len(body)
    recommendation_text = compact_text(body[:main_end])
    recommendation_text = compact_text(EVIDENCE_BLOCK_RE.sub("", recommendation_text).strip(" ,，;；。"))

    reason, implementation_advice = _split_reason_and_advice(body, reason_match, advice_match)

    section = find_section_for_page(sections, page_start)
    section_title = section.title if section else ""

    return RecommendationRecord(
        recommendation_id=make_node_id("Recommendation", number),
        number=number,
        title=f"推荐意见{number}",
        text=recommendation_text,
        evidence_grade=evidence_grade,
        recommendation_strength=recommendation_strength,
        is_bps=is_bps,
        reason=reason,
        implementation_advice=implementation_advice,
        section=section_title,
        page_start=page_start,
        page_end=page_end,
        source_text=compact_text(chunk),
    )


def parse_recommendations(
    pages: list[TextPage],
    sections: list[SectionRecord] | None = None,
) -> list[RecommendationRecord]:
    """Parse recommendation cards from cleaned pages."""

    sections = sections or []
    document_text, page_spans = _build_document_text(pages)
    matches = list(HEADER_RE.finditer(document_text))
    recommendations: list[RecommendationRecord] = []

    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(document_text)
        end = _trim_trailing_content(document_text, start, end)
        raw_chunk = document_text[start:end]
        page_start, page_end = _pages_for_span(page_spans, start, end)
        record = _parse_chunk(match.group(1), raw_chunk, page_start, page_end, sections)
        recommendations.append(record)

    return recommendations
