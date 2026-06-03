from __future__ import annotations

from typing import Iterable

try:
    import regex as re
except ImportError:  # pragma: no cover
    import re  # type: ignore

from .models import TextPage


def normalize_fullwidth(text: str) -> str:
    """Normalize full-width ASCII chars, numbers, brackets and colons."""

    chars: list[str] = []
    for char in text:
        code = ord(char)
        if char == "\u3000":
            chars.append(" ")
        elif 0xFF01 <= code <= 0xFF5E:
            chars.append(chr(code - 0xFEE0))
        else:
            chars.append(char)
    return "".join(chars)


def normalize_text(text: str) -> str:
    """Normalize punctuation and whitespace while preserving Chinese text."""

    text = normalize_fullwidth(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("﹙", "(").replace("﹚", ")")
    text = text.replace("：", ":").replace("（", "(").replace("）", ")")
    text = text.replace("，", ",")
    text = re.sub(r"[ \t]+", " ", text)
    return text


def clean_page_text(text: str) -> str:
    """Clean one extracted PDF page and keep structural line breaks."""

    text = normalize_text(text)
    raw_lines = [line.strip() for line in text.splitlines()]
    kept_lines: list[str] = []
    for line in raw_lines:
        if not line:
            continue
        if re.fullmatch(r"[-_—–]*\d+[-_—–]*", line):
            continue
        if re.fullmatch(r"[·\s]+", line):
            continue
        kept_lines.append(line)

    text = "\n".join(kept_lines)
    text = re.sub(r"-\n", "", text)
    text = re.sub(
        r"(?<![。！？!?；;:])\n(?!\s*(推荐意见|推荐理由|实施建议|[一二三四五六七八九十]+[、.]|[(][一二三四五六七八九十]+[)]|[0-9]+[.、]))",
        "",
        text,
    )
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def compact_text(text: str, max_len: int | None = None) -> str:
    """Collapse whitespace for CSV-friendly source snippets."""

    text = normalize_text(text)
    text = re.sub(r"\s+", " ", text).strip()
    if max_len is not None and len(text) > max_len:
        return text[: max_len - 1].rstrip() + "…"
    return text


def clean_pages(pages: Iterable[TextPage]) -> list[TextPage]:
    return [
        TextPage(page_number=page.page_number, text=clean_page_text(page.text))
        for page in pages
    ]

