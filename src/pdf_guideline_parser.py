from __future__ import annotations

import json
import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class ParsedPage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    page_number: int = Field(ge=1)
    text: str


class GuidelineSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    text: str


def parse_pdf_text(pdf_path: str | Path) -> list[ParsedPage]:
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() in {".txt", ".md"}:
        return _parse_text_file(path)

    try:
        import fitz  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - depends on local optional dependency
        raise RuntimeError("PyMuPDF/fitz is required to parse PDF files in this environment.") from exc

    pages: list[ParsedPage] = []
    with fitz.open(path) as document:
        for index, page in enumerate(document, 1):
            text = _normalize_text(page.get_text("text"))
            pages.append(ParsedPage(page_number=index, text=text))
    return pages


def split_sections(pages: list[ParsedPage]) -> list[GuidelineSection]:
    if not pages:
        return []

    section_chunks: list[dict[str, object]] = []
    current_title = "未分章节"
    current_page_start = pages[0].page_number
    current_lines: list[str] = []
    current_page_end = pages[0].page_number

    for page in pages:
        for line in page.text.splitlines():
            heading = detect_section_heading(line)
            if heading and current_lines:
                section_chunks.append(
                    {
                        "title": current_title,
                        "page_start": current_page_start,
                        "page_end": current_page_end,
                        "text": "\n".join(current_lines).strip(),
                    }
                )
                current_title = heading
                current_page_start = page.page_number
                current_lines = [line.strip()]
            else:
                if heading:
                    current_title = heading
                    current_page_start = page.page_number
                current_lines.append(line.strip())
            current_page_end = page.page_number

    if current_lines:
        section_chunks.append(
            {
                "title": current_title,
                "page_start": current_page_start,
                "page_end": current_page_end,
                "text": "\n".join(current_lines).strip(),
            }
        )

    return [
        GuidelineSection(
            title=str(chunk["title"]),
            page_start=int(chunk["page_start"]),
            page_end=int(chunk["page_end"]),
            text=str(chunk["text"]),
        )
        for chunk in section_chunks
        if str(chunk["text"]).strip()
    ]


def parse_guideline_sections(pdf_path: str | Path) -> list[GuidelineSection]:
    return split_sections(parse_pdf_text(pdf_path))


def save_sections_json(sections: list[GuidelineSection], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps([section.model_dump(mode="json") for section in sections], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def detect_section_heading(line: str) -> str | None:
    stripped = line.strip()
    if not stripped or len(stripped) > 80:
        return None
    patterns = (
        r"^第[一二三四五六七八九十百]+[章节篇]\s*.+",
        r"^[一二三四五六七八九十]+[、.．]\s*.+",
        r"^\d+(?:\.\d+)*[、.．]\s*[\u4e00-\u9fffA-Za-z].+",
        r"^#{1,6}\s*.+",
    )
    if any(re.match(pattern, stripped) for pattern in patterns):
        return re.sub(r"^#{1,6}\s*", "", stripped)
    common_headings = {"诊断", "治疗", "鉴别诊断", "疾病评估", "监测", "随访", "营养", "外科治疗"}
    if stripped in common_headings:
        return stripped
    return None


def _parse_text_file(path: Path) -> list[ParsedPage]:
    text = _normalize_text(path.read_text(encoding="utf-8"))
    parts = re.split(r"(?m)^##\s*Page\s+(\d+)\s*$", text)
    if len(parts) > 1:
        pages: list[ParsedPage] = []
        prefix = parts[0].strip()
        if prefix:
            pages.append(ParsedPage(page_number=1, text=prefix))
        for index in range(1, len(parts), 2):
            page_number = int(parts[index])
            page_text = parts[index + 1] if index + 1 < len(parts) else ""
            pages.append(ParsedPage(page_number=page_number, text=page_text.strip()))
        return pages
    return [ParsedPage(page_number=1, text=text)]


def _normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
