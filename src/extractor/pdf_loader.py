from __future__ import annotations

from pathlib import Path

from .models import TextPage


class PDFExtractionError(RuntimeError):
    """Raised when PDF text extraction cannot produce usable text."""


def load_pdf_pages(pdf_path: str | Path) -> list[TextPage]:
    """Load a PDF with PyMuPDF and return one TextPage per page."""

    path = Path(pdf_path)
    if not path.exists():
        raise PDFExtractionError(f"PDF 文件不存在: {path}")

    try:
        import fitz
    except ImportError as exc:
        raise PDFExtractionError(
            "缺少 PyMuPDF 依赖。请先运行: python -m pip install -r requirements.txt"
        ) from exc

    try:
        document = fitz.open(path)
    except Exception as exc:  # pragma: no cover - depends on external file
        raise PDFExtractionError(f"无法打开 PDF: {path}") from exc

    pages: list[TextPage] = []
    for index, page in enumerate(document, start=1):
        text = page.get_text("text")
        pages.append(TextPage(page_number=index, text=text or ""))

    if not any(page.text.strip() for page in pages):
        raise PDFExtractionError(
            "PDF 文本抽取结果为空。该文件可能是扫描件，需要先 OCR 后再运行本项目。"
        )

    return pages

