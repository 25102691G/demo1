from __future__ import annotations

from pathlib import Path

from .baidu_doc_parser import BaiduDocParserClient, BaiduDocParserConfig
from .models import PDFDocumentLoadResult, TextPage
from .quality import assess_pdf_text_quality


class PDFExtractionError(RuntimeError):
    """PDF 文本提取失败。"""


def load_pdf_pages(pdf_path: str | Path) -> list[TextPage]:
    """读取 PDF 并返回按页文本。"""

    return load_pdf_document(pdf_path).pages


def load_pdf_document(
    pdf_path: str | Path,
    *,
    ocr_mode: str = "auto",
    ocr_provider: str = "baidu_doc_parser",
    baidu_api_key_env: str = "BAIDU_API_KEY",
    baidu_secret_key_env: str = "BAIDU_SECRET_KEY",
) -> PDFDocumentLoadResult:
    """读取 PDF，必要时自动切换到百度智能云文档解析。"""

    mode = _normalize_ocr_mode(ocr_mode)
    provider = _normalize_ocr_provider(ocr_provider)
    pages = _load_pdf_pages_with_pymupdf(pdf_path)
    quality = assess_pdf_text_quality(pages)

    if mode == "never" or (mode == "auto" and quality.status == "ok"):
        return PDFDocumentLoadResult(
            pages=pages,
            quality=quality,
            extraction_method="pymupdf",
            ocr_used=False,
            ocr_provider="",
        )

    if provider != "baidu_doc_parser":
        raise PDFExtractionError(f"不支持的 OCR provider: {ocr_provider}")

    try:
        ocr_pages = BaiduDocParserClient(
            BaiduDocParserConfig(
                api_key_env=baidu_api_key_env,
                secret_key_env=baidu_secret_key_env,
            )
        ).parse_pdf(pdf_path)
    except Exception as exc:
        if mode == "always":
            raise PDFExtractionError("百度智能云文档解析失败") from exc
        raise PDFExtractionError(
            f"{quality.reason or 'PDF 文本质量较差'}，且百度智能云文档解析失败"
        ) from exc

    ocr_quality = assess_pdf_text_quality(ocr_pages)
    if ocr_quality.status != "ok":
        raise PDFExtractionError(
            f"百度智能云文档解析结果质量仍然较差: {ocr_quality.reason}"
        )

    return PDFDocumentLoadResult(
        pages=ocr_pages,
        quality=ocr_quality,
        extraction_method="baidu_doc_parser",
        ocr_used=True,
        ocr_provider=provider,
    )


def _load_pdf_pages_with_pymupdf(pdf_path: str | Path) -> list[TextPage]:
    """使用 PyMuPDF 提取 PDF 原生文本。"""

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


def _normalize_ocr_mode(value: str) -> str:
    mode = value.strip().lower()
    if mode not in {"auto", "never", "always"}:
        raise PDFExtractionError(f"不支持的 OCR mode: {value}")
    return mode


def _normalize_ocr_provider(value: str) -> str:
    provider = value.strip().lower()
    if provider in {"baidu", "baidu_doc_parser", "baidu_document_parser"}:
        return "baidu_doc_parser"
    return provider
