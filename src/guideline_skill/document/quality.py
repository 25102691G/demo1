from __future__ import annotations

import unicodedata
from collections.abc import Sequence

from .models import PDFTextQualityReport, TextPage


def assess_pdf_text_quality(pages: Sequence[TextPage]) -> PDFTextQualityReport:
    """评估 PDF 原生文本提取结果是否可用。"""

    text = "\n".join(page.text or "" for page in pages)
    total_chars = len(text.strip())
    if total_chars == 0:
        return PDFTextQualityReport(
            status="poor",
            score=0.0,
            total_chars=0,
            chinese_ratio=0.0,
            suspicious_ratio=1.0,
            control_ratio=0.0,
            reason="PDF 文本提取结果为空",
        )

    chinese_chars = sum(1 for char in text if _is_chinese(char))
    suspicious_chars = sum(1 for char in text if _is_suspicious(char))
    control_chars = sum(1 for char in text if _is_control(char))
    chinese_ratio = chinese_chars / total_chars
    suspicious_ratio = suspicious_chars / total_chars
    control_ratio = control_chars / total_chars

    score = 1.0
    if total_chars < 200:
        score -= 0.35
    if chinese_ratio < 0.2:
        score -= 0.45
    elif chinese_ratio < 0.35:
        score -= 0.2
    if suspicious_ratio > 0.12:
        score -= 0.45
    elif suspicious_ratio > 0.06:
        score -= 0.25
    if control_ratio > 0.01:
        score -= 0.2

    score = max(0.0, min(1.0, score))
    status = "ok" if score >= 0.6 else "poor"
    reason = ""
    if status == "poor":
        reason = (
            f"PDF 文本质量较差: 总字符数={total_chars}, "
            f"中文比例={chinese_ratio:.2f}, 异常字符比例={suspicious_ratio:.2f}"
        )

    return PDFTextQualityReport(
        status=status,
        score=round(score, 4),
        total_chars=total_chars,
        chinese_ratio=round(chinese_ratio, 4),
        suspicious_ratio=round(suspicious_ratio, 4),
        control_ratio=round(control_ratio, 4),
        reason=reason,
    )


def _is_chinese(char: str) -> bool:
    return "\u4e00" <= char <= "\u9fff"


def _is_control(char: str) -> bool:
    if char in "\n\r\t":
        return False
    return unicodedata.category(char) in {"Cc", "Cf"}


def _is_suspicious(char: str) -> bool:
    if char.isspace() or _is_chinese(char):
        return False
    if char.isascii():
        return False
    category = unicodedata.category(char)
    if category in {"Cc", "Cf", "Co", "Cs"}:
        return True
    if category.startswith("S"):
        return True
    name = unicodedata.name(char, "")
    return "PRIVATE USE" in name or "REPLACEMENT" in name
