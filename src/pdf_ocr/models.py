from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TextPage:
    page_number: int
    text: str
