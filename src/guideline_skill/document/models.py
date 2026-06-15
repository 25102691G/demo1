from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Optional

from pydantic import BaseModel, Field

try:
    from pydantic import ConfigDict
except ImportError:  # pragma: no cover - for pydantic v1 users
    ConfigDict = None  # type: ignore[assignment]


class GraphBaseModel(BaseModel):
    """Base pydantic model shared by graph extraction records."""

    if ConfigDict is not None:
        model_config = ConfigDict(extra="forbid")
    else:  # pragma: no cover - for pydantic v1 users
        class Config:
            extra = "forbid"


class TextPage(GraphBaseModel):
    page_number: int
    text: str


class PDFTextQualityReport(GraphBaseModel):
    status: str
    score: float
    total_chars: int
    chinese_ratio: float
    suspicious_ratio: float
    control_ratio: float
    reason: str = ""


class PDFDocumentLoadResult(GraphBaseModel):
    pages: list[TextPage]
    quality: PDFTextQualityReport
    extraction_method: str
    ocr_used: bool = False
    ocr_provider: str = ""


class SectionRecord(GraphBaseModel):
    section_id: str
    parent_section_id: str = ""
    title: str
    level: int
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    source_text: str = ""


class RecommendationRecord(GraphBaseModel):
    recommendation_id: str
    number: str
    title: str
    text: str
    evidence_grade: str = ""
    recommendation_strength: str = ""
    is_bps: bool = False
    reason: str = ""
    implementation_advice: str = ""
    section: str = ""
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    source_text: str = ""


class EntityMention(GraphBaseModel):
    node_id: str
    label: str
    name: str
    normalized_name: str
    aliases: list[str] = Field(default_factory=list)
    description: str = ""
    section: str = ""
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    source_text: str = ""
    confidence: float = 0.85


class NodeRecord(GraphBaseModel):
    node_id: str
    label: str
    name: str
    normalized_name: str = ""
    aliases: list[str] = Field(default_factory=list)
    description: str = ""
    section: str = ""
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    source_text: str = ""
    evidence_grade: str = ""
    recommendation_strength: str = ""
    confidence: float = 1.0


class EdgeRecord(GraphBaseModel):
    edge_id: str
    source_id: str
    target_id: str
    relation_type: str
    description: str = ""
    section: str = ""
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    source_text: str = ""
    evidence_grade: str = ""
    recommendation_strength: str = ""
    confidence: float = 0.75


class GraphBuildResult(GraphBaseModel):
    nodes: list[NodeRecord]
    edges: list[EdgeRecord]


def model_to_dict(model: BaseModel) -> dict[str, Any]:
    """Return a pydantic v1/v2 compatible dict."""

    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()  # pragma: no cover - for pydantic v1 users


def stable_token(value: str, max_len: int = 64) -> str:
    """Create a deterministic ASCII token for IDs."""

    normalized = value.strip().lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    normalized = normalized.strip("_")
    if normalized:
        return normalized[:max_len]
    digest = hashlib.md5(value.encode("utf-8")).hexdigest()[:12]
    return f"id_{digest}"


def make_node_id(label: str, normalized_name: str) -> str:
    return f"{label}:{stable_token(normalized_name)}"


def make_edge_id(source_id: str, relation_type: str, target_id: str, context: str = "") -> str:
    seed = f"{source_id}|{relation_type}|{target_id}|{context}"
    digest = hashlib.md5(seed.encode("utf-8")).hexdigest()[:16]
    return f"Edge:{digest}"


def aliases_to_json(aliases: list[str]) -> str:
    return json.dumps(aliases, ensure_ascii=False)
