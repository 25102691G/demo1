from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .anchors import AnchorRegistry


GuidelineDocType = Literal["structured_guideline", "narrative_guideline"]


@dataclass(frozen=True)
class ClassificationResult:
    doc_type: GuidelineDocType
    total_score: int
    unit_score: int
    field_score: int
    unit_anchor_count: int
    field_anchor_count: int
    primary_unit_anchor: str | None
    matched_counts: dict[str, int]


class GuidelineClassifier:
    def __init__(self, anchor_registry: AnchorRegistry) -> None:
        self.anchor_registry = anchor_registry

    def classify(self, text: str) -> ClassificationResult:
        score = self.anchor_registry.score(text)
        is_structured = self.anchor_registry.is_structured(text)
        doc_type: GuidelineDocType = "structured_guideline" if is_structured else "narrative_guideline"

        return ClassificationResult(
            doc_type=doc_type,
            total_score=score.total_score,
            unit_score=score.unit_score,
            field_score=score.field_score,
            unit_anchor_count=score.unit_anchor_count,
            field_anchor_count=score.field_anchor_count,
            primary_unit_anchor=self.anchor_registry.choose_primary_unit_anchor(text) if is_structured else None,
            matched_counts=score.matched_counts,
        )
