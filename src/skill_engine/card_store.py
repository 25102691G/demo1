from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .utils import as_text_list, clean_text, shorten


class CardStoreError(ValueError):
    """Raised when recommendation cards cannot be loaded."""


@dataclass(frozen=True)
class CardStore:
    cards: list[dict[str, Any]]
    cards_by_id: dict[str, dict[str, Any]]

    @classmethod
    def from_jsonl(cls, path: Path) -> "CardStore":
        cards = load_cards_jsonl(path)
        cards_by_id: dict[str, dict[str, Any]] = {}
        for card in cards:
            card_id = clean_text(card.get("card_id"))
            if not card_id:
                raise CardStoreError(f"{path}: card missing card_id")
            if card_id in cards_by_id:
                raise CardStoreError(f"{path}: duplicate card_id {card_id!r}")
            cards_by_id[card_id] = card
        return cls(cards=cards, cards_by_id=cards_by_id)

    def select_for_subskill(self, subskill: dict[str, Any], *, limit: int | None = None) -> list[dict[str, Any]]:
        selection = subskill.get("card_selection") or {}
        ids = [
            *as_text_list(selection.get("required")),
            *as_text_list(selection.get("optional")),
        ]
        selected = [self.cards_by_id[card_id] for card_id in ids if card_id in self.cards_by_id]
        if limit is not None and limit > 0:
            selected = selected[:limit]
        return selected

    def select_by_filter(self, card_filter: Mapping[str, Any], *, limit: int | None = None) -> list[dict[str, Any]]:
        selected = [card for card in self.cards if _matches_card_filter(card, card_filter)]
        if limit is not None and limit > 0:
            selected = selected[:limit]
        return selected

    def get(self, card_id: str) -> dict[str, Any] | None:
        return self.cards_by_id.get(card_id)


def load_cards_jsonl(path: Path) -> list[dict[str, Any]]:
    cards_path = Path(path)
    if not cards_path.exists():
        raise CardStoreError(f"cards file does not exist: {cards_path}")
    cards: list[dict[str, Any]] = []
    with cards_path.open("r", encoding="utf-8-sig") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CardStoreError(
                    f"{cards_path}: line {line_no}: invalid JSON: {exc.msg}"
                ) from exc
            if not isinstance(payload, dict):
                raise CardStoreError(
                    f"{cards_path}: line {line_no}: expected object card"
                )
            cards.append(payload)
    return cards


def card_to_evidence_snippet(card: dict[str, Any]) -> str:
    label = clean_text(card.get("recommendation_label") or card.get("source_statement_id"))
    text = clean_text(card.get("action") or card.get("statement_text") or card.get("condition"))
    if label and text:
        return shorten(f"{label}: {text}", limit=260)
    return shorten(text or label or clean_text(card.get("card_id")), limit=260)


def card_to_recommendation(card: dict[str, Any]) -> str:
    return shorten(
        clean_text(card.get("action") or card.get("statement_text") or card.get("clinical_task")),
        limit=320,
    )


def _matches_card_filter(card: dict[str, Any], card_filter: Mapping[str, Any]) -> bool:
    if not card_filter:
        return True
    for field, expected in card_filter.items():
        if field in {"limit", "top_k"}:
            continue
        actual = clean_text(card.get(field))
        expected_values = as_text_list(expected)
        if not expected_values:
            continue
        if actual not in expected_values:
            return False
    return True
