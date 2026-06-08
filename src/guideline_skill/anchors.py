from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass(frozen=True)
class AnchorMatch:
    name: str
    anchor_type: str
    pattern: str
    text: str
    start: int
    end: int
    weight: int
    metadata: dict[str, Any]


@dataclass(frozen=True)
class AnchorScore:
    total_score: int
    unit_score: int
    field_score: int
    unit_anchor_count: int
    field_anchor_count: int
    matched_counts: dict[str, int]
    matched_unit_counts: dict[str, int]
    matched_field_counts: dict[str, int]


class AnchorRegistry:
    """Load configurable guideline anchor rules and score text matches."""

    def __init__(self, rules_path: str | Path | None = None) -> None:
        self.rules_path = Path(rules_path) if rules_path is not None else _default_config_path("anchor_rules.yaml")
        payload = _load_yaml_mapping(self.rules_path)

        classification = payload.get("classification") or {}
        if not isinstance(classification, Mapping):
            raise ValueError("anchor_rules.yaml 的 classification 必须是映射对象。")

        # 以下两个值来源于configs\anchor_rules.yaml。
        self.structured_threshold = int(classification.get("structured_threshold", 15))
        self.min_unit_anchor_count = int(classification.get("min_unit_anchor_count", 2))
        
        self._unit_anchors = _normalize_anchor_rules(payload.get("unit_anchors"), "unit_anchors")
        self._field_anchors = _normalize_anchor_rules(payload.get("field_anchors"), "field_anchors")
        self._compiled_unit_anchors = _compile_anchor_rules(self._unit_anchors, anchor_type="unit")
        self._compiled_field_anchors = _compile_anchor_rules(self._field_anchors, anchor_type="field")
        self._unit_anchor_by_name = {str(anchor["name"]): anchor for anchor in self._unit_anchors}

    def find_matches(self, text: str) -> list[AnchorMatch]:
        matches: list[AnchorMatch] = []
        for anchor_type, compiled_rules in (
            ("unit", self._compiled_unit_anchors),
            ("field", self._compiled_field_anchors),
        ):
            for anchor, pattern_text, pattern in compiled_rules:
                metadata = {
                    key: value
                    for key, value in anchor.items()
                    if key not in {"patterns", "weight"}
                }
                for match in pattern.finditer(text):
                    matches.append(
                        AnchorMatch(
                            name=str(anchor["name"]),
                            anchor_type=anchor_type,
                            pattern=pattern_text,
                            text=match.group(0),
                            start=match.start(),
                            end=match.end(),
                            weight=int(anchor.get("weight", 1)),
                            metadata=dict(metadata),
                        )
                    )
        return sorted(matches, key=lambda item: (item.start, item.end, item.anchor_type, item.name))

    def score(self, text: str) -> AnchorScore:
        matches = self.find_matches(text)
        unit_matches = [match for match in matches if match.anchor_type == "unit"]
        field_matches = [match for match in matches if match.anchor_type == "field"]

        matched_counts = Counter(match.name for match in matches)
        matched_unit_counts = Counter(match.name for match in unit_matches)
        matched_field_counts = Counter(match.name for match in field_matches)
        unit_score = sum(match.weight for match in unit_matches)
        field_score = sum(match.weight for match in field_matches)

        return AnchorScore(
            total_score=unit_score + field_score,
            unit_score=unit_score,
            field_score=field_score,
            unit_anchor_count=len(unit_matches),
            field_anchor_count=len(field_matches),
            matched_counts=dict(matched_counts),
            matched_unit_counts=dict(matched_unit_counts),
            matched_field_counts=dict(matched_field_counts),
        )

    def is_structured(self, text: str) -> bool:
        score = self.score(text)
        return (
            score.total_score >= self.structured_threshold
            and score.unit_anchor_count >= self.min_unit_anchor_count
        )

    def choose_primary_unit_anchor(self, text: str) -> str | None:
        unit_matches = [match for match in self.find_matches(text) if match.anchor_type == "unit"]
        if not unit_matches:
            return None

        first_seen = {match.name: match.start for match in unit_matches}
        weighted_counts: dict[str, int] = {}
        plain_counts: dict[str, int] = {}
        for match in unit_matches:
            weighted_counts[match.name] = weighted_counts.get(match.name, 0) + match.weight
            plain_counts[match.name] = plain_counts.get(match.name, 0) + 1

        return min(
            weighted_counts,
            key=lambda name: (-weighted_counts[name], -plain_counts[name], first_seen[name], name),
        )

    def get_unit_anchor(self, name: str) -> dict[str, Any] | None:
        anchor = self._unit_anchor_by_name.get(name)
        return dict(anchor) if anchor is not None else None

    def get_field_anchors(self) -> list[dict[str, Any]]:
        return [dict(anchor) for anchor in self._field_anchors]


def _default_config_path(filename: str) -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / filename


def _load_yaml_mapping(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} 顶层必须是映射对象。")
    return payload


def _normalize_anchor_rules(value: Any, field_name: str) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"anchor_rules.yaml 的 {field_name} 必须是列表。")

    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(value, 1):
        if not isinstance(item, Mapping):
            raise ValueError(f"{field_name}[{index}] 必须是映射对象。")
        if not item.get("name"):
            raise ValueError(f"{field_name}[{index}] 缺少 name。")
        patterns = item.get("patterns")
        if not isinstance(patterns, list) or not all(isinstance(pattern, str) for pattern in patterns):
            raise ValueError(f"{field_name}[{index}].patterns 必须是字符串列表。")
        normalized.append(dict(item))
    return normalized


def _compile_anchor_rules(
    anchors: list[dict[str, Any]],
    *,
    anchor_type: str,
) -> list[tuple[dict[str, Any], str, re.Pattern[str]]]:
    compiled: list[tuple[dict[str, Any], str, re.Pattern[str]]] = []
    for anchor in anchors:
        for pattern_text in anchor["patterns"]:
            try:
                pattern = re.compile(pattern_text, re.IGNORECASE | re.MULTILINE)
            except re.error as exc:
                raise ValueError(
                    f"{anchor_type} anchor {anchor.get('name')!r} 的正则无效：{pattern_text!r}"
                ) from exc
            compiled.append((anchor, pattern_text, pattern))
    return compiled
