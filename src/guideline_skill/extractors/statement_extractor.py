from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Pattern

from guideline_skill.segmenters.statement_segmenter import StatementSegment


@dataclass(frozen=True)
class ExtractedStatementFields:
    original_label: str
    statement_type: str
    statement_text: str
    clinical_question: str | None
    evidence_quality_raw: str | None
    strength_raw: str | None
    consensus_level: str | None
    implementation_advice: str | None
    rationale: str | None


class StatementExtractor:
    def __init__(self, field_anchors: list[dict[str, Any]]) -> None:
        self.field_anchors = field_anchors
        self._compiled = _compile_field_anchors(field_anchors)

    def extract(
        self,
        segment: StatementSegment,
        surrounding_context: str | None = None,
    ) -> ExtractedStatementFields:
        text = segment.raw_text
        label_end = _label_end(text, segment.original_label)
        field_matches = _field_matches(text, self._compiled, start=label_end)
        first_field_start = field_matches[0].start if field_matches else len(text)

        fields: dict[str, str | None] = {
            "clinical_question": None,
            "evidence_quality_raw": None,
            "strength_raw": None,
            "consensus_level": None,
            "implementation_advice": None,
            "rationale": None,
        }

        for index, match in enumerate(field_matches):
            field = str(match.anchor.get("field") or "")
            if field not in fields or fields[field]:
                continue
            next_start = field_matches[index + 1].start if index + 1 < len(field_matches) else len(text)
            fields[field] = _extract_value(text, match, next_start)

        if not fields["clinical_question"] and surrounding_context:
            fields["clinical_question"] = _extract_context_clinical_question(
                surrounding_context,
                self._compiled,
            )

        return ExtractedStatementFields(
            original_label=segment.original_label,
            statement_type=_infer_statement_type(segment.original_label),
            statement_text=_clean_value(text[label_end:first_field_start]),
            clinical_question=fields["clinical_question"],
            evidence_quality_raw=fields["evidence_quality_raw"],
            strength_raw=fields["strength_raw"],
            consensus_level=fields["consensus_level"],
            implementation_advice=fields["implementation_advice"],
            rationale=fields["rationale"],
        )


@dataclass(frozen=True)
class _CompiledFieldAnchor:
    anchor: Mapping[str, Any]
    pattern_text: str
    pattern: Pattern[str]


@dataclass(frozen=True)
class _FieldMatch:
    anchor: Mapping[str, Any]
    pattern_text: str
    text: str
    start: int
    end: int


def _compile_field_anchors(field_anchors: list[dict[str, Any]]) -> list[_CompiledFieldAnchor]:
    compiled: list[_CompiledFieldAnchor] = []
    for anchor in field_anchors:
        for pattern_text in anchor.get("patterns", []):
            compiled.append(
                _CompiledFieldAnchor(
                    anchor=anchor,
                    pattern_text=str(pattern_text),
                    pattern=re.compile(str(pattern_text), re.IGNORECASE | re.MULTILINE),
                )
            )
    return compiled


def _field_matches(
    text: str,
    compiled: list[_CompiledFieldAnchor],
    *,
    start: int,
) -> list[_FieldMatch]:
    candidates: list[_FieldMatch] = []
    for item in compiled:
        for match in item.pattern.finditer(text, pos=start):
            candidates.append(
                _FieldMatch(
                    anchor=item.anchor,
                    pattern_text=item.pattern_text,
                    text=match.group(0),
                    start=match.start(),
                    end=match.end(),
                )
            )
    candidates.sort(key=lambda item: (item.start, item.end))

    matches: list[_FieldMatch] = []
    last_end = -1
    for match in candidates:
        if match.start < last_end:
            continue
        matches.append(match)
        last_end = match.end
    return matches


def _label_end(text: str, original_label: str) -> int:
    index = text.find(original_label)
    return index + len(original_label) if index >= 0 else 0


def _extract_value(text: str, match: _FieldMatch, next_start: int) -> str | None:
    if _anchor_has_explicit_value(match.text):
        value = text[match.end:next_start]
    else:
        value = match.text
    cleaned = _clean_value(value)
    return cleaned or None


def _anchor_has_explicit_value(anchor_text: str) -> bool:
    return anchor_text.rstrip().endswith((":", "："))


def _extract_context_clinical_question(
    surrounding_context: str,
    compiled: list[_CompiledFieldAnchor],
) -> str | None:
    clinical_matches = [
        match
        for match in _field_matches(surrounding_context, compiled, start=0)
        if match.anchor.get("field") == "clinical_question"
    ]
    if not clinical_matches:
        return None
    match = clinical_matches[-1]
    following_matches = [
        item
        for item in _field_matches(surrounding_context, compiled, start=match.end)
        if item.start > match.start
    ]
    next_start = following_matches[0].start if following_matches else len(surrounding_context)
    return _extract_value(surrounding_context, match, next_start)


def _infer_statement_type(original_label: str) -> str:
    if re.search(r"共识意见", original_label):
        return "consensus"
    if re.search(r"陈述|声明", original_label):
        return "statement"
    return "recommendation"


def _clean_value(value: str) -> str:
    text = re.sub(r"\s+", " ", value.replace("\u3000", " "))
    return text.strip(" \t\r\n,，;；。)")
