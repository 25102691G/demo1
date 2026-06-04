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
            fields[field] = _extract_value(text, match, next_start, field=field)

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


def _extract_value(text: str, match: _FieldMatch, next_start: int, *, field: str | None = None) -> str | None:
    if _anchor_has_explicit_value(match.text):
        value = text[match.end:next_start]
    else:
        value = match.text
    cleaned = _clean_field_value(value, field)
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
    return _extract_value(surrounding_context, match, next_start, field="clinical_question")


def _infer_statement_type(original_label: str) -> str:
    if re.search(r"共识意见", original_label):
        return "consensus"
    if re.search(r"陈述|声明", original_label):
        return "statement"
    return "recommendation"


def _clean_value(value: str) -> str:
    text = re.sub(r"\s+", " ", value.replace("\u3000", " "))
    text = re.sub(r"^[\s,，;；。]+|[\s,，;；。]+$", "", text)
    text = re.sub(r"[（(]\s*$", "", text)
    return re.sub(r"^[\s,，;；。]+|[\s,，;；。]+$", "", text)


def _clean_field_value(value: str, field: str | None) -> str:
    text = _cut_noise(value)
    if field == "evidence_quality_raw":
        return _clean_evidence_quality_raw(text)
    if field == "strength_raw":
        return _clean_strength_raw(text)
    return _clean_value(text)


def _cut_noise(value: str) -> str:
    text = value
    noise_patterns = [
        r"——中华消化杂志.*",
        r"中华消化杂志\d{4}年.*",
        r"ChinJDig,.*",
        r"##\s*Page\s+\d+.*",
        r"表\s*\d+.*",
    ]
    for pattern in noise_patterns:
        text = re.split(pattern, text, maxsplit=1, flags=re.IGNORECASE | re.DOTALL)[0]
    return text


def _clean_evidence_quality_raw(value: str) -> str:
    text = _clean_value(value)
    match = re.search(r"\b([1-4])\b|([A-D])级?|([高中过]?低|极低|很低|中等)", text, flags=re.IGNORECASE)
    if match:
        return match.group(0).strip()
    return text


def _clean_strength_raw(value: str) -> str:
    text = _clean_value(value)
    if re.search(r"\bBPS\b|最佳临床实践", text, flags=re.IGNORECASE):
        return "BPS"
    if "强" in text:
        return "强"
    if "弱" in text:
        return "弱"
    match = re.search(r"\b(strong|weak)\b", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).lower()
    return text
