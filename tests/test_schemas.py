from __future__ import annotations

import json

from guideline_skill.schemas import (
    ClinicalInfoUnit,
    ClinicalInfoUnitBody,
    GuidelineMeta,
    SourceLocation,
    StatementUnit,
    StatementUnitBody,
)


def test_statement_unit_serializes_without_ascii_escaping() -> None:
    statement_unit = _statement_unit()

    payload = json.loads(statement_unit.to_json())

    assert payload["unit"]["statement_text"] == "建议完善检查。"
    assert "\\u5efa" not in statement_unit.to_json()


def test_statement_unit_record_type_is_fixed() -> None:
    statement_unit = _statement_unit()

    assert statement_unit.record_type == "statement_unit"


def test_clinical_info_unit_serializes_without_ascii_escaping() -> None:
    clinical_info_unit = _clinical_info_unit()

    payload = json.loads(clinical_info_unit.to_json())

    assert payload["unit"]["raw_text"] == "诊断标准包括临床表现和检查结果。"
    assert "\\u8bca" not in clinical_info_unit.to_json()


def test_clinical_info_unit_record_type_is_fixed() -> None:
    clinical_info_unit = _clinical_info_unit()

    assert clinical_info_unit.record_type == "clinical_info_unit"


def _statement_unit() -> StatementUnit:
    return StatementUnit(
        guideline_meta=GuidelineMeta(
            title="测试指南",
            source_file="test.pdf",
            doc_type="structured_guideline",
        ),
        unit=StatementUnitBody(
            id="statement_unit_001",
            original_label="推荐意见1：",
            statement_type="recommendation",
            statement_text="建议完善检查。",
            clinical_question=None,
            evidence_quality_raw="2",
            evidence_quality_normalized="moderate",
            strength_raw="强",
            strength_normalized="strong",
            consensus_level=None,
            implementation_advice=None,
            rationale=None,
            source_location=SourceLocation(page_start=1, page_end=1, section="诊断"),
            confidence=0.9,
            needs_human_review=False,
            review_reasons=[],
        ),
    )


def _clinical_info_unit() -> ClinicalInfoUnit:
    return ClinicalInfoUnit(
        guideline_meta=GuidelineMeta(
            title="测试指南",
            source_file="test.pdf",
            doc_type="narrative_guideline",
        ),
        unit=ClinicalInfoUnitBody(
            id="clinical_info_unit_001",
            section_path=["诊断"],
            title="诊断标准",
            raw_text="诊断标准包括临床表现和检查结果。",
            unit_type="diagnostic_criteria",
            clinical_topic="diagnosis",
            action=None,
            condition=None,
            indication=[],
            contraindication=[],
            diagnostic_criteria=["临床表现", "检查结果"],
            differential_diagnosis=[],
            drug=None,
            dose=None,
            route=None,
            frequency=None,
            duration=None,
            source_location=SourceLocation(page_start=1, page_end=1, section="诊断"),
            confidence=0.9,
            needs_human_review=False,
            review_reasons=[],
        ),
    )
