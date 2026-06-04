from __future__ import annotations

import json

from guideline_skill.schemas import (
    ClinicalInfoUnit,
    ClinicalInfoUnitBody,
    GuidelineMeta,
    SourceLocation,
    StatementEvidence,
    StatementUnit,
)


def test_statement_unit_serializes_recommendation_card_shape() -> None:
    statement_unit = _statement_unit()

    payload = json.loads(statement_unit.to_json())

    assert payload["record_type"] == "recommendation_card"
    assert "unit" not in payload
    assert payload["guideline"]["title"] == "Test guideline"
    assert payload["card_id"] == "statement_unit_001"
    assert payload["source_statement_id"] == "Recommendation 1:"
    assert payload["disease"] == "Crohn disease"
    assert payload["action"] == "Complete testing."
    assert payload["clinical_task"] == "testing"
    assert payload["do_not"] == ["Do not rely on one test."]
    assert payload["required_inputs"] == ["symptoms"]
    assert payload["supporting_features"] == ["abdominal pain"]
    assert payload["recommended_tests"] == ["endoscopy"]
    assert payload["evidence"]["recommendation_strength_raw"] == "strong"
    assert "\\u" not in statement_unit.to_json()


def test_statement_unit_record_type_is_fixed() -> None:
    statement_unit = _statement_unit()

    assert statement_unit.record_type == "recommendation_card"


def test_clinical_info_unit_serializes_without_ascii_escaping() -> None:
    clinical_info_unit = _clinical_info_unit()

    payload = json.loads(clinical_info_unit.to_json())

    assert payload["unit"]["raw_text"] == "diagnostic criteria"
    assert "\\u" not in clinical_info_unit.to_json()


def test_clinical_info_unit_record_type_is_fixed() -> None:
    clinical_info_unit = _clinical_info_unit()

    assert clinical_info_unit.record_type == "clinical_info_unit"


def _statement_unit() -> StatementUnit:
    return StatementUnit(
        guideline=GuidelineMeta(
            title="Test guideline",
            source_file="test.pdf",
            doc_type="structured_guideline",
        ),
        card_id="statement_unit_001",
        source_statement_id="Recommendation 1:",
        disease="Crohn disease",
        statement_type="recommendation",
        statement_text="Complete testing.",
        clinical_question=None,
        clinical_stage="diagnosis",
        clinical_task="testing",
        population="suspected Crohn disease",
        condition="when diagnosis is unclear",
        action="Complete testing.",
        do_not=["Do not rely on one test."],
        required_inputs=["symptoms"],
        supporting_features=["abdominal pain"],
        recommended_tests=["endoscopy"],
        evidence=StatementEvidence(
            evidence_quality_raw="2",
            evidence_quality_normalized="moderate",
            recommendation_strength_raw="strong",
            recommendation_strength_normalized="strong",
            consensus_level=None,
        ),
        implementation_advice=None,
        rationale=None,
        source_location=SourceLocation(page_start=1, page_end=1, section="diagnosis"),
        confidence=0.9,
        needs_human_review=False,
        review_reasons=[],
    )


def _clinical_info_unit() -> ClinicalInfoUnit:
    return ClinicalInfoUnit(
        guideline_meta=GuidelineMeta(
            title="Test guideline",
            source_file="test.pdf",
            doc_type="narrative_guideline",
        ),
        unit=ClinicalInfoUnitBody(
            id="clinical_info_unit_001",
            section_path=["diagnosis"],
            title="diagnostic criteria",
            raw_text="diagnostic criteria",
            unit_type="diagnostic_criteria",
            clinical_topic="diagnosis",
            action=None,
            condition=None,
            indication=[],
            contraindication=[],
            diagnostic_criteria=["clinical features", "test results"],
            differential_diagnosis=[],
            drug=None,
            dose=None,
            route=None,
            frequency=None,
            duration=None,
            source_location=SourceLocation(page_start=1, page_end=1, section="diagnosis"),
            confidence=0.9,
            needs_human_review=False,
            review_reasons=[],
        ),
    )
