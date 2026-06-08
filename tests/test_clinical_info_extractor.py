from __future__ import annotations

import inspect

from guideline_skill.extractors.clinical_info_extractor import ClinicalInfoExtractor
from guideline_skill.segmenters.heading_segmenter import HeadingSegment


class MockDeepSeekClient:
    def __init__(self, payload: dict[str, object] | None = None, error: Exception | None = None) -> None:
        self.payload = payload or {}
        self.error = error
        self.calls: list[tuple[str, str]] = []

    def chat_json(self, system_prompt: str, user_prompt: str) -> dict[str, object]:
        self.calls.append((system_prompt, user_prompt))
        if self.error:
            raise self.error
        return self.payload


def test_clinical_info_extractor_calls_deepseek_client() -> None:
    client = MockDeepSeekClient(_valid_payload())
    extractor = ClinicalInfoExtractor(client)

    extractor.extract(_segment())

    assert client.calls
    assert "section_path" in client.calls[0][1]
    assert "raw_text" in client.calls[0][1]


def test_clinical_info_extractor_builds_unit_from_valid_llm_json() -> None:
    client = MockDeepSeekClient(_valid_payload())
    extractor = ClinicalInfoExtractor(client)

    unit = extractor.extract(_segment(), title="测试指南", source_file="test.pdf")

    assert unit.record_type == "recommendation_card"
    assert unit.guideline.doc_type == "narrative_guideline"
    assert unit.guideline.title == "测试指南"
    assert unit.card_id.startswith("clinical_info_unit_")
    assert unit.source_statement_id == unit.card_id
    assert unit.statement_text == _segment().raw_text
    assert unit.statement_type == "test_order"
    assert unit.clinical_stage == "诊断 / 实验室检查"
    assert unit.clinical_task == "diagnosis"
    assert unit.action == "建议完善血常规。"
    assert unit.confidence == 0.86
    assert unit.needs_human_review is False
    assert unit.evidence.evidence_quality_normalized == "unknown"
    assert unit.evidence.recommendation_strength_normalized == "unknown"


def test_clinical_info_extractor_falls_back_when_llm_call_fails() -> None:
    client = MockDeepSeekClient(error=RuntimeError("boom"))
    extractor = ClinicalInfoExtractor(client)

    unit = extractor.extract(_segment())

    assert unit.statement_type == "other"
    assert unit.clinical_task == "other"
    assert unit.statement_text == _segment().raw_text
    assert unit.confidence == 0.0
    assert unit.needs_human_review is True
    assert unit.review_reasons[0].startswith("clinical_info_llm_failed:")


def test_clinical_info_extractor_has_no_rule_extraction_logic() -> None:
    source = inspect.getsource(ClinicalInfoExtractor)

    assert "split_medical_list" not in source
    assert "import re" not in source
    assert "re." not in source
    assert "dose_pattern" not in source
    assert "unit_type ==" not in source


def _segment() -> HeadingSegment:
    return HeadingSegment(
        section_path=["诊断", "实验室检查"],
        title="实验室检查",
        raw_text="实验室检查\n疑似患者应完善血常规、炎症指标等检查。",
        start=10,
        end=40,
        page_start=1,
        page_end=1,
        heading_pattern_name="arabic_decimal_heading",
        heading_rank=30,
    )


def _valid_payload() -> dict[str, object]:
    return {
        "unit_type": "test_order",
        "clinical_topic": "diagnosis",
        "action": "建议完善血常规。",
        "condition": "疑似患者",
        "indication": [],
        "contraindication": [],
        "diagnostic_criteria": [],
        "differential_diagnosis": [],
        "drug": None,
        "dose": None,
        "route": None,
        "frequency": None,
        "duration": None,
        "confidence": 0.86,
        "needs_human_review": False,
        "review_reasons": [],
    }
