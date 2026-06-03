from __future__ import annotations

from guideline_skill.anchors import AnchorRegistry
from guideline_skill.classifier import GuidelineClassifier
from guideline_skill.normalizer import LLMNormalizer
from guideline_skill.pipelines.structured_pipeline import StructuredGuidelinePipeline


class MockDeepSeekClient:
    def __init__(self) -> None:
        self.calls = 0

    def chat_json(self, system_prompt: str, user_prompt: str) -> dict[str, object]:
        self.calls += 1
        return {
            "evidence_quality_normalized": "moderate",
            "strength_normalized": "strong",
            "confidence": 0.92,
            "needs_human_review": False,
            "review_reasons": [],
        }


def test_structured_pipeline_extracts_statement_units_with_llm_normalization() -> None:
    anchor_registry = AnchorRegistry()
    classifier = GuidelineClassifier(anchor_registry)
    client = MockDeepSeekClient()
    pipeline = StructuredGuidelinePipeline(
        anchor_registry=anchor_registry,
        normalizer=LLMNormalizer(client),
    )
    text = (
        "临床问题1：如何诊断？\n"
        "推荐意见1：建议完善内镜检查。证据等级：2，推荐强度：强。实施建议：记录病变范围。\n"
        "推荐意见2：建议结合影像学检查。证据等级：2，推荐强度：强。推荐理由：可评估小肠。"
    )

    units = pipeline.run(text, classifier.classify(text), title="测试指南")

    assert client.calls == 2
    assert len(units) == 2
    assert units[0].record_type == "statement_unit"
    assert units[0].unit.statement_text == "建议完善内镜检查"
    assert units[0].unit.clinical_question == "如何诊断？"
    assert units[0].unit.evidence_quality_raw == "2"
    assert units[0].unit.strength_raw == "强"
    assert units[0].unit.evidence_quality_normalized == "moderate"
    assert units[0].unit.strength_normalized == "strong"
    assert units[0].unit.needs_human_review is False
