from __future__ import annotations

from guideline_skill.anchors import AnchorRegistry
from guideline_skill.classifier import ClassificationResult
from guideline_skill.normalizer import LLMNormalizer
from guideline_skill.pipelines.structured_pipeline import StructuredGuidelinePipeline


class MockDeepSeekClient:
    def __init__(self) -> None:
        self.calls = 0

    def chat_json(self, system_prompt: str, user_prompt: str) -> dict[str, object]:
        self.calls += 1
        if "disease name" in system_prompt:
            return {"disease": "test disease"}
        if "推荐卡字段抽取助手" in system_prompt:
            return {
                "action": "summarized action",
                "clinical_task": "检查选择",
                "population": "疑似 CD 患者",
                "condition": "需要评估病变范围时",
                "do_not": ["不要单独依赖单一检查"],
                "required_inputs": ["临床表现"],
                "supporting_features": ["腹痛"],
                "recommended_tests": ["内镜检查"],
            }
        if "evidence quality values" in system_prompt:
            return {"normalizations": {}}
        if "recommendation strength values" in system_prompt:
            return {"normalizations": {}}
        return {
            "evidence_quality_normalized": "moderate",
            "strength_normalized": "strong",
            "confidence": 0.92,
            "needs_human_review": False,
            "review_reasons": [],
        }


def test_structured_pipeline_extracts_recommendation_cards_with_llm_enrichment() -> None:
    anchor_registry = AnchorRegistry()
    client = MockDeepSeekClient()
    pipeline = StructuredGuidelinePipeline(
        anchor_registry=anchor_registry,
        normalizer=LLMNormalizer(client),
    )
    text = (
        "Clinical question 1: how to diagnose?\n"
        "Recommendation 1: Complete endoscopy. Evidence quality: 2. Recommendation strength: strong. "
        "Implementation advice: Record lesion extent.\n"
        "Recommendation 2: Combine imaging. Evidence quality: 2. Recommendation strength: strong."
    )

    classification = ClassificationResult(
        doc_type="structured_guideline",
        total_score=10,
        unit_score=10,
        field_score=0,
        unit_anchor_count=2,
        field_anchor_count=0,
        primary_unit_anchor="english_recommendation",
        matched_counts={},
    )

    units = pipeline.run(text, classification, title="Test guideline", source_file="crohn.pdf")

    assert client.calls == 3
    assert len(units) == 2
    assert units[0].record_type == "recommendation_card"
    assert units[0].guideline.title == "Test guideline"
    assert units[0].card_id.startswith("statement_unit_001_")
    assert units[0].source_statement_id == "Recommendation 1:"
    assert units[0].disease == "test disease"
    assert units[0].clinical_stage == "unknown"
    assert units[0].clinical_task == "检查选择"
    assert units[0].population == "疑似 CD 患者"
    assert units[0].condition == "需要评估病变范围时"
    assert units[0].action == "summarized action"
    assert units[0].do_not == ["不要单独依赖单一检查"]
    assert units[0].required_inputs == ["临床表现"]
    assert units[0].supporting_features == ["腹痛"]
    assert units[0].recommended_tests == ["内镜检查"]
    assert units[0].statement_text.startswith("Complete endoscopy.")
    assert units[0].evidence.evidence_quality_raw is None
    assert units[0].evidence.evidence_quality_normalized == "unknown"
    assert units[0].evidence.recommendation_strength_raw is None
    assert units[0].evidence.recommendation_strength_normalized == "unknown"
    assert units[0].needs_human_review is True
