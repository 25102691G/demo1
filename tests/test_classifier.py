from __future__ import annotations

from guideline_skill.anchors import AnchorRegistry
from guideline_skill.classifier import GuidelineClassifier


def test_classifies_text_with_multiple_unit_and_field_anchors_as_structured() -> None:
    classifier = GuidelineClassifier(AnchorRegistry())
    text = (
        "推荐意见1：建议完善内镜检查。证据等级：2，推荐强度：强。\n"
        "推荐意见2：建议结合影像学检查。证据等级：2，推荐强度：强。"
    )

    result = classifier.classify(text)

    assert result.doc_type == "structured_guideline"
    assert result.primary_unit_anchor == "recommendation"
    assert result.unit_anchor_count == 2


def test_classifies_field_only_text_as_narrative() -> None:
    classifier = GuidelineClassifier(AnchorRegistry())

    result = classifier.classify("证据等级：2。推荐强度：强。推荐理由：样本质量较好。")

    assert result.doc_type == "narrative_guideline"
    assert result.primary_unit_anchor is None
    assert result.unit_anchor_count == 0


def test_classifies_plain_clinical_text_as_narrative() -> None:
    classifier = GuidelineClassifier(AnchorRegistry())

    result = classifier.classify("接诊时应询问症状持续时间、既往病史和近期用药情况。")

    assert result.doc_type == "narrative_guideline"
    assert result.total_score == 0
