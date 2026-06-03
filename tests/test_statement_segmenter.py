from __future__ import annotations

from guideline_skill.anchors import AnchorRegistry
from guideline_skill.segmenters.statement_segmenter import StatementSegmenter


def test_segments_by_recommendation_unit_anchors() -> None:
    segmenter = StatementSegmenter(AnchorRegistry())
    text = "推荐意见1：第一条。证据等级：2。\n推荐意见2：第二条。证据等级：1。"

    segments = segmenter.segment(text, "recommendation")

    assert len(segments) == 2
    assert segments[0].raw_text.startswith("推荐意见1：")
    assert segments[1].raw_text.startswith("推荐意见2：")


def test_does_not_segment_by_field_anchors() -> None:
    segmenter = StatementSegmenter(AnchorRegistry())
    text = "推荐意见1：第一条。证据等级：2。推荐强度：强。推荐理由：理由。"

    segments = segmenter.segment(text, "recommendation")

    assert len(segments) == 1
    assert "证据等级：2" in segments[0].raw_text
    assert "推荐强度：强" in segments[0].raw_text


def test_preserves_original_label() -> None:
    segmenter = StatementSegmenter(AnchorRegistry())

    segments = segmenter.segment("推荐意见12.1：建议随访。", "recommendation")

    assert segments[0].original_label == "推荐意见12.1："
