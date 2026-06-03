from __future__ import annotations

from guideline_skill.anchors import AnchorRegistry


def test_loads_unit_and_field_anchors_from_yaml() -> None:
    registry = AnchorRegistry()

    assert registry.get_unit_anchor("recommendation") is not None
    assert any(anchor["name"] == "evidence_quality" for anchor in registry.get_field_anchors())


def test_matches_recommendation_unit_anchor() -> None:
    registry = AnchorRegistry()

    matches = registry.find_matches("推荐意见1：建议结合临床表现进行综合判断。")

    assert any(match.name == "recommendation" and match.anchor_type == "unit" for match in matches)


def test_matches_evidence_quality_field_anchor() -> None:
    registry = AnchorRegistry()

    matches = registry.find_matches("证据等级：2，推荐强度：强。")

    assert any(match.name == "evidence_quality" and match.anchor_type == "field" for match in matches)


def test_field_anchor_is_not_counted_as_unit_anchor() -> None:
    registry = AnchorRegistry()

    score = registry.score("证据等级：2，推荐强度：强，推荐理由：证据充分。")

    assert score.field_anchor_count >= 1
    assert score.unit_anchor_count == 0
    assert not registry.is_structured("证据等级：2，推荐强度：强，推荐理由：证据充分。")


def test_choose_primary_unit_anchor_only_uses_unit_anchors() -> None:
    registry = AnchorRegistry()
    text = "证据等级：2。推荐意见1：建议检查。推荐意见2：建议随访。"

    assert registry.choose_primary_unit_anchor(text) == "recommendation"
