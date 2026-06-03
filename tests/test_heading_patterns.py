from __future__ import annotations

from guideline_skill.segmenters.heading_segmenter import HeadingPatternRegistry, HeadingSegmenter


def test_loads_heading_rules_from_yaml() -> None:
    registry = HeadingPatternRegistry()

    assert registry.match_heading("一、诊断") is not None


def test_matches_common_heading_patterns() -> None:
    registry = HeadingPatternRegistry()

    examples = {
        "一、诊断": "chinese_section",
        "1. 一般检查": "arabic_section",
        "1.1 实验室检查": "arabic_decimal_heading",
        "I.1 胃肠道结核的好发部位": "roman_decimal_heading",
        "II.6.1 化疗药物": "roman_decimal_heading",
        "（三）鉴别诊断": "chinese_parentheses",
        "（A）诊断标准": "uppercase_parentheses",
        "① 血常规": "circled_number",
    }

    for line, expected_name in examples.items():
        match = registry.match_heading(line)
        assert match is not None
        assert match.name == expected_name


def test_heading_segmenter_builds_section_path_from_rank() -> None:
    segmenter = HeadingSegmenter()
    text = """## Page 1
1. 一般检查
第一段。
1.1 实验室检查
第二段。
（A）诊断标准
第三段。
2. 治疗
第四段。
"""

    segments = segmenter.segment(text)

    assert [segment.section_path for segment in segments] == [
        ["一般检查"],
        ["一般检查", "实验室检查"],
        ["一般检查", "实验室检查", "诊断标准"],
        ["治疗"],
    ]
    assert segments[0].page_start == 1
    assert segments[1].heading_rank == 30
