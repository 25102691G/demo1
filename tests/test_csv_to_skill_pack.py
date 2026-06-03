from __future__ import annotations

from typing import Any, Mapping

import pytest

from guideline_skill.csv_to_skill_pack import (
    CsvRecommendation,
    LlmSemanticEnricher,
    RuleBasedSemanticEnricher,
    build_recommendation_card,
    validate_llm_enrichment_payload,
)


class FakeLlmClient:
    def __init__(self, payload: Mapping[str, Any] | Exception) -> None:
        self.payload = payload
        self.prompts: list[str] = []

    def complete_json(self, prompt: str) -> Mapping[str, Any]:
        self.prompts.append(prompt)
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


def test_llm_semantic_enricher_validates_and_returns_payload() -> None:
    recommendation = _recommendation()
    client = FakeLlmClient(
        {
            "clinical_stage": "诊断证据整合",
            "clinical_task": "结肠镜和多肠段活检",
            "population": "疑似克罗恩病患者",
            "condition": "当患者需要完善诊断证据时。",
            "required_inputs": ["症状与体征", "结肠镜报告"],
            "safety_notes": ["本条内容不能替代医生诊断。"],
        }
    )

    enriched = LlmSemanticEnricher(client=client).enrich(recommendation)

    assert enriched["clinical_task"] == "结肠镜和多肠段活检"
    assert enriched["required_inputs"] == ["症状与体征", "结肠镜报告"]
    assert "推荐正文" in client.prompts[0]


def test_llm_semantic_enricher_can_fallback_to_rule_based() -> None:
    recommendation = _recommendation()
    enricher = LlmSemanticEnricher(
        client=FakeLlmClient(RuntimeError("temporary failure")),
        fallback_enricher=RuleBasedSemanticEnricher(),
    )

    enriched = enricher.enrich(recommendation)

    assert enriched["clinical_stage"] == "诊断证据整合"
    assert "结肠镜报告" in enriched["required_inputs"]


def test_llm_semantic_enricher_fills_empty_safety_notes() -> None:
    recommendation = _recommendation()
    enriched = LlmSemanticEnricher(
        client=FakeLlmClient(
            {
                "clinical_stage": "\u8bca\u65ad\u8bc1\u636e\u6574\u5408",
                "clinical_task": "\u5185\u955c\u8bc1\u636e\u6838\u5bf9",
                "population": "\u7591\u4f3c\u514b\u7f57\u6069\u75c5\u60a3\u8005",
                "condition": "\u9700\u8981\u5b8c\u5584\u8bca\u65ad\u8bc1\u636e\u65f6",
                "required_inputs": ["\u5185\u955c\u68c0\u67e5"],
                "safety_notes": [],
            }
        )
    ).enrich(recommendation)

    assert enriched["clinical_task"] == "\u5185\u955c\u8bc1\u636e\u6838\u5bf9"
    assert enriched["required_inputs"] == ["\u5185\u955c\u68c0\u67e5"]
    assert enriched["safety_notes"]


def test_llm_semantic_enricher_fills_empty_condition_from_stage() -> None:
    recommendation = _recommendation()
    enriched = LlmSemanticEnricher(
        client=FakeLlmClient(
            {
                "clinical_stage": "\u836f\u7269\u4e0e\u8425\u517b\u6cbb\u7597\u9009\u62e9",
                "clinical_task": "\u751f\u7269\u5236\u5242\u8bf1\u5bfc\u7f13\u89e3",
                "population": "\u4e2d\u91cd\u5ea6\u6d3b\u52a8\u671f\u514b\u7f57\u6069\u75c5\u60a3\u8005",
                "condition": "",
                "required_inputs": ["\u6d3b\u52a8\u5ea6", "\u611f\u67d3\u98ce\u9669"],
                "safety_notes": ["\u9700\u7531\u533b\u751f\u8bc4\u4f30\u611f\u67d3\u98ce\u9669"],
            }
        )
    ).enrich(recommendation)

    assert enriched["condition"] == (
        "\u5f53\u533b\u751f\u5df2\u8003\u8651\u514b\u7f57\u6069\u75c5\u6cbb\u7597\uff0c"
        "\u5e76\u9700\u8981\u7ed3\u5408\u75c5\u60c5\u4e25\u91cd\u7a0b\u5ea6"
        "\u548c\u5b89\u5168\u524d\u63d0\u9009\u62e9\u65b9\u6848\u65f6\u3002"
    )


def test_llm_enrichment_accepts_delimited_string_lists() -> None:
    enriched = validate_llm_enrichment_payload(
        {
            "clinical_stage": "诊断证据整合",
            "clinical_task": "结肠镜和多肠段活检",
            "population": "疑似克罗恩病患者",
            "condition": "当患者需要完善诊断证据时。",
            "required_inputs": "症状与体征；结肠镜报告；病理结果",
            "safety_notes": "本条内容不能替代医生诊断；信息不足时不得输出最终诊断",
        }
    )

    assert enriched["required_inputs"] == ["症状与体征", "结肠镜报告", "病理结果"]
    assert enriched["safety_notes"] == ["本条内容不能替代医生诊断", "信息不足时不得输出最终诊断"]


def test_llm_enrichment_accepts_json_string_lists() -> None:
    enriched = validate_llm_enrichment_payload(
        {
            "clinical_stage": "\u8bca\u65ad\u8bc1\u636e\u6574\u5408",
            "clinical_task": "\u5185\u955c\u548c\u5f71\u50cf\u8bc1\u636e\u6838\u5bf9",
            "population": "\u7591\u4f3c\u514b\u7f57\u6069\u75c5\u60a3\u8005",
            "condition": "\u9700\u8981\u5b8c\u5584\u8bca\u65ad\u8bc1\u636e\u65f6",
            "required_inputs": (
                "[\"\u75c7\u72b6\u4e0e\u4f53\u5f81\", "
                "\"\u5185\u955c\u68c0\u67e5\", "
                "\"\u75c5\u7406\u7ed3\u679c\"]"
            ),
            "safety_notes": (
                "[\"\u4e0d\u80fd\u66ff\u4ee3\u533b\u751f\u8bca\u65ad\", "
                "\"\u4fe1\u606f\u4e0d\u8db3\u65f6\u9700\u8865\u5145\u8bc1\u636e\"]"
            ),
        }
    )

    assert enriched["required_inputs"] == [
        "\u75c7\u72b6\u4e0e\u4f53\u5f81",
        "\u5185\u955c\u68c0\u67e5",
        "\u75c5\u7406\u7ed3\u679c",
    ]
    assert enriched["safety_notes"] == [
        "\u4e0d\u80fd\u66ff\u4ee3\u533b\u751f\u8bca\u65ad",
        "\u4fe1\u606f\u4e0d\u8db3\u65f6\u9700\u8865\u5145\u8bc1\u636e",
    ]


def test_llm_enrichment_accepts_wrapped_object_lists() -> None:
    enriched = validate_llm_enrichment_payload(
        {
            "clinical_stage": "\u6cbb\u7597\u524d\u8bc4\u4f30",
            "clinical_task": "\u6cbb\u7597\u524d\u4fe1\u606f\u6838\u5bf9",
            "population": "\u786e\u8bca\u514b\u7f57\u6069\u75c5\u60a3\u8005",
            "condition": "\u51c6\u5907\u5236\u5b9a\u6cbb\u7597\u65b9\u6848\u65f6",
            "required_inputs": {
                "items": [
                    {"name": "\u6d3b\u52a8\u5ea6", "source": "\u4e34\u5e8a\u8bc4\u4f30"},
                    {"name": "\u611f\u67d3\u98ce\u9669"},
                ]
            },
            "safety_notes": {
                "1": "\u9700\u8bc4\u4f30\u7981\u5fcc\u8bc1",
                "2": "\u4e0d\u5f97\u8f93\u51fa\u6307\u5357\u5916\u4e8b\u5b9e",
            },
        }
    )

    assert enriched["required_inputs"] == [
        "\u6d3b\u52a8\u5ea6",
        "\u4e34\u5e8a\u8bc4\u4f30",
        "\u611f\u67d3\u98ce\u9669",
    ]
    assert enriched["safety_notes"] == [
        "\u9700\u8bc4\u4f30\u7981\u5fcc\u8bc1",
        "\u4e0d\u5f97\u8f93\u51fa\u6307\u5357\u5916\u4e8b\u5b9e",
    ]


def test_llm_enrichment_localizes_english_abbreviations() -> None:
    enriched = validate_llm_enrichment_payload(
        {
            "clinical_stage": "\u8bca\u65ad\u8bc1\u636e\u6574\u5408",
            "clinical_task": "\u5c0f\u80a0\u5f71\u50cf\u8bc4\u4f30",
            "population": "\u7591\u4f3c\u514b\u7f57\u6069\u75c5\u60a3\u8005",
            "condition": "\u9700\u8981\u8bc4\u4f30\u5c0f\u80a0\u75c5\u53d8\u65f6",
            "required_inputs": ["\u75c7\u72b6\u4e0e\u4f53\u5f81", "MRE"],
            "safety_notes": ["\u4e0d\u80fd\u66ff\u4ee3\u533b\u751f\u8bca\u65ad"],
        }
    )

    assert enriched["required_inputs"] == [
        "\u75c7\u72b6\u4e0e\u4f53\u5f81",
        "\u78c1\u5171\u632f\u5c0f\u80a0\u6210\u50cf\uff08MRE\uff09",
    ]


def test_llm_enrichment_localizes_drug_abbreviations() -> None:
    enriched = validate_llm_enrichment_payload(
        {
            "clinical_stage": "\u836f\u7269\u4e0e\u8425\u517b\u6cbb\u7597\u9009\u62e9",
            "clinical_task": "\u751f\u7269\u5236\u5242\u6cbb\u7597\u8bc4\u4f30",
            "population": "\u786e\u8bca\u514b\u7f57\u6069\u75c5\u60a3\u8005",
            "condition": "\u9700\u8981\u8bc4\u4f30\u6297TNF\u6cbb\u7597\u65f6",
            "required_inputs": ["IFX"],
            "safety_notes": ["\u9700\u7531\u533b\u751f\u8bc4\u4f30\u611f\u67d3\u98ce\u9669"],
        }
    )

    assert enriched["required_inputs"] == ["\u82f1\u592b\u5229\u897f\u5355\u6297\uff08IFX\uff09"]


def test_llm_enrichment_payload_rejects_missing_fields() -> None:
    with pytest.raises(ValueError, match="缺少字段"):
        validate_llm_enrichment_payload({"clinical_stage": "诊断证据整合"})


def test_llm_enrichment_builds_recommendation_card() -> None:
    recommendation = _recommendation()
    semantic_fields = LlmSemanticEnricher(
        client=FakeLlmClient(
            {
                "clinical_stage": "诊断证据整合",
                "clinical_task": "结肠镜和多肠段活检",
                "population": "疑似克罗恩病患者",
                "condition": "当患者需要完善诊断证据时。",
                "required_inputs": ["症状与体征", "结肠镜报告"],
                "safety_notes": ["本条内容不能替代医生诊断。"],
            }
        )
    ).enrich(recommendation)

    card = build_recommendation_card(
        recommendation,
        semantic_fields=semantic_fields,
        semantic_overrides={},
    )

    assert card.recommendation_id == "推荐意见3"
    assert card.clinical_task == "结肠镜和多肠段活检"
    assert card.evidence_level == "证据等级2"
    assert card.recommendation_strength == "强推荐"


def _recommendation() -> CsvRecommendation:
    return CsvRecommendation(
        recommendation_id="Recommendation:3",
        number="3",
        title="推荐意见3",
        text="结肠镜应作为常规检查方法用于CD诊断，建议尽量进入回肠末段。",
        evidence_grade="2",
        recommendation_strength="强",
        section="诊断及评估",
        page_start=4,
        page_end=4,
    )
