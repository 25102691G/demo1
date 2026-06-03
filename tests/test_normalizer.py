from __future__ import annotations

import guideline_skill.normalizer as normalizer_module
from guideline_skill.normalizer import LLMNormalizer


class MockDeepSeekClient:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.calls: list[tuple[str, str]] = []

    def chat_json(self, system_prompt: str, user_prompt: str) -> dict[str, object]:
        self.calls.append((system_prompt, user_prompt))
        return self.payload


def test_normalizer_calls_deepseek_client() -> None:
    client = MockDeepSeekClient(
        {
            "evidence_quality_normalized": "moderate",
            "strength_normalized": "weak",
            "confidence": 0.91,
            "needs_human_review": False,
            "review_reasons": [],
        }
    )

    result = LLMNormalizer(client).normalize_statement_fields("1", "强", "recommendation")

    assert client.calls
    assert result["evidence_quality_normalized"] == "moderate"
    assert result["strength_normalized"] == "weak"


def test_normalizer_returns_valid_llm_json_values() -> None:
    client = MockDeepSeekClient(
        {
            "evidence_quality_normalized": "high",
            "strength_normalized": "strong",
            "confidence": 0.88,
            "needs_human_review": False,
            "review_reasons": [],
        }
    )

    result = LLMNormalizer(client).normalize_statement_fields("A级证据", "强推荐")

    assert result["evidence_quality_normalized"] == "high"
    assert result["strength_normalized"] == "strong"
    assert result["confidence"] == 0.88


def test_normalizer_falls_back_when_llm_returns_invalid_enums() -> None:
    client = MockDeepSeekClient(
        {
            "evidence_quality_normalized": "grade_a",
            "strength_normalized": "forceful",
            "confidence": 0.9,
            "needs_human_review": False,
            "review_reasons": [],
        }
    )

    result = LLMNormalizer(client).normalize_statement_fields("A级证据", "强推荐")

    assert result["evidence_quality_normalized"] == "unknown"
    assert result["strength_normalized"] == "unknown"
    assert result["needs_human_review"] is True
    assert result["review_reasons"][0].startswith("normalization_llm_failed:")


def test_normalizer_module_does_not_expose_fixed_rule_mapping_functions() -> None:
    assert not hasattr(normalizer_module, "normalize_evidence_quality")
    assert not hasattr(normalizer_module, "normalize_strength")
