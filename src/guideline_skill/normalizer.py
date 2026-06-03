from __future__ import annotations

import json
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .schemas import EvidenceQualityNormalized, StrengthNormalized


class JsonChatClient(Protocol):
    def chat_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        ...


class NormalizationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_quality_normalized: EvidenceQualityNormalized | None = None
    strength_normalized: StrengthNormalized | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    needs_human_review: bool
    review_reasons: list[str] = Field(default_factory=list)


NORMALIZATION_SYSTEM_PROMPT = """你是医学指南字段归一化助手。请将原始证据等级和推荐强度归一化为标准枚举。
只输出 JSON，不要输出解释。
不要编造原文没有的信息。
不能确定时输出 unknown。

evidence_quality_normalized 只能从以下枚举中选择：
- high
- moderate
- low
- very_low
- unknown
- null

strength_normalized 只能从以下枚举中选择：
- strong
- weak
- best_practice_statement
- consensus_statement
- unknown
- null

归一化时参考语义：
- 高质量、高、1、A级证据等通常对应 high
- 中等质量、中等、2、B级证据等通常对应 moderate
- 低质量、低、3、C级证据等通常对应 low
- 极低质量、极低、4、D级证据等通常对应 very_low
- 强、强推荐、A级推荐等通常对应 strong
- 弱、弱推荐、B级推荐等通常对应 weak
- BPS、最佳临床实践通常对应 best_practice_statement
- 共识意见、推荐级别但无强弱时通常对应 consensus_statement"""


class LLMNormalizer:
    def __init__(self, deepseek_client: JsonChatClient) -> None:
        self.deepseek_client = deepseek_client

    def normalize_statement_fields(
        self,
        evidence_quality_raw: str | None,
        strength_raw: str | None,
        statement_type: str | None = None,
    ) -> dict[str, Any]:
        user_prompt = json.dumps(
            {
                "evidence_quality_raw": evidence_quality_raw,
                "strength_raw": strength_raw,
                "statement_type": statement_type,
            },
            ensure_ascii=False,
        )
        try:
            payload = self.deepseek_client.chat_json(NORMALIZATION_SYSTEM_PROMPT, user_prompt)
            result = NormalizationResult.model_validate(payload)
        except (Exception, ValidationError) as exc:
            return _failed_normalization(exc)

        return result.model_dump(mode="json")


def _failed_normalization(exc: Exception) -> dict[str, Any]:
    return {
        "evidence_quality_normalized": "unknown",
        "strength_normalized": "unknown",
        "confidence": 0.0,
        "needs_human_review": True,
        "review_reasons": [f"normalization_llm_failed: {exc}"],
    }
