from __future__ import annotations

import json
from typing import Any, Protocol, Sequence

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


class DiseaseExtractionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    disease: str


class RecommendationActionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: str


class RecommendationSemanticFieldsResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: str
    clinical_task: str | None = None
    population: str | None = None
    condition: str | None = None
    do_not: list[str] = Field(default_factory=list)
    required_inputs: list[str] = Field(default_factory=list)
    recommended_tests: list[str] = Field(default_factory=list)


class EvidenceQualityBatchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    normalizations: dict[str, EvidenceQualityNormalized]


class StrengthBatchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    normalizations: dict[str, StrengthNormalized]


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


DISEASE_EXTRACTION_SYSTEM_PROMPT = """You extract the disease name from a medical guideline PDF filename.
Return only JSON with this shape: {"disease": "..."}.
Use the language of the filename. If no disease can be identified, return {"disease": "unknown"}."""

ACTION_SUMMARY_SYSTEM_PROMPT = """你是医学指南推荐卡字段抽取助手。
请只依据输入中的原文信息生成字段，不能补充、推断或泛化原文没有写出的信息。
输出必须是 JSON 对象，且只包含以下字段：
{
  "action": "中文推荐动作",
  "clinical_task": null,
  "population": null,
  "condition": null,
  "do_not": [],
  "required_inputs": [],
  "recommended_tests": []
}

字段要求：
- action：给用户看的中文推荐动作；允许保留 PDF 原文中的英文缩写、药名、检查名，如 CD、CTE、MRE、MRI、TNF。
- clinical_task：这条推荐对应的具体医疗任务，需比 clinical_stage 更细；没有明确依据则为 null。
- population：适用患者人群；原文没有明确人群则为 null。
- condition：什么情况下应该考虑这条推荐；原文没有明确条件则为 null。
- do_not：原文明确不建议、不能单独依赖、需要避免的做法；没有则为空数组。
- required_inputs：使用这条推荐前需要先知道的病例信息；没有则为空数组。
- recommended_tests：原文推荐检查或建议补充的信息；没有则为空数组。
所有中文字段必须使用中文表达；英文缩写可以原样保留。"""

EVIDENCE_BATCH_SYSTEM_PROMPT = """You normalize distinct raw evidence quality values from medical guidelines.
Return only JSON with this shape: {"normalizations": {"raw value": "normalized value"}}.
Each normalized value must be one of: high, moderate, low, very_low, unknown.
Do not invent raw values that are absent from the input."""

STRENGTH_BATCH_SYSTEM_PROMPT = """You normalize distinct raw recommendation strength values from medical guidelines.
Return only JSON with this shape: {"normalizations": {"raw value": "normalized value"}}.
Each normalized value must be one of: strong, weak, best_practice_statement, consensus_statement, unknown.
Do not invent raw values that are absent from the input."""


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

    def extract_disease_from_filename(self, filename: str | None) -> dict[str, Any]:
        user_prompt = json.dumps({"filename": filename}, ensure_ascii=False)
        try:
            payload = self.deepseek_client.chat_json(DISEASE_EXTRACTION_SYSTEM_PROMPT, user_prompt)
            result = DiseaseExtractionResult.model_validate(payload)
        except (Exception, ValidationError) as exc:
            return {
                "disease": "unknown",
                "review_reasons": [f"disease_llm_failed: {exc}"],
            }

        disease = result.disease.strip() or "unknown"
        return {"disease": disease, "review_reasons": []}

    def summarize_recommendation_action(
        self,
        *,
        statement_text: str,
        implementation_advice: str | None,
        rationale: str | None = None,
        clinical_stage: str | None = None,
    ) -> dict[str, Any]:
        user_prompt = json.dumps(
            {
                "statement_text": statement_text,
                "implementation_advice": implementation_advice,
                "rationale": rationale,
                "clinical_stage": clinical_stage,
            },
            ensure_ascii=False,
        )
        try:
            payload = self.deepseek_client.chat_json(ACTION_SUMMARY_SYSTEM_PROMPT, user_prompt)
            result = RecommendationSemanticFieldsResult.model_validate(payload)
        except (Exception, ValidationError) as exc:
            return {
                "action": statement_text,
                "clinical_task": "",
                "population": None,
                "condition": None,
                "do_not": [],
                "required_inputs": [],
                "recommended_tests": [],
                "review_reasons": [f"action_llm_failed: {exc}"],
            }

        action = result.action.strip() or statement_text
        return {
            "action": action,
            "clinical_task": _clean_text(result.clinical_task),
            "population": _clean_optional_text(result.population),
            "condition": _clean_optional_text(result.condition),
            "do_not": _clean_text_list(result.do_not),
            "required_inputs": _clean_text_list(result.required_inputs),
            "recommended_tests": _clean_text_list(result.recommended_tests),
            "review_reasons": [],
        }

    def normalize_evidence_quality_values(self, raw_values: Sequence[str | None]) -> dict[str, Any]:
        values = _unique_text_values(raw_values)
        if not values:
            return {"normalizations": {}, "review_reasons": []}
        user_prompt = json.dumps({"raw_values": values}, ensure_ascii=False)
        try:
            payload = self.deepseek_client.chat_json(EVIDENCE_BATCH_SYSTEM_PROMPT, user_prompt)
            result = EvidenceQualityBatchResult.model_validate(payload)
        except (Exception, ValidationError) as exc:
            return {
                "normalizations": {value: "unknown" for value in values},
                "review_reasons": [f"evidence_quality_batch_llm_failed: {exc}"],
            }
        return {
            "normalizations": {value: result.normalizations.get(value, "unknown") for value in values},
            "review_reasons": [],
        }

    def normalize_strength_values(self, raw_values: Sequence[str | None]) -> dict[str, Any]:
        values = _unique_text_values(raw_values)
        if not values:
            return {"normalizations": {}, "review_reasons": []}
        user_prompt = json.dumps({"raw_values": values}, ensure_ascii=False)
        try:
            payload = self.deepseek_client.chat_json(STRENGTH_BATCH_SYSTEM_PROMPT, user_prompt)
            result = StrengthBatchResult.model_validate(payload)
        except (Exception, ValidationError) as exc:
            return {
                "normalizations": {value: "unknown" for value in values},
                "review_reasons": [f"strength_batch_llm_failed: {exc}"],
            }
        return {
            "normalizations": {value: result.normalizations.get(value, "unknown") for value in values},
            "review_reasons": [],
        }


def _failed_normalization(exc: Exception) -> dict[str, Any]:
    return {
        "evidence_quality_normalized": "unknown",
        "strength_normalized": "unknown",
        "confidence": 0.0,
        "needs_human_review": True,
        "review_reasons": [f"normalization_llm_failed: {exc}"],
    }


def _unique_text_values(values: Sequence[str | None]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _clean_optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    text = value.strip()
    return text or None


def _clean_text(value: str | None) -> str:
    if value is None:
        return ""
    return value.strip()


def _clean_text_list(values: Sequence[str]) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        cleaned.append(text)
    return cleaned
