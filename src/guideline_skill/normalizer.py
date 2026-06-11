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


class DiseaseExtractionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    disease: str


class RecommendationActionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: str


class RecommendationSemanticFieldsResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    action: str
    clinical_task: str | None = None
    population: str | None = None
    condition: str | None = None
    required_inputs: list[str] = Field(default_factory=list)


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


DISEASE_EXTRACTION_SYSTEM_PROMPT = """你需要从医学指南 PDF 文件名中提取疾病名称。
只返回符合此结构的 JSON：{"disease": "..."}。
使用文件名本身的语言。如果无法识别疾病名称，返回 {"disease": "unknown"}。"""

ACTION_SUMMARY_SYSTEM_PROMPT = """你是医学指南推荐卡字段抽取助手。
请只依据输入中的原文信息生成字段，不能补充、推断或泛化原文没有写出的信息。
输出必须是 JSON 对象，且只包含以下字段：
{
  "action": "中文推荐动作",
  "clinical_task": null,
  "population": null,
  "condition": null,
  "required_inputs": []
}

字段要求：
- action：给用户看的中文推荐动作；允许保留 PDF 原文中的英文缩写、药名、检查名，如 CD、CTE、MRE、MRI、TNF。
- clinical_task：这条推荐对应的具体医疗任务，需比 clinical_stage 更细；没有明确依据则为 null。
- population：适用患者人群；原文没有明确人群则为 null。
- condition：什么情况下应该考虑这条推荐；原文没有明确条件则为 null。
- required_inputs：使用这条推荐前需要先知道的病例信息；没有则为空数组。
所有中文字段必须使用中文表达；英文缩写可以原样保留。"""

ACTION_SUMMARY_SYSTEM_PROMPT = """你是医学指南推荐卡字段抽取助手。
请只依据输入中的原文信息生成字段，不能补充、推断或泛化原文没有写出的信息。
输出必须是 JSON 对象，且只包含以下字段：
{
  "action": "中文推荐动作",
  "clinical_task": null,
  "population": null,
  "condition": null,
  "required_inputs": []
}

字段要求：
- action：给用户看的中文推荐动作；允许保留 PDF 原文中的英文缩写、药名、检查名，如 CD、CTE、MRE、MRI、TNF。
- clinical_task：这条推荐对应的具体医疗任务，需比 clinical_stage 更细；没有明确依据则为 null。
- population：适用患者人群，只能取儿童、青少年、成人、老年人、孕妇、没有明确人群之一；原文没有明确人群则为 没有明确人群。
- condition：什么情况下应该考虑这条推荐；原文没有明确条件则为 null。
- required_inputs：使用这条推荐前需要先知道的病例信息；没有则为空数组。
所有中文字段必须使用中文表达；英文缩写可以原样保留。"""

ACTION_SUMMARY_SYSTEM_PROMPT_WITHOUT_POPULATION = """你是医学指南推荐卡字段抽取助手。
请只依据输入中的原文信息生成字段，不能补充、推断或泛化原文没有写出的信息。
输出必须是 JSON 对象，且只包含以下字段：
{
  "action": "中文推荐动作",
  "clinical_task": null,
  "condition": null,
  "required_inputs": []
}

字段要求：
- action：给用户看的中文推荐动作；允许保留 PDF 原文中的英文缩写、药名、检查名，如 CD、CTE、MRE、MRI、TNF。
- clinical_task：这条推荐对应的具体医疗任务，需比 clinical_stage 更细；没有明确依据则为 null。
- condition：什么情况下应该考虑这条推荐；原文没有明确条件则为 null。
- required_inputs：使用这条推荐前需要先知道的病例信息；没有则为空数组。
所有中文字段必须使用中文表达；英文缩写可以原样保留。"""

EVIDENCE_BATCH_SYSTEM_PROMPT = """你需要对医学指南中不同的原始证据质量取值进行归一化。
只返回符合此结构的 JSON：{"normalizations": {"raw value": "normalized value"}}。
每个归一化后的值必须是以下之一：high、moderate、low、very_low、unknown。
不要编造输入中不存在的原始值。"""

STRENGTH_BATCH_SYSTEM_PROMPT = """你需要对医学指南中不同的原始推荐强度取值进行归一化。
只返回符合此结构的 JSON：{"normalizations": {"raw value": "normalized value"}}。
每个归一化后的值必须是以下之一：strong、weak、best_practice_statement、consensus_statement、unknown。
不要编造输入中不存在的原始值。"""


class LLMNormalizer:
    def __init__(self, deepseek_client: JsonChatClient) -> None:
        self.deepseek_client = deepseek_client
        self._disease_cache: dict[str, dict[str, Any]] = {}
        self._population_cache: dict[str, str] = {}

    def normalize_statement_fields(
        self,
        evidence_quality_raw: str | None,
        strength_raw: str | None,
    ) -> dict[str, Any]:
        user_prompt = json.dumps(
            {
                "evidence_quality_raw": evidence_quality_raw,
                "strength_raw": strength_raw,
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
        """从文件名中提取疾病名称。"""
        cache_key = str(filename or "").strip()
        if cache_key in self._disease_cache:
            return dict(self._disease_cache[cache_key])

        user_prompt = json.dumps({"filename": filename}, ensure_ascii=False)
        try:
            payload = self.deepseek_client.chat_json(DISEASE_EXTRACTION_SYSTEM_PROMPT, user_prompt)
            result = DiseaseExtractionResult.model_validate(payload)
        except (Exception, ValidationError) as exc:
            disease_payload = {
                "disease": "unknown",
                "review_reasons": [f"disease_llm_failed: {exc}"],
            }
            self._disease_cache[cache_key] = disease_payload
            return dict(disease_payload)

        disease = result.disease.strip() or "unknown"
        disease_payload = {"disease": disease, "review_reasons": []}
        self._disease_cache[cache_key] = disease_payload
        return dict(disease_payload)

    def summarize_recommendation(
        self,
        *,
        raw_chunk_text: str,
        statement_text: str,
    ) -> dict[str, Any]:
        """从 raw_chunk_text 中提取推荐语义字段。"""
        cached_population = self._population_cache.get("population")
        system_prompt = (
            ACTION_SUMMARY_SYSTEM_PROMPT_WITHOUT_POPULATION
            if cached_population
            else ACTION_SUMMARY_SYSTEM_PROMPT
        )
        user_prompt = json.dumps(
            {
                "raw_chunk_text": raw_chunk_text,
            },
            ensure_ascii=False,
        )
        try:
            payload = self.deepseek_client.chat_json(system_prompt, user_prompt)
            result = RecommendationSemanticFieldsResult.model_validate(payload)
        except (Exception, ValidationError) as exc:
            return {
                "action": statement_text,
                "clinical_task": "",
                "population": cached_population,
                "condition": None,
                "required_inputs": [],
            }

        action = result.action.strip() or statement_text
        population = _clean_optional_text(result.population) or cached_population
        if population:
            self._population_cache["population"] = population
        return {
            "action": action,
            "clinical_task": _clean_text(result.clinical_task),
            "population": population,
            "condition": _clean_optional_text(result.condition),
            "required_inputs": _clean_text_list(result.required_inputs),
        }

    def normalize_evidence_quality_values(self, raw_values: Sequence[str | None]) -> dict[str, Any]:
        values = _unique_text_values(raw_values)
        if not values:
            return {"normalizations": {}, "review_reasons": []}
        """对证据质量值进行标准化。"""
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
        """对强度值进行标准化。"""
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
