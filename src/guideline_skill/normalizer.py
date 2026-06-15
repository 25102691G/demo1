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


class EvidenceAndStrengthBatchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_quality_normalizations: dict[str, EvidenceQualityNormalized]
    strength_normalizations: dict[str, StrengthNormalized]

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

EVIDENCE_SCORE_BATCH_SYSTEM_PROMPT = """你需要对同一个医学指南 PDF 中出现的原始证据质量取值进行统一打分标准化。
只返回符合此结构的 JSON：{"normalizations": {"raw value": 0.2}}。
每个分数必须是 0 到 1 之间的数字；无法判断时返回 0.5。
必须只为输入 raw_values 中出现的原始值生成映射，不要新增原始值。

打分原则：
- 先识别该 PDF 内部的证据质量等级体系，再按从低到高映射到 0 到 1。
- 如果原始值是 1、2、3、4、5，通常分别对应 0.2、0.4、0.6、0.8、1.0。
- 如果原始值是 A、B、C、D，且 A 表示最高质量，通常分别对应 1.0、0.75、0.5、0.25。
- 如果原始值是高、中、低、极低，通常分别对应 1.0、0.67、0.33、0.1。
- 如果 PDF 明确说明数字或字母方向与上述不同，优先遵循 PDF 内部语义。"""

STRENGTH_SCORE_BATCH_SYSTEM_PROMPT = """你需要对同一个医学指南 PDF 中出现的原始推荐强度取值进行统一打分标准化。
只返回符合此结构的 JSON：{"normalizations": {"raw value": 1.0}}。
每个分数必须是 0 到 1 之间的数字；无法判断时返回 0.5。
必须只为输入 raw_values 中出现的原始值生成映射，不要新增原始值。

打分原则：
- 先识别该 PDF 内部的推荐强度等级体系，再按从弱到强映射到 0 到 1。
- 如果原始值是弱、中、强，通常分别对应 0.3、0.65、1.0。
- 如果原始值只有弱、强，通常分别对应 0.3、1.0。
- 如果原始值是 A、B、C，且 A 表示最强推荐，通常分别对应 1.0、0.65、0.3。
- 如果原始推荐强度中出现 BPS 或最佳临床实践，直接映射为 1.0。
- 共识声明如果没有强弱语义，无法转换为强弱分数时返回 0.5。
- 如果 PDF 明确说明等级方向与上述不同，优先遵循 PDF 内部语义。"""

EVIDENCE_AND_STRENGTH_SCORE_BATCH_SYSTEM_PROMPT = """你需要对同一个医学指南 PDF 中出现的原始证据质量和原始推荐强度取值进行统一打分标准化。
只返回符合此结构的 JSON：
{
  "evidence_quality_normalizations": {"raw evidence value": 0.2},
  "strength_normalizations": {"raw strength value": 1.0}
}
每个分数必须是 0 到 1 之间的数字；无法判断时返回 0.5。
必须只为输入 evidence_quality_raw_values 和 strength_raw_values 中出现的原始值生成映射，不要新增原始值。

输入 items 表示同一个 recommendation card 中成对出现的原始证据质量和原始推荐强度。
如果同一个 item 的 evidence_quality_raw 或 strength_raw 任意一个字段中出现 BPS 或最佳临床实践，则该 item 对应的两个标准化结果都输出为 1.0。

证据质量打分原则：
- 先识别该 PDF 内部的证据质量等级体系，再按从低到高映射到 0 到 1。
- 如果原始值是 1、2、3、4、5，通常分别对应 0.2、0.4、0.6、0.8、1.0。
- 如果原始值是 A、B、C、D，且 A 表示最高质量，通常分别对应 1.0、0.75、0.5、0.25。
- 如果原始值是高、中、低、极低，通常分别对应 1.0、0.67、0.33、0.1。
- 如果 PDF 明确说明数字或字母方向与上述不同，优先遵循 PDF 内部语义。

推荐强度打分原则：
- 先识别该 PDF 内部的推荐强度等级体系，再按从弱到强映射到 0 到 1。
- 如果原始值是弱、中、强，通常分别对应 0.3、0.65、1.0。
- 如果原始值只有弱、强，通常分别对应 0.3、1.0。
- 如果原始值是 A、B、C，且 A 表示最强推荐，通常分别对应 1.0、0.65、0.3。
- 如果原始推荐强度中出现 BPS 或最佳临床实践，直接映射为 1.0。
- 共识声明如果没有强弱语义，无法转换为强弱分数时返回 0.5。
- 如果 PDF 明确说明等级方向与上述不同，优先遵循 PDF 内部语义。"""

NORMALIZATION_SYSTEM_PROMPT = """你是医学指南字段标准化助手。请将单条推荐中的原始证据质量和原始推荐强度转换为 0 到 1 之间的数字分数。
只输出 JSON，不要输出解释。
输出结构必须为：
{
  "evidence_quality_normalized": 0.5,
  "strength_normalized": 0.5
}
无法判断时返回 0.5。
如果 evidence_quality_raw 或 strength_raw 任意一个字段中出现 BPS 或最佳临床实践，则两个字段都返回 1.0。"""


class LLMNormalizer:
    def __init__(self, deepseek_client: JsonChatClient) -> None:
        self.deepseek_client = deepseek_client
        self._disease_cache: dict[str, dict[str, Any]] = {}
        self._population_cache: dict[str, str] = {}
        self._evidence_quality_cache: dict[str, dict[str, EvidenceQualityNormalized]] = {}
        self._strength_cache: dict[str, dict[str, StrengthNormalized]] = {}

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

        return {
            "evidence_quality_normalized": result.evidence_quality_normalized
            if result.evidence_quality_normalized is not None
            else 0.5,
            "strength_normalized": result.strength_normalized if result.strength_normalized is not None else 0.5,
        }

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

    def normalize_evidence_and_strength_values(
        self,
        evidence_quality_raw_values: Sequence[str | None],
        strength_raw_values: Sequence[str | None],
        *,
        source_file: str | None = None,
    ) -> dict[str, Any]:
        evidence_values = _unique_text_values(evidence_quality_raw_values)
        strength_values = _unique_text_values(strength_raw_values)
        if not evidence_values and not strength_values:
            return {
                "evidence_quality_normalizations": {},
                "strength_normalizations": {},
                "review_reasons": [],
            }

        cache_key = _pdf_cache_key(source_file)
        evidence_cached = self._evidence_quality_cache.setdefault(cache_key, {})
        strength_cached = self._strength_cache.setdefault(cache_key, {})
        items = _paired_raw_values(evidence_quality_raw_values, strength_raw_values)

        for item in items:
            evidence_raw = item["evidence_quality_raw"]
            strength_raw = item["strength_raw"]
            if not _has_bps_value(evidence_raw, strength_raw):
                continue
            if evidence_raw:
                evidence_cached[evidence_raw] = 1.0
            if strength_raw:
                strength_cached[strength_raw] = 1.0

        missing_evidence_values = [value for value in evidence_values if value not in evidence_cached]
        missing_strength_values = [value for value in strength_values if value not in strength_cached]
        if not missing_evidence_values and not missing_strength_values:
            return {
                "evidence_quality_normalizations": {
                    value: evidence_cached.get(value, 0.5) for value in evidence_values
                },
                "strength_normalizations": {value: strength_cached.get(value, 0.5) for value in strength_values},
                "review_reasons": [],
            }

        user_prompt = json.dumps(
            {
                "source_file": source_file,
                "evidence_quality_raw_values": evidence_values,
                "strength_raw_values": strength_values,
                "items": items,
            },
            ensure_ascii=False,
        )
        try:
            payload = self.deepseek_client.chat_json(
                EVIDENCE_AND_STRENGTH_SCORE_BATCH_SYSTEM_PROMPT,
                user_prompt,
            )
            result = EvidenceAndStrengthBatchResult.model_validate(payload)
        except (Exception, ValidationError) as exc:
            for value in missing_evidence_values:
                evidence_cached[value] = 0.5
            for value in missing_strength_values:
                strength_cached[value] = 0.5
            return {
                "evidence_quality_normalizations": {
                    value: evidence_cached.get(value, 0.5) for value in evidence_values
                },
                "strength_normalizations": {value: strength_cached.get(value, 0.5) for value in strength_values},
                "review_reasons": [f"evidence_strength_batch_llm_failed: {exc}"],
            }

        for value in missing_evidence_values:
            evidence_cached[value] = result.evidence_quality_normalizations.get(value, 0.5)
        for value in missing_strength_values:
            strength_cached[value] = result.strength_normalizations.get(value, 0.5)
        return {
            "evidence_quality_normalizations": {
                value: evidence_cached.get(value, 0.5) for value in evidence_values
            },
            "strength_normalizations": {value: strength_cached.get(value, 0.5) for value in strength_values},
            "review_reasons": [],
        }

    def normalize_evidence_quality_values(
        self,
        raw_values: Sequence[str | None],
        *,
        source_file: str | None = None,
    ) -> dict[str, Any]:
        values = _unique_text_values(raw_values)
        if not values:
            return {"normalizations": {}, "review_reasons": []}
        """对证据质量值进行标准化。"""
        cache_key = _pdf_cache_key(source_file)
        cached = self._evidence_quality_cache.setdefault(cache_key, {})
        missing_values = [value for value in values if value not in cached]
        if not missing_values:
            return {
                "normalizations": {value: cached.get(value, 0.5) for value in values},
                "review_reasons": [],
            }

        user_prompt = json.dumps(
            {
                "source_file": source_file,
                "raw_values": values,
            },
            ensure_ascii=False,
        )
        try:
            payload = self.deepseek_client.chat_json(EVIDENCE_SCORE_BATCH_SYSTEM_PROMPT, user_prompt)
            result = EvidenceQualityBatchResult.model_validate(payload)
        except (Exception, ValidationError) as exc:
            for value in missing_values:
                cached[value] = 0.5
            return {
                "normalizations": {value: cached.get(value, 0.5) for value in values},
                "review_reasons": [f"evidence_quality_batch_llm_failed: {exc}"],
            }
        for value in values:
            cached[value] = result.normalizations.get(value, 0.5)
        return {
            "normalizations": {value: cached.get(value, 0.5) for value in values},
            "review_reasons": [],
        }

    def normalize_strength_values(
        self,
        raw_values: Sequence[str | None],
        *,
        source_file: str | None = None,
    ) -> dict[str, Any]:
        """对强度值进行标准化。"""
        values = _unique_text_values(raw_values)
        if not values:
            return {"normalizations": {}, "review_reasons": []}
        cache_key = _pdf_cache_key(source_file)
        cached = self._strength_cache.setdefault(cache_key, {})
        missing_values = [value for value in values if value not in cached]
        if not missing_values:
            return {
                "normalizations": {value: cached.get(value, 0.5) for value in values},
                "review_reasons": [],
            }

        user_prompt = json.dumps(
            {
                "source_file": source_file,
                "raw_values": values,
            },
            ensure_ascii=False,
        )
        try:
            payload = self.deepseek_client.chat_json(STRENGTH_SCORE_BATCH_SYSTEM_PROMPT, user_prompt)
            result = StrengthBatchResult.model_validate(payload)
        except (Exception, ValidationError) as exc:
            for value in missing_values:
                cached[value] = 0.5
            return {
                "normalizations": {value: cached.get(value, 0.5) for value in values},
                "review_reasons": [f"strength_batch_llm_failed: {exc}"],
            }
        for value in values:
            cached[value] = result.normalizations.get(value, 0.5)
        return {
            "normalizations": {value: cached.get(value, 0.5) for value in values},
            "review_reasons": [],
        }


def _pdf_cache_key(source_file: str | None) -> str:
    text = str(source_file or "").strip()
    return text or "__unknown_pdf__"


def _paired_raw_values(
    evidence_quality_raw_values: Sequence[str | None],
    strength_raw_values: Sequence[str | None],
) -> list[dict[str, str | None]]:
    items: list[dict[str, str | None]] = []
    total = max(len(evidence_quality_raw_values), len(strength_raw_values))
    for index in range(total):
        evidence_raw = _clean_optional_text(evidence_quality_raw_values[index]) if index < len(evidence_quality_raw_values) else None
        strength_raw = _clean_optional_text(strength_raw_values[index]) if index < len(strength_raw_values) else None
        items.append(
            {
                "evidence_quality_raw": evidence_raw,
                "strength_raw": strength_raw,
            }
        )
    return items


def _has_bps_value(*values: str | None) -> bool:
    return any(value and ("BPS" in value.upper() or "最佳临床实践" in value) for value in values)


def _failed_normalization(exc: Exception) -> dict[str, Any]:
    return {
        "evidence_quality_normalized": 0.5,
        "strength_normalized": 0.5,
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
