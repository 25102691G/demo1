from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]

from skill_engine.icd_features import build_mapped_icd_features
from skill_engine.llm_client import JsonChatClient
from skill_engine.utils import clean_text, normalize_key


ICD_EXTRACTION_SYSTEM_PROMPT_FROM_CASE = """你是一名专攻消化内科与表型提取的医学专家。
请根据患者临床文本，仅提取该患者相关的表型信息，包括症状、体征、实验室异常、影像学异常、内镜异常、病理异常等。

每个表型必须输出section_name，用于表示该表型对应的消化系统疾病分类。
section_name 必须从以下枚举中选择，不允许自由生成：

肠的其他疾病，
胆囊、胆道和胰腺疾患，
非感染性小肠炎和结肠炎，
腹膜疾病，
肝疾病，
口腔、涎腺和颌疾病，
阑尾疾病，
疝，
食管、胃和十二指肠疾病，
消化系统的其他疾病。


只输出提取得到的诊断内容并写为 json，格式如下：
{"diagnoses": [{"diagnosis": "原文中的表型短语", "section_name": "消化系统疾病分类"}]}。
诊断内容请使用中文书写。禁止输出其他任何无关信息。"""

ICD_EXTRACTION_SYSTEM_PROMPT_FROM_CARDS = """你是一名专攻消化内科与表型提取的医学专家。
请根据疾病指南片段，仅提取该片段涉及的表型信息，包括症状、体征、实验室异常、影像学异常、内镜异常、病理异常等。
请将表型信息分为阳性表型和阴性表型：
阳性表型指片段中确认存在、支持诊断、检查发现异常的表现。
阴性表型指片段中明确否认、未见、无、排除、不支持的表现。
不得根据常识或上下文自行推断，必须来自原文明确表述。

每个表型必须输出section_name，用于表示该表型对应的消化系统疾病分类。
section_name 必须从以下枚举中选择，不允许自由生成：

肠的其他疾病，
胆囊、胆道和胰腺疾患，
非感染性小肠炎和结肠炎，
腹膜疾病，
肝疾病，
口腔、涎腺和颌疾病，
阑尾疾病，
疝，
食管、胃和十二指肠疾病，
消化系统的其他疾病。


只输出提取得到的诊断内容并写为 json，格式如下：
{"positive_features": [{"diagnosis": "原文中的阳性表型短语", "section_name": "消化系统疾病分类"}], "negative_features": [{"diagnosis": "原文中的阴性表型短语", "section_name": "消化系统疾病分类"}]}。
没有对应内容时输出空数组。
诊断内容请使用中文书写。禁止输出其他任何无关信息。"""

DEFAULT_MODEL_PATH = ROOT / "data" / "qwen3-embedding-8b"
DEFAULT_ICD10_PATH = ROOT / "data" / "ICD10" / "ICD10.json"
DEFAULT_ICD10_EMBEDDINGS_PATH = ROOT / "data" / "ICD10" / "ICD10_embeddings.pt"
DEFAULT_ICD_SIMILARITY_THRESHOLD = 0.8
DEFAULT_ICD_TOP_K = 5
DEFAULT_ICD_QUERY_INSTRUCTION = (
    "Given a clinical diagnosis phrase in Chinese, retrieve the matching ICD-10 diagnosis name"
)

ICD_RECORD_FIELDS = (
    "chapter",
    "chapter_code_range",
    "chapter_name",
    "section_code_range",
    "section_name",
    "category_code",
    "category_name",
    "subcategory_code",
    "subcategory_name",
    "diagnosis_code",
    "diagnosis_name",
)


@dataclass(frozen=True)
class IcdResources:
    model: Any
    tokenizer: Any
    pooling_mode: str
    records: list[dict[str, str]]
    record_embeddings: Any
    record_keys: list[str]


class IcdExtractor:
    def __init__(
        self,
        resources: IcdResources,
        *,
        similarity_threshold: float = DEFAULT_ICD_SIMILARITY_THRESHOLD,
        batch_size: int = 30,
        max_length: int = 128,
    ) -> None:
        self.resources = resources
        self.similarity_threshold = similarity_threshold
        self.batch_size = batch_size
        self.max_length = max_length
        self._last_summary: dict[str, Any] = _empty_icd_summary()

    @classmethod
    def from_paths(
        cls,
        *,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        icd10_path: str | Path = DEFAULT_ICD10_PATH,
        icd10_embeddings_path: str | Path = DEFAULT_ICD10_EMBEDDINGS_PATH,
        similarity_threshold: float = DEFAULT_ICD_SIMILARITY_THRESHOLD,
        batch_size: int = 30,
        max_length: int = 128,
    ) -> IcdExtractor:
        torch = _load_torch()
        AutoTokenizer, AutoModel = _load_transformers()
        model_path = Path(model_path)
        pooling_mode = _embedding_pooling_mode(model_path)

        tokenizer_kwargs: dict[str, Any] = {"local_files_only": True}
        if pooling_mode == "last_token":
            tokenizer_kwargs["padding_side"] = "left"
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), **tokenizer_kwargs)
        model = AutoModel.from_pretrained(str(model_path), local_files_only=True)
        records = _load_icd10_records(icd10_path)
        record_embeddings = torch.load(str(icd10_embeddings_path), map_location="cpu")
        if int(record_embeddings.shape[0]) != len(records):
            raise ValueError(
                f"{icd10_embeddings_path}: embedding row count "
                f"{int(record_embeddings.shape[0])} does not match ICD10 record count {len(records)}"
            )
        resources = IcdResources(
            model=model,
            tokenizer=tokenizer,
            pooling_mode=pooling_mode,
            records=records,
            record_embeddings=record_embeddings,
            record_keys=[record["diagnosis_name"] for record in records],
        )
        return cls(
            resources,
            similarity_threshold=similarity_threshold,
            batch_size=batch_size,
            max_length=max_length,
        )

    def extract_icd_from_case(
        self,
        text: str,
        deepseek_client: JsonChatClient,
        prompt: str,
    ) -> dict[str, Any]:
        diagnoses = self.extract_diagnoses(text, deepseek_client, prompt)
        mappings = self.map_diagnoses_to_icd(diagnoses, source_type="case")
        return {"features": build_mapped_icd_features(mappings)}

    def extract_icd_from_cards(
        self,
        cards: Sequence[Mapping[str, Any]],
        deepseek_client: JsonChatClient,
        *,
        llm_workers: int = 1,
        prompt: str,
    ) -> dict[str, list[dict[str, Any]]]:
        diagnosis_source_groups = _extract_icd_diagnosis_sources_from_cards(
            cards,
            icd_extractor=self,
            deepseek_client=deepseek_client,
            llm_workers=llm_workers,
            prompt=prompt,
        )
        positive_features, positive_summary = self._map_card_diagnosis_sources(
            diagnosis_source_groups["positive_features"],
            source_type="cards_positive",
        )
        negative_features, negative_summary = self._map_card_diagnosis_sources(
            diagnosis_source_groups["negative_features"],
            source_type="cards_negative",
        )
        self._last_summary = {
            "source_type": "cards",
            "positive_features": positive_summary,
            "negative_features": negative_summary,
        }
        return {
            "positive_features": positive_features,
            "negative_features": negative_features,
        }

    def extract_diagnoses(
        self,
        text: str,
        deepseek_client: JsonChatClient,
        prompt: str,
    ) -> list[dict[str, str]]:
        if not str(prompt or "").strip():
            raise ValueError("extract_diagnoses requires a non-empty prompt")
        if not str(text or "").strip():
            return []
        user_prompt = json.dumps({"clinical_text": text}, ensure_ascii=False)
        payload = deepseek_client.chat_json(prompt, user_prompt)
        return _parse_diagnoses(payload)

    def extract_diagnosis_groups(
        self,
        text: str,
        deepseek_client: JsonChatClient,
        prompt: str,
    ) -> dict[str, list[dict[str, str]]]:
        if not str(prompt or "").strip():
            raise ValueError("extract_diagnosis_groups requires a non-empty prompt")
        if not str(text or "").strip():
            return _empty_diagnosis_groups()
        user_prompt = json.dumps({"clinical_text": text}, ensure_ascii=False)
        payload = deepseek_client.chat_json(prompt, user_prompt)
        return _parse_diagnosis_groups(payload)

    def _map_card_diagnosis_sources(
        self,
        diagnosis_sources: Sequence[Mapping[str, Any]],
        *,
        source_type: str,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        diagnoses = _dedupe_diagnosis_items(diagnosis_sources)
        mappings = self.map_diagnoses_to_icd(diagnoses, source_type=source_type)
        _attach_card_ids_to_mapping_results(mappings, diagnosis_sources)
        _attach_card_ids_to_icd_summary(self._last_summary, diagnosis_sources)
        icd_features = build_mapped_icd_features(mappings)
        _attach_card_ids_to_icd_features(icd_features, diagnosis_sources)
        return icd_features, self.get_last_summary()

    def map_diagnoses_to_icd(
        self,
        diagnoses: Sequence[Any],
        *,
        source_type: str = "unknown",
    ) -> list[dict[str, Any]]:
        diagnosis_items = _dedupe_diagnosis_items(diagnoses)
        if not diagnosis_items:
            self._last_summary = _empty_icd_summary(source_type=source_type)
            return []
        torch = _load_torch()
        device = _get_device(torch)
        resources = self.resources
        model = resources.model
        tokenizer = resources.tokenizer
        record_embeddings = resources.record_embeddings
        cleaned = _embedding_query_texts(diagnosis_items, pooling_mode=resources.pooling_mode)

        try:
            model = model.to(device)
            record_embeddings = record_embeddings.to(device)
        except Exception:
            device = torch.device("cpu")
            model = model.to(device)
            record_embeddings = record_embeddings.to(device)

        diagnosis_embeddings = []
        for start in range(0, len(cleaned), self.batch_size):
            batch = cleaned[start : start + self.batch_size]
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            batch_embeddings = _pool_embeddings(
                outputs.last_hidden_state,
                inputs["attention_mask"],
                pooling_mode=resources.pooling_mode,
                torch=torch,
            )
            batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1).float()
            diagnosis_embeddings.append(batch_embeddings)

        query_embeddings = torch.cat(diagnosis_embeddings, 0)
        topk = min(DEFAULT_ICD_TOP_K, len(resources.records))
        topk_indices, topk_values = topk_similarity(query_embeddings, record_embeddings, k=topk)
        topk_indices = topk_indices.cpu().numpy().tolist()
        topk_values = topk_values.cpu().float().numpy().tolist()

        results: list[dict[str, Any]] = []
        seen_codes: set[str] = set()
        for index, item in enumerate(diagnosis_items):
            diagnosis = item["diagnosis"]
            section_name = clean_text(item.get("section_name"))
            candidates = _icd_candidates(topk_indices[index], topk_values[index], resources.records)
            above_threshold = [
                candidate
                for candidate in candidates
                if candidate["similarity_score"] >= self.similarity_threshold
            ]
            selected = above_threshold[0] if above_threshold else None
            # 暂时关闭 section_name 强匹配逻辑。
            # selected = _select_section_name_candidate(above_threshold, section_name)
            best_candidate = candidates[0] if candidates else {}
            similarity_score = float(best_candidate.get("similarity_score") or 0.0)
            diagnosis_code = clean_text(best_candidate.get("diagnosis_code")) or None
            matched_section_name = clean_text(best_candidate.get("section_name"))

            if not above_threshold:
                results.append(
                    _mapping_result(
                        diagnosis=diagnosis,
                        section_name=section_name,
                        record=best_candidate,
                        matched_section_name=matched_section_name,
                        similarity_score=similarity_score,
                        status="low_similarity",
                        candidates=candidates,
                    )
                )
                continue

            # 暂时关闭 section_name 不匹配拦截。
            # if selected is None:
            #     results.append(
            #         _mapping_result(
            #             diagnosis=diagnosis,
            #             section_name=section_name,
            #             record=best_candidate,
            #             matched_section_name=matched_section_name,
            #             similarity_score=similarity_score,
            #             status="section_name_mismatch",
            #             candidates=candidates,
            #         )
            #     )
            #     continue

            diagnosis_code = clean_text(selected.get("diagnosis_code")) or None
            matched_section_name = clean_text(selected.get("section_name"))
            similarity_score = float(selected.get("similarity_score") or 0.0)

            if diagnosis_code in seen_codes:
                results.append(
                    _mapping_result(
                        diagnosis=diagnosis,
                        section_name=section_name,
                        record=selected,
                        matched_section_name=matched_section_name,
                        similarity_score=similarity_score,
                        status="duplicate",
                        candidates=candidates,
                    )
                )
                continue

            seen_codes.add(diagnosis_code)
            results.append(
                _mapping_result(
                    diagnosis=diagnosis,
                    section_name=section_name,
                    record=selected,
                    matched_section_name=matched_section_name,
                    similarity_score=similarity_score,
                    status="mapped",
                    candidates=candidates,
                )
            )
        self._last_summary = _build_icd_summary(
            source_type=source_type,
            input_count=len(diagnoses),
            deduped_count=len(diagnosis_items),
            results=results,
            similarity_threshold=self.similarity_threshold,
            top_k=topk,
        )
        return results

    def get_last_summary(self) -> dict[str, Any]:
        return json.loads(json.dumps(self._last_summary, ensure_ascii=False))

    def write_last_summary(self, path: str | Path) -> None:
        summary_path = Path(path)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(self._last_summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def _extract_icd_diagnosis_sources_from_cards(
    cards: Sequence[Mapping[str, Any]],
    *,
    icd_extractor: IcdExtractor,
    deepseek_client: JsonChatClient,
    llm_workers: int,
    prompt: str,
) -> dict[str, list[dict[str, str]]]:
    candidates = [
        (clean_text(card.get("card_id")), clean_text(card.get("raw_chunk_text")))
        for card in cards
    ]
    total = len(candidates)
    workers = max(1, int(llm_workers or 1))
    if workers <= 1 or len(candidates) <= 1:
        diagnosis_source_groups = _empty_diagnosis_groups()
        for index, (card_id, text) in enumerate(candidates, start=1):
            if card_id:
                extracted_groups = icd_extractor.extract_diagnosis_groups(
                    text,
                    deepseek_client,
                    prompt,
                )
                for group_name, diagnoses in extracted_groups.items():
                    diagnosis_source_groups[group_name].extend(
                        {
                            "diagnosis": diagnosis["diagnosis"],
                            "section_name": clean_text(diagnosis.get("section_name")),
                            "card_id": card_id,
                        }
                        for diagnosis in diagnoses
                    )
            _log_icd_cards_progress(index, total)
        return _dedupe_diagnosis_source_groups(diagnosis_source_groups)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_index = {
            executor.submit(
                icd_extractor.extract_diagnosis_groups,
                text,
                deepseek_client,
                prompt,
            ): index
            for index, (_card_id, text) in enumerate(candidates)
        }
        diagnosis_groups_by_card = [_empty_diagnosis_groups() for _ in candidates]
        completed = 0
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            diagnosis_groups_by_card[index] = future.result()
            completed += 1
            _log_icd_cards_progress(completed, total)
    diagnosis_source_groups = _empty_diagnosis_groups()
    for (card_id, _text), extracted_groups in zip(
        candidates,
        diagnosis_groups_by_card,
        strict=False,
    ):
        if not card_id:
            continue
        for group_name, diagnoses in extracted_groups.items():
            diagnosis_source_groups[group_name].extend(
                {
                    "diagnosis": diagnosis["diagnosis"],
                    "section_name": clean_text(diagnosis.get("section_name")),
                    "card_id": card_id,
                }
                for diagnosis in diagnoses
            )
    return _dedupe_diagnosis_source_groups(diagnosis_source_groups)


def _log_icd_cards_progress(completed: int, total: int) -> None:
    if total <= 0:
        return
    if completed % 10 == 0 or completed == total:
        print(f"ICD cards progress: {completed}/{total}", flush=True)


def _parse_diagnoses(payload: Mapping[str, Any]) -> list[dict[str, str]]:
    return _dedupe_diagnosis_items(_parse_diagnosis_items(payload.get("diagnoses", [])))


def _parse_diagnosis_groups(payload: Mapping[str, Any]) -> dict[str, list[dict[str, str]]]:
    groups = {
        "positive_features": _dedupe_diagnosis_items(
            _parse_diagnosis_items(payload.get("positive_features", []))
        ),
        "negative_features": _dedupe_diagnosis_items(
            _parse_diagnosis_items(payload.get("negative_features", []))
        ),
    }
    if not groups["positive_features"] and not groups["negative_features"]:
        groups["positive_features"] = _parse_diagnoses(payload)
    return groups


def _parse_diagnosis_items(values: Any) -> list[dict[str, str]]:
    if not isinstance(values, list):
        return []

    diagnoses: list[dict[str, str]] = []
    for item in values:
        if isinstance(item, str):
            diagnoses.append({"diagnosis": clean_text(item), "section_name": ""})
        elif isinstance(item, Mapping):
            value = (
                item.get("diagnosis")
                or item.get("Diagnosis")
                or item.get("diagnosis_name")
                or item.get("disease_name")
                or item.get("name")
            )
            if value is not None:
                diagnoses.append(
                    {
                        "diagnosis": clean_text(value),
                        "section_name": clean_text(
                            item.get("section_name") or item.get("SectionName")
                        ),
                    }
                )
    return diagnoses


def _empty_diagnosis_groups() -> dict[str, list[dict[str, str]]]:
    return {"positive_features": [], "negative_features": []}


def _dedupe_diagnosis_items(values: Sequence[Any]) -> list[dict[str, str]]:
    seen: set[tuple[str, str]] = set()
    deduped: list[dict[str, str]] = []
    for value in values:
        if isinstance(value, Mapping):
            diagnosis = clean_text(
                value.get("diagnosis")
                or value.get("Diagnosis")
                or value.get("diagnosis_name")
                or value.get("disease_name")
                or value.get("name")
                or value.get("original_diagnosis")
            )
            section_name = clean_text(value.get("section_name") or value.get("SectionName"))
        else:
            diagnosis = clean_text(value)
            section_name = ""
        key = (normalize_key(diagnosis), normalize_key(section_name))
        if not key[0] or key in seen:
            continue
        seen.add(key)
        deduped.append({"diagnosis": diagnosis, "section_name": section_name})
    return deduped


def _dedupe_diagnosis_sources(values: Iterable[Mapping[str, Any]]) -> list[dict[str, str]]:
    seen: set[tuple[str, str, str]] = set()
    deduped: list[dict[str, str]] = []
    for value in values:
        diagnosis = clean_text(value.get("diagnosis"))
        section_name = clean_text(value.get("section_name"))
        card_id = clean_text(value.get("card_id"))
        key = (normalize_key(diagnosis), normalize_key(section_name), card_id)
        if not key[0] or not key[2] or key in seen:
            continue
        seen.add(key)
        deduped.append(
            {"diagnosis": diagnosis, "section_name": section_name, "card_id": card_id}
        )
    return deduped


def _dedupe_diagnosis_source_groups(
    groups: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, list[dict[str, str]]]:
    return {
        "positive_features": _dedupe_diagnosis_sources(groups.get("positive_features") or []),
        "negative_features": _dedupe_diagnosis_sources(groups.get("negative_features") or []),
    }


def _attach_card_ids_to_icd_features(
    features: Sequence[dict[str, Any]],
    diagnosis_sources: Sequence[Mapping[str, Any]],
) -> None:
    sources_by_diagnosis = _card_ids_by_diagnosis(diagnosis_sources)
    for feature in features:
        key = _diagnosis_section_name_key(
            clean_text(feature.get("name")),
            clean_text(feature.get("section_name")),
        )
        card_ids = sources_by_diagnosis.get(key)
        if card_ids:
            feature["card_id"] = list(card_ids)


def _attach_card_ids_to_mapping_results(
    mappings: Sequence[dict[str, Any]],
    diagnosis_sources: Sequence[Mapping[str, Any]],
) -> None:
    sources_by_diagnosis = _card_ids_by_diagnosis(diagnosis_sources)
    for mapping in mappings:
        key = _diagnosis_section_name_key(
            clean_text(mapping.get("original_diagnosis")),
            clean_text(mapping.get("section_name")),
        )
        card_ids = sources_by_diagnosis.get(key)
        if card_ids:
            mapping["card_id"] = list(card_ids)


def _attach_card_ids_to_icd_summary(
    summary: dict[str, Any],
    diagnosis_sources: Sequence[Mapping[str, Any]],
) -> None:
    sources_by_diagnosis = _card_ids_by_diagnosis(diagnosis_sources)
    for item in summary.get("items") or []:
        if not isinstance(item, dict):
            continue
        key = _diagnosis_section_name_key(
            clean_text(item.get("diagnosis")),
            clean_text(item.get("section_name")),
        )
        card_ids = sources_by_diagnosis.get(key)
        if card_ids:
            item["card_id"] = list(card_ids)


def _card_ids_by_diagnosis(
    diagnosis_sources: Sequence[Mapping[str, Any]],
) -> dict[str, list[str]]:
    sources_by_diagnosis: dict[str, list[str]] = defaultdict(list)
    seen_by_diagnosis: dict[str, set[str]] = defaultdict(set)
    for source in diagnosis_sources:
        key = _diagnosis_section_name_key(
            clean_text(source.get("diagnosis")),
            clean_text(source.get("section_name")),
        )
        card_id = clean_text(source.get("card_id"))
        if not key or not card_id or card_id in seen_by_diagnosis[key]:
            continue
        seen_by_diagnosis[key].add(card_id)
        sources_by_diagnosis[key].append(card_id)
    return sources_by_diagnosis


def topk_similarity(query_embeddings: Any, record_embeddings: Any, *, k: int = 1) -> tuple[Any, Any]:
    torch = _load_torch()
    query_embeddings = torch.nn.functional.normalize(query_embeddings, p=2, dim=1).float()
    record_embeddings = torch.nn.functional.normalize(record_embeddings, p=2, dim=1).float()
    similarities = torch.matmul(query_embeddings, record_embeddings.T)
    topk_values, topk_indices = torch.topk(similarities, k, dim=1)
    return topk_indices, topk_values


def _embedding_query_texts(
    diagnosis_items: Sequence[Mapping[str, str]],
    *,
    pooling_mode: str,
) -> list[str]:
    texts = [item["diagnosis"] for item in diagnosis_items]
    if pooling_mode != "last_token":
        return texts
    return [f"Instruct: {DEFAULT_ICD_QUERY_INSTRUCTION}\nQuery:{text}" for text in texts]


def _pool_embeddings(
    last_hidden_state: Any,
    attention_mask: Any,
    *,
    pooling_mode: str,
    torch: Any,
) -> Any:
    if pooling_mode != "last_token":
        return last_hidden_state[:, 0, :]
    left_padding = bool((attention_mask[:, -1].sum() == attention_mask.shape[0]).item())
    if left_padding:
        return last_hidden_state[:, -1, :]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_state.shape[0]
    return last_hidden_state[torch.arange(batch_size, device=last_hidden_state.device), sequence_lengths]


def _embedding_pooling_mode(model_path: Path) -> str:
    pooling_config_path = model_path / "1_Pooling" / "config.json"
    if pooling_config_path.exists():
        with pooling_config_path.open("r", encoding="utf-8") as handle:
            pooling_config = json.load(handle)
        if pooling_config.get("pooling_mode_lasttoken") is True:
            return "last_token"
    config_path = model_path / "config.json"
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as handle:
            model_config = json.load(handle)
        if str(model_config.get("model_type") or "").lower() == "qwen3":
            return "last_token"
    return "cls"


def _icd_candidates(
    indices: Sequence[int],
    values: Sequence[float],
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for index, value in zip(indices, values, strict=False):
        candidate = {field: clean_text(records[index].get(field)) for field in ICD_RECORD_FIELDS}
        candidate["similarity_score"] = _normalize_similarity_score(value)
        candidates.append(candidate)
    return candidates


def _normalize_similarity_score(value: Any) -> float:
    return max(0.0, min(float(value or 0.0), 1.0))


def _select_section_name_candidate(
    candidates: Sequence[Mapping[str, Any]],
    section_name: str,
) -> Mapping[str, Any] | None:
    if not clean_text(section_name):
        return candidates[0] if candidates else None
    for candidate in candidates:
        if normalize_key(section_name) == normalize_key(clean_text(candidate.get("section_name"))):
            return candidate
    return None


def _diagnosis_section_name_key(diagnosis: str, section_name: str) -> str:
    diagnosis_key = normalize_key(diagnosis)
    if not diagnosis_key:
        return ""
    return f"{diagnosis_key}\0{normalize_key(section_name)}"


def _mapping_result(
    *,
    diagnosis: str,
    section_name: str,
    record: Mapping[str, Any],
    matched_section_name: str,
    similarity_score: float,
    status: str,
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {"original_diagnosis": diagnosis, "section_name": section_name}
    for field in ICD_RECORD_FIELDS:
        if field == "section_name":
            continue
        result[field] = clean_text(record.get(field)) or None
    result["matched_section_name"] = matched_section_name
    result["similarity_score"] = similarity_score
    result["status"] = status
    result["candidates"] = [dict(candidate) for candidate in candidates]
    return result


def _build_icd_summary(
    *,
    source_type: str,
    input_count: int,
    deduped_count: int,
    results: Sequence[Mapping[str, Any]],
    similarity_threshold: float,
    top_k: int,
) -> dict[str, Any]:
    counts = {
        "mapped_count": 0,
        "low_similarity_count": 0,
        "section_name_mismatch_count": 0,
        "duplicate_count": 0,
    }
    for result in results:
        status = clean_text(result.get("status"))
        key = f"{status}_count"
        if key in counts:
            counts[key] += 1
    summary_items = [_summary_item(result) for result in results]

    return {
        "source_type": source_type,
        "similarity_threshold": similarity_threshold,
        "top_k": top_k,
        "input_count": input_count,
        "deduped_count": deduped_count,
        **counts,
        "items": summary_items,
        "mapped_items": [item for item in summary_items if item.get("status") == "mapped"],
        "other_items": [item for item in summary_items if item.get("status") != "mapped"],
    }


def _summary_item(result: Mapping[str, Any]) -> dict[str, Any]:
    item = {
        "diagnosis": clean_text(result.get("original_diagnosis")),
        "section_name": clean_text(result.get("section_name")),
        "matched_section_name": clean_text(result.get("matched_section_name")),
        "similarity_score": result.get("similarity_score"),
        "status": clean_text(result.get("status")),
    }
    for field in ICD_RECORD_FIELDS:
        if field == "section_name":
            continue
        item[field] = clean_text(result.get(field))
    item["top_candidates"] = [dict(candidate) for candidate in result.get("candidates") or []]
    return item


def _empty_icd_summary(source_type: str = "unknown") -> dict[str, Any]:
    return {
        "source_type": source_type,
        "similarity_threshold": None,
        "top_k": DEFAULT_ICD_TOP_K,
        "input_count": 0,
        "deduped_count": 0,
        "mapped_count": 0,
        "low_similarity_count": 0,
        "section_name_mismatch_count": 0,
        "duplicate_count": 0,
        "items": [],
    }


def _load_icd10_records(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"{path}: ICD10 JSON must be an array")

    records: list[dict[str, str]] = []
    for item in data:
        if not isinstance(item, Mapping):
            continue
        record = {field: clean_text(item.get(field)) for field in ICD_RECORD_FIELDS}
        if not record["diagnosis_name"]:
            continue
        records.append(record)
    return records


def _get_device(torch: Any) -> Any:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_torch() -> Any:
    try:
        import torch
    except Exception as exc:
        raise RuntimeError(
            "ICD extraction requires the optional dependency 'torch'. "
            "Install it before calling ICD embedding mapping."
        ) from exc
    return torch


def _load_transformers() -> tuple[Any, Any]:
    try:
        from transformers import AutoModel, AutoTokenizer
    except Exception as exc:
        raise RuntimeError(
            "ICD extraction requires the optional dependency 'transformers'. "
            "Install it before loading ICD resources."
        ) from exc
    return AutoTokenizer, AutoModel
