from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]

from skill_engine.hpo_features import build_mapped_hpo_features
from skill_engine.llm_client import JsonChatClient
from skill_engine.utils import clean_text, normalize_key


HPO_EXTRACTION_SYSTEM_PROMPT_FROM_CASE = """你是一名专攻消化内科与表型提取的医学专家。
请根据患者临床文本，仅提取该患者相关的表型信息，包括症状、体征、实验室异常、影像学异常、内镜异常、病理异常等。
对照人类表型本体论（HPO）数据库判定对应表型。

每个表型必须输出 body_site，用于表示该表型对应的解剖部位。
body_site 必须从以下枚举中选择，不允许自由生成：

未知, 不适用, 全身, 腹部,
口腔, 食管, 胃, 十二指肠,
小肠, 空肠, 回肠, 回盲部,
结肠, 直肠, 肛门肛周, 肠道, 胃肠道,
肝, 胆道, 胰腺, 脾,
腹膜, 肠系膜, 门静脉系统。

只输出提取得到的表型内容并写为json，格式如下：
{"phenotypes": [{"phenotype": "原文中的表型短语", "body_site": "枚举中的部位"}]}。
描述内容请使用中文书写。
禁止输出其他任何无关信息。"""

HPO_EXTRACTION_SYSTEM_PROMPT_FROM_CARDS = """你是一名专攻消化内科与表型提取的医学专家。
请根据疾病指南片段，仅提取该疾病相关的表型信息，包括症状、体征、实验室异常、影像学异常、内镜异常、病理异常等。
对照人类表型本体论（HPO）数据库判定对应表型。

每个表型必须输出 body_site，用于表示该表型对应的解剖部位。
body_site 必须从以下枚举中选择，不允许自由生成：

未知, 不适用, 全身, 腹部,
口腔, 食管, 胃, 十二指肠,
小肠, 空肠, 回肠, 回盲部,
结肠, 直肠, 肛门肛周, 肠道, 胃肠道,
肝, 胆道, 胰腺, 脾,
腹膜, 肠系膜, 门静脉系统。

只输出提取得到的表型内容并写为json，格式如下：
{"phenotypes": [{"phenotype": "原文中的表型短语", "body_site": "枚举中的部位"}]}。
描述内容请使用中文书写。
禁止输出其他任何无关信息。"""

DEFAULT_MODEL_PATH = ROOT / "data" / "qwen3-embedding-8b"
DEFAULT_DEFINITION2ID_PATH = ROOT / "data" / "ontology" / "hpo.json"
DEFAULT_DEFINITION_EMBEDDINGS_PATH = ROOT / "data" / "ontology" / "hpo_embeddings.pt"
DEFAULT_HPO_SIMILARITY_THRESHOLD = 0.8
DEFAULT_HPO_BODY_SITE_TOP_K = 5


@dataclass(frozen=True)
class HpoResources:
    model: Any
    tokenizer: Any
    definition2id: dict[str, str]
    definition_body_sites: dict[str, str]
    definition_embeddings: Any
    definition_keys: list[str]


class HpoExtractor:
    """LLM phenotype extraction plus BioLORD embedding mapping to HPO IDs.

    This module is intentionally standalone. Importing it does not import torch or
    transformers; those optional dependencies are loaded only by ``from_paths`` or
    mapping methods.
    """

    def __init__(
        self,
        resources: HpoResources,
        *,
        similarity_threshold: float = DEFAULT_HPO_SIMILARITY_THRESHOLD,
        batch_size: int = 30,
        max_length: int = 128,
    ) -> None:
        self.resources = resources
        self.similarity_threshold = similarity_threshold
        self.batch_size = batch_size
        self.max_length = max_length
        self._last_summary: dict[str, Any] = _empty_hpo_summary()

    @classmethod
    def from_paths(
        cls,
        *,
        model_path: str | Path,
        definition2id_path: str | Path,
        definition_embeddings_path: str | Path,
        similarity_threshold: float = DEFAULT_HPO_SIMILARITY_THRESHOLD,
        batch_size: int = 30,
        max_length: int = 128,
    ) -> HpoExtractor:
        torch = _load_torch()
        AutoTokenizer, AutoModel = _load_transformers()

        tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
        model = AutoModel.from_pretrained(str(model_path), local_files_only=True)
        definition2id, definition_body_sites = _load_definition2id(definition2id_path)
        definition_embeddings = torch.load(str(definition_embeddings_path), map_location="cpu")
        resources = HpoResources(
            model=model,
            tokenizer=tokenizer,
            definition2id=definition2id,
            definition_body_sites=definition_body_sites,
            definition_embeddings=definition_embeddings,
            definition_keys=list(definition2id.keys()),
        )
        return cls(
            resources,
            similarity_threshold=similarity_threshold,
            batch_size=batch_size,
            max_length=max_length,
        )

    def extract_hpo_from_case(
        self,
        text: str,
        deepseek_client: JsonChatClient,
        prompt: str,
    ) -> dict[str, Any]:
        # HPO 提取核心方法：提取病人case
        phenotypes = self.extract_phenotypes(text, deepseek_client, prompt)
        mappings = self.map_phenotypes_to_hpo(phenotypes, source_type="case")
        return {"symptoms": build_mapped_hpo_features(mappings)}

    def extract_hpo_from_cards(
        self,
        cards: Sequence[Mapping[str, Any]],
        deepseek_client: JsonChatClient,
        *,
        llm_workers: int = 1,
        prompt: str,
    ) -> list[dict[str, Any]]:
        # HPO 提取核心方法：提取recommendation cards
        phenotype_sources = _extract_hpo_phenotype_sources_from_cards(
            cards,
            hpo_extractor=self,
            deepseek_client=deepseek_client,
            llm_workers=llm_workers,
            prompt=prompt,
        )
        phenotypes = _dedupe_phenotype_items(phenotype_sources)
        mappings = self.map_phenotypes_to_hpo(phenotypes, source_type="cards")
        _attach_card_ids_to_mapping_results(mappings, phenotype_sources)
        _attach_card_ids_to_hpo_summary(self._last_summary, phenotype_sources)
        hpo_features = build_mapped_hpo_features(mappings)
        _attach_card_ids_to_hpo_features(hpo_features, phenotype_sources)
        return hpo_features

    def extract_phenotypes(
        self,
        text: str,
        deepseek_client: JsonChatClient,
        prompt: str,
    ) -> list[dict[str, str]]:
        if not str(prompt or "").strip():
            raise ValueError("extract_phenotypes requires a non-empty prompt")
        if not str(text or "").strip():
            return []
        user_prompt = json.dumps({"clinical_text": text}, ensure_ascii=False)
        payload = deepseek_client.chat_json(prompt, user_prompt)
        return _parse_phenotypes(payload)

    def map_phenotypes_to_hpo(
        self,
        phenotypes: Sequence[Any],
        *,
        source_type: str = "unknown",
    ) -> list[dict[str, Any]]:
        phenotype_items = _dedupe_phenotype_items(phenotypes)
        if not phenotype_items:
            self._last_summary = _empty_hpo_summary(source_type=source_type)
            return []
        cleaned = [item["phenotype"] for item in phenotype_items]

        torch = _load_torch()
        device = _get_device(torch)
        resources = self.resources
        model = resources.model
        tokenizer = resources.tokenizer
        definition_embeddings = resources.definition_embeddings

        try:
            model = model.to(device)
            definition_embeddings = definition_embeddings.to(device)
        except Exception:
            device = torch.device("cpu")
            model = model.to(device)
            definition_embeddings = definition_embeddings.to(device)

        phenotype_embeddings = []
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
            phenotype_embeddings.append(outputs.last_hidden_state[:, 0, :])

        query_embeddings = torch.cat(phenotype_embeddings, 0)
        topk = min(DEFAULT_HPO_BODY_SITE_TOP_K, len(resources.definition_keys))
        topk_indices, topk_values = topk_similarity(query_embeddings, definition_embeddings, k=topk)
        topk_indices = topk_indices.cpu().numpy().tolist()
        topk_values = topk_values.cpu().numpy().tolist()

        definition_values = list(resources.definition2id.values())
        results: list[dict[str, Any]] = []
        seen_hpo_codes: set[str] = set()
        for index, item in enumerate(phenotype_items):
            phenotype = item["phenotype"]
            body_site = clean_text(item.get("body_site"))
            candidates = _hpo_candidates(
                topk_indices[index],
                topk_values[index],
                definition_values,
                resources.definition_keys,
                resources.definition_body_sites,
            )
            above_threshold = [
                candidate
                for candidate in candidates
                if candidate["similarity_score"] >= self.similarity_threshold
            ]
            selected = _select_body_site_candidate(above_threshold, body_site)
            best_candidate = candidates[0] if candidates else {}
            similarity_score = float(best_candidate.get("similarity_score") or 0.0)
            hpo_code = clean_text(best_candidate.get("hpo_code")) or None
            hpo_term = clean_text(best_candidate.get("hpo_term")) or None
            matched_body_site = clean_text(best_candidate.get("matched_body_site"))

            if not above_threshold:
                results.append(
                    _mapping_result(
                        phenotype=phenotype,
                        body_site=body_site,
                        hpo_code=None,
                        hpo_term=None,
                        matched_body_site=matched_body_site,
                        similarity_score=similarity_score,
                        status="low_similarity",
                        candidates=candidates,
                    )
                )
                continue

            if selected is None:
                results.append(
                    _mapping_result(
                        phenotype=phenotype,
                        body_site=body_site,
                        hpo_code=hpo_code,
                        hpo_term=hpo_term,
                        matched_body_site=matched_body_site,
                        similarity_score=similarity_score,
                        status="body_site_mismatch",
                        candidates=candidates,
                    )
                )
                continue

            hpo_code = clean_text(selected.get("hpo_code")) or None
            hpo_term = clean_text(selected.get("hpo_term")) or None
            matched_body_site = clean_text(selected.get("matched_body_site"))
            similarity_score = float(selected.get("similarity_score") or 0.0)

            if hpo_code in seen_hpo_codes:
                results.append(
                    _mapping_result(
                        phenotype=phenotype,
                        body_site=body_site,
                        hpo_code=hpo_code,
                        hpo_term=hpo_term,
                        matched_body_site=matched_body_site,
                        similarity_score=similarity_score,
                        status="duplicate",
                        candidates=candidates,
                    )
                )
                continue

            seen_hpo_codes.add(hpo_code)
            results.append(
                _mapping_result(
                    phenotype=phenotype,
                    body_site=body_site,
                    hpo_code=hpo_code,
                    hpo_term=hpo_term,
                    matched_body_site=matched_body_site,
                    similarity_score=similarity_score,
                    status="mapped",
                    candidates=candidates,
                )
            )
        self._last_summary = _build_hpo_summary(
            source_type=source_type,
            input_count=len(phenotypes),
            deduped_count=len(phenotype_items),
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


def _extract_hpo_phenotype_sources_from_cards(
    cards: Sequence[Mapping[str, Any]],
    *,
    hpo_extractor: HpoExtractor,
    deepseek_client: JsonChatClient,
    llm_workers: int,
    prompt: str,
) -> list[dict[str, str]]:
    candidates = [
        (clean_text(card.get("card_id")), clean_text(card.get("raw_chunk_text")))
        for card in cards
    ]
    total = len(candidates)
    workers = max(1, int(llm_workers or 1))
    if workers <= 1 or len(candidates) <= 1:
        phenotype_sources: list[dict[str, str]] = []
        for index, (card_id, text) in enumerate(candidates, start=1):
            if card_id:
                phenotype_sources.extend(
                    {
                        "phenotype": phenotype["phenotype"],
                        "body_site": clean_text(phenotype.get("body_site")),
                        "card_id": card_id,
                    }
                    for phenotype in hpo_extractor.extract_phenotypes(
                        text,
                        deepseek_client,
                        prompt,
                    )
                )
            _log_hpo_cards_progress(index, total)
        return _dedupe_phenotype_sources(phenotype_sources)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_index = {
            executor.submit(
                hpo_extractor.extract_phenotypes,
                text,
                deepseek_client,
                prompt,
            ): index
            for index, (_card_id, text) in enumerate(candidates)
        }
        phenotype_groups: list[list[dict[str, str]]] = [[] for _ in candidates]
        completed = 0
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            phenotype_groups[index] = future.result()
            completed += 1
            _log_hpo_cards_progress(completed, total)
    return _dedupe_phenotype_sources(
        {
            "phenotype": phenotype["phenotype"],
            "body_site": clean_text(phenotype.get("body_site")),
            "card_id": card_id,
        }
        for (card_id, _text), group in zip(candidates, phenotype_groups, strict=False)
        if card_id
        for phenotype in group
    )


def _log_hpo_cards_progress(completed: int, total: int) -> None:
    if total <= 0:
        return
    if completed % 10 == 0 or completed == total:
        print(f"HPO cards progress: {completed}/{total}", flush=True)


def _dedupe_phenotype_sources(values: Iterable[Mapping[str, Any]]) -> list[dict[str, str]]:
    seen: set[tuple[str, str, str]] = set()
    deduped: list[dict[str, str]] = []
    for value in values:
        phenotype = clean_text(value.get("phenotype"))
        body_site = clean_text(value.get("body_site"))
        card_id = clean_text(value.get("card_id"))
        key = (normalize_key(phenotype), normalize_key(body_site), card_id)
        if not key[0] or not key[2] or key in seen:
            continue
        seen.add(key)
        deduped.append({"phenotype": phenotype, "body_site": body_site, "card_id": card_id})
    return deduped


def _attach_card_ids_to_hpo_features(
    features: Sequence[dict[str, Any]],
    phenotype_sources: Sequence[Mapping[str, Any]],
) -> None:
    sources_by_phenotype: dict[str, list[str]] = defaultdict(list)
    seen_by_phenotype: dict[str, set[str]] = defaultdict(set)
    for source in phenotype_sources:
        phenotype = clean_text(source.get("phenotype"))
        body_site = clean_text(source.get("body_site"))
        card_id = clean_text(source.get("card_id"))
        key = _phenotype_body_site_key(phenotype, body_site)
        if not key or not card_id or card_id in seen_by_phenotype[key]:
            continue
        seen_by_phenotype[key].add(card_id)
        sources_by_phenotype[key].append(card_id)

    for feature in features:
        key = _phenotype_body_site_key(
            clean_text(feature.get("name")),
            clean_text(feature.get("body_site")),
        )
        card_ids = sources_by_phenotype.get(key)
        if card_ids:
            feature["card_id"] = list(card_ids)


def _attach_card_ids_to_mapping_results(
    mappings: Sequence[dict[str, Any]],
    phenotype_sources: Sequence[Mapping[str, Any]],
) -> None:
    sources_by_phenotype: dict[str, list[str]] = defaultdict(list)
    seen_by_phenotype: dict[str, set[str]] = defaultdict(set)
    for source in phenotype_sources:
        key = _phenotype_body_site_key(
            clean_text(source.get("phenotype")),
            clean_text(source.get("body_site")),
        )
        card_id = clean_text(source.get("card_id"))
        if not key or not card_id or card_id in seen_by_phenotype[key]:
            continue
        seen_by_phenotype[key].add(card_id)
        sources_by_phenotype[key].append(card_id)

    for mapping in mappings:
        key = _phenotype_body_site_key(
            clean_text(mapping.get("original_phenotype")),
            clean_text(mapping.get("body_site")),
        )
        card_ids = sources_by_phenotype.get(key)
        if card_ids:
            mapping["card_id"] = list(card_ids)


def _attach_card_ids_to_hpo_summary(
    summary: dict[str, Any],
    phenotype_sources: Sequence[Mapping[str, Any]],
) -> None:
    sources_by_phenotype: dict[str, list[str]] = defaultdict(list)
    seen_by_phenotype: dict[str, set[str]] = defaultdict(set)
    for source in phenotype_sources:
        key = _phenotype_body_site_key(
            clean_text(source.get("phenotype")),
            clean_text(source.get("body_site")),
        )
        card_id = clean_text(source.get("card_id"))
        if not key or not card_id or card_id in seen_by_phenotype[key]:
            continue
        seen_by_phenotype[key].add(card_id)
        sources_by_phenotype[key].append(card_id)

    for item in summary.get("items") or []:
        if not isinstance(item, dict):
            continue
        key = _phenotype_body_site_key(
            clean_text(item.get("phenotype")),
            clean_text(item.get("body_site")),
        )
        card_ids = sources_by_phenotype.get(key)
        if card_ids:
            item["card_id"] = list(card_ids)


def topk_similarity(query_embeddings: Any, definition_embeddings: Any, *, k: int = 1) -> tuple[Any, Any]:
    torch = _load_torch()
    query_embeddings = torch.nn.functional.normalize(query_embeddings, p=2, dim=1)
    definition_embeddings = torch.nn.functional.normalize(definition_embeddings, p=2, dim=1)
    similarities = torch.matmul(query_embeddings, definition_embeddings.T)
    topk_values, topk_indices = torch.topk(similarities, k, dim=1)
    return topk_indices, topk_values


def _parse_phenotypes(payload: Mapping[str, Any]) -> list[dict[str, str]]:
    values = payload.get("phenotypes", [])
    if not isinstance(values, list):
        return []

    phenotypes: list[dict[str, str]] = []
    for item in values:
        if isinstance(item, str):
            phenotypes.append({"phenotype": clean_text(item), "body_site": ""})
        elif isinstance(item, Mapping):
            value = item.get("phenotype") or item.get("Phenotype") or item.get("description")
            if value is not None:
                phenotypes.append(
                    {
                        "phenotype": clean_text(value),
                        "body_site": clean_text(item.get("body_site") or item.get("BodySite")),
                    }
                )
    return _dedupe_phenotype_items(phenotypes)


def _dedupe_phenotype_items(values: Sequence[Any]) -> list[dict[str, str]]:
    seen: set[tuple[str, str]] = set()
    deduped: list[dict[str, str]] = []
    for value in values:
        if isinstance(value, Mapping):
            phenotype = clean_text(
                value.get("phenotype")
                or value.get("Phenotype")
                or value.get("description")
                or value.get("original_phenotype")
            )
            body_site = clean_text(value.get("body_site") or value.get("BodySite"))
        else:
            phenotype = clean_text(value)
            body_site = ""
        key = (normalize_key(phenotype), normalize_key(body_site))
        if not key[0] or key in seen:
            continue
        seen.add(key)
        deduped.append({"phenotype": phenotype, "body_site": body_site})
    return deduped


def _dedupe_texts(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        text = str(value or "").strip()
        key = " ".join(text.casefold().split())
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(text)
    return deduped


def _hpo_candidates(
    indices: Sequence[int],
    values: Sequence[float],
    definition_values: Sequence[str],
    definition_keys: Sequence[str],
    definition_body_sites: Mapping[str, str],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for index, value in zip(indices, values, strict=False):
        hpo_term = definition_keys[index]
        candidates.append(
            {
                "hpo_term": hpo_term,
                "hpo_code": definition_values[index],
                "matched_body_site": clean_text(definition_body_sites.get(hpo_term)),
                "similarity_score": float(value),
            }
        )
    return candidates


def _select_body_site_candidate(
    candidates: Sequence[Mapping[str, Any]],
    body_site: str,
) -> Mapping[str, Any] | None:
    if not clean_text(body_site):
        return candidates[0] if candidates else None
    for candidate in candidates:
        if normalize_key(body_site) == normalize_key(clean_text(candidate.get("matched_body_site"))):
            return candidate
    return None


def _phenotype_body_site_key(phenotype: str, body_site: str) -> str:
    phenotype_key = normalize_key(phenotype)
    if not phenotype_key:
        return ""
    return f"{phenotype_key}\0{normalize_key(body_site)}"


def _mapping_result(
    *,
    phenotype: str,
    body_site: str,
    hpo_code: str | None,
    hpo_term: str | None,
    matched_body_site: str,
    similarity_score: float,
    status: str,
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "original_phenotype": phenotype,
        "body_site": body_site,
        "hpo_code": hpo_code,
        "hpo_term": hpo_term,
        "matched_body_site": matched_body_site,
        "similarity_score": similarity_score,
        "status": status,
        "candidates": [dict(candidate) for candidate in candidates],
    }


def _build_hpo_summary(
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
        "body_site_mismatch_count": 0,
        "duplicate_count": 0,
    }
    for result in results:
        status = clean_text(result.get("status"))
        key = f"{status}_count"
        if key in counts:
            counts[key] += 1

    return {
        "source_type": source_type,
        "similarity_threshold": similarity_threshold,
        "top_k": top_k,
        "input_count": input_count,
        "deduped_count": deduped_count,
        **counts,
        "items": [_summary_item(result) for result in results],
    }


def _summary_item(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "phenotype": clean_text(result.get("original_phenotype")),
        "body_site": clean_text(result.get("body_site")),
        "matched_term": clean_text(result.get("hpo_term")),
        "hpo_code": clean_text(result.get("hpo_code")),
        "matched_body_site": clean_text(result.get("matched_body_site")),
        "similarity_score": result.get("similarity_score"),
        "status": clean_text(result.get("status")),
        "top_candidates": [dict(candidate) for candidate in result.get("candidates") or []],
    }


def _empty_hpo_summary(source_type: str = "unknown") -> dict[str, Any]:
    return {
        "source_type": source_type,
        "similarity_threshold": None,
        "top_k": DEFAULT_HPO_BODY_SITE_TOP_K,
        "input_count": 0,
        "deduped_count": 0,
        "mapped_count": 0,
        "low_similarity_count": 0,
        "body_site_mismatch_count": 0,
        "duplicate_count": 0,
        "items": [],
    }


def _load_definition2id(path: str | Path) -> tuple[dict[str, str], dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: definition2id JSON must be an object")

    definition2id: dict[str, str] = {}
    definition_body_sites: dict[str, str] = {}
    for key, value in data.items():
        if isinstance(value, list):
            if not value:
                continue
            hpo_id = value[0]
            body_site = ""
        elif isinstance(value, Mapping):
            hpo_ids = value.get("hpo_ids")
            if isinstance(hpo_ids, list):
                if not hpo_ids:
                    continue
                hpo_id = hpo_ids[0]
            else:
                hpo_id = value.get("hpo_id") or value.get("id")
            body_site = clean_text(value.get("body_site"))
        else:
            hpo_id = value
            body_site = ""
        if hpo_id is None:
            continue
        term = str(key)
        definition2id[term] = str(hpo_id)
        definition_body_sites[term] = body_site
    return definition2id, definition_body_sites


def _get_device(torch: Any) -> Any:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_torch() -> Any:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on optional local package
        raise RuntimeError(
            "HPO extraction requires the optional dependency 'torch'. "
            "Install it before calling BioLORD mapping."
        ) from exc
    return torch


def _load_transformers() -> tuple[Any, Any]:
    try:
        from transformers import AutoModel, AutoTokenizer
    except Exception as exc:  # pragma: no cover - depends on optional local package
        raise RuntimeError(
            "HPO extraction requires the optional dependency 'transformers'. "
            "Install it before loading BioLORD resources."
        ) from exc
    return AutoTokenizer, AutoModel
