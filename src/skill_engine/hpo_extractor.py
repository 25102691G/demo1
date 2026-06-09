from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from skill_engine.llm_client import (
    JsonChatClient,
    LlmConfig,
    OpenAICompatibleJsonChatClient,
    load_llm_config_from_env,
)


HPO_EXTRACTION_SYSTEM_PROMPT = """你是医学表型抽取助手。
请从患者临床文本中抽取明确出现的表型、症状或体征。
只输出 JSON，格式如下：
{"phenotypes": [{"phenotype": "原文中的表型短语"}]}。
不要翻译，不要改写为英文，不要标准化为 HPO 术语。
输入是中文时，phenotype 必须保留中文原文表达。
只抽取原文明确支持的阳性表型；否定表述不要抽取，例如“无发热”不要输出“发热”。
不要输出诊断、药物、检查项目或治疗方式，除非它们本身直接描述患者表型。"""

DEFAULT_MODEL_PATH = ROOT / "data" / "bge-large-zh-v1.5"
DEFAULT_DEFINITION2ID_PATH = ROOT / "data" / "ontology" / "definition2id.json"
DEFAULT_DEFINITION_EMBEDDINGS_PATH = ROOT / "data" / "ontology" / "definition_embeddings.pt"


@dataclass(frozen=True)
class HpoResources:
    model: Any
    tokenizer: Any
    definition2id: dict[str, str]
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
        similarity_threshold: float = 0.8,
        batch_size: int = 30,
        max_length: int = 128,
    ) -> None:
        self.resources = resources
        self.similarity_threshold = similarity_threshold
        self.batch_size = batch_size
        self.max_length = max_length

    @classmethod
    def from_paths(
        cls,
        *,
        model_path: str | Path,
        definition2id_path: str | Path,
        definition_embeddings_path: str | Path,
        similarity_threshold: float = 0.8,
        batch_size: int = 30,
        max_length: int = 128,
    ) -> HpoExtractor:
        torch = _load_torch()
        AutoTokenizer, AutoModel = _load_transformers()

        tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
        model = AutoModel.from_pretrained(str(model_path), local_files_only=True)
        definition2id = _load_definition2id(definition2id_path)
        definition_embeddings = torch.load(str(definition_embeddings_path), map_location="cpu")
        resources = HpoResources(
            model=model,
            tokenizer=tokenizer,
            definition2id=definition2id,
            definition_embeddings=definition_embeddings,
            definition_keys=list(definition2id.keys()),
        )
        return cls(
            resources,
            similarity_threshold=similarity_threshold,
            batch_size=batch_size,
            max_length=max_length,
        )

    def extract_from_text(self, text: str, deepseek_client: JsonChatClient) -> dict[str, Any]:
        phenotypes = self.extract_phenotypes(text, deepseek_client)
        mappings = self.map_phenotypes_to_hpo(phenotypes)
        mapped = [item for item in mappings if item["status"] == "mapped"]
        return {
            "phenotypes": phenotypes,
            "hpo_codes": [item["hpo_code"] for item in mapped],
            "hpo_descriptions": [item["hpo_term"] for item in mapped],
            "hpo_mappings": mappings,
        }

    def extract_phenotypes(self, text: str, deepseek_client: JsonChatClient) -> list[str]:
        if not str(text or "").strip():
            return []
        user_prompt = json.dumps({"clinical_text": text}, ensure_ascii=False)
        payload = deepseek_client.chat_json(HPO_EXTRACTION_SYSTEM_PROMPT, user_prompt)
        return _parse_phenotypes(payload)

    def map_phenotypes_to_hpo(self, phenotypes: Sequence[str]) -> list[dict[str, Any]]:
        cleaned = _dedupe_texts(phenotypes)
        if not cleaned:
            return []

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
        topk_indices, topk_values = topk_similarity(query_embeddings, definition_embeddings, k=1)
        topk_indices = topk_indices.cpu().numpy().tolist()
        topk_values = topk_values.cpu().numpy().tolist()

        definition_values = list(resources.definition2id.values())
        results: list[dict[str, Any]] = []
        seen_hpo_codes: set[str] = set()
        for index, phenotype in enumerate(cleaned):
            best_match_index = topk_indices[index][0]
            similarity_score = float(topk_values[index][0])
            hpo_code = definition_values[best_match_index]
            hpo_term = resources.definition_keys[best_match_index]

            if similarity_score < self.similarity_threshold:
                results.append(
                    _mapping_result(
                        phenotype=phenotype,
                        hpo_code=None,
                        hpo_term=None,
                        similarity_score=similarity_score,
                        status="low_similarity",
                    )
                )
                continue

            if hpo_code in seen_hpo_codes:
                results.append(
                    _mapping_result(
                        phenotype=phenotype,
                        hpo_code=hpo_code,
                        hpo_term=hpo_term,
                        similarity_score=similarity_score,
                        status="duplicate",
                    )
                )
                continue

            seen_hpo_codes.add(hpo_code)
            results.append(
                _mapping_result(
                    phenotype=phenotype,
                    hpo_code=hpo_code,
                    hpo_term=hpo_term,
                    similarity_score=similarity_score,
                    status="mapped",
                )
            )
        return results


def topk_similarity(query_embeddings: Any, definition_embeddings: Any, *, k: int = 1) -> tuple[Any, Any]:
    torch = _load_torch()
    query_embeddings = torch.nn.functional.normalize(query_embeddings, p=2, dim=1)
    definition_embeddings = torch.nn.functional.normalize(definition_embeddings, p=2, dim=1)
    similarities = torch.matmul(query_embeddings, definition_embeddings.T)
    topk_values, topk_indices = torch.topk(similarities, k, dim=1)
    return topk_indices, topk_values


def _parse_phenotypes(payload: Mapping[str, Any]) -> list[str]:
    values = payload.get("phenotypes", [])
    if not isinstance(values, list):
        return []

    phenotypes: list[str] = []
    for item in values:
        if isinstance(item, str):
            phenotypes.append(item)
        elif isinstance(item, Mapping):
            value = item.get("phenotype") or item.get("Phenotype") or item.get("description")
            if value is not None:
                phenotypes.append(str(value))
    return _dedupe_texts(phenotypes)


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


def _mapping_result(
    *,
    phenotype: str,
    hpo_code: str | None,
    hpo_term: str | None,
    similarity_score: float,
    status: str,
) -> dict[str, Any]:
    return {
        "original_phenotype": phenotype,
        "hpo_code": hpo_code,
        "hpo_term": hpo_term,
        "similarity_score": similarity_score,
        "status": status,
    }


def _load_definition2id(path: str | Path) -> dict[str, str]:
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: definition2id JSON must be an object")

    definition2id: dict[str, str] = {}
    for key, value in data.items():
        if isinstance(value, list):
            if not value:
                continue
            hpo_id = value[0]
        else:
            hpo_id = value
        if hpo_id is None:
            continue
        definition2id[str(key)] = str(hpo_id)
    return definition2id


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


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Extract phenotypes from clinical text with an LLM and map them to HPO IDs."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input-text", help="Raw clinical text.")
    input_group.add_argument("--input-file", help="UTF-8 text file containing raw clinical text.")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH), help="Local BGE model path.")
    parser.add_argument("--definition2id-path", default=str(DEFAULT_DEFINITION2ID_PATH))
    parser.add_argument("--definition-embeddings-path", default=str(DEFAULT_DEFINITION_EMBEDDINGS_PATH))
    parser.add_argument("--similarity-threshold", type=float, default=0.8)
    parser.add_argument("--batch-size", type=int, default=30)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--api-key", default=os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--base-url", default=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    parser.add_argument("--llm-model", default=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"))
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args(argv)

    try:
        text = _read_cli_input(args)
        llm_config = _cli_llm_config(args)
        extractor = HpoExtractor.from_paths(
            model_path=_resolve_cli_path(args.model_path),
            definition2id_path=_resolve_cli_path(args.definition2id_path),
            definition_embeddings_path=_resolve_cli_path(args.definition_embeddings_path),
            similarity_threshold=args.similarity_threshold,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
        llm_client = OpenAICompatibleJsonChatClient(llm_config)
        result = extractor.extract_from_text(text, llm_client)
    except Exception as exc:
        print(f"hpo_extractor: error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _read_cli_input(args: argparse.Namespace) -> str:
    if args.input_text:
        return args.input_text
    if args.input_file:
        return _resolve_cli_path(args.input_file).read_text(encoding="utf-8-sig")
    raise ValueError("provide --input-text or --input-file")


def _resolve_cli_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _cli_llm_config(args: argparse.Namespace) -> LlmConfig:
    if args.api_key:
        return LlmConfig(
            api_key=args.api_key,
            base_url=args.base_url,
            model=args.llm_model,
            temperature=args.temperature,
        )
    return load_llm_config_from_env(temperature=args.temperature)


if __name__ == "__main__":
    raise SystemExit(main())
