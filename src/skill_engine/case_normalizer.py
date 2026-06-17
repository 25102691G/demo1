from __future__ import annotations

import json
import uuid
import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .hpo_extractor import HPO_EXTRACTION_SYSTEM_PROMPT_FROM_CASE, HpoExtractor
from .llm_client import JsonChatClient
from .schemas import build_required_defaults, load_json_schema, validate_json
from .utils import clean_text


def normalize_case(
    raw_input: str,
    schema_path: Path,
    *,
    hpo_extractor: HpoExtractor,
    deepseek_client: JsonChatClient,
) -> dict[str, Any]:
    schema = load_json_schema(Path(schema_path))
    canonical = _default_case(raw_input, schema)
    _apply_hpo_extraction(canonical, raw_input, hpo_extractor, deepseek_client)
    validate_json(canonical, schema, label="canonical_case")
    return canonical


def normalize_case_from_json(
    data: dict[str, Any],
    raw_input: str | None,
    schema_path: Path,
    *,
    hpo_extractor: HpoExtractor,
    deepseek_client: JsonChatClient,
) -> dict[str, Any]:
    schema = load_json_schema(Path(schema_path))
    effective_raw = raw_input or clean_text(data.get("raw_input"))
    canonical = _default_case(effective_raw, schema)
    _apply_hpo_extraction(canonical, effective_raw, hpo_extractor, deepseek_client)
    _deep_merge(canonical, data)
    canonical["raw_input"] = clean_text(canonical.get("raw_input")) or effective_raw
    validate_json(canonical, schema, label="canonical_case")
    return canonical


def load_case_json(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: case JSON must be an object")
    return data


def _default_case(raw_input: str, schema: Mapping[str, Any]) -> dict[str, Any]:
    canonical = build_required_defaults(schema)
    canonical["raw_input"] = raw_input or ""
    if not canonical.get("case_id"):
        canonical["case_id"] = f"case_{uuid.uuid4().hex}"

    return canonical


def _apply_hpo_extraction(
    canonical: dict[str, Any],
    raw_input: str,
    hpo_extractor: HpoExtractor,
    deepseek_client: JsonChatClient,
) -> None:
    positive_features = hpo_extractor.extract_hpo_from_case(
        raw_input or "",
        deepseek_client,
        HPO_EXTRACTION_SYSTEM_PROMPT_FROM_CASE,
    )
    symptoms = positive_features.get("symptoms", [])
    canonical["symptoms"] = symptoms if isinstance(symptoms, list) else []


def _deep_merge(base: dict[str, Any], incoming: Mapping[str, Any]) -> None:
    for key, value in incoming.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
