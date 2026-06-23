from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from .utils import clean_text


HPO_FEATURE_KEYS = (
    "name",
    "hpo_code",
    "hpo_term",
    "similarity_score",
    "status",
)


def build_hpo_feature(mapping: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "name": mapping["original_phenotype"],
        "hpo_code": mapping["hpo_code"],
        "hpo_term": mapping["hpo_term"],
        "similarity_score": mapping["similarity_score"],
        "status": mapping["status"],
    }


def build_mapped_hpo_features(mappings: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [build_hpo_feature(item) for item in mappings if item.get("status") == "mapped"]


def pick_hpo_feature_payload(feature: Mapping[str, Any]) -> dict[str, Any]:
    payload = {key: feature[key] for key in HPO_FEATURE_KEYS if key in feature}
    if "name" not in payload:
        payload["name"] = clean_text(feature.get("name"))
    return payload
