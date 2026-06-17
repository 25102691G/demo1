from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from .utils import clean_text


ICD_FEATURE_KEYS = (
    "name",
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
    "similarity_score",
    "status",
)


def build_icd_feature(mapping: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "name": mapping["original_diagnosis"],
        "chapter": clean_text(mapping.get("chapter")),
        "chapter_code_range": clean_text(mapping.get("chapter_code_range")),
        "chapter_name": clean_text(mapping.get("chapter_name")),
        "section_code_range": clean_text(mapping.get("section_code_range")),
        "section_name": clean_text(mapping.get("section_name")),
        "category_code": clean_text(mapping.get("category_code")),
        "category_name": clean_text(mapping.get("category_name")),
        "subcategory_code": clean_text(mapping.get("subcategory_code")),
        "subcategory_name": clean_text(mapping.get("subcategory_name")),
        "diagnosis_code": clean_text(mapping.get("diagnosis_code")) or None,
        "diagnosis_name": clean_text(mapping.get("diagnosis_name")) or None,
        "similarity_score": mapping["similarity_score"],
        "status": mapping["status"],
    }


def build_mapped_icd_features(mappings: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [build_icd_feature(item) for item in mappings if item.get("status") == "mapped"]


def pick_icd_feature_payload(feature: Mapping[str, Any]) -> dict[str, Any]:
    payload = {key: feature[key] for key in ICD_FEATURE_KEYS if key in feature}
    if "name" not in payload:
        payload["name"] = clean_text(feature.get("name"))
    return payload
