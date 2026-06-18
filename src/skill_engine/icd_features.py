from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from .utils import clean_text


ICD_FEATURE_KEYS = (
    "name",
    "similarity_score",
    "status",
    "icd10_mapping",
    "hpo_mapping",
)

ICD10_MAPPING_KEYS = (
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


def build_icd_feature(mapping: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "name": mapping["original_diagnosis"],
        "similarity_score": mapping["similarity_score"],
        "status": mapping["status"],
        "icd10_mapping": _icd10_mapping(mapping),
        "hpo_mapping": {},
    }


def build_mapped_icd_features(mappings: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [build_icd_feature(item) for item in mappings if item.get("status") == "mapped"]


def pick_icd_feature_payload(feature: Mapping[str, Any]) -> dict[str, Any]:
    if "icd10_mapping" in feature:
        payload = {key: feature[key] for key in ICD_FEATURE_KEYS if key in feature}
    else:
        payload = {
            "name": clean_text(feature.get("name")),
            "similarity_score": feature.get("similarity_score"),
            "status": clean_text(feature.get("status")),
            "icd10_mapping": _icd10_mapping(feature),
            "hpo_mapping": {},
        }
    if "name" not in payload:
        payload["name"] = clean_text(feature.get("name"))
    return payload


def _icd10_mapping(mapping: Mapping[str, Any]) -> dict[str, Any]:
    result = {key: clean_text(mapping.get(key)) for key in ICD10_MAPPING_KEYS}
    result["diagnosis_code"] = result["diagnosis_code"] or None
    result["diagnosis_name"] = result["diagnosis_name"] or None
    return result
