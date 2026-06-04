from __future__ import annotations

import copy
import json
import re
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .schemas import load_json_schema, validate_json
from .utils import clean_text, dedupe_texts


SYMPTOM_KEYWORDS = (
    "腹痛",
    "腹泻",
    "发热",
    "便血",
    "体重下降",
    "乏力",
    "恶心",
    "呕吐",
    "肛瘘",
    "肛周脓肿",
    "关节痛",
    "皮疹",
)
SIGN_KEYWORDS = ("压痛", "反跳痛", "包块", "水肿")
LAB_KEYWORDS = (
    "CRP",
    "ESR",
    "血红蛋白",
    "白蛋白",
    "粪便钙卫蛋白",
    "钙卫蛋白",
    "白细胞",
    "血小板",
    "肝功能",
    "肾功能",
)
IMAGING_KEYWORDS = ("CTE", "MRE", "CT", "MRI", "超声", "X线", "造影")
ENDOSCOPY_KEYWORDS = ("结肠镜", "胃镜", "肠镜", "胶囊内镜", "小肠镜", "内镜")
PATHOLOGY_KEYWORDS = ("病理", "活检", "组织学", "肉芽肿")
MEDICATION_KEYWORDS = (
    "美沙拉嗪",
    "激素",
    "糖皮质激素",
    "英夫利昔单抗",
    "阿达木单抗",
    "乌司奴单抗",
    "维得利珠单抗",
    "抗生素",
    "免疫抑制剂",
)
PROCEDURE_KEYWORDS = ("手术", "切除", "造瘘", "引流", "活检")
RED_FLAG_KEYWORDS = (
    "剧烈腹痛",
    "高热",
    "休克",
    "黑便",
    "大量便血",
    "意识障碍",
    "严重脱水",
    "肠梗阻",
    "穿孔",
    "中毒性巨结肠",
)
CRITICAL_RED_FLAGS = ("休克", "穿孔", "意识障碍", "中毒性巨结肠")


def normalize_case(raw_input: str, schema_path: Path) -> dict[str, Any]:
    canonical = _default_case(raw_input)
    _apply_rule_extraction(canonical, raw_input)
    _record_missing_fields(canonical)
    schema = load_json_schema(Path(schema_path))
    validate_json(canonical, schema, label="canonical_case")
    return canonical


def normalize_case_from_json(
    data: dict[str, Any],
    raw_input: str | None,
    schema_path: Path,
) -> dict[str, Any]:
    effective_raw = raw_input or clean_text(data.get("raw_input"))
    canonical = _default_case(effective_raw)
    _apply_rule_extraction(canonical, effective_raw)
    _deep_merge(canonical, data)
    canonical["raw_input"] = clean_text(canonical.get("raw_input")) or effective_raw
    _repair_container_shapes(canonical)
    _record_missing_fields(canonical)
    schema = load_json_schema(Path(schema_path))
    validate_json(canonical, schema, label="canonical_case")
    return canonical


def load_case_json(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: case JSON must be an object")
    return data


def _default_case(raw_input: str) -> dict[str, Any]:
    return {
        "case_id": f"case_{uuid.uuid4().hex}",
        "raw_input": raw_input or "",
        "input_language": "zh-CN",
        "demographics": {
            "age": None,
            "sex": "unknown",
            "pregnancy_status": "unknown",
        },
        "chief_complaint": None,
        "history_of_present_illness": None,
        "symptoms": [],
        "signs": [],
        "vitals": {
            "temperature": {"value": None, "unit": "℃", "interpretation": None},
            "heart_rate": {"value": None, "unit": "次/分", "interpretation": None},
            "respiratory_rate": {"value": None, "unit": "次/分", "interpretation": None},
            "blood_pressure": {
                "systolic": None,
                "diastolic": None,
                "unit": "mmHg",
                "interpretation": None,
            },
            "oxygen_saturation": {"value": None, "unit": "%", "interpretation": None},
        },
        "labs": {"items": []},
        "imaging": {"items": []},
        "endoscopy": {"items": []},
        "pathology": {"items": []},
        "diagnoses": [],
        "medications": [],
        "procedures": [],
        "allergies": [],
        "comorbidities": [],
        "family_history": [],
        "extra_manifestations": [],
        "scores": [],
        "red_flags": [],
        "patient_goal": None,
        "extraction_quality": {
            "confidence": 0.5,
            "missing_or_uncertain": [],
            "normalization_notes": ["rule_based_normalization_v1"],
        },
    }


def _apply_rule_extraction(canonical: dict[str, Any], raw_input: str) -> None:
    text = raw_input or ""
    _extract_demographics(canonical, text)
    canonical["chief_complaint"] = _short_clause(text)
    canonical["history_of_present_illness"] = text or None
    canonical["symptoms"] = [_clinical_item(keyword, text) for keyword in _matched(SYMPTOM_KEYWORDS, text)]
    canonical["signs"] = [_clinical_item(keyword, text) for keyword in _matched(SIGN_KEYWORDS, text)]
    canonical["labs"]["items"] = [_lab_item(keyword, text) for keyword in _matched(LAB_KEYWORDS, text)]
    canonical["imaging"]["items"] = [
        _imaging_item(keyword, text) for keyword in _matched(IMAGING_KEYWORDS, text)
    ]
    canonical["endoscopy"]["items"] = [
        _endoscopy_item(keyword, text) for keyword in _matched(ENDOSCOPY_KEYWORDS, text)
    ]
    canonical["pathology"]["items"] = [
        _pathology_item(keyword, text) for keyword in _matched(PATHOLOGY_KEYWORDS, text)
    ]
    canonical["diagnoses"] = _extract_diagnoses(text)
    canonical["medications"] = [
        {"name": keyword, "status": "unknown", "dose": None, "frequency": None, "source_text": keyword}
        for keyword in _matched(MEDICATION_KEYWORDS, text)
    ]
    canonical["procedures"] = [
        {"name": keyword, "date": None, "result": None, "source_text": keyword}
        for keyword in _matched(PROCEDURE_KEYWORDS, text)
    ]
    canonical["red_flags"] = [
        {
            "name": keyword,
            "source": "rule_detected",
            "severity": "critical" if keyword in CRITICAL_RED_FLAGS else "high",
            "source_text": keyword,
        }
        for keyword in _matched(RED_FLAG_KEYWORDS, text)
    ]
    _dedupe_case_lists(canonical)


def _extract_demographics(canonical: dict[str, Any], text: str) -> None:
    age_match = re.search(r"(?<!\d)(\d{1,3})\s*岁", text)
    if age_match:
        canonical["demographics"]["age"] = int(age_match.group(1))
    if re.search(r"男|男性", text):
        canonical["demographics"]["sex"] = "male"
    elif re.search(r"女|女性", text):
        canonical["demographics"]["sex"] = "female"


def _clinical_item(keyword: str, text: str) -> dict[str, Any]:
    return {
        "name": keyword,
        "standard_name": None,
        "status": "present",
        "duration": _nearby_duration(keyword, text),
        "severity": _nearby_severity(keyword, text),
        "value": None,
        "source_text": keyword,
    }


def _lab_item(keyword: str, text: str) -> dict[str, Any]:
    return {
        "name": keyword,
        "standard_name": None,
        "value": _nearby_value(keyword, text),
        "unit": None,
        "reference_range": None,
        "interpretation": _nearby_interpretation(keyword, text),
        "date": None,
        "source_text": keyword,
    }


def _imaging_item(keyword: str, text: str) -> dict[str, Any]:
    return {
        "modality": keyword,
        "body_part": None,
        "findings": _nearby_findings(keyword, text),
        "impression": None,
        "date": None,
        "source_text": keyword,
    }


def _endoscopy_item(keyword: str, text: str) -> dict[str, Any]:
    return {
        "type": keyword,
        "findings": _nearby_findings(keyword, text),
        "biopsy_taken": "unknown",
        "date": None,
        "source_text": keyword,
    }


def _pathology_item(keyword: str, text: str) -> dict[str, Any]:
    return {
        "specimen": None,
        "findings": _nearby_findings(keyword, text),
        "diagnosis": None,
        "date": None,
        "source_text": keyword,
    }


def _extract_diagnoses(text: str) -> list[dict[str, Any]]:
    diagnoses: list[dict[str, Any]] = []
    patterns = (
        (r"(?:确诊|诊断为|明确诊断为)([^，。；;]{2,30})", "confirmed"),
        (r"(?:高度怀疑|高度疑似)([^，。；;]{2,30})", "highly_suspected"),
        (r"(?:疑似|疑诊|考虑)([^，。；;]{2,30})", "suspected"),
        (r"(?:既往诊断|既往有)([^，。；;]{2,30})", "history"),
    )
    for pattern, status in patterns:
        for match in re.finditer(pattern, text):
            name = _clean_diagnosis_name(match.group(1))
            if name:
                diagnoses.append(
                    {"name": name, "status": status, "date": None, "source_text": match.group(0)}
                )
    return _dedupe_dicts_by_name(diagnoses)


def _matched(keywords: tuple[str, ...], text: str) -> list[str]:
    return [keyword for keyword in keywords if keyword and keyword.casefold() in text.casefold()]


def _nearby_duration(keyword: str, text: str) -> str | None:
    pattern = rf"{re.escape(keyword)}[^，。；;]{{0,12}}?((?:\d+|半)[天周月年])"
    match = re.search(pattern, text)
    return match.group(1) if match else None


def _nearby_severity(keyword: str, text: str) -> str:
    window = _window(keyword, text)
    if any(word in window for word in ("剧烈", "严重", "重度")):
        return "severe"
    if any(word in window for word in ("中度", "明显")):
        return "moderate"
    if any(word in window for word in ("轻", "轻度")):
        return "mild"
    return "unknown"


def _nearby_value(keyword: str, text: str) -> str | None:
    window = _window(keyword, text, size=18)
    match = re.search(r"([<>]?\d+(?:\.\d+)?)\s*([A-Za-zμ/%]+(?:/[A-Za-z]+)?)?", window)
    if not match:
        return None
    unit = clean_text(match.group(2))
    return f"{match.group(1)} {unit}".strip()


def _nearby_interpretation(keyword: str, text: str) -> str:
    window = _window(keyword, text, size=18)
    if any(word in window for word in ("升高", "增高", "高")):
        return "elevated"
    if any(word in window for word in ("降低", "下降", "低")):
        return "decreased"
    if "阳性" in window:
        return "positive"
    if "阴性" in window:
        return "negative"
    return "unknown"


def _nearby_findings(keyword: str, text: str) -> list[str]:
    window = _window(keyword, text, size=28)
    findings = [
        word
        for word in ("溃疡", "狭窄", "瘘管", "脓肿", "水肿", "糜烂", "肉芽肿", "炎症")
        if word in window
    ]
    return findings or [keyword]


def _window(keyword: str, text: str, *, size: int = 12) -> str:
    index = text.casefold().find(keyword.casefold())
    if index < 0:
        return ""
    return text[max(0, index - size) : index + len(keyword) + size]


def _short_clause(text: str) -> str | None:
    cleaned = clean_text(text)
    if not cleaned:
        return None
    return re.split(r"[。；;\n]", cleaned, maxsplit=1)[0][:80]


def _clean_diagnosis_name(value: str) -> str:
    text = clean_text(value).strip(" ：:，。；;")
    text = re.sub(r"(患者|可能|相关|表现)$", "", text)
    return text[:40]


def _deep_merge(base: dict[str, Any], incoming: Mapping[str, Any]) -> None:
    for key, value in incoming.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)


def _repair_container_shapes(canonical: dict[str, Any]) -> None:
    for field in ("labs", "imaging", "endoscopy", "pathology"):
        value = canonical.get(field)
        if isinstance(value, list):
            canonical[field] = {"items": value}
        elif not isinstance(value, dict):
            canonical[field] = {"items": []}
        else:
            value.setdefault("items", [])
    for field in (
        "symptoms",
        "signs",
        "diagnoses",
        "medications",
        "procedures",
        "red_flags",
        "allergies",
        "comorbidities",
        "family_history",
        "extra_manifestations",
        "scores",
    ):
        if not isinstance(canonical.get(field), list):
            canonical[field] = []


def _record_missing_fields(canonical: dict[str, Any]) -> None:
    missing = []
    if canonical["demographics"].get("age") is None:
        missing.append("demographics.age")
    if canonical["demographics"].get("sex") in {None, "unknown"}:
        missing.append("demographics.sex")
    for field in ("symptoms", "signs", "diagnoses"):
        if not canonical.get(field):
            missing.append(field)
    for field in ("labs", "imaging", "endoscopy", "pathology"):
        if not canonical.get(field, {}).get("items"):
            missing.append(f"{field}.items")
    quality = canonical.setdefault("extraction_quality", {})
    quality["missing_or_uncertain"] = dedupe_texts(
        [*quality.get("missing_or_uncertain", []), *missing]
    )
    quality.setdefault("normalization_notes", []).append(
        "未抽取到的信息保留为空值或空数组。"
    )
    confidence = quality.get("confidence", 0.5)
    try:
        quality["confidence"] = max(0.0, min(float(confidence), 1.0))
    except (TypeError, ValueError):
        quality["confidence"] = 0.5


def _dedupe_case_lists(canonical: dict[str, Any]) -> None:
    for field in ("symptoms", "signs", "diagnoses", "medications", "procedures", "red_flags"):
        canonical[field] = _dedupe_dicts_by_name(canonical.get(field, []))


def _dedupe_dicts_by_name(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for item in items:
        key = clean_text(item.get("name")).casefold()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped
