from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple | set):
        return list(value)
    return [value]


def as_text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = clean_text(value)
        return [text] if text else []
    if isinstance(value, Mapping):
        texts: list[str] = []
        for item in value.values():
            texts.extend(as_text_list(item))
        return texts
    if isinstance(value, Iterable):
        texts = []
        for item in value:
            texts.extend(as_text_list(item))
        return texts
    text = clean_text(value)
    return [text] if text else []


def dedupe_texts(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        text = clean_text(value)
        key = normalize_key(text)
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(text)
    return deduped


def normalize_key(value: str) -> str:
    return re.sub(r"[\s\-_'/()（）【】\[\]{}，,。.;；:：]+", "", value.casefold())


def is_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list | tuple | set):
        return any(is_present(item) for item in value)
    if isinstance(value, Mapping):
        if "items" in value and len(value) <= 2:
            return is_present(value.get("items"))
        return any(is_present(item) for item in value.values())
    return True


def get_path(root: Any, path: str) -> Any:
    if not path:
        return root
    value: Any = root
    for part in path.split("."):
        if part == "":
            continue
        if isinstance(value, Mapping):
            value = value.get(part)
        elif isinstance(value, list):
            try:
                value = value[int(part)]
            except (ValueError, IndexError):
                return None
        else:
            return None
    return value


def resolve_case_path(canonical_case: Mapping[str, Any], path: str) -> Any:
    case_path = path.removeprefix("case.")
    direct = get_path(canonical_case, case_path)
    if is_present(direct):
        return direct

    compact = normalize_key(case_path)
    if compact in {"lab", "labs", "laboratory"}:
        return get_path(canonical_case, "labs.items")
    if compact in {"imaging", "image"}:
        return get_path(canonical_case, "imaging.items")
    if compact in {"endoscopy", "scope"}:
        return get_path(canonical_case, "endoscopy.items")
    if compact in {"pathology", "biopsy"}:
        return get_path(canonical_case, "pathology.items")
    if compact in {"ctemre", "ctemri", "ct_mre", "cte_mre"}:
        return _find_case_text(canonical_case, ("CTE", "MRE"))
    if compact in {"fc", "fecalcalprotectin"}:
        return _find_case_text(canonical_case, ("粪便钙卫蛋白", "钙卫蛋白", "FC"))
    return direct


def resolve_prefixed_path(
    *,
    path: str,
    result: Mapping[str, Any] | None,
    context: Mapping[str, Any],
    canonical_case: Mapping[str, Any],
    state: Mapping[str, Any],
) -> Any:
    if path.startswith("result."):
        return get_path(result or {}, path.removeprefix("result."))
    if path.startswith("context."):
        return get_path(context, path.removeprefix("context."))
    if path.startswith("case."):
        return resolve_case_path(canonical_case, path)
    if path.startswith("state."):
        return get_path(state, path.removeprefix("state."))
    return get_path(result or {}, path)


def text_contains_term(text: str, term: str) -> bool:
    haystack = clean_text(text)
    needle = clean_text(term)
    if not haystack or not needle:
        return False
    if re.fullmatch(r"[A-Za-z0-9 .'\-/]+", needle):
        pattern = re.escape(needle).replace(r"\ ", r"\s+")
        return re.search(rf"(?<![A-Za-z0-9]){pattern}(?![A-Za-z0-9])", haystack, re.I) is not None

    hay_key = normalize_key(haystack)
    needle_key = normalize_key(needle)
    if not hay_key or not needle_key:
        return False
    if needle_key in hay_key or hay_key in needle_key:
        return True

    reduced = _strip_generic_clinical_words(needle_key)
    if len(reduced) >= 2 and reduced in hay_key:
        return True
    return _has_meaningful_overlap(hay_key, reduced or needle_key)


def flatten_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return " ".join(flatten_text(item) for item in value.values())
    if isinstance(value, Iterable):
        return " ".join(flatten_text(item) for item in value)
    return str(value)


def shorten(text: str, *, limit: int = 240) -> str:
    cleaned = clean_text(text)
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1] + "..."


def _find_case_text(canonical_case: Mapping[str, Any], terms: tuple[str, ...]) -> str | None:
    all_text = flatten_text(canonical_case)
    for term in terms:
        if text_contains_term(all_text, term):
            return term
    return None


def _strip_generic_clinical_words(value: str) -> str:
    reduced = value
    for word in (
        "检查",
        "检测",
        "浓度",
        "评估",
        "情况",
        "建议",
        "患者",
        "进行",
        "用于",
        "水平",
        "结果",
        "常规",
    ):
        reduced = reduced.replace(word, "")
    return reduced


def _has_meaningful_overlap(haystack: str, needle: str) -> bool:
    if len(needle) < 4:
        return False
    min_size = 3 if re.search(r"[\u4e00-\u9fff]", needle) else 4
    for size in range(min(8, len(needle)), min_size - 1, -1):
        for start in range(0, len(needle) - size + 1):
            chunk = needle[start : start + size]
            if chunk in haystack:
                return True
    return False
