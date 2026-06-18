from __future__ import annotations

import json
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

from .hpo_features import pick_hpo_feature_payload
from .icd_features import pick_icd_feature_payload
from .skill_loader import SkillPack
from .utils import clean_text, flatten_text, is_present, resolve_case_path, text_contains_term


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DEFINITION2ID_PATH = ROOT / "data" / "ontology" / "definition2id.json"

def route_skills(
    canonical_case: dict[str, Any],
    skill_packs: list[SkillPack],
    *,
    top_k: int | None = None,
    min_score: float | None = None,
    feature_mode: str = "hpo",
) -> list[dict[str, Any]]:
    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be at least 1")
    candidates = [
        _score_skill(canonical_case, pack, min_score=min_score, feature_mode=feature_mode)
        for pack in skill_packs
    ]
    candidates.sort(key=lambda item: item["score"], reverse=True)
    default_top_k = top_k or _default_top_k(skill_packs)
    selected = candidates[:default_top_k]
    for rank, candidate in enumerate(selected, start=1):
        candidate["rank"] = rank
    return selected


def _score_skill(
    canonical_case: dict[str, Any],
    pack: SkillPack,
    *,
    min_score: float | None,
    feature_mode: str,
) -> dict[str, Any]:
    routing = pack.skill.get("routing_profile") or {}
    positive_features = routing.get("positive_features") or {}
    matched_features: list[dict[str, Any]] = []
    raw_score = 0.0

    for feature in _iter_positive_features(positive_features):
        if feature_mode == "hpo":
            matched = _match_symptom_feature(canonical_case, feature)
        elif feature_mode == "icd10":
            matched = _match_icd10_feature(canonical_case, feature)
        else:
            raise ValueError(f"unsupported feature mode: {feature_mode}")
        if not matched:
            continue
        weight = float(feature.get("weight") or 0.2)
        raw_score += weight
        matched_features.append(_matched_feature_payload(feature))

    penalties: list[str] = []
    # penalty_score = 0.0
    # for feature in routing.get("negative_features") or []:
    #     if not isinstance(feature, Mapping):
    #         continue
    #     name = clean_text(feature.get("name"))
    #     if text_contains_term(searchable_text, name):
    #         penalty = float(feature.get("penalty") or 0)
    #         penalty_score += penalty
    #         penalties.append(name)

    # score = _normalize_score(raw_score - penalty_score, routing.get("scoring") or {})
    score = raw_score
    threshold = _candidate_threshold(routing, min_score=min_score)
    missing_key_evidence = _missing_key_evidence(canonical_case, pack)
    disease_name = pack.disease_name or clean_text((routing.get("disease_identity") or {}).get("primary_name"))
    return {
        "skill_id": pack.skill_id,
        "disease_name": disease_name,
        "score": score,
        "rank": 0,
        "matched_features": matched_features,
        "negative_features": penalties,
        "missing_key_evidence": missing_key_evidence,
        "reasoning_summary": _reasoning_summary(score, threshold, matched_features, penalties),
        "candidate_threshold": threshold,
        "strong_candidate_threshold": _strong_threshold(routing),
        "raw_score": raw_score,
    }


def _iter_positive_features(positive_features: Any) -> list[Mapping[str, Any]]:
    if isinstance(positive_features, list):
        return [feature for feature in positive_features if isinstance(feature, Mapping)]
    if isinstance(positive_features, Mapping):
        return [
            feature
            for features in positive_features.values()
            if isinstance(features, list)
            for feature in features
            if isinstance(feature, Mapping)
        ]
    return []


def _case_searchable_text(canonical_case: Mapping[str, Any]) -> str:
    parts: list[str] = [flatten_text(canonical_case.get("raw_input"))]
    parts.extend(clean_text(item.get("name")) for item in canonical_case.get("symptoms") or [])
    parts.extend(clean_text(item.get("name")) for item in canonical_case.get("signs") or [])
    parts.extend(clean_text(item.get("name")) for item in canonical_case.get("diagnoses") or [])
    for feature in canonical_case.get("features") or []:
        icd10_mapping = feature.get("icd10_mapping") if isinstance(feature, Mapping) else {}
        parts.extend(
            [
                clean_text(feature.get("name")) if isinstance(feature, Mapping) else "",
                clean_text(icd10_mapping.get("diagnosis_name")) if isinstance(icd10_mapping, Mapping) else "",
                clean_text(icd10_mapping.get("category_name")) if isinstance(icd10_mapping, Mapping) else "",
                clean_text(icd10_mapping.get("section_name")) if isinstance(icd10_mapping, Mapping) else "",
            ]
        )
    for lab in (canonical_case.get("labs") or {}).get("items") or []:
        parts.extend([clean_text(lab.get("name")), clean_text(lab.get("source_text"))])
    for imaging in (canonical_case.get("imaging") or {}).get("items") or []:
        parts.extend(
            [
                clean_text(imaging.get("modality")),
                flatten_text(imaging.get("findings")),
                clean_text(imaging.get("impression")),
            ]
        )
    for endoscopy in (canonical_case.get("endoscopy") or {}).get("items") or []:
        parts.extend([clean_text(endoscopy.get("type")), flatten_text(endoscopy.get("findings"))])
    for pathology in (canonical_case.get("pathology") or {}).get("items") or []:
        parts.extend([flatten_text(pathology.get("findings")), clean_text(pathology.get("diagnosis"))])
    return " ".join(part for part in parts if part)


def _match_feature(searchable_text: str, feature: Mapping[str, Any]) -> str | None:
    names = [clean_text(feature.get("name"))]
    names.extend(clean_text(item) for item in feature.get("synonyms") or [])
    for name in names:
        if text_contains_term(searchable_text, name):
            return name
    return None


def _match_symptom_feature(canonical_case: Mapping[str, Any], feature: Mapping[str, Any]) -> bool:
    feature_code = clean_text(feature.get("hpo_code"))
    if not feature_code:
        return False
    return any(
        _hpo_codes_match(clean_text(symptom.get("hpo_code")), feature_code)
        for symptom in canonical_case.get("symptoms") or []
        if isinstance(symptom, Mapping)
    )


def _match_icd10_feature(canonical_case: Mapping[str, Any], feature: Mapping[str, Any]) -> bool:
    feature_code = _feature_diagnosis_code(feature)
    if not feature_code:
        return False
    return any(
        clean_text(code) == feature_code
        for code in _case_diagnosis_codes(canonical_case)
    )


def _feature_diagnosis_code(feature: Mapping[str, Any]) -> str:
    icd10_mapping = feature.get("icd10_mapping")
    if isinstance(icd10_mapping, Mapping):
        return clean_text(icd10_mapping.get("diagnosis_code"))
    return clean_text(feature.get("diagnosis_code"))


def _case_diagnosis_codes(canonical_case: Mapping[str, Any]) -> list[str]:
    codes: list[str] = []
    for feature in canonical_case.get("features") or []:
        if not isinstance(feature, Mapping):
            continue
        code = _feature_diagnosis_code(feature)
        if code:
            codes.append(code)
    for diagnosis in canonical_case.get("diagnoses") or []:
        if not isinstance(diagnosis, Mapping):
            continue
        code = _feature_diagnosis_code(diagnosis)
        if code:
            codes.append(code)
    return codes


def _hpo_codes_match(case_code: str, feature_code: str) -> bool:
    if not case_code or not feature_code:
        return False
    if case_code == feature_code:
        return True
    code_terms = _load_hpo_code_terms()
    return bool(code_terms.get(case_code) and code_terms.get(case_code) == code_terms.get(feature_code))


@lru_cache(maxsize=1)
def _load_hpo_code_terms(path: Path = DEFAULT_DEFINITION2ID_PATH) -> dict[str, str]:
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        return {}

    code_terms: dict[str, str] = {}
    for term, value in data.items():
        term_text = clean_text(term)
        if not term_text:
            continue
        if isinstance(value, Mapping):
            raw_codes = value.get("hpo_ids")
            if raw_codes is None:
                raw_codes = value.get("hpo_id") or value.get("id")
            codes = raw_codes if isinstance(raw_codes, list) else [raw_codes]
        else:
            codes = value if isinstance(value, list) else [value]
        for code in codes:
            code_text = clean_text(code)
            if code_text and code_text not in code_terms:
                code_terms[code_text] = term_text
    return code_terms


def _matched_feature_payload(feature: Mapping[str, Any]) -> dict[str, Any]:
    if "diagnosis_code" in feature or "icd10_mapping" in feature:
        return pick_icd_feature_payload(feature)
    return pick_hpo_feature_payload(feature)


def _normalize_score(score: float, scoring: Mapping[str, Any]) -> float:
    if clean_text(scoring.get("normalization")) in {"sum_max_1", "", "clip_0_1"}:
        return round(max(0.0, min(score, 1.0)), 4)
    return round(max(0.0, score), 4)


def _candidate_threshold(routing: Mapping[str, Any], *, min_score: float | None) -> float:
    if min_score is not None:
        return float(min_score)
    thresholds = (routing.get("scoring") or {}).get("thresholds") or {}
    return float(thresholds.get("candidate", 0.0))


def _strong_threshold(routing: Mapping[str, Any]) -> float:
    thresholds = (routing.get("scoring") or {}).get("thresholds") or {}
    return float(thresholds.get("strong_candidate", 1.0))


def _default_top_k(skill_packs: list[SkillPack]) -> int:
    values = [
        int(((pack.skill.get("routing_profile") or {}).get("scoring") or {}).get("top_k_default") or 0)
        for pack in skill_packs
    ]
    return max([3, *values])


def _missing_key_evidence(canonical_case: Mapping[str, Any], pack: SkillPack) -> list[str]:
    missing: list[str] = []
    for subskill in pack.skill.get("subskills") or []:
        if not isinstance(subskill, Mapping):
            continue
        if clean_text(subskill.get("subskill_id")) != "diagnostic_integration":
            continue
        requirements = (subskill.get("input_requirements") or {}).get("required_for_high_confidence") or []
        for requirement in requirements:
            if not isinstance(requirement, Mapping):
                continue
            path = clean_text(requirement.get("path"))
            label = clean_text(requirement.get("label")) or path
            if path and not is_present(resolve_case_path(canonical_case, path)):
                missing.append(label)
    return missing


def _reasoning_summary(
    score: float,
    threshold: float,
    matched_features: list[dict[str, Any]],
    penalties: list[str],
) -> str:
    if not matched_features:
        return (
            "No declared positive routing features matched; returned as a candidate "
            "only because top-k fallback is enabled."
        )
    names = ", ".join(feature["name"] for feature in matched_features[:5])
    penalty_text = f" Penalties: {', '.join(penalties)}." if penalties else ""
    return f"Matched declared features: {names}. Score {score:.2f}, candidate threshold {threshold:.2f}.{penalty_text}"
