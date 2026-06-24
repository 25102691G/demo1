from __future__ import annotations

import json
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

try:
    from rapidfuzz import fuzz
except Exception:
    fuzz = None

from .hpo_extractor import DEFAULT_DEFINITION2ID_PATH
from .hpo_features import pick_hpo_feature_payload
from .icd_extractor import (
    DEFAULT_MODEL_PATH as DEFAULT_ICD_MODEL_PATH,
    _embedding_pooling_mode,
    _load_torch,
    _load_transformers,
    _pool_embeddings,
)
from .icd_features import pick_icd_feature_payload
from .skill_loader import SkillPack
from .utils import clean_text, dedupe_texts, flatten_text, is_present, normalize_key, resolve_case_path, text_contains_term


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LITERAL_MATCH_THRESHOLD = 0.8
DEFAULT_SEMANTIC_MATCH_THRESHOLD = 0.8
DEFAULT_MEDICAL_EXAMINATION_MATCH_THRESHOLD = 0.75
MEDICAL_EXAMINATION_CASE_FIELDS = {
    "实验室检查": ("lab_tests", "raw_input"),
    "影像学检查": ("imaging_tests", "raw_input"),
    "内镜检查": ("endoscopy", "raw_input"),
    "病理": ("pathology", "raw_input"),
    "综合诊断": (
        "clinical_presentation",
        "lab_tests",
        "imaging_tests",
        "endoscopy",
        "pathology",
        "raw_input",
    ),
}

def route_skills(
    canonical_case: dict[str, Any],
    skill_packs: list[SkillPack],
    *,
    top_k: int | None = None,
    min_score: float | None = None,
    feature_mode: str = "hpo",
    semantic_resources: Any | None = None,
) -> list[dict[str, Any]]:
    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be at least 1")
    candidates = []
    for index, pack in enumerate(skill_packs, start=1):
        candidate = _score_skill(
            canonical_case,
            pack,
            min_score=min_score,
            feature_mode=feature_mode,
            semantic_resources=semantic_resources,
        )
        candidates.append(candidate)
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
    semantic_resources: Any | None,
) -> dict[str, Any]:
    routing = pack.skill.get("routing_profile") or {}
    positive_features = routing.get("positive_features") or {}
    negative_features = routing.get("negative_features") or []
    evidence_mapping = _evidence_mapping(routing)
    matched_positive_features: list[dict[str, Any]] = []
    raw_score = 0.0

    for feature in _iter_features(positive_features):
        match = _match_routing_feature(
            canonical_case,
            feature,
            feature_mode,
            semantic_resources=semantic_resources,
        )
        if not match:
            continue
        weight = float(feature.get("weight") or 0.2)
        similarity_score = float(match["similarity_score"])
        evidence_quality_score = _feature_evidence_quality_score(feature, evidence_mapping)
        raw_score += weight * similarity_score * evidence_quality_score
        matched_payload = _matched_feature_payload(feature, match)
        matched_payload["evidence_quality_score"] = evidence_quality_score
        matched_positive_features.append(matched_payload)

    matched_negative_features: list[dict[str, Any]] = []
    # TODO：阴性症状分数计算方式需要修改（目前会出现skill中无腹痛匹配到case中腹痛，不合理）
    # for feature in _iter_features(negative_features):
    #     match = _match_routing_feature(
    #         canonical_case,
    #         feature,
    #         feature_mode,
    #         semantic_resources=semantic_resources,
    #     )
    #     if not match:
    #         continue
    #     weight = float(feature.get("weight") or -0.1)
    #     similarity_score = float(match["similarity_score"])
    #     evidence_quality_score = _feature_evidence_quality_score(feature, evidence_mapping)
    #     raw_score += weight * similarity_score * evidence_quality_score
    #     matched_payload = _matched_feature_payload(feature, match)
    #     matched_payload["evidence_quality_score"] = evidence_quality_score
    #     matched_negative_features.append(matched_payload)

    # score = _normalize_score(raw_score - penalty_score, routing.get("scoring") or {})
    score = raw_score
    threshold = _candidate_threshold(routing, min_score=min_score)
    missing_key_evidence = _missing_key_evidence(canonical_case, pack)
    missing_medical_examinations = _missing_medical_examinations(
        canonical_case,
        pack,
        semantic_resources=semantic_resources,
    )
    disease_name = pack.disease_name or clean_text((routing.get("disease_identity") or {}).get("primary_name"))
    return {
        "skill_id": pack.skill_id,
        "disease_name": disease_name,
        "score": score,
        "rank": 0,
        "matched_positive_features": matched_positive_features,
        "matched_negative_features": matched_negative_features,
        "missing_key_evidence": missing_key_evidence,
        "missing_medical_examinations": missing_medical_examinations,
        "reasoning_summary": _reasoning_summary(
            score,
            threshold,
            matched_positive_features,
            matched_negative_features,
        ),
        "candidate_threshold": threshold,
        "strong_candidate_threshold": _strong_threshold(routing),
        "raw_score": raw_score,
    }


def _iter_features(features_value: Any) -> list[Mapping[str, Any]]:
    if isinstance(features_value, list):
        return [feature for feature in features_value if isinstance(feature, Mapping)]
    if isinstance(features_value, Mapping):
        return [
            feature
            for features in features_value.values()
            if isinstance(features, list)
            for feature in features
            if isinstance(feature, Mapping)
        ]
    return []


def _evidence_mapping(routing: Mapping[str, Any]) -> Mapping[str, Any]:
    scoring = routing.get("scoring")
    if not isinstance(scoring, Mapping):
        return {}
    mapping = scoring.get("mapping")
    return mapping if isinstance(mapping, Mapping) else {}


def _feature_evidence_quality_score(
    feature: Mapping[str, Any],
    evidence_mapping: Mapping[str, Any],
    default: float = 0.5,
) -> float:
    scores = [
        score
        for card_id in _feature_card_ids(feature)
        if (score := _card_evidence_quality_score(evidence_mapping.get(card_id))) is not None
    ]
    return max(scores) if scores else default


def _feature_card_ids(feature: Mapping[str, Any]) -> list[str]:
    value = feature.get("card_id")
    if isinstance(value, list):
        return dedupe_texts(clean_text(item) for item in value)
    card_id_value = clean_text(value)
    return [card_id_value] if card_id_value else []


def _card_evidence_quality_score(evidence: Any) -> float | None:
    if not isinstance(evidence, Mapping):
        return None
    value = evidence.get("evidence_quality_normalized")
    if value is None:
        return None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(score, 1.0))


def _match_routing_feature(
    canonical_case: Mapping[str, Any],
    feature: Mapping[str, Any],
    feature_mode: str,
    *,
    semantic_resources: Any | None = None,
) -> dict[str, Any] | None:
    if feature_mode == "hpo":
        return _match_hpo_feature(canonical_case, feature, semantic_resources=semantic_resources)
    if feature_mode == "icd10":
        return _match_icd10_feature(canonical_case, feature)
    raise ValueError(f"unsupported feature mode: {feature_mode}")


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


def _match_hpo_feature(
    canonical_case: Mapping[str, Any],
    feature: Mapping[str, Any],
    *,
    semantic_resources: Any | None,
) -> dict[str, Any] | None:
    exact_match = _match_hpo_exact_feature(canonical_case, feature)
    if exact_match:
        return exact_match

    literal_match = _match_hpo_literal_feature(canonical_case, feature)
    if literal_match:
        return literal_match

    semantic_match = _match_hpo_semantic_feature(
        canonical_case,
        feature,
        semantic_resources=semantic_resources,
    )
    if semantic_match:
        return semantic_match

    return _match_hpo_code_feature(canonical_case, feature)


def _match_hpo_exact_feature(
    canonical_case: Mapping[str, Any],
    feature: Mapping[str, Any],
) -> dict[str, Any] | None:
    return _exact_text_match(_case_symptom_text_features(canonical_case), _routing_hpo_feature_texts(feature))


def _match_hpo_literal_feature(
    canonical_case: Mapping[str, Any],
    feature: Mapping[str, Any],
) -> dict[str, Any] | None:
    return _literal_text_match(_case_symptom_text_features(canonical_case), _routing_hpo_feature_texts(feature))


def _match_hpo_semantic_feature(
    canonical_case: Mapping[str, Any],
    feature: Mapping[str, Any],
    *,
    semantic_resources: Any | None,
) -> dict[str, Any] | None:
    return _semantic_text_match(
        _case_symptom_text_features(canonical_case),
        _routing_hpo_feature_texts(feature),
        matcher=_hpo_semantic_matcher(semantic_resources),
    )


def _match_hpo_code_feature(
    canonical_case: Mapping[str, Any],
    feature: Mapping[str, Any],
) -> dict[str, Any] | None:
    feature_code = clean_text(feature.get("hpo_code"))
    if not feature_code:
        return None
    for symptom in canonical_case.get("symptoms") or []:
        if not isinstance(symptom, Mapping):
            continue
        case_code = clean_text(symptom.get("hpo_code"))
        if _hpo_codes_match(case_code, feature_code):
            return _match_result(
                match_type="hpo_code",
                similarity_score=0.7,
                case_feature=symptom,
                matched_text=case_code,
                target_text=feature_code,
            )
    return None


def _match_icd10_feature(
    canonical_case: Mapping[str, Any],
    feature: Mapping[str, Any],
) -> dict[str, Any] | None:
    case_features = _case_text_features(canonical_case)
    target_texts = _routing_feature_texts(feature)
    if not case_features or not target_texts:
        return None

    exact_match = _exact_text_match(case_features, target_texts)
    if exact_match:
        return exact_match

    literal_match = _literal_text_match(case_features, target_texts)
    if literal_match:
        return literal_match

    return _semantic_text_match(case_features, target_texts)


def _case_text_features(canonical_case: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    features: list[Mapping[str, Any]] = []
    for feature in canonical_case.get("features") or []:
        if not isinstance(feature, Mapping):
            continue
        if clean_text(feature.get("name")):
            features.append(feature)
    return features


def _case_symptom_text_features(canonical_case: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    features: list[Mapping[str, Any]] = []
    for symptom in canonical_case.get("symptoms") or []:
        if not isinstance(symptom, Mapping):
            continue
        if clean_text(symptom.get("name")):
            features.append(symptom)
    return features


def _routing_feature_texts(feature: Mapping[str, Any]) -> list[str]:
    texts = [clean_text(feature.get("name"))]
    texts.extend(clean_text(item) for item in feature.get("synonyms") or [])
    icd10_mapping = feature.get("icd10_mapping")
    if isinstance(icd10_mapping, Mapping):
        texts.extend(
            [
                clean_text(icd10_mapping.get("diagnosis_name")),
                clean_text(icd10_mapping.get("category_name")),
                clean_text(icd10_mapping.get("section_name")),
            ]
        )
    return dedupe_texts(texts)


def _routing_hpo_feature_texts(feature: Mapping[str, Any]) -> list[str]:
    return dedupe_texts([clean_text(feature.get("name"))])


def _exact_text_match(
    case_features: list[Mapping[str, Any]],
    target_texts: list[str],
) -> dict[str, Any] | None:
    for case_feature in case_features:
        case_text = clean_text(case_feature.get("name"))
        case_key = normalize_key(case_text)
        if not case_key:
            continue
        for target_text in target_texts:
            if case_key == normalize_key(target_text):
                return _match_result(
                    match_type="exact",
                    similarity_score=1.0,
                    case_feature=case_feature,
                    matched_text=case_text,
                    target_text=target_text,
                )
    return None


def _literal_text_match(
    case_features: list[Mapping[str, Any]],
    target_texts: list[str],
) -> dict[str, Any] | None:
    best_match: dict[str, Any] | None = None
    for case_feature in case_features:
        case_text = clean_text(case_feature.get("name"))
        if not case_text:
            continue
        for target_text in target_texts:
            score = _literal_similarity(case_text, target_text)
            if score < DEFAULT_LITERAL_MATCH_THRESHOLD:
                continue
            if best_match is None or score > best_match["similarity_score"]:
                best_match = _match_result(
                    match_type="literal",
                    similarity_score=score,
                    case_feature=case_feature,
                    matched_text=case_text,
                    target_text=target_text,
                )
    return best_match


def _literal_similarity(left: str, right: str) -> float:
    left_key = normalize_key(left)
    right_key = normalize_key(right)
    if not left_key or not right_key:
        return 0.0
    contains_score = 0.95 if left_key in right_key or right_key in left_key else 0.0
    fuzzy_score = 0.0
    if fuzz is not None:
        fuzzy_score = float(fuzz.WRatio(left, right) or 0.0) / 100.0
    return max(contains_score, fuzzy_score)


def _semantic_text_match(
    case_features: list[Mapping[str, Any]],
    target_texts: list[str],
    *,
    matcher: Any | None = None,
) -> dict[str, Any] | None:
    if matcher is None:
        matcher = _semantic_matcher()
    if matcher is None:
        return None
    best_match: dict[str, Any] | None = None
    for case_feature in case_features:
        case_text = clean_text(case_feature.get("name"))
        if not case_text:
            continue
        semantic_match = matcher.best_match(case_text, target_texts)
        if semantic_match is None:
            continue
        target_text, score = semantic_match
        if score < DEFAULT_SEMANTIC_MATCH_THRESHOLD:
            continue
        if best_match is None or score > best_match["similarity_score"]:
            best_match = _match_result(
                match_type="semantic",
                similarity_score=score,
                case_feature=case_feature,
                matched_text=case_text,
                target_text=target_text,
            )
    return best_match


def _match_result(
    *,
    match_type: str,
    similarity_score: float,
    case_feature: Mapping[str, Any],
    matched_text: str,
    target_text: str,
) -> dict[str, Any]:
    return {
        "match_type": match_type,
        "similarity_score": max(0.0, min(float(similarity_score), 1.0)),
        "case_feature": dict(case_feature),
        "matched_text": matched_text,
        "target_text": target_text,
    }


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


class _SemanticTextMatcher:
    def __init__(
        self,
        *,
        model_path: Path = DEFAULT_ICD_MODEL_PATH,
        batch_size: int = 16,
        max_length: int = 128,
    ) -> None:
        self.model_path = model_path
        self.batch_size = batch_size
        self.max_length = max_length
        self._disabled = False
        self._torch: Any | None = None
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._device: Any | None = None
        self._pooling_mode = ""
        self._embedding_cache: dict[tuple[str, ...], Any] = {}

    def best_match(self, query: str, targets: list[str]) -> tuple[str, float] | None:
        target_texts = dedupe_texts(targets)
        if not clean_text(query) or not target_texts:
            return None
        try:
            query_embeddings = self._embed([query])
            target_embeddings = self._embed(target_texts)
            torch = self._torch
            similarities = torch.matmul(query_embeddings, target_embeddings.T)
            values, indices = torch.max(similarities, dim=1)
            index = int(indices[0].item())
            score = max(0.0, min(float(values[0].item()), 1.0))
            return target_texts[index], score
        except Exception:
            self._disabled = True
            return None

    def _embed(self, texts: list[str]) -> Any:
        key = tuple(texts)
        if key in self._embedding_cache:
            return self._embedding_cache[key]
        self._ensure_loaded()
        if self._disabled:
            raise RuntimeError("semantic matcher is unavailable")

        torch = self._torch
        batches = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            inputs = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self._device)
            with torch.no_grad():
                outputs = self._model(**inputs)
            batch_embeddings = _pool_embeddings(
                outputs.last_hidden_state,
                inputs["attention_mask"],
                pooling_mode=self._pooling_mode,
                torch=torch,
            )
            batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1).float()
            batches.append(batch_embeddings.cpu())

        embeddings = torch.cat(batches, dim=0)
        self._embedding_cache[key] = embeddings
        return embeddings

    def _ensure_loaded(self) -> None:
        if self._disabled:
            raise RuntimeError("semantic matcher is unavailable")
        if self._model is not None:
            return
        torch = _load_torch()
        AutoTokenizer, AutoModel = _load_transformers()
        pooling_mode = _embedding_pooling_mode(self.model_path)
        tokenizer_kwargs: dict[str, Any] = {"local_files_only": True}
        if pooling_mode == "last_token":
            tokenizer_kwargs["padding_side"] = "left"
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = AutoTokenizer.from_pretrained(str(self.model_path), **tokenizer_kwargs)
        self._model = AutoModel.from_pretrained(str(self.model_path), local_files_only=True).to(
            self._device
        )
        self._model.eval()
        self._torch = torch
        self._pooling_mode = pooling_mode


class _ReusableSemanticTextMatcher:
    def __init__(
        self,
        resources: Any,
        *,
        batch_size: int = 16,
        max_length: int = 128,
    ) -> None:
        self.resources = resources
        self.batch_size = batch_size
        self.max_length = max_length
        self._disabled = False
        self._torch: Any | None = None
        self._device: Any | None = None
        self._model: Any | None = None
        self._embedding_cache: dict[tuple[str, ...], Any] = {}

    def best_match(self, query: str, targets: list[str]) -> tuple[str, float] | None:
        target_texts = dedupe_texts(targets)
        if not clean_text(query) or not target_texts:
            return None
        try:
            query_embeddings = self._embed([query])
            target_embeddings = self._embed(target_texts)
            torch = self._torch
            similarities = torch.matmul(query_embeddings, target_embeddings.T)
            values, indices = torch.max(similarities, dim=1)
            index = int(indices[0].item())
            score = max(0.0, min(float(values[0].item()), 1.0))
            return target_texts[index], score
        except Exception:
            self._disabled = True
            return None

    def _embed(self, texts: list[str]) -> Any:
        key = tuple(texts)
        if key in self._embedding_cache:
            return self._embedding_cache[key]
        self._ensure_ready()
        if self._disabled:
            raise RuntimeError("semantic matcher is unavailable")

        torch = self._torch
        tokenizer = self.resources.tokenizer
        model = self._model
        pooling_mode = clean_text(getattr(self.resources, "pooling_mode", ""))
        batches = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self._device)
            with torch.no_grad():
                outputs = model(**inputs)
            batch_embeddings = _pool_embeddings(
                outputs.last_hidden_state,
                inputs["attention_mask"],
                pooling_mode=pooling_mode,
                torch=torch,
            )
            batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1).float()
            batches.append(batch_embeddings.cpu())

        embeddings = torch.cat(batches, dim=0)
        self._embedding_cache[key] = embeddings
        return embeddings

    def _ensure_ready(self) -> None:
        if self._disabled:
            raise RuntimeError("semantic matcher is unavailable")
        if self._torch is not None:
            return
        torch = _load_torch()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = self.resources.model.to(self._device)
        self._model.eval()
        self._torch = torch


@lru_cache(maxsize=1)
def _semantic_matcher() -> _SemanticTextMatcher | None:
    return _SemanticTextMatcher()


_HPO_SEMANTIC_MATCHERS: dict[int, _ReusableSemanticTextMatcher] = {}


def _hpo_semantic_matcher(resources: Any | None) -> _ReusableSemanticTextMatcher | None:
    if resources is None:
        return None
    key = id(resources)
    matcher = _HPO_SEMANTIC_MATCHERS.get(key)
    if matcher is None:
        matcher = _ReusableSemanticTextMatcher(resources)
        _HPO_SEMANTIC_MATCHERS[key] = matcher
    return matcher


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


def _matched_feature_payload(
    feature: Mapping[str, Any],
    match: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "skill_feature": _feature_evidence_payload(feature),
    }
    if match:
        case_feature = match.get("case_feature")
        if isinstance(case_feature, Mapping):
            payload["case_evidence"] = _feature_evidence_payload(case_feature)
        payload["match"] = {
            "type": clean_text(match.get("match_type")),
            "score": match.get("similarity_score"),
            "case_text": clean_text(match.get("matched_text")),
            "skill_text": clean_text(match.get("target_text")),
        }
    return payload


def _feature_evidence_payload(feature: Mapping[str, Any]) -> dict[str, Any]:
    if "diagnosis_code" in feature or "icd10_mapping" in feature:
        payload = pick_icd_feature_payload(feature)
    else:
        payload = pick_hpo_feature_payload(feature)
    if "similarity_score" in payload:
        payload["mapping_score"] = payload.pop("similarity_score")
    return payload


def _matched_feature_name(feature: Mapping[str, Any]) -> str:
    skill_feature = feature.get("skill_feature")
    if isinstance(skill_feature, Mapping):
        return clean_text(skill_feature.get("name"))
    return clean_text(feature.get("name"))


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


def _missing_medical_examinations(
    canonical_case: Mapping[str, Any],
    pack: SkillPack,
    *,
    semantic_resources: Any | None,
) -> list[dict[str, Any]]:
    medical_examinations = pack.skill.get("medical_examinations")
    if not isinstance(medical_examinations, Mapping):
        return []
    matcher = _hpo_semantic_matcher(semantic_resources) if semantic_resources is not None else _semantic_matcher()
    if matcher is None:
        return []

    case_texts = _case_medical_examination_texts(canonical_case)
    missing: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for clinical_task, raw_items in medical_examinations.items():
        task = clean_text(clinical_task)
        if task not in MEDICAL_EXAMINATION_CASE_FIELDS or not isinstance(raw_items, list):
            continue
        comparable_case_texts = [
            item
            for item in (
                case_texts.get(field)
                for field in MEDICAL_EXAMINATION_CASE_FIELDS[task]
            )
            if item["text"]
        ]
        for item in raw_items:
            if not isinstance(item, Mapping):
                continue
            context_texts = _medical_examination_context_texts(item)
            examination = _medical_examination_name(item)
            if not examination:
                continue
            query_text = " ".join([examination, *context_texts]).strip()
            best_field, best_score = _best_medical_examination_case_match(
                query_text,
                comparable_case_texts,
                matcher,
            )
            if best_score >= DEFAULT_MEDICAL_EXAMINATION_MATCH_THRESHOLD:
                continue
            key = (task, examination, _source_cards_key(item.get("source_cards")))
            if key in seen:
                continue
            seen.add(key)
            missing.append(
                {
                    "clinical_task": task,
                    "examination": examination,
                    "source_cards": _medical_examination_source_cards(item),
                    "matched_case_field": best_field,
                    "similarity_score": round(best_score, 4),
                    "reason": "skill 中建议该检查，但 canonical_case.raw 中未匹配到相近内容",
                }
                )
    return missing


def _case_medical_examination_texts(canonical_case: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    raw_input = canonical_case.get("raw_input")
    raw_values = raw_input if isinstance(raw_input, Mapping) else {}
    result: dict[str, dict[str, str]] = {}
    for field in (
        "clinical_presentation",
        "lab_tests",
        "imaging_tests",
        "endoscopy",
        "pathology",
    ):
        result[field] = {
            "field": field,
            "text": clean_text(raw_values.get(field)) or clean_text(canonical_case.get(field)),
        }
    result["raw_input"] = {
        "field": "raw_input",
        "text": flatten_text(raw_input),
    }
    return result


def _medical_examination_name(item: Mapping[str, Any]) -> str:
    return clean_text(item.get("examination"))


def _medical_examination_source_cards(item: Mapping[str, Any]) -> list[dict[str, Any]]:
    source_cards = []
    for source in item.get("source_cards") or []:
        if not isinstance(source, Mapping):
            continue
        card_id = clean_text(source.get("card_id"))
        if not card_id:
            continue
        source_cards.append(
            {
                "card_id": card_id,
                "recommendation_label": clean_text(source.get("recommendation_label")) or None,
            }
        )
    return source_cards


def _source_cards_key(value: Any) -> str:
    if not isinstance(value, list):
        return ""
    return "|".join(
        dedupe_texts(
            clean_text(source.get("card_id"))
            for source in value
            if isinstance(source, Mapping) and clean_text(source.get("card_id"))
        )
    )


def _medical_examination_context_texts(item: Mapping[str, Any]) -> list[str]:
    texts: list[str] = []
    for field in ("key_symptoms", "attention_points"):
        texts.extend(clean_text(value) for value in item.get(field) or [] if clean_text(value))
    return dedupe_texts(texts)


def _best_medical_examination_case_match(
    query_text: str,
    case_texts: list[dict[str, str]],
    matcher: Any,
) -> tuple[str | None, float]:
    best_field: str | None = None
    best_score = 0.0
    for case_text in case_texts:
        semantic_match = matcher.best_match(case_text["text"], [query_text])
        if semantic_match is None:
            continue
        _target, score = semantic_match
        if score > best_score:
            best_score = score
            best_field = case_text["field"]
    return best_field, best_score


def _reasoning_summary(
    score: float,
    threshold: float,
    matched_positive_features: list[dict[str, Any]],
    matched_negative_features: list[dict[str, Any]],
) -> str:
    if not matched_positive_features:
        return (
            "No declared positive routing features matched; returned as a candidate "
            "only because top-k fallback is enabled."
        )
    names = ", ".join(
        name
        for name in (_matched_feature_name(feature) for feature in matched_positive_features[:5])
        if name
    )
    negative_names = [
        name
        for name in (_matched_feature_name(feature) for feature in matched_negative_features[:5])
        if name
    ]
    penalty_text = f" Penalties: {', '.join(negative_names)}." if negative_names else ""
    return f"Matched declared features: {names}. Score {score:.2f}, candidate threshold {threshold:.2f}.{penalty_text}"
