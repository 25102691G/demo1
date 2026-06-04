from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]


class BuildSkillPackError(ValueError):
    """Raised when a skill pack cannot be built from the supplied cards."""


@dataclass(frozen=True)
class LoadedCard:
    path: Path
    line_no: int
    data: dict[str, Any]


@dataclass(frozen=True)
class TaxonomyEntry:
    subskill_id: str
    name: str
    type: str
    keywords: tuple[str, ...]


DEFAULT_TAXONOMY: "OrderedDict[str, TaxonomyEntry]" = OrderedDict(
    [
        (
            "initial_triage",
            TaxonomyEntry(
                subskill_id="initial_triage",
                name="Initial triage",
                type="safety_triage",
                keywords=("初筛", "急诊", "危险", "警示", "红旗", "转诊"),
            ),
        ),
        (
            "diagnostic_integration",
            TaxonomyEntry(
                subskill_id="diagnostic_integration",
                name="Diagnostic integration",
                type="evidence_check",
                keywords=("诊断", "检查", "实验室", "内镜", "影像", "病理", "综合判断"),
            ),
        ),
        (
            "differential_diagnosis",
            TaxonomyEntry(
                subskill_id="differential_diagnosis",
                name="Differential diagnosis",
                type="differential_check",
                keywords=("鉴别", "排除", "区别", "误诊"),
            ),
        ),
        (
            "staging_assessment",
            TaxonomyEntry(
                subskill_id="staging_assessment",
                name="Staging and assessment",
                type="evidence_check",
                keywords=("分型", "分期", "活动度", "严重程度", "并发症", "风险评估"),
            ),
        ),
        (
            "management_plan",
            TaxonomyEntry(
                subskill_id="management_plan",
                name="Management plan",
                type="plan_generation",
                keywords=("治疗", "药物", "手术", "营养", "诱导", "维持", "管理", "方案"),
            ),
        ),
        (
            "monitoring_follow_up",
            TaxonomyEntry(
                subskill_id="monitoring_follow_up",
                name="Monitoring and follow-up",
                type="plan_generation",
                keywords=("监测", "随访", "复查", "复发", "长期管理"),
            ),
        ),
        (
            "patient_support",
            TaxonomyEntry(
                subskill_id="patient_support",
                name="Patient support",
                type="plan_generation",
                keywords=("患者教育", "生活方式", "心理", "依从性", "饮食"),
            ),
        ),
        (
            "general_guideline_support",
            TaxonomyEntry(
                subskill_id="general_guideline_support",
                name="General guideline support",
                type="card_retrieval",
                keywords=(),
            ),
        ),
    ]
)

WORKFLOW_REQUIRED_SUBSKILLS = (
    "initial_triage",
    "diagnostic_integration",
    "differential_diagnosis",
    "staging_assessment",
    "management_plan",
)

CARD_TEXT_FIELDS_FOR_CLASSIFICATION = (
    "clinical_task",
    "clinical_stage",
    "recommendation_label",
    "condition",
    "action",
    "statement_text",
)

LAB_KEYWORDS = ("血", "CRP", "ESR", "粪便", "指标", "抗体", "生化", "白蛋白", "实验室", "钙卫蛋白")
IMAGING_KEYWORDS = ("CT", "MRI", "MRE", "CTE", "超声", "影像", "造影")
ENDOSCOPY_KEYWORDS = ("内镜", "结肠镜", "胃镜", "肠镜", "小肠镜")
PATHOLOGY_KEYWORDS = ("病理", "活检", "活体", "组织学")
EMERGENCY_KEYWORDS = (
    "急诊",
    "急症",
    "急性",
    "急腹症",
    "大出血",
    "出血",
    "穿孔",
    "梗阻",
    "休克",
    "脓毒症",
    "中毒",
    "危及生命",
    "立即",
    "紧急",
)
DIFFERENTIAL_KEYWORDS = ("鉴别", "排除", "区别", "需排除")


def load_jsonl(path: str | Path) -> list[LoadedCard]:
    cards_path = Path(path)
    if not cards_path.exists():
        raise BuildSkillPackError(f"cards file does not exist: {cards_path}")
    if not cards_path.is_file():
        raise BuildSkillPackError(f"cards path is not a file: {cards_path}")

    loaded: list[LoadedCard] = []
    with cards_path.open("r", encoding="utf-8-sig") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise BuildSkillPackError(
                    f"{cards_path}: line {line_no}: invalid JSON: {exc.msg}"
                ) from exc
            if not isinstance(payload, dict):
                raise BuildSkillPackError(
                    f"{cards_path}: line {line_no}: expected JSON object, got {type(payload).__name__}"
                )
            loaded.append(LoadedCard(path=cards_path, line_no=line_no, data=payload))

    if not loaded:
        raise BuildSkillPackError(f"{cards_path}: no cards found")
    return loaded


def discover_cards_sources(path: str | Path) -> list[Path]:
    cards_path = Path(path)
    if not cards_path.exists():
        raise BuildSkillPackError(f"cards path does not exist: {cards_path}")
    if cards_path.is_file():
        if cards_path.suffix.lower() != ".jsonl":
            raise BuildSkillPackError(f"cards file must be a .jsonl file: {cards_path}")
        return [cards_path]
    if not cards_path.is_dir():
        raise BuildSkillPackError(f"cards path is neither a file nor a directory: {cards_path}")

    sources = sorted(
        child
        for child in cards_path.rglob("*.jsonl")
        if child.is_file() and not child.name.startswith(".")
    )
    if not sources:
        raise BuildSkillPackError(f"{cards_path}: no .jsonl files found")
    return sources


def validate_cards(
    loaded_cards: Sequence[LoadedCard],
    card_schema_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    seen: dict[str, LoadedCard] = {}
    schema: Mapping[str, Any] | None = None
    validator: Any | None = None

    if card_schema_path:
        schema_path = Path(card_schema_path)
        if schema_path.exists():
            schema = _load_json(schema_path)
            validator = _make_jsonschema_validator(schema, schema_path)

    cards: list[dict[str, Any]] = []
    for loaded in loaded_cards:
        card = normalize_card_payload(loaded.data)
        card_id = _clean_text(card.get("card_id"))
        if not card_id:
            raise BuildSkillPackError(
                f"{loaded.path}: line {loaded.line_no}: missing required card_id"
            )
        if card_id in seen:
            previous = seen[card_id]
            raise BuildSkillPackError(
                f"{loaded.path}: line {loaded.line_no}: duplicate card_id {card_id!r}; "
                f"first seen at {previous.path}: line {previous.line_no}"
            )
        seen[card_id] = loaded

        if validator is not None:
            errors = sorted(validator.iter_errors(card), key=lambda error: list(error.path))
            if errors:
                error = errors[0]
                validation_path = _format_validation_path(error.path)
                raise BuildSkillPackError(
                    f"{loaded.path}: line {loaded.line_no}: card_id {card_id!r}: "
                    f"card schema validation failed at {validation_path}: {error.message}"
                )
        cards.append(dict(card))

    return cards


def normalize_card_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    record_type = _clean_text(payload.get("record_type"))
    if record_type == "recommendation_card" or "card_id" in payload:
        return dict(payload)
    unit = payload.get("unit")
    if not isinstance(unit, Mapping):
        return dict(payload)
    if record_type == "clinical_info_unit":
        return _clinical_info_unit_to_card(payload, unit)
    if record_type != "statement_unit":
        return dict(payload)

    return _statement_unit_to_card(payload, unit)


def _statement_unit_to_card(payload: Mapping[str, Any], unit: Mapping[str, Any]) -> dict[str, Any]:
    guideline = payload.get("guideline_meta") if isinstance(payload.get("guideline_meta"), Mapping) else {}
    source_location = unit.get("source_location") if isinstance(unit.get("source_location"), Mapping) else {}
    title = _clean_text(guideline.get("title"))
    section = _clean_text(source_location.get("section"))
    statement_text = _clean_text(unit.get("statement_text"))

    return {
        "record_type": "recommendation_card",
        "card_id": _clean_text(unit.get("id")),
        "source_statement_id": _first_text(unit.get("original_label"), unit.get("id")),
        "disease": _first_text(payload.get("disease"), _infer_disease_name_from_title(title)),
        "guideline": {
            "title": title or "unknown guideline",
            "source_file": guideline.get("source_file"),
            "doc_type": guideline.get("doc_type"),
        },
        "clinical_stage": section or "general_guideline_support",
        "clinical_task": unit.get("clinical_question"),
        "population": None,
        "condition": None,
        "action": statement_text or "See statement_text.",
        "do_not": [],
        "required_inputs": [],
        "supporting_features": [],
        "recommended_tests": [],
        "safety_notes": [],
        "evidence": {
            "evidence_quality_raw": unit.get("evidence_quality_raw"),
            "evidence_quality_normalized": _normalized_evidence_value(
                unit.get("evidence_quality_normalized")
            ),
            "recommendation_strength_raw": unit.get("strength_raw"),
            "recommendation_strength_normalized": _normalized_strength_value(
                unit.get("strength_normalized")
            ),
            "consensus_level": unit.get("consensus_level"),
        },
        "rationale": unit.get("rationale"),
        "source_location": {
            "pdf": guideline.get("source_file"),
            "page_start": _positive_int(source_location.get("page_start"), default=1),
            "page_end": _positive_int(
                source_location.get("page_end"),
                default=_positive_int(source_location.get("page_start"), default=1),
            ),
            "section": section or None,
        },
        "statement_text": statement_text,
        "needs_human_review": unit.get("needs_human_review", True),
        "review_reasons": unit.get("review_reasons", ["converted_from_statement_unit"]),
    }


def _clinical_info_unit_to_card(payload: Mapping[str, Any], unit: Mapping[str, Any]) -> dict[str, Any]:
    guideline = payload.get("guideline_meta") if isinstance(payload.get("guideline_meta"), Mapping) else {}
    source_location = unit.get("source_location") if isinstance(unit.get("source_location"), Mapping) else {}
    title = _clean_text(guideline.get("title"))
    section = _clean_text(source_location.get("section"))
    raw_text = _first_text(unit.get("raw_text"), unit.get("title"))
    action = _first_text(unit.get("action"), unit.get("title"), raw_text, "See raw_text.")
    clinical_topic = _clean_text(unit.get("clinical_topic"))

    return {
        "record_type": "recommendation_card",
        "card_id": _clean_text(unit.get("id")),
        "source_statement_id": _clean_text(unit.get("id")),
        "disease": _first_text(payload.get("disease"), _infer_disease_name_from_title(title)),
        "guideline": {
            "title": title or "unknown guideline",
            "source_file": guideline.get("source_file"),
            "doc_type": guideline.get("doc_type"),
        },
        "clinical_stage": section or clinical_topic or "general_guideline_support",
        "clinical_task": clinical_topic or unit.get("unit_type"),
        "population": None,
        "condition": unit.get("condition"),
        "action": action,
        "do_not": _as_text_list(unit.get("contraindication")),
        "required_inputs": [],
        "supporting_features": _as_text_list(unit.get("indication")),
        "recommended_tests": _recommended_tests_from_clinical_info(unit),
        "contraindications_or_cautions": _as_text_list(unit.get("contraindication")),
        "safety_notes": [],
        "evidence": {
            "evidence_quality_raw": None,
            "evidence_quality_normalized": "unknown",
            "recommendation_strength_raw": None,
            "recommendation_strength_normalized": "unknown",
            "consensus_level": None,
        },
        "rationale": None,
        "source_location": {
            "pdf": guideline.get("source_file"),
            "page_start": _positive_int(source_location.get("page_start"), default=1),
            "page_end": _positive_int(
                source_location.get("page_end"),
                default=_positive_int(source_location.get("page_start"), default=1),
            ),
            "section": section or None,
        },
        "statement_text": raw_text,
        "must_differentiate": _as_text_list(unit.get("differential_diagnosis")),
        "needs_human_review": unit.get("needs_human_review", True),
        "review_reasons": unit.get("review_reasons", ["converted_from_clinical_info_unit"]),
    }


def _recommended_tests_from_clinical_info(unit: Mapping[str, Any]) -> list[str]:
    unit_type = _clean_text(unit.get("unit_type"))
    if unit_type not in {"test_order", "instrumental_exam", "imaging_exam", "endoscopy_exam"}:
        return []
    return _dedupe_texts(
        _as_text_list(unit.get("title"))
        + _as_text_list(unit.get("raw_text"))
        + _as_text_list(unit.get("diagnostic_criteria"))
    )


def infer_metadata(cards: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    disease_name = _most_common_non_empty(_clean_text(card.get("disease")) for card in cards)
    if not disease_name:
        disease_name = "unknown disease"

    aliases = _dedupe_texts(
        [disease_name]
        + [
            alias
            for card in cards
            for field in ("disease_aliases", "aliases")
            for alias in _as_text_list(card.get(field))
        ]
    )

    guideline_name = _most_common_non_empty(
        _first_text(
            _path_value(card, "guideline_meta.title"),
            _path_value(card, "guideline.title"),
            card.get("source_guideline"),
            card.get("guideline_name"),
        )
        for card in cards
    )
    if not guideline_name:
        guideline_name = f"{disease_name} guideline"

    source_pdf = _most_common_non_empty(
        _first_text(
            _path_value(card, "guideline_meta.source_file"),
            _path_value(card, "guideline.source_file"),
            card.get("source_file"),
            _path_value(card, "source_location.pdf"),
        )
        for card in cards
    )
    if not source_pdf:
        source_pdf = "unknown"

    version = _extract_year(guideline_name) or _extract_year(source_pdf) or "unknown"
    publication_year = _extract_publication_year(guideline_name, source_pdf, version)
    skill_id = slugify_skill_id(disease_name, guideline_name, version, source_pdf)

    return {
        "skill_id": skill_id,
        "disease_name": disease_name,
        "disease_aliases": aliases,
        "guideline": {
            "name": guideline_name,
            "version": version,
            "source_pdf": source_pdf,
            "publication_year": publication_year,
            "language": "zh-CN",
        },
        "target_users": ["clinicians", "clinical decision support developers"],
        "intended_use": [
            "guideline-based clinical decision support",
            "recommendation retrieval",
            "structured guideline workflow execution",
        ],
        "scope": (
            "This skill pack is generated from recommendation cards in the input cards.jsonl "
            "and supports guideline retrieval, evidence checking, and management recommendation drafting."
        ),
        "disclaimer": "本工具仅用于指南知识组织和临床决策支持，不替代医生诊断、处方或急诊处理。",
    }


def slugify_skill_id(
    disease_name: str,
    guideline_name: str | None = None,
    guideline_version: str | None = None,
    source_pdf: str | None = None,
) -> str:
    disease_slug = _ascii_slug(disease_name)
    guideline_slug = _ascii_slug(guideline_name or "")
    version_slug = _ascii_slug(guideline_version or "")
    source_stem = Path(source_pdf).stem if source_pdf and source_pdf != "unknown" else ""
    source_slug = _ascii_slug(source_stem)
    source_text = " ".join(
        text
        for text in (disease_name, guideline_version, guideline_name, source_stem)
        if text and text != "unknown"
    )

    if re.search(r"[a-z]", disease_slug):
        slug = _join_slug_parts([disease_slug, version_slug, guideline_slug or source_slug])
    elif re.search(r"[a-z]", guideline_slug):
        slug = _join_slug_parts([guideline_slug, version_slug])
    else:
        slug = ""

    if slug and re.search(r"[a-z]", slug):
        return slug[:96].strip("_") or _hashed_skill_id(source_text)
    return _hashed_skill_id(source_text or disease_name)


def infer_output_package_name(cards_path: str | Path) -> str:
    path = Path(cards_path)
    generic_file_names = {"result", "cards", "recommendation_cards", "recommendations"}
    source_name = path.parent.name if path.stem.lower() in generic_file_names else path.stem
    package_name = _safe_path_name(source_name)
    if not package_name:
        package_name = _hashed_skill_id(str(path.resolve()))
    return package_name


def build_routing_profile(cards: Sequence[Mapping[str, Any]], metadata: Mapping[str, Any]) -> dict[str, Any]:
    positive_features: dict[str, list[dict[str, Any]]] = {
        "symptoms": [],
        "signs": [],
        "labs": [],
        "imaging": [],
        "endoscopy": [],
        "pathology": [],
        "findings": [],
    }
    feature_seen: dict[str, set[str]] = {key: set() for key in positive_features}

    negative_features: list[dict[str, Any]] = []
    negative_seen: set[str] = set()
    red_flags: list[dict[str, Any]] = []
    red_seen: set[str] = set()
    must_differentiate: list[dict[str, Any]] = []
    differential_seen: set[str] = set()

    disease_name = _clean_text(metadata.get("disease_name"))

    for card in cards:
        for feature in _as_text_list(card.get("supporting_features")):
            _add_feature(positive_features, feature_seen, "findings", feature, weight=0.2)

        for test in _as_text_list(card.get("recommended_tests")):
            bucket = _classify_test_feature(test)
            _add_feature(positive_features, feature_seen, bucket, test, weight=0.2)

        for field in ("do_not", "not_recommended", "avoid", "contraindications_or_cautions"):
            for value in _as_text_list(card.get(field)):
                key = _normalize_key(value)
                if not key or key in negative_seen:
                    continue
                negative_seen.add(key)
                negative_features.append(
                    {"name": value, "penalty": 0.3, "match_type": "semantic_or_exact"}
                )

        for value in _safety_candidate_texts(card):
            if _contains_any(value, EMERGENCY_KEYWORDS):
                key = _normalize_key(value)
                if key and key not in red_seen:
                    red_seen.add(key)
                    red_flags.append(
                        {
                            "name": _shorten_text(value, limit=80),
                            "severity": _red_flag_severity(value),
                            "action": "Recommend urgent medical evaluation.",
                        }
                    )

        for disease in _extract_differentials(card, current_disease=disease_name):
            key = _normalize_key(disease)
            if not key or key in differential_seen:
                continue
            differential_seen.add(key)
            must_differentiate.append(
                {
                    "disease_name": disease,
                    "reason": "Mentioned in guideline text as a condition to distinguish or exclude.",
                }
            )

    return {
        "routing_version": "0.1",
        "disease_identity": {
            "primary_name": metadata["disease_name"],
            "aliases": metadata["disease_aliases"],
            "abbreviations": [],
        },
        "positive_features": positive_features,
        "negative_features": negative_features,
        "red_flags": red_flags,
        "must_differentiate": must_differentiate,
        "scoring": {
            "method": "hybrid_weighted_semantic",
            "normalization": "sum_max_1",
            "thresholds": {
                "candidate": 0.35,
                "strong_candidate": 0.6,
                "very_likely": 0.8,
            },
            "top_k_default": 5,
            "safety_override": True,
        },
    }


def load_taxonomy(taxonomy_path: str | Path | None = None) -> "OrderedDict[str, TaxonomyEntry]":
    taxonomy: "OrderedDict[str, TaxonomyEntry]" = OrderedDict(DEFAULT_TAXONOMY)
    if not taxonomy_path:
        return taxonomy

    path = Path(taxonomy_path)
    if not path.exists():
        return taxonomy

    try:
        import yaml
    except ImportError as exc:
        raise BuildSkillPackError(
            "PyYAML is required to read taxonomy YAML files. Install PyYAML or omit --taxonomy."
        ) from exc

    with path.open("r", encoding="utf-8-sig") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, Mapping):
        raise BuildSkillPackError(f"{path}: taxonomy must be a YAML object")
    if any(key in raw for key in ("disease", "disease_name", "disease_aliases")):
        raise BuildSkillPackError(f"{path}: taxonomy must be generic and must not define disease identity")

    entries = raw.get("subskills", raw)
    if not isinstance(entries, Mapping):
        raise BuildSkillPackError(f"{path}: taxonomy subskills must be a mapping")

    for subskill_id, entry in entries.items():
        subskill_key = str(subskill_id)
        if subskill_key not in taxonomy:
            taxonomy[subskill_key] = TaxonomyEntry(
                subskill_id=subskill_key,
                name=_humanize_identifier(subskill_key),
                type="card_retrieval",
                keywords=(),
            )
        current = taxonomy[subskill_key]
        keywords, replace_keywords, name, subskill_type = _parse_taxonomy_entry(path, subskill_key, entry)
        merged_keywords = tuple(_dedupe_texts(keywords if replace_keywords else [*current.keywords, *keywords]))
        taxonomy[subskill_key] = TaxonomyEntry(
            subskill_id=subskill_key,
            name=name or current.name,
            type=subskill_type or current.type,
            keywords=merged_keywords,
        )
    return taxonomy


def classify_card_to_subskill(
    card: Mapping[str, Any],
    taxonomy: Mapping[str, TaxonomyEntry],
) -> str:
    for field_name in CARD_TEXT_FIELDS_FOR_CLASSIFICATION:
        text = _stringify_for_matching(card.get(field_name))
        if not text:
            continue
        text_lower = text.lower()
        for subskill_id, entry in taxonomy.items():
            if subskill_id == "general_guideline_support":
                continue
            for keyword in entry.keywords:
                if keyword.lower() in text_lower:
                    return subskill_id
    return "general_guideline_support"


def build_subskills(
    cards: Sequence[Mapping[str, Any]],
    taxonomy: Mapping[str, TaxonomyEntry],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for card in cards:
        grouped[classify_card_to_subskill(card, taxonomy)].append(card)

    subskill_ids: list[str] = []
    for subskill_id in taxonomy:
        if subskill_id in grouped or subskill_id in WORKFLOW_REQUIRED_SUBSKILLS:
            subskill_ids.append(subskill_id)
    if not subskill_ids:
        subskill_ids.append("general_guideline_support")

    subskills: list[dict[str, Any]] = []
    for subskill_id in subskill_ids:
        entry = taxonomy.get(
            subskill_id,
            TaxonomyEntry(subskill_id, _humanize_identifier(subskill_id), "card_retrieval", ()),
        )
        selected_cards = grouped.get(subskill_id, [])
        subskills.append(
            {
                "subskill_id": subskill_id,
                "name": entry.name,
                "type": entry.type,
                "input_requirements": _build_input_requirements(selected_cards),
                "card_selection": {
                    "mode": "by_recommendation_id",
                    "required": [],
                    "optional": [_clean_text(card.get("card_id")) for card in selected_cards],
                },
                "output_schema_ref": "schemas/output.schema.json",
            }
        )
    return subskills


def build_workflow() -> dict[str, Any]:
    return {
        "workflow_version": "0.1",
        "entrypoint": "triage",
        "global_policies": {
            "stop_on_emergency": True,
            "require_citations": True,
            "no_prescription": True,
        },
        "steps": [
            {
                "step_id": "triage",
                "type": "safety_triage",
                "description": "Check emergency red flags and routing suitability.",
                "config": {
                    "red_flags_ref": "routing_profile.red_flags",
                    "subskill_ref": "initial_triage",
                },
                "transitions": [
                    {"when": {"op": "exists", "path": "result.red_flags"}, "to": "emergency_safety"},
                    {
                        "when": {"op": "gte", "left": "context.route_score", "right": 0.35},
                        "to": "diagnostic_evidence_check",
                    },
                    {"when": {"op": "default"}, "to": "low_probability_output"},
                ],
            },
            {
                "step_id": "diagnostic_evidence_check",
                "type": "evidence_check",
                "description": "Check diagnostic evidence and missing required information.",
                "config": {"subskill_ref": "diagnostic_integration"},
                "transitions": [
                    {
                        "when": {"op": "exists", "path": "result.missing_required_evidence"},
                        "to": "missing_diagnostic_information_output",
                    },
                    {"when": {"op": "default"}, "to": "differential_check"},
                ],
            },
            {
                "step_id": "differential_check",
                "type": "differential_check",
                "description": "Check important differential diagnoses.",
                "config": {"subskill_ref": "differential_diagnosis"},
                "transitions": [
                    {
                        "when": {"op": "exists", "path": "result.differential_warnings"},
                        "to": "differential_needed_output",
                    },
                    {"when": {"op": "default"}, "to": "staging_assessment"},
                ],
            },
            {
                "step_id": "staging_assessment",
                "type": "evidence_check",
                "description": "Assess staging, severity, complications, or risk profile.",
                "config": {"subskill_ref": "staging_assessment"},
                "transitions": [{"when": {"op": "default"}, "to": "management_gate"}],
            },
            {
                "step_id": "management_gate",
                "type": "management_gate",
                "description": "Decide whether management plan generation is allowed.",
                "config": {
                    "diagnosis_required_policy": "safety_constraints.treatment_policy.diagnosis_required"
                },
                "transitions": [
                    {
                        "when": {"op": "eq", "left": "result.allow_management_plan", "right": True},
                        "to": "management_plan_generation",
                    },
                    {"when": {"op": "default"}, "to": "candidate_summary_output"},
                ],
            },
            {
                "step_id": "management_plan_generation",
                "type": "plan_generation",
                "description": "Generate guideline-based management plan.",
                "config": {"subskill_ref": "management_plan"},
                "transitions": [{"when": {"op": "default"}, "to": "final_output"}],
            },
            {
                "step_id": "emergency_safety",
                "type": "terminal_output",
                "description": "Emergency safety output.",
                "config": {"output_template": "emergency_safety"},
                "transitions": [],
            },
            {
                "step_id": "low_probability_output",
                "type": "terminal_output",
                "description": "Low probability output.",
                "config": {"output_template": "low_probability"},
                "transitions": [],
            },
            {
                "step_id": "missing_diagnostic_information_output",
                "type": "terminal_output",
                "description": "Missing diagnostic information output.",
                "config": {"output_template": "missing_information"},
                "transitions": [],
            },
            {
                "step_id": "differential_needed_output",
                "type": "terminal_output",
                "description": "Differential diagnosis needed output.",
                "config": {"output_template": "differential_needed"},
                "transitions": [],
            },
            {
                "step_id": "candidate_summary_output",
                "type": "terminal_output",
                "description": "Candidate disease summary output.",
                "config": {"output_template": "candidate_summary"},
                "transitions": [],
            },
            {
                "step_id": "final_output",
                "type": "terminal_output",
                "description": "Final guideline-based output.",
                "config": {"output_template": "final_recommendation"},
                "transitions": [],
            },
        ],
    }


def build_knowledge_base() -> dict[str, Any]:
    return {
        "cards_path": "cards.jsonl",
        "card_schema_ref": "schema/recommendation_card.schema.json",
        "retrieval": {
            "default_mode": "hybrid",
            "modes": ["by_recommendation_id", "semantic", "metadata_filter", "hybrid"],
            "default_top_k": 8,
            "citation_required": True,
        },
    }


def build_safety_constraints(cards: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    general = [
        {
            "id": "no_diagnosis_replacement",
            "rule": "This skill pack supports guideline-based information organization and does not replace clinician diagnosis.",
            "severity": "high",
        },
        {
            "id": "no_prescription",
            "rule": "Do not provide prescriptions or individualized drug dosing without qualified physician review.",
            "severity": "high",
        },
        {
            "id": "emergency_referral",
            "rule": "If emergency red flags are present, recommend urgent medical evaluation.",
            "severity": "critical",
        },
    ]
    seen_rules = {_normalize_key(rule["rule"]) for rule in general}
    for card in cards:
        for note in _as_text_list(card.get("safety_notes")):
            key = _normalize_key(note)
            if not key or key in seen_rules:
                continue
            seen_rules.add(key)
            general.append(
                {
                    "id": f"card_safety_note_{len(general) - 2}",
                    "rule": note,
                    "severity": "critical" if _contains_any(note, EMERGENCY_KEYWORDS) else "medium",
                }
            )

    return {
        "safety_version": "0.1",
        "general": general,
        "treatment_policy": {
            "diagnosis_required": "confirmed_or_high_confidence",
            "allow_general_management_advice": True,
            "allow_specific_drug_dosing": False,
            "allow_prescription": False,
            "require_physician_review": True,
        },
        "emergency_policy": {
            "trigger_from": ["routing_profile.red_flags", "safety_constraints.general"],
            "output_template": "emergency_safety",
            "stop_workflow": True,
        },
    }


def build_output_templates() -> dict[str, Any]:
    return {
        "emergency_safety": {
            "audience": "clinician",
            "structure": ["status", "red_flags", "urgent_action", "disclaimer"],
        },
        "low_probability": {
            "audience": "clinician",
            "structure": [
                "status",
                "matched_features",
                "reason",
                "suggested_next_steps",
                "disclaimer",
            ],
        },
        "missing_information": {
            "audience": "clinician",
            "structure": [
                "status",
                "missing_information",
                "recommended_next_steps",
                "used_cards",
                "citations",
                "disclaimer",
            ],
        },
        "differential_needed": {
            "audience": "clinician",
            "structure": [
                "status",
                "differential_diagnoses",
                "key_distinguishing_features",
                "recommended_next_steps",
                "used_cards",
                "citations",
                "disclaimer",
            ],
        },
        "candidate_summary": {
            "audience": "clinician",
            "structure": [
                "status",
                "matched_features",
                "evidence_summary",
                "missing_information",
                "used_cards",
                "citations",
                "disclaimer",
            ],
        },
        "final_recommendation": {
            "audience": "clinician",
            "structure": [
                "status",
                "evidence_summary",
                "guideline_recommendations",
                "management_considerations",
                "monitoring_plan",
                "used_cards",
                "citations",
                "disclaimer",
            ],
        },
    }


def build_validation_block() -> dict[str, Any]:
    return {
        "required_checks": [
            "skill_pack_schema_validation",
            "card_schema_validation",
            "card_id_uniqueness",
            "workflow_transition_targets_exist",
            "workflow_subskill_refs_exist",
            "output_templates_exist",
            "selected_card_ids_exist",
        ],
        "review_policy": {
            "requires_human_review": True,
            "reviewer_role": "clinician_or_guideline_curator",
            "notes": "Automatically generated from cards.jsonl; clinical review is required before production use.",
        },
        "test_cases_path": "tests/fixtures/skill_pack_cases.jsonl",
    }


def build_skill_pack(
    cards: Sequence[Mapping[str, Any]],
    taxonomy: Mapping[str, TaxonomyEntry],
    *,
    schema_version: str = "0.3",
) -> dict[str, Any]:
    metadata = infer_metadata(cards)
    skill = {
        "schema_version": schema_version,
        "metadata": metadata,
        "routing_profile": build_routing_profile(cards, metadata),
        "knowledge_base": build_knowledge_base(),
        "workflow": build_workflow(),
        "subskills": build_subskills(cards, taxonomy),
        "safety_constraints": build_safety_constraints(cards),
        "output_templates": build_output_templates(),
        "validation": build_validation_block(),
    }
    return skill


def validate_skill_schema(skill_dict: Mapping[str, Any], skill_schema_path: str | Path) -> None:
    schema_path = Path(skill_schema_path)
    if not schema_path.exists():
        raise BuildSkillPackError(f"skill schema does not exist: {schema_path}")
    schema = _load_json(schema_path)
    validator = _make_jsonschema_validator(schema, schema_path)
    errors = sorted(validator.iter_errors(skill_dict), key=lambda error: list(error.path))
    if errors:
        details = []
        for error in errors[:8]:
            details.append(f"{_format_validation_path(error.path)}: {error.message}")
        raise BuildSkillPackError(
            f"skill schema validation failed against {schema_path}:\n" + "\n".join(details)
        )


def validate_cross_references(
    skill_dict: Mapping[str, Any],
    cards: Sequence[Mapping[str, Any]],
    *,
    output_dir: str | Path | None = None,
    output_package_name: str | None = None,
) -> None:
    errors: list[str] = []
    metadata = _require_mapping(skill_dict.get("metadata"), "metadata")
    skill_id = _clean_text(metadata.get("skill_id"))
    if not skill_id:
        errors.append("metadata.skill_id must not be empty")

    workflow = _require_mapping(skill_dict.get("workflow"), "workflow")
    steps = workflow.get("steps") or []
    if not isinstance(steps, Sequence):
        errors.append("workflow.steps must be a list")
        steps = []
    step_ids = {_clean_text(step.get("step_id")) for step in steps if isinstance(step, Mapping)}
    entrypoint = _clean_text(workflow.get("entrypoint"))
    if entrypoint not in step_ids:
        errors.append(f"workflow.entrypoint {entrypoint!r} does not exist in workflow.steps")

    subskills = skill_dict.get("subskills") or []
    if not isinstance(subskills, Sequence) or not subskills:
        errors.append("subskills must contain at least one subskill")
        subskills = []
    subskill_ids = {
        _clean_text(subskill.get("subskill_id"))
        for subskill in subskills
        if isinstance(subskill, Mapping)
    }
    output_templates = skill_dict.get("output_templates") or {}
    template_ids = set(output_templates.keys()) if isinstance(output_templates, Mapping) else set()

    for step in steps:
        if not isinstance(step, Mapping):
            continue
        step_id = _clean_text(step.get("step_id"))
        for transition in step.get("transitions") or []:
            if not isinstance(transition, Mapping):
                continue
            target = _clean_text(transition.get("to"))
            if target not in step_ids:
                errors.append(f"workflow step {step_id!r} transition.to {target!r} does not exist")
        config = step.get("config") or {}
        if isinstance(config, Mapping):
            subskill_ref = _clean_text(config.get("subskill_ref"))
            if subskill_ref and subskill_ref not in subskill_ids:
                errors.append(
                    f"workflow step {step_id!r} config.subskill_ref {subskill_ref!r} does not exist"
                )
            output_template = _clean_text(config.get("output_template"))
            if output_template and output_template not in template_ids:
                errors.append(
                    f"workflow step {step_id!r} config.output_template {output_template!r} does not exist"
                )

    card_ids = {_clean_text(card.get("card_id")) for card in cards}
    for subskill in subskills:
        if not isinstance(subskill, Mapping):
            continue
        subskill_id = _clean_text(subskill.get("subskill_id"))
        selection = subskill.get("card_selection") or {}
        if not isinstance(selection, Mapping):
            continue
        for field_name in ("required", "optional"):
            for card_id in selection.get(field_name) or []:
                if _clean_text(card_id) not in card_ids:
                    errors.append(
                        f"subskill {subskill_id!r} card_selection.{field_name} "
                        f"references unknown card_id {card_id!r}"
                    )

    knowledge_base = _require_mapping(skill_dict.get("knowledge_base"), "knowledge_base")
    cards_path = _clean_text(knowledge_base.get("cards_path"))
    if cards_path != "cards.jsonl":
        errors.append("knowledge_base.cards_path must be exactly 'cards.jsonl'")
    if output_dir is not None and cards_path:
        package_name = output_package_name or skill_id
        expected = Path(output_dir) / package_name / cards_path
        if expected.name != "cards.jsonl":
            errors.append(f"knowledge_base.cards_path does not point to output cards.jsonl: {expected}")

    if errors:
        raise BuildSkillPackError("cross-reference validation failed:\n" + "\n".join(errors))


def write_skill_pack(
    skill_dict: Mapping[str, Any],
    cards: Sequence[Mapping[str, Any]],
    out_dir: str | Path,
    *,
    output_package_name: str | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> Path:
    try:
        import yaml
    except ImportError as exc:
        raise BuildSkillPackError("PyYAML is required to write skill.yaml. Install PyYAML.") from exc

    skill_id = _clean_text(_require_mapping(skill_dict.get("metadata"), "metadata").get("skill_id"))
    if not skill_id:
        raise BuildSkillPackError("metadata.skill_id must not be empty")

    package_name = output_package_name or skill_id
    target_dir = Path(out_dir) / package_name
    if target_dir.exists() and not force and not dry_run:
        raise BuildSkillPackError(f"output directory already exists: {target_dir}; use --force to overwrite")
    if dry_run:
        return target_dir

    target_dir.mkdir(parents=True, exist_ok=True)
    skill_path = target_dir / "skill.yaml"
    cards_path = target_dir / "cards.jsonl"

    with skill_path.open("w", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(
            dict(skill_dict),
            handle,
            allow_unicode=True,
            sort_keys=False,
            indent=2,
            width=1000,
        )

    with cards_path.open("w", encoding="utf-8", newline="\n") as handle:
        for card in cards:
            handle.write(json.dumps(card, ensure_ascii=False, sort_keys=False))
            handle.write("\n")
    return target_dir


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Build a generic disease skill pack from recommendation cards JSONL."
    )
    parser.add_argument(
        "--cards",
        required=True,
        help="Input cards.jsonl file path, or a directory to recursively process all .jsonl files.",
    )
    parser.add_argument("--out-dir", default="data/skill", help="Output skill pack root directory.")
    parser.add_argument(
        "--skill-schema",
        default="schema/skill_pack.schema.json",
        help="Path to skill_pack.schema.json.",
    )
    parser.add_argument(
        "--card-schema",
        default="schema/recommendation_card.schema.json",
        help="Path to recommendation_card.schema.json; cards are validated when the file exists.",
    )
    parser.add_argument(
        "--taxonomy",
        default="config/subskill_taxonomy.yaml",
        help="Optional subskill taxonomy YAML path.",
    )
    parser.add_argument("--schema-version", default="0.3", help="Skill schema_version, e.g. 0.3.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output files.")
    parser.add_argument("--dry-run", action="store_true", help="Build and validate without writing files.")
    args = parser.parse_args(argv)

    try:
        card_sources = discover_cards_sources(args.cards)
        taxonomy = load_taxonomy(args.taxonomy)
        results = []
        for cards_source in card_sources:
            loaded_cards = load_jsonl(cards_source)
            cards = validate_cards(loaded_cards, args.card_schema)
            skill_dict = build_skill_pack(cards, taxonomy, schema_version=args.schema_version)
            package_name = infer_output_package_name(cards_source)
            validate_cross_references(
                skill_dict,
                cards,
                output_dir=args.out_dir,
                output_package_name=package_name,
            )
            validate_skill_schema(skill_dict, args.skill_schema)
            target_dir = write_skill_pack(
                skill_dict,
                cards,
                args.out_dir,
                output_package_name=package_name,
                force=args.force,
                dry_run=args.dry_run,
            )
            results.append(
                {
                    "cards_source": str(cards_source),
                    "package_name": package_name,
                    "skill_id": skill_dict["metadata"]["skill_id"],
                    "card_count": len(cards),
                    "subskill_count": len(skill_dict["subskills"]),
                    "output_dir": str(target_dir),
                    "skill_yaml": str(target_dir / "skill.yaml"),
                    "cards_jsonl": str(target_dir / "cards.jsonl"),
                    "schema_validation": "pass",
                }
            )
    except BuildSkillPackError as exc:
        print("build_skill_pack: fail", file=sys.stderr)
        print(f"error: {exc}", file=sys.stderr)
        return 1

    summary = {
        "status": "dry_run_ok" if args.dry_run else "ok",
        "processed_count": len(results),
        "results": results,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        raise BuildSkillPackError(f"{path}: invalid JSON schema: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise BuildSkillPackError(f"{path}: expected a JSON object")
    return payload


def _make_jsonschema_validator(schema: Mapping[str, Any], schema_path: Path) -> Any:
    try:
        from jsonschema import Draft202012Validator
    except ImportError as exc:
        raise BuildSkillPackError(
            "jsonschema is required for schema validation. Install jsonschema>=4."
        ) from exc
    try:
        Draft202012Validator.check_schema(schema)
    except Exception as exc:  # pragma: no cover - jsonschema provides several validation errors.
        raise BuildSkillPackError(f"{schema_path}: invalid JSON Schema: {exc}") from exc
    return Draft202012Validator(schema)


def _format_validation_path(path_parts: Iterable[Any]) -> str:
    parts = [str(part) for part in path_parts]
    return "$" if not parts else "$." + ".".join(parts)


def _most_common_non_empty(values: Iterable[str]) -> str | None:
    counter: Counter[str] = Counter(value for value in values if value)
    if not counter:
        return None
    return counter.most_common(1)[0][0]


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _first_text(*values: Any) -> str:
    for value in values:
        text = _clean_text(value)
        if text:
            return text
    return ""


def _path_value(payload: Mapping[str, Any], dotted_path: str) -> Any:
    value: Any = payload
    for part in dotted_path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def _as_text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [_clean_text(value)] if _clean_text(value) else []
    if isinstance(value, Mapping):
        texts: list[str] = []
        for item in value.values():
            texts.extend(_as_text_list(item))
        return texts
    if isinstance(value, Iterable):
        texts = []
        for item in value:
            texts.extend(_as_text_list(item))
        return texts
    text = _clean_text(value)
    return [text] if text else []


def _dedupe_texts(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        text = _clean_text(value)
        key = _normalize_key(text)
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(text)
    return deduped


def _normalize_key(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().lower()


def _extract_year(text: str) -> str | None:
    match = re.search(r"(?:19|20)\d{2}", text or "")
    return match.group(0) if match else None


def _extract_publication_year(*texts: str | None) -> int | None:
    for text in texts:
        year = _extract_year(text or "")
        if year:
            return int(year)
    return None


def _infer_disease_name_from_title(title: str) -> str:
    text = _clean_text(title)
    if not text:
        return "unknown disease"
    for marker in ("诊治指南", "诊断和治疗共识意见", "专家共识意见", "筛查与早诊早治指南", "指南", "共识"):
        if marker in text:
            text = text.split(marker, 1)[0]
            break
    text = re.sub(r"[（(](?:19|20)\d{2}.*?[）)]", "", text).strip()
    return text or title


def _normalized_evidence_value(value: Any) -> str:
    text = _clean_text(value)
    allowed = {
        "high",
        "moderate",
        "low",
        "very_low",
        "level_1",
        "level_2",
        "level_3",
        "level_4",
        "level_5",
        "not_applicable",
        "unknown",
    }
    return text if text in allowed else "unknown"


def _normalized_strength_value(value: Any) -> str:
    text = _clean_text(value)
    allowed = {
        "strong",
        "weak",
        "grade_a",
        "grade_b",
        "best_practice_statement",
        "not_applicable",
        "unknown",
    }
    return text if text in allowed else "unknown"


def _positive_int(value: Any, *, default: int) -> int:
    try:
        integer = int(value)
    except (TypeError, ValueError):
        return default
    return integer if integer >= 1 else default


def _ascii_slug(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii").lower()
    slug = re.sub(r"[^a-z0-9]+", "_", ascii_text).strip("_")
    slug = re.sub(r"_+", "_", slug)
    return slug


def _join_slug_parts(parts: Sequence[str]) -> str:
    tokens: list[str] = []
    seen: set[str] = set()
    for part in parts:
        for token in part.split("_"):
            if not token or token in seen:
                continue
            seen.add(token)
            tokens.append(token)
    return "_".join(tokens)


def _safe_path_name(value: str) -> str:
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", value).strip()
    name = re.sub(r"\s+", " ", name)
    name = name.rstrip(". ")
    return name[:120]


def _hashed_skill_id(value: str) -> str:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:10]
    return f"disease_skill_{digest}"


def _classify_test_feature(test_name: str) -> str:
    if _contains_any(test_name, LAB_KEYWORDS):
        return "labs"
    if _contains_any(test_name, IMAGING_KEYWORDS):
        return "imaging"
    if _contains_any(test_name, ENDOSCOPY_KEYWORDS):
        return "endoscopy"
    if _contains_any(test_name, PATHOLOGY_KEYWORDS):
        return "pathology"
    return "findings"


def _contains_any(text: str, keywords: Iterable[str]) -> bool:
    text_upper = text.upper()
    text_lower = text.lower()
    return any(keyword.upper() in text_upper or keyword.lower() in text_lower for keyword in keywords)


def _add_feature(
    feature_buckets: dict[str, list[dict[str, Any]]],
    seen: dict[str, set[str]],
    bucket: str,
    name: str,
    *,
    weight: float,
) -> None:
    text = _clean_text(name)
    key = _normalize_key(text)
    if not key or key in seen[bucket]:
        return
    seen[bucket].add(key)
    feature_buckets[bucket].append(
        {
            "name": text,
            "operator": "semantic_match",
            "weight": weight,
            "match_type": "semantic_or_exact",
        }
    )


def _safety_candidate_texts(card: Mapping[str, Any]) -> list[str]:
    values: list[str] = []
    values.extend(_as_text_list(card.get("safety_notes")))
    for field in ("action", "condition", "statement_text"):
        text = _clean_text(card.get(field))
        if text:
            values.append(text)
    return values


def _red_flag_severity(text: str) -> str:
    critical_keywords = ("休克", "穿孔", "大出血", "危及生命", "脓毒症")
    return "critical" if _contains_any(text, critical_keywords) else "high"


def _extract_differentials(card: Mapping[str, Any], *, current_disease: str) -> list[str]:
    explicit: list[str] = []
    for field in ("must_differentiate", "differential_diagnoses", "differential_disease"):
        explicit.extend(_as_text_list(card.get(field)))
    if explicit:
        return _clean_differential_names(explicit, current_disease=current_disease)

    text = "。".join(
        _clean_text(card.get(field))
        for field in ("clinical_task", "action", "condition", "statement_text")
        if _clean_text(card.get(field))
    )
    if not _contains_any(text, DIFFERENTIAL_KEYWORDS):
        return []
    candidates: list[str] = []
    for match in re.finditer(r"(?:鉴别|排除|区别|需排除)([^。；;]{2,80})", text):
        phrase = match.group(1)
        phrase = re.sub(r"^(诊断|疾病|病因|是否|有无|与)", "", phrase)
        phrase = re.sub(r"(等疾病|等病变|等情况|疾病|病变)$", "", phrase)
        candidates.extend(re.split(r"[、,，/及和与或]", phrase))
    return _clean_differential_names(candidates, current_disease=current_disease)


def _clean_differential_names(values: Iterable[str], *, current_disease: str) -> list[str]:
    names: list[str] = []
    for value in values:
        text = _clean_text(value).strip(" ：:，,。；;（）()[]【】")
        if not text or len(text) < 2 or len(text) > 30:
            continue
        if current_disease and _normalize_key(text) == _normalize_key(current_disease):
            continue
        if any(skip in text for skip in ("诊断", "检查", "患者", "表现", "充分", "高度怀疑")):
            continue
        names.append(text)
    return _dedupe_texts(names)


def _parse_taxonomy_entry(
    path: Path,
    subskill_id: str,
    entry: Any,
) -> tuple[list[str], bool, str | None, str | None]:
    if isinstance(entry, list):
        return _as_text_list(entry), False, None, None
    if isinstance(entry, Mapping):
        if any(key in entry for key in ("disease", "disease_name", "disease_aliases")):
            raise BuildSkillPackError(
                f"{path}: taxonomy entry {subskill_id!r} must not define disease identity"
            )
        keywords = _as_text_list(entry.get("keywords"))
        if not keywords and "add_keywords" in entry:
            keywords = _as_text_list(entry.get("add_keywords"))
        replace = bool(entry.get("replace") or entry.get("mode") == "replace")
        name = _clean_text(entry.get("name")) or None
        subskill_type = _clean_text(entry.get("type")) or None
        return keywords, replace, name, subskill_type
    raise BuildSkillPackError(f"{path}: taxonomy entry {subskill_id!r} must be a list or object")


def _build_input_requirements(cards: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    items: list[dict[str, str]] = []
    seen: set[str] = set()
    for card in cards:
        for required_input in _as_required_inputs(card.get("required_inputs")):
            key = _normalize_key(required_input["path"] + "|" + required_input["label"])
            if key in seen:
                continue
            seen.add(key)
            items.append(required_input)
    return {"required_for_high_confidence": items}


def _as_required_inputs(value: Any) -> list[dict[str, str]]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        path = _clean_text(value.get("path"))
        label = _clean_text(value.get("label") or value.get("name") or value.get("field"))
        if path or label:
            return [{"path": path or _case_path(label), "label": label or path}]
        return [
            required_input
            for item in value.values()
            for required_input in _as_required_inputs(item)
        ]
    if isinstance(value, str):
        text = _clean_text(value)
        return [{"path": _case_path(text), "label": text}] if text else []
    if isinstance(value, Iterable):
        return [
            required_input
            for item in value
            for required_input in _as_required_inputs(item)
        ]
    text = _clean_text(value)
    return [{"path": _case_path(text), "label": text}] if text else []


def _case_path(label: str) -> str:
    slug = _ascii_slug(label)
    if not slug:
        slug = "required_input_" + hashlib.sha1(label.encode("utf-8")).hexdigest()[:8]
    return f"case.{slug}"


def _humanize_identifier(value: str) -> str:
    return value.replace("_", " ").strip().title()


def _stringify_for_matching(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, Mapping):
        return " ".join(_stringify_for_matching(item) for item in value.values())
    if isinstance(value, str):
        return value
    if isinstance(value, Iterable):
        return " ".join(_stringify_for_matching(item) for item in value)
    return str(value)


def _shorten_text(text: str, *, limit: int) -> str:
    text = _clean_text(text)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BuildSkillPackError(f"{name} must be an object")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
