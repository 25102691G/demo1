from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter, OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from skill_engine.hpo_extractor import (
    DEFAULT_DEFINITION2ID_PATH,
    DEFAULT_DEFINITION_EMBEDDINGS_PATH,
    DEFAULT_MODEL_PATH as DEFAULT_HPO_MODEL_PATH,
    HPO_EXTRACTION_SYSTEM_PROMPT_FROM_CARDS,
    HpoExtractor,
)
from skill_engine.icd_extractor import (
    DEFAULT_ICD10_EMBEDDINGS_PATH,
    DEFAULT_ICD10_PATH,
    DEFAULT_MODEL_PATH as DEFAULT_ICD_MODEL_PATH,
    ICD_EXTRACTION_SYSTEM_PROMPT_FROM_CARDS,
    IcdResources,
    IcdExtractor,
    _load_icd10_records,
    _load_torch,
)
from skill_engine.llm_client import OpenAICompatibleJsonChatClient, load_llm_config_from_env

NORMALIZE_BRANCH_COUNTS: Counter[str] = Counter()
DEFAULT_TAXONOMY_PATH = ROOT / "configs" / "subskill_taxonomy.yaml"


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
MEDICAL_EXAMINATION_STAGE = "诊断评估流程"
MEDICAL_EXAMINATION_TASKS = (
    "初步筛查与临床表现评估",
    "实验室检查",
    "影像学检查",
    "内镜检查",
    "病理",
    "综合诊断",
)
MEDICAL_EXAMINATION_SYSTEM_PROMPT = """你是一名专攻消化内科指南结构化的医学专家。
请根据疾病指南片段，提取该片段明确提到的诊断评估内容，包括需要做的医学检查、重点关注症状体征、实验室/影像/内镜/病理关注点、综合诊断判断依据。
不得根据常识或上下文自行推断，必须来自原文明确表述。

只输出 JSON 对象，格式如下：
{"examinations":["需要做的检查"],"key_symptoms":["重点关注症状或体征"],"attention_points":["其他诊断关注点"]}。
没有对应内容时输出空数组。内容请使用中文书写。禁止输出其他任何无关信息。"""


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
        card = dict(loaded.data)
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
    return dict(payload)

def infer_metadata(
    cards: Sequence[Mapping[str, Any]],
    *,
    metadata_icd_extractor: IcdExtractor | None = None,
) -> dict[str, Any]:
    fallback_disease_name = _most_common_non_empty(_clean_text(card.get("disease")) for card in cards)
    if not fallback_disease_name:
        fallback_disease_name = "unknown disease"

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
    disease_query = _metadata_disease_query(guideline_name, source_pdf, fallback_disease_name)
    disease_name = (
        _resolve_metadata_disease_name(disease_query, metadata_icd_extractor)
        or fallback_disease_name
    )
    skill_id = slugify_skill_id(disease_name, guideline_name, version, source_pdf)

    return {
        "skill_id": skill_id,
        "disease_name": disease_name,
        "guideline": {
            "name": guideline_name,
            "version": version,
            "source_pdf": source_pdf,
            "publication_year": publication_year,
            "language": "zh-CN",
        },
    }


def slugify_skill_id(
    disease_name: str,
    guideline_name: str | None = None,
    guideline_version: str | None = None,
    source_pdf: str | None = None,
) -> str:
    source_stem = Path(source_pdf).stem if source_pdf and source_pdf != "unknown" else ""
    source_text = " ".join(
        text
        for text in (disease_name, guideline_version, guideline_name, source_stem)
        if text and text != "unknown"
    )
    original_name = _safe_skill_id(source_stem or Path(guideline_name or "").stem or disease_name)
    if original_name:
        return original_name

    disease_slug = _ascii_slug(disease_name)
    guideline_slug = _ascii_slug(guideline_name or "")
    version_slug = _ascii_slug(guideline_version or "")
    source_slug = _ascii_slug(source_stem)

    if re.search(r"[a-z]", disease_slug):
        slug = _join_slug_parts([disease_slug, version_slug, guideline_slug or source_slug])
    elif re.search(r"[a-z]", guideline_slug):
        slug = _join_slug_parts([guideline_slug, version_slug])
    else:
        slug = ""

    if slug and re.search(r"[a-z]", slug):
        return slug[:96].strip("_") or _hashed_skill_id(source_text)
    return _hashed_skill_id(source_text or disease_name)


def _metadata_disease_query(guideline_name: str, source_pdf: str, fallback_disease_name: str) -> str:
    source_stem = Path(source_pdf).stem if source_pdf and source_pdf != "unknown" else ""
    guideline_stem = Path(guideline_name).stem if guideline_name else ""
    return _clean_text(source_stem or guideline_stem or fallback_disease_name)


def _resolve_metadata_disease_name(
    disease_query: str,
    metadata_icd_extractor: IcdExtractor | None,
) -> str | None:
    query = _clean_text(disease_query)
    if not query or metadata_icd_extractor is None:
        return None
    try:
        mappings = metadata_icd_extractor.map_diagnoses_to_icd(
            [{"diagnosis": query}],
            source_type="metadata",
        )
    except Exception as exc:
        print(
            f"metadata disease_name ICD10 embedding failed for {query!r}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return None
    if not mappings:
        return None
    return _clean_text(mappings[0].get("diagnosis_name")) or None


def infer_output_package_name(cards_path: str | Path) -> str:
    path = Path(cards_path)
    generic_file_names = {"result", "cards", "recommendation_cards", "recommendations"}
    source_name = path.parent.name if path.stem.lower() in generic_file_names else path.stem
    package_name = _safe_path_name(source_name)
    if not package_name:
        package_name = _hashed_skill_id(str(path.resolve()))
    return package_name


def build_routing_profile(
    cards: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
    *,
    feature_extractor: Any,
    feature_mode: str,
    deepseek_client: Any,
    llm_workers: int = 1,
) -> dict[str, Any]:
    extracted_features = _extract_features_from_cards(
        cards=cards,
        feature_extractor=feature_extractor,
        feature_mode=feature_mode,
        deepseek_client=deepseek_client,
        llm_workers=llm_workers,
    )
    if feature_mode in {"hpo", "icd10"}:
        if not isinstance(extracted_features, Mapping):
            raise BuildSkillPackError(f"{feature_mode} feature extractor must return feature groups")
        positive_features: list[dict[str, Any]] | dict[str, list[dict[str, Any]]] = (
            _dedupe_positive_features(extracted_features.get("positive_features") or [])
        )
        negative_features = _dedupe_positive_features(
            extracted_features.get("negative_features") or []
        )
        _attach_clinical_sources(positive_features, cards)
        _attach_clinical_sources(negative_features, cards)
    else:
        raise BuildSkillPackError(f"unsupported feature mode: {feature_mode}")

    return {
        "population": _dedupe_texts(card.get("population") for card in cards),
        "positive_features": positive_features,
        "negative_features": negative_features,
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
            "mapping": _build_card_evidence_mapping(cards),
        },
    }

def build_medical_examinations(
    cards: Sequence[Mapping[str, Any]],
    deepseek_client: Any,
    *,
    llm_workers: int = 1,
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {task: [] for task in MEDICAL_EXAMINATION_TASKS}
    candidates = [
        card
        for card in cards
        if _clean_text(card.get("clinical_stage")) == MEDICAL_EXAMINATION_STAGE
        and _clean_text(card.get("clinical_task")) in result
        and _clean_text(card.get("raw_chunk_text"))
    ]
    if not candidates:
        return result

    workers = max(1, int(llm_workers or 1))
    extracted_by_index: list[dict[str, list[str]]] = [
        {"examinations": [], "key_symptoms": [], "attention_points": []}
        for _ in candidates
    ]
    if workers <= 1 or len(candidates) <= 1:
        for index, card in enumerate(candidates):
            extracted_by_index[index] = extract_medical_examination_item(card, deepseek_client)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_index = {
                executor.submit(extract_medical_examination_item, card, deepseek_client): index
                for index, card in enumerate(candidates)
            }
            for future in as_completed(future_to_index):
                extracted_by_index[future_to_index[future]] = future.result()

    for card, extracted in zip(candidates, extracted_by_index, strict=False):
        if not any(extracted.values()):
            continue
        task = _clean_text(card.get("clinical_task"))
        result[task].append(
            {
                "card_id": _clean_text(card.get("card_id")),
                "recommendation_label": _clean_text(card.get("recommendation_label")) or None,
                "examinations": extracted["examinations"],
                "key_symptoms": extracted["key_symptoms"],
                "attention_points": extracted["attention_points"],
            }
        )
    return result


def extract_medical_examination_item(
    card: Mapping[str, Any],
    deepseek_client: Any,
) -> dict[str, list[str]]:
    user_prompt = json.dumps(
        {
            "clinical_task": _clean_text(card.get("clinical_task")),
            "raw_chunk_text": _clean_text(card.get("raw_chunk_text")),
        },
        ensure_ascii=False,
    )
    payload = deepseek_client.chat_json(MEDICAL_EXAMINATION_SYSTEM_PROMPT, user_prompt)
    return {
        "examinations": _dedupe_texts(_as_text_list(payload.get("examinations"))),
        "key_symptoms": _dedupe_texts(_as_text_list(payload.get("key_symptoms"))),
        "attention_points": _dedupe_texts(_as_text_list(payload.get("attention_points"))),
    }


def _build_card_evidence_mapping(cards: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    mapping: dict[str, dict[str, Any]] = {}
    for card in cards:
        card_id = _clean_text(card.get("card_id"))
        evidence = card.get("evidence")
        if not card_id or not isinstance(evidence, Mapping):
            continue
        mapping[card_id] = dict(evidence)
    return mapping


def load_taxonomy(taxonomy_path: str | Path | None = None) -> "OrderedDict[str, TaxonomyEntry]":
    path = Path(taxonomy_path) if taxonomy_path else DEFAULT_TAXONOMY_PATH
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        raise BuildSkillPackError(f"taxonomy file does not exist: {path}")

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

    taxonomy: "OrderedDict[str, TaxonomyEntry]" = OrderedDict()
    for subskill_id, entry in entries.items():
        subskill_key = str(subskill_id)
        keywords, replace_keywords, name, subskill_type = _parse_taxonomy_entry(path, subskill_key, entry)
        del replace_keywords
        taxonomy[subskill_key] = TaxonomyEntry(
            subskill_id=subskill_key,
            name=name or _humanize_identifier(subskill_key),
            type=subskill_type or "card_retrieval",
            keywords=tuple(_dedupe_texts(keywords)),
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


def build_workflow(cards: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    diagnostic_tasks = [
        "初步筛查与临床表现评估",
        "实验室检查",
        "影像学检查",
        "内镜检查",
        "病理",
        "综合诊断",
    ]
    staging_tasks = ["影像学检查", "内镜检查", "病理", "综合诊断"]
    management_tasks = ["一般治疗", "药物治疗", "手术治疗"]
    diagnostic_filter = {
        "clinical_stage": "诊断评估流程",
        "clinical_task": diagnostic_tasks,
    }
    differential_filter = {
        "clinical_stage": "诊断评估流程",
        "clinical_task": ["综合诊断"],
    }
    staging_filter = {
        "clinical_stage": "诊断评估流程",
        "clinical_task": staging_tasks,
    }
    management_filter = {
        "clinical_stage": "治疗流程",
        "clinical_task": management_tasks,
    }
    monitoring_filter = {
        "clinical_stage": "治疗流程",
        "clinical_task": ["随访与监测"],
    }
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
                "config": {
                    "card_filter": diagnostic_filter,
                    "input_requirements": _build_input_requirements(
                        _cards_matching_filter(cards or [], diagnostic_filter)
                    ),
                },
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
                "config": {
                    "card_filter": differential_filter,
                    "input_requirements": _build_input_requirements(
                        _cards_matching_filter(cards or [], differential_filter)
                    ),
                },
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
                "config": {
                    "card_filter": staging_filter,
                    "input_requirements": _build_input_requirements(
                        _cards_matching_filter(cards or [], staging_filter)
                    ),
                },
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
                "config": {
                    "card_filter": management_filter,
                    "monitoring_card_filter": monitoring_filter,
                    "input_requirements": _build_input_requirements(
                        _cards_matching_filter(cards or [], management_filter)
                    ),
                },
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
        "cards_path": "recommendation_card.jsonl",
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
                "matched_positive_features",
                "matched_negative_features",
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
                "matched_positive_features",
                "matched_negative_features",
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
            "notes": "Automatically generated from result.jsonl; clinical review is required before production use.",
        },
        "test_cases_path": "tests/fixtures/skill_pack_cases.jsonl",
    }


def build_skill_pack(
    cards: Sequence[Mapping[str, Any]],
    taxonomy: Mapping[str, TaxonomyEntry],
    *,
    schema_version: str = "0.3",
    feature_extractor: Any,
    feature_mode: str,
    deepseek_client: Any,
    metadata_icd_extractor: IcdExtractor | None = None,
    llm_workers: int = 1,
) -> dict[str, Any]:
    metadata = infer_metadata(cards, metadata_icd_extractor=metadata_icd_extractor)
    skill = {
        "schema_version": schema_version,
        "metadata": metadata,
        "routing_profile": build_routing_profile(
            cards,
            metadata,
            feature_extractor=feature_extractor,
            feature_mode=feature_mode,
            deepseek_client=deepseek_client,
            llm_workers=llm_workers,
        ),
        "medical_examinations": build_medical_examinations(
            cards,
            deepseek_client,
            llm_workers=llm_workers,
        ),
        "knowledge_base": build_knowledge_base(),
        "workflow": build_workflow(cards),
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
    if not isinstance(subskills, Sequence):
        errors.append("subskills must be a list when present")
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
    if output_dir is not None and cards_path:
        expected_dir = Path(output_dir)
        if output_package_name:
            expected_dir = expected_dir / output_package_name

    if errors:
        raise BuildSkillPackError("cross-reference validation failed:\n" + "\n".join(errors))


def write_skill_pack(
    skill_dict: Mapping[str, Any],
    cards: Sequence[Mapping[str, Any]],
    out_dir: str | Path,
    *,
    output_package_name: str | None = None,
    skill_filename: str = "skill.yaml",
    force: bool = False,
    dry_run: bool = False,
) -> Path:
    del cards
    try:
        import yaml
    except ImportError as exc:
        raise BuildSkillPackError("PyYAML is required to write skill.yaml. Install PyYAML.") from exc

    skill_id = _clean_text(_require_mapping(skill_dict.get("metadata"), "metadata").get("skill_id"))
    if not skill_id:
        raise BuildSkillPackError("metadata.skill_id must not be empty")

    target_dir = Path(out_dir)
    if output_package_name:
        target_dir = target_dir / output_package_name
    skill_path = target_dir / skill_filename
    if skill_path.exists() and not force and not dry_run:
        raise BuildSkillPackError(f"{skill_filename} already exists: {skill_path}; use --force to overwrite")
    if dry_run:
        return target_dir

    target_dir.mkdir(parents=True, exist_ok=True)

    with skill_path.open("w", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(
            dict(skill_dict),
            handle,
            allow_unicode=True,
            sort_keys=False,
            indent=2,
            width=1000,
        )

    return target_dir


def build_default_hpo_dependencies(
    similarity_threshold: float | None = None,
) -> tuple[HpoExtractor, OpenAICompatibleJsonChatClient]:
    hpo_kwargs: dict[str, Any] = {}
    if similarity_threshold is not None:
        hpo_kwargs["similarity_threshold"] = similarity_threshold
    hpo_extractor = HpoExtractor.from_paths(
        model_path=DEFAULT_HPO_MODEL_PATH,
        definition2id_path=DEFAULT_DEFINITION2ID_PATH,
        definition_embeddings_path=DEFAULT_DEFINITION_EMBEDDINGS_PATH,
        **hpo_kwargs,
    )
    deepseek_client = OpenAICompatibleJsonChatClient(load_llm_config_from_env())
    return hpo_extractor, deepseek_client


def build_default_icd10_dependencies(
    similarity_threshold: float | None = None,
    model_path: str | Path | None = None,
) -> tuple[IcdExtractor, OpenAICompatibleJsonChatClient]:
    icd_kwargs: dict[str, Any] = {}
    if similarity_threshold is not None:
        icd_kwargs["similarity_threshold"] = similarity_threshold
    icd_extractor = IcdExtractor.from_paths(
        model_path=model_path or DEFAULT_ICD_MODEL_PATH,
        icd10_path=DEFAULT_ICD10_PATH,
        icd10_embeddings_path=DEFAULT_ICD10_EMBEDDINGS_PATH,
        **icd_kwargs,
    )
    deepseek_client = OpenAICompatibleJsonChatClient(load_llm_config_from_env())
    return icd_extractor, deepseek_client


def build_metadata_icd_extractor(
    feature_extractor: Any,
    *,
    feature_mode: str,
    model_path: str | Path | None = None,
) -> IcdExtractor:
    if feature_mode == "icd10" and isinstance(feature_extractor, IcdExtractor):
        return feature_extractor
    if feature_mode == "hpo" and isinstance(feature_extractor, HpoExtractor):
        torch = _load_torch()
        resources = feature_extractor.resources
        records = _load_icd10_records(DEFAULT_ICD10_PATH)
        record_embeddings = torch.load(str(DEFAULT_ICD10_EMBEDDINGS_PATH), map_location="cpu")
        if int(record_embeddings.shape[0]) != len(records):
            raise BuildSkillPackError(
                f"{DEFAULT_ICD10_EMBEDDINGS_PATH}: embedding row count "
                f"{int(record_embeddings.shape[0])} does not match ICD10 record count {len(records)}"
            )
        icd_resources = IcdResources(
            model=resources.model,
            tokenizer=resources.tokenizer,
            pooling_mode=resources.pooling_mode,
            records=records,
            record_embeddings=record_embeddings,
            record_keys=[record["diagnosis_name"] for record in records],
        )
        return IcdExtractor(icd_resources, similarity_threshold=0.0, batch_size=1)
    return IcdExtractor.from_paths(
        model_path=model_path or DEFAULT_ICD_MODEL_PATH,
        icd10_path=DEFAULT_ICD10_PATH,
        icd10_embeddings_path=DEFAULT_ICD10_EMBEDDINGS_PATH,
        similarity_threshold=0.0,
        batch_size=1,
    )


def build_default_feature_dependencies(
    feature_mode: str,
    similarity_threshold: float | None = None,
    model_path: str | Path | None = None,
) -> tuple[Any, OpenAICompatibleJsonChatClient]:
    if feature_mode == "hpo":
        return build_default_hpo_dependencies(similarity_threshold)
    if feature_mode == "icd10":
        return build_default_icd10_dependencies(similarity_threshold, model_path=model_path)
    raise BuildSkillPackError(f"unsupported feature mode: {feature_mode}")


def _extract_features_from_cards(
    *,
    cards: Sequence[Mapping[str, Any]],
    feature_extractor: Any,
    feature_mode: str,
    deepseek_client: Any,
    llm_workers: int,
) -> Any:
    if feature_mode == "hpo":
        return feature_extractor.extract_hpo_from_cards(
            cards,
            deepseek_client,
            llm_workers=llm_workers,
            prompt=HPO_EXTRACTION_SYSTEM_PROMPT_FROM_CARDS,
        )
    if feature_mode == "icd10":
        return feature_extractor.extract_icd_from_cards(
            cards,
            deepseek_client,
            llm_workers=llm_workers,
            prompt=ICD_EXTRACTION_SYSTEM_PROMPT_FROM_CARDS,
        )
    raise BuildSkillPackError(f"unsupported feature mode: {feature_mode}")


def log_build_start(index: int, total: int, cards_source: Path) -> None:
    remaining = max(total - index, 0)
    print(f"[{index}/{total}] 开始构建: {cards_source}，剩余 {remaining} 个", flush=True)


def log_build_step(index: int, total: int, message: str) -> None:
    print(f"[{index}/{total}] {message}", flush=True)


def log_build_done(
    index: int,
    total: int,
    cards_source: Path,
    *,
    card_count: int,
    skill_yaml: Path,
) -> None:
    print(
        f"[{index}/{total}] 完成: {cards_source}，{card_count} 张 card，输出 {skill_yaml}",
        flush=True,
    )


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
        help="Input result.jsonl file path, or a directory to recursively process all .jsonl files.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Optional output root. Defaults to writing skill_hpo.yaml or skill_icd10.yaml next to each input result.jsonl.",
    )
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
        default="configs/subskill_taxonomy.yaml",
        help="Optional subskill taxonomy YAML path.",
    )
    feature_group = parser.add_mutually_exclusive_group(required=True)
    feature_group.add_argument("--hpo", action="store_true", help="Use HPO feature extraction.")
    feature_group.add_argument("--icd10", action="store_true", help="Use ICD10 feature extraction.")
    parser.add_argument("--schema-version", default="0.3", help="Skill schema_version, e.g. 0.3.")
    parser.add_argument(
        "--hpo-summary-output",
        default=None,
        help="Optional feature summary JSON path. Defaults to hpo_skill_summary.json or icd10_skill_summary.json next to each input JSONL.",
    )
    parser.add_argument(
        "--llm-workers",
        type=int,
        default=20,
        help="Concurrent LLM calls for feature extraction. Defaults to 20.",
    )
    parser.add_argument("--similarity-threshold", type=_similarity_threshold)
    parser.add_argument(
        "--model-path",
        default=None,
        help="Optional ICD10 embedding model path. Use the same model that built ICD10_embeddings.pt.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing output files.")
    parser.add_argument("--dry-run", action="store_true", help="Build and validate without writing files.")
    args = parser.parse_args(argv)

    try:
        card_sources = discover_cards_sources(args.cards)
        taxonomy = load_taxonomy(args.taxonomy)
        feature_mode = _feature_mode_from_args(args)
        skill_filename = _skill_filename_for_mode(feature_mode)
        feature_extractor, deepseek_client = build_default_feature_dependencies(
            feature_mode,
            args.similarity_threshold,
            model_path=args.model_path,
        )
        metadata_icd_extractor = build_metadata_icd_extractor(
            feature_extractor,
            feature_mode=feature_mode,
            model_path=args.model_path,
        )
        results = []
        total_sources = len(card_sources)
        for index, cards_source in enumerate(card_sources, 1):
            log_build_start(index, total_sources, cards_source)
            log_build_step(index, total_sources, "读取 JSONL...")
            loaded_cards = load_jsonl(cards_source)
            log_build_step(index, total_sources, f"读取完成: {len(loaded_cards)} 张 card")
            log_build_step(index, total_sources, "校验 recommendation_card schema...")
            cards = validate_cards(loaded_cards, args.card_schema)
            log_build_step(
                index,
                total_sources,
                f"构建 skill pack，feature={feature_mode}, workers={args.llm_workers}...",
            )
            skill_dict = build_skill_pack(
                cards,
                taxonomy,
                schema_version=args.schema_version,
                feature_extractor=feature_extractor,
                feature_mode=feature_mode,
                deepseek_client=deepseek_client,
                metadata_icd_extractor=metadata_icd_extractor,
                llm_workers=args.llm_workers,
            )
            feature_summary_path = _hpo_summary_output_path(
                cards_source,
                args.hpo_summary_output,
                feature_mode=feature_mode,
            )
            package_name = infer_output_package_name(cards_source) if args.out_dir else None
            output_dir = Path(args.out_dir) if args.out_dir else cards_source.parent
            log_build_step(index, total_sources, "校验 skill pack...")
            validate_cross_references(
                skill_dict,
                cards,
                output_dir=output_dir,
                output_package_name=package_name,
            )
            validate_skill_schema(skill_dict, args.skill_schema)
            log_build_step(index, total_sources, f"写出 {skill_filename}...")
            target_dir = write_skill_pack(
                skill_dict,
                cards,
                output_dir,
                output_package_name=package_name,
                skill_filename=skill_filename,
                force=args.force,
                dry_run=args.dry_run,
            )
            if not args.dry_run:
                feature_extractor.write_last_summary(feature_summary_path)
            results.append(
                {
                    "cards_source": str(cards_source),
                    "feature_mode": feature_mode,
                    "package_name": package_name or cards_source.parent.name,
                    "skill_id": skill_dict["metadata"]["skill_id"],
                    "card_count": len(cards),
                    "output_dir": str(target_dir),
                    "skill_yaml": str(target_dir / skill_filename),
                    "result_jsonl": str(cards_source),
                    "feature_summary": str(feature_summary_path),
                    "schema_validation": "pass",
                }
            )
            log_build_done(
                index,
                total_sources,
                cards_source,
                card_count=len(cards),
                skill_yaml=target_dir / skill_filename,
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
    print(f"build_skill_pack 完成: 处理 {len(results)} 个 JSONL", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _hpo_summary_output_path(
    cards_source: Path,
    explicit_path: str | None,
    *,
    feature_mode: str,
) -> Path:
    if explicit_path:
        return Path(explicit_path)
    return cards_source.parent / f"{feature_mode}_skill_summary.json"


def _feature_mode_from_args(args: argparse.Namespace) -> str:
    if args.hpo:
        return "hpo"
    if args.icd10:
        return "icd10"
    raise BuildSkillPackError("provide exactly one of --hpo or --icd10")


def _skill_filename_for_mode(feature_mode: str) -> str:
    if feature_mode == "hpo":
        return "skill_hpo.yaml"
    if feature_mode == "icd10":
        return "skill_icd10.yaml"
    raise BuildSkillPackError(f"unsupported feature mode: {feature_mode}")


def _similarity_threshold(value: str) -> float:
    try:
        threshold = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number between 0 and 1") from exc
    if not 0 <= threshold <= 1:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return threshold


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


def _safe_skill_id(value: str) -> str:
    text = _clean_text(value)
    if not text or text == "unknown":
        return ""
    return _safe_path_name(text)


def _hashed_skill_id(value: str) -> str:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:10]
    return f"disease_skill_{digest}"


def _contains_any(text: str, keywords: Iterable[str]) -> bool:
    text_upper = text.upper()
    text_lower = text.lower()
    return any(keyword.upper() in text_upper or keyword.lower() in text_lower for keyword in keywords)


def _dedupe_positive_features(features: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for feature in features:
        if not isinstance(feature, Mapping):
            continue
        name = _clean_text(feature.get("name"))
        key = _normalize_key(name)
        if not key or key in seen:
            continue
        seen.add(key)
        copied = dict(feature)
        copied["name"] = name
        result.append(copied)
    return result


def _attach_clinical_sources(
    features: Sequence[dict[str, Any]],
    cards: Sequence[Mapping[str, Any]],
) -> None:
    card_index = {
        _clean_text(card.get("card_id")): card
        for card in cards
        if _clean_text(card.get("card_id"))
    }
    for feature in features:
        sources: list[dict[str, str]] = []
        seen: set[tuple[str, str, str]] = set()
        for card_id in _as_text_list(feature.get("card_id")):
            card = card_index.get(card_id)
            if card is None:
                continue
            clinical_stage = _clean_text(card.get("clinical_stage"))
            clinical_task = _clean_text(card.get("clinical_task"))
            if not clinical_stage and not clinical_task:
                continue
            key = (card_id, clinical_stage, clinical_task)
            if key in seen:
                continue
            seen.add(key)
            sources.append(
                {
                    "card_id": card_id,
                    "clinical_stage": clinical_stage,
                    "clinical_task": clinical_task,
                }
            )
        if sources:
            feature["clinical_sources"] = sources


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
    for field in ():
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


def _cards_matching_filter(
    cards: Sequence[Mapping[str, Any]],
    card_filter: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    return [card for card in cards if _card_matches_filter(card, card_filter)]


def _card_matches_filter(card: Mapping[str, Any], card_filter: Mapping[str, Any]) -> bool:
    for field, expected in card_filter.items():
        if field in {"limit", "top_k"}:
            continue
        actual = _clean_text(card.get(field))
        expected_values = _as_text_list(expected)
        if not expected_values:
            continue
        if actual not in expected_values:
            return False
    return True


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
