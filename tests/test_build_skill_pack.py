from __future__ import annotations

import json
import importlib.util
import sys
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SKILL_SCHEMA = ROOT / "schema" / "skill_pack.schema.json"
CARD_SCHEMA = ROOT / "schema" / "recommendation_card.schema.json"

_SPEC = importlib.util.spec_from_file_location(
    "build_skill_pack", ROOT / "scripts" / "build_skill_pack.py"
)
assert _SPEC and _SPEC.loader
builder = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = builder
_SPEC.loader.exec_module(builder)


def test_build_skill_pack_from_result_jsonl(tmp_path: Path) -> None:
    source_dir = tmp_path / "Example guideline"
    source_dir.mkdir()
    cards_path = source_dir / "result.jsonl"
    _write_jsonl(cards_path, [_card("CARD-001"), _card("CARD-002", clinical_task="治疗方案")])

    cards = builder.validate_cards(builder.load_jsonl(cards_path), CARD_SCHEMA)
    hpo_extractor, deepseek_client = _mock_hpo_dependencies()
    skill = builder.build_skill_pack(
        cards,
        builder.load_taxonomy(),
        schema_version="0.3",
        hpo_extractor=hpo_extractor,
        deepseek_client=deepseek_client,
    )
    builder.validate_cross_references(
        skill,
        cards,
        output_dir=source_dir,
    )
    builder.validate_skill_schema(skill, SKILL_SCHEMA)
    output_dir = builder.write_skill_pack(
        skill,
        cards,
        source_dir,
        force=True,
    )

    assert output_dir.name == "Example guideline"
    raw_skill = yaml.safe_load((output_dir / "skill.yaml").read_text(encoding="utf-8"))
    assert raw_skill["knowledge_base"]["cards_path"] == "result.jsonl"
    assert (output_dir / "result.jsonl").exists()
    assert not (output_dir / "cards.jsonl").exists()
    assert raw_skill["metadata"]["disease_name"] == "Example disease"
    assert hpo_extractor.calls == [
        (cards[0]["raw_chunk_text"], deepseek_client),
        (cards[1]["raw_chunk_text"], deepseek_client),
    ]
    assert raw_skill["routing_profile"]["positive_features"] == {
        "symptoms": [{"name": "HPO phenotype", "weight": 0.2}]
    }


def test_workflow_references_existing_steps_subskills_and_templates(tmp_path: Path) -> None:
    cards_path = tmp_path / "cards.jsonl"
    _write_jsonl(cards_path, [_card("CARD-001")])
    cards = builder.validate_cards(builder.load_jsonl(cards_path), CARD_SCHEMA)
    hpo_extractor, deepseek_client = _mock_hpo_dependencies()
    skill = builder.build_skill_pack(
        cards,
        builder.load_taxonomy(),
        schema_version="0.3",
        hpo_extractor=hpo_extractor,
        deepseek_client=deepseek_client,
    )

    step_ids = {step["step_id"] for step in skill["workflow"]["steps"]}
    subskill_ids = {subskill["subskill_id"] for subskill in skill["subskills"]}
    template_ids = set(skill["output_templates"])

    assert all(
        transition["to"] in step_ids
        for step in skill["workflow"]["steps"]
        for transition in step["transitions"]
    )
    assert all(
        step.get("config", {}).get("subskill_ref") in subskill_ids
        for step in skill["workflow"]["steps"]
        if "subskill_ref" in step.get("config", {})
    )
    assert all(
        step.get("config", {}).get("output_template") in template_ids
        for step in skill["workflow"]["steps"]
        if "output_template" in step.get("config", {})
    )
    builder.validate_cross_references(skill, cards)


def test_duplicate_card_id_fails(tmp_path: Path) -> None:
    cards_path = tmp_path / "cards.jsonl"
    _write_jsonl(cards_path, [_card("CARD-001"), _card("CARD-001")])

    with pytest.raises(builder.BuildSkillPackError, match="duplicate card_id"):
        builder.validate_cards(builder.load_jsonl(cards_path), CARD_SCHEMA)


def test_card_schema_validation_reports_missing_required_field(tmp_path: Path) -> None:
    card = _card("CARD-001")
    del card["action"]
    cards_path = tmp_path / "cards.jsonl"
    _write_jsonl(cards_path, [card])

    with pytest.raises(builder.BuildSkillPackError, match="'action' is a required property"):
        builder.validate_cards(builder.load_jsonl(cards_path), CARD_SCHEMA)


def test_validate_cards_keeps_result_jsonl_payload_unchanged(tmp_path: Path) -> None:
    cards_path = tmp_path / "result.jsonl"
    raw_cards = [
        _card("CARD-001"),
        _card("CARD-002", clinical_task="治疗方案"),
    ]
    _write_jsonl(
        cards_path,
        raw_cards,
    )

    cards = builder.validate_cards(builder.load_jsonl(cards_path), CARD_SCHEMA)

    assert cards == raw_cards


def test_discover_cards_sources_accepts_file_or_directory(tmp_path: Path) -> None:
    first_dir = tmp_path / "First guideline"
    second_dir = tmp_path / "Second guideline"
    first_dir.mkdir()
    second_dir.mkdir()
    first_cards = first_dir / "result.jsonl"
    second_cards = second_dir / "custom_cards.jsonl"
    _write_jsonl(first_cards, [_card("CARD-001")])
    _write_jsonl(second_cards, [_card("CARD-002")])
    (first_dir / "summary.json").write_text("{}", encoding="utf-8")

    assert builder.discover_cards_sources(first_cards) == [first_cards]
    assert builder.discover_cards_sources(tmp_path) == [first_cards, second_cards]
    assert builder.infer_output_package_name(first_cards) == "First guideline"
    assert builder.infer_output_package_name(second_cards) == "custom_cards"


class MockHpoExtractor:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def extract_hpo_from_text(self, text: str, deepseek_client: object) -> dict[str, list[dict[str, object]]]:
        self.calls.append((text, deepseek_client))
        return {"symptoms": [{"name": "HPO phenotype", "weight": 0.2}]}


class MockDeepSeekClient:
    pass


def _mock_hpo_dependencies() -> tuple[MockHpoExtractor, MockDeepSeekClient]:
    return MockHpoExtractor(), MockDeepSeekClient()


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def _card(card_id: str, *, clinical_task: str = "诊断检查") -> dict[str, object]:
    return {
        "record_type": "recommendation_card",
        "card_id": card_id,
        "source_statement_id": "REC-1",
        "disease": "Example disease",
        "guideline": {
            "title": "Example disease guideline 2024 Springfield",
            "source_file": "example-guideline-2024.pdf",
            "doc_type": "structured_guideline",
        },
        "clinical_stage": "诊断及评估",
        "clinical_task": clinical_task,
        "population": "suspected patients",
        "condition": "需要完善诊断证据时",
        "raw_chunk_text": "推荐意见1：结合血常规、结肠镜和病理检查进行综合判断。",
        "action": "结合血常规、结肠镜和病理检查进行综合判断",
        "do_not": ["不要替代医生诊断"],
        "required_inputs": ["症状", "实验室检查"],
        "supporting_features": ["腹痛"],
        "recommended_tests": ["血常规", "结肠镜", "病理活检"],
        "safety_notes": ["本条内容不能替代医生诊断。"],
        "evidence": {
            "evidence_quality_normalized": "unknown",
            "recommendation_strength_normalized": "unknown",
        },
        "source_location": {
            "page_start": 1,
            "page_end": 1,
            "section": "诊断及评估",
        },
    }
