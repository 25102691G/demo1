from __future__ import annotations

from pathlib import Path

from skill_engine.case_normalizer import normalize_case
from skill_engine.output_builder import build_workflow_output
from skill_engine.router import route_skills
from skill_engine.schemas import load_json_schema, validate_json
from skill_engine.skill_loader import SkillPack, load_skill_pack
from skill_engine.workflow_engine import WorkflowEngine


ROOT = Path(__file__).resolve().parents[1]
CASE_SCHEMA = ROOT / "schema" / "canonical_case.schema.json"
SKILL_SCHEMA = ROOT / "schema" / "skill_pack.schema.json"
OUTPUT_SCHEMA = ROOT / "schema" / "workflow_output.schema.json"
CROHN_SKILL_DIR = ROOT / "data" / "skills" / "中国克罗恩病诊治指南（2023年·广州）"


def test_case_normalizer_minimal_text() -> None:
    canonical_case = normalize_case("42岁男性，腹痛腹泻半年，CRP升高。", CASE_SCHEMA)

    for field in (
        "case_id",
        "raw_input",
        "demographics",
        "symptoms",
        "signs",
        "vitals",
        "labs",
        "imaging",
        "endoscopy",
        "pathology",
        "diagnoses",
        "medications",
        "procedures",
        "red_flags",
        "extraction_quality",
    ):
        assert field in canonical_case
    assert canonical_case["labs"]["items"]
    assert canonical_case["imaging"] == {"items": []}
    assert canonical_case["endoscopy"] == {"items": []}
    assert canonical_case["pathology"] == {"items": []}
    assert 0 <= canonical_case["extraction_quality"]["confidence"] <= 1
    validate_json(canonical_case, load_json_schema(CASE_SCHEMA), label="canonical_case")


def test_skill_loader_loads_generated_skill() -> None:
    pack = _load_crohn_pack()

    assert pack.skill_id == "disease_skill_5098a99979"
    assert pack.cards
    assert len(pack.cards_by_id) == len(pack.cards)
    step_ids = {step["step_id"] for step in pack.skill["workflow"]["steps"]}
    subskill_ids = {subskill["subskill_id"] for subskill in pack.skill["subskills"]}
    assert pack.skill["workflow"]["entrypoint"] in step_ids
    assert all(
        transition["to"] in step_ids
        for step in pack.skill["workflow"]["steps"]
        for transition in step["transitions"]
    )
    assert all(
        step.get("config", {}).get("subskill_ref") in subskill_ids
        for step in pack.skill["workflow"]["steps"]
        if "subskill_ref" in step.get("config", {})
    )


def test_router_returns_schema_candidate() -> None:
    canonical_case = normalize_case(
        "腹痛腹泻半年，体重下降，肛瘘，粪便钙卫蛋白升高。",
        CASE_SCHEMA,
    )
    candidate = route_skills(canonical_case, [_load_crohn_pack()], top_k=1)[0]

    assert candidate["skill_id"] == "disease_skill_5098a99979"
    assert candidate["disease_name"] == "克罗恩病"
    assert candidate["score"] >= 0
    assert candidate["matched_features"]
    assert "missing_key_evidence" in candidate
    allowed_sources = {
        "symptom",
        "sign",
        "lab",
        "imaging",
        "endoscopy",
        "pathology",
        "diagnosis",
        "text",
        "other",
    }
    assert all(feature["source"] in allowed_sources for feature in candidate["matched_features"])


def test_workflow_engine_runs_without_disease_hardcoding() -> None:
    pack = _load_crohn_pack()
    canonical_case = normalize_case("腹痛腹泻半年，粪便钙卫蛋白升高。", CASE_SCHEMA)
    candidate = route_skills(canonical_case, [pack], top_k=1)[0]

    skill_output = WorkflowEngine().run(pack, canonical_case, candidate)

    assert skill_output["skill_id"] == pack.skill_id
    assert skill_output["workflow_status"] in {
        "completed",
        "stopped_for_safety",
        "missing_information",
        "candidate_only",
        "error",
    }
    assert skill_output["executed_steps"][-1]["type"] == "terminal_output"


def test_workflow_output_schema_validation() -> None:
    pack = _load_crohn_pack()
    canonical_case = normalize_case(
        "腹痛腹泻半年，体重下降，肛瘘，粪便钙卫蛋白升高。",
        CASE_SCHEMA,
    )
    top_candidates = route_skills(canonical_case, [pack], top_k=1)
    selected_outputs = [WorkflowEngine().run(pack, canonical_case, top_candidates[0])]

    output = build_workflow_output(
        canonical_case=canonical_case,
        top_candidates=top_candidates,
        selected_skill_outputs=selected_outputs,
        skill_packs=[pack],
        output_schema_path=OUTPUT_SCHEMA,
        debug=True,
    )

    validate_json(output, load_json_schema(OUTPUT_SCHEMA), label="workflow_output")
    assert output["case_id"] == canonical_case["case_id"]


def test_safety_red_flag_stops_workflow() -> None:
    pack = _load_crohn_pack()
    canonical_case = normalize_case("剧烈腹痛、高热、休克，腹泻三天。", CASE_SCHEMA)
    top_candidates = route_skills(canonical_case, [pack], top_k=1)
    selected_outputs = [WorkflowEngine().run(pack, canonical_case, top_candidates[0])]

    output = build_workflow_output(
        canonical_case=canonical_case,
        top_candidates=top_candidates,
        selected_skill_outputs=selected_outputs,
        skill_packs=[pack],
        output_schema_path=OUTPUT_SCHEMA,
        debug=True,
    )

    assert output["status"] == "stopped_for_safety"
    assert output["safety"]["has_red_flags"] is True
    assert output["safety"]["workflow_stopped"] is True


def _load_crohn_pack() -> SkillPack:
    return load_skill_pack(CROHN_SKILL_DIR, SKILL_SCHEMA)
