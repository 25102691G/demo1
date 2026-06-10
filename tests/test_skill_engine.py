from __future__ import annotations

from pathlib import Path

from skill_engine.case_normalizer import normalize_case, normalize_case_from_json
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


class MockHpoExtractor:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def extract_hpo_from_text(self, text: str, deepseek_client: object) -> dict[str, list[dict[str, object]]]:
        self.calls.append((text, deepseek_client))
        return {"symptoms": [{"name": "腹痛", "weight": 0.2}, {"name": "腹泻", "weight": 0.2}]}


class MockDeepSeekClient:
    pass


def _normalize_case(text: str) -> dict[str, object]:
    return normalize_case(
        text,
        CASE_SCHEMA,
        hpo_extractor=MockHpoExtractor(),
        deepseek_client=MockDeepSeekClient(),
    )


def test_case_normalizer_minimal_text() -> None:
    hpo_extractor = MockHpoExtractor()
    deepseek_client = MockDeepSeekClient()
    canonical_case = normalize_case(
        "42岁男性，腹痛腹泻半年，CRP升高。",
        CASE_SCHEMA,
        hpo_extractor=hpo_extractor,
        deepseek_client=deepseek_client,
    )

    assert set(canonical_case) == {"case_id", "raw_input", "symptoms"}
    assert hpo_extractor.calls == [("42岁男性，腹痛腹泻半年，CRP升高。", deepseek_client)]
    assert canonical_case["symptoms"] == [{"name": "腹痛", "weight": 0.2}, {"name": "腹泻", "weight": 0.2}]
    validate_json(canonical_case, load_json_schema(CASE_SCHEMA), label="canonical_case")


def test_case_normalizer_from_json_overrides_hpo_symptoms() -> None:
    canonical_case = normalize_case_from_json(
        {"raw_input": "structured case", "symptoms": [{"name": "结构化症状", "weight": 0.8}]},
        None,
        CASE_SCHEMA,
        hpo_extractor=MockHpoExtractor(),
        deepseek_client=MockDeepSeekClient(),
    )

    assert canonical_case["raw_input"] == "structured case"
    assert canonical_case["symptoms"] == [{"name": "结构化症状", "weight": 0.8}]
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
    canonical_case = _normalize_case("腹痛腹泻半年，体重下降，肛瘘，粪便钙卫蛋白升高。")
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
    canonical_case = _normalize_case("腹痛腹泻半年，粪便钙卫蛋白升高。")
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
    canonical_case = _normalize_case("腹痛腹泻半年，体重下降，肛瘘，粪便钙卫蛋白升高。")
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
    assert "input_text" not in output
    assert output["canonical_case"] == canonical_case


def test_safety_red_flags_are_not_rule_extracted_by_normalizer() -> None:
    pack = _load_crohn_pack()
    canonical_case = _normalize_case("剧烈腹痛、高热、休克，腹泻三天。")
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

    assert "red_flags" not in canonical_case
    assert output["safety"]["has_red_flags"] is False
    assert output["safety"]["workflow_stopped"] is False


def _load_crohn_pack() -> SkillPack:
    return load_skill_pack(CROHN_SKILL_DIR, SKILL_SCHEMA)
