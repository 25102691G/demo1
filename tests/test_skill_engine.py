from __future__ import annotations

from pathlib import Path

from skill_engine import router as router_module
from skill_engine.case_normalizer import normalize_case, normalize_case_from_json
from skill_engine.output_builder import build_workflow_output
from skill_engine.router import route_skills
from skill_engine.schemas import load_json_schema, validate_json
from skill_engine.skill_loader import SkillPack
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
        return {
            "symptoms": [
                {
                    "name": "腹痛",
                    "hpo_code": "HP:0002027",
                    "hpo_term": "腹痛",
                    "similarity_score": 1.0,
                    "status": "mapped",
                },
                {
                    "name": "腹泻",
                    "hpo_code": "HP:0002014",
                    "hpo_term": "腹泻",
                    "similarity_score": 1.0,
                    "status": "mapped",
                },
            ]
        }


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
    assert canonical_case["symptoms"] == [
        {
            "name": "腹痛",
            "hpo_code": "HP:0002027",
            "hpo_term": "腹痛",
            "similarity_score": 1.0,
            "status": "mapped",
        },
        {
            "name": "腹泻",
            "hpo_code": "HP:0002014",
            "hpo_term": "腹泻",
            "similarity_score": 1.0,
            "status": "mapped",
        },
    ]
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


def test_skill_pack_fixture_has_workflow_shape() -> None:
    pack = _load_crohn_pack()

    assert pack.skill_id == "workflow_test_skill"
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
    canonical_case = {
        "case_id": "case_router",
        "raw_input": "腹痛",
        "symptoms": [{"name": "腹痛", "hpo_code": "HP:0002027"}],
    }
    pack = _make_router_pack(
        [
            {
                "name": "腹痛",
                "hpo_code": "HP:0002027",
                "hpo_term": "腹痛",
                "similarity_score": 1.0,
                "status": "mapped",
                "weight": 0.2,
            }
        ]
    )
    candidate = route_skills(canonical_case, [pack], top_k=1)[0]

    assert candidate["skill_id"] == "test_skill"
    assert candidate["disease_name"] == "测试病"
    assert candidate["score"] == 0.2
    assert candidate["matched_features"] == [
        {
            "name": "腹痛",
            "hpo_code": "HP:0002027",
            "hpo_term": "腹痛",
            "similarity_score": 1.0,
            "status": "mapped",
        }
    ]
    assert "missing_key_evidence" in candidate


def test_router_matches_symptom_hpo_codes_with_same_definition_term(monkeypatch) -> None:
    monkeypatch.setattr(
        router_module,
        "_load_hpo_code_terms",
        lambda: {"HP:CASE": "同一术语", "HP:FEATURE": "同一术语"},
    )
    canonical_case = {
        "case_id": "case_router",
        "raw_input": "同义症状",
        "symptoms": [{"name": "同义症状", "hpo_code": "HP:CASE"}],
    }
    pack = _make_router_pack(
        [
            {
                "name": "同义特征",
                "hpo_code": "HP:FEATURE",
                "hpo_term": "同一术语",
                "similarity_score": 0.9,
                "status": "mapped",
                "weight": 0.4,
            }
        ]
    )

    candidate = route_skills(canonical_case, [pack], top_k=1)[0]

    assert candidate["score"] == 0.4
    assert candidate["matched_features"] == [
        {
            "name": "同义特征",
            "hpo_code": "HP:FEATURE",
            "hpo_term": "同一术语",
            "similarity_score": 0.9,
            "status": "mapped",
        }
    ]


def test_router_does_not_match_symptom_text_without_hpo_match(monkeypatch) -> None:
    monkeypatch.setattr(
        router_module,
        "_load_hpo_code_terms",
        lambda: {"HP:CASE": "病例术语", "HP:FEATURE": "技能术语"},
    )
    canonical_case = {
        "case_id": "case_router",
        "raw_input": "腹痛",
        "symptoms": [{"name": "腹痛", "hpo_code": "HP:CASE"}],
    }
    pack = _make_router_pack(
        [
            {
                "name": "腹痛",
                "hpo_code": "HP:FEATURE",
                "hpo_term": "技能术语",
                "similarity_score": 1.0,
                "status": "mapped",
                "weight": 0.2,
            }
        ]
    )

    candidate = route_skills(canonical_case, [pack], top_k=1)[0]

    assert candidate["score"] == 0.0
    assert candidate["matched_features"] == []


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
    return _make_workflow_pack()


def _make_router_pack(symptoms: list[dict[str, object]]) -> SkillPack:
    return SkillPack(
        skill_dir=ROOT,
        skill={
            "metadata": {"skill_id": "test_skill", "disease_name": "测试病"},
            "routing_profile": {
                "positive_features": {"symptoms": symptoms},
                "scoring": {
                    "normalization": "sum_max_1",
                    "thresholds": {"candidate": 0.1, "strong_candidate": 0.5},
                    "top_k_default": 1,
                },
            },
        },
        skill_id="test_skill",
        disease_name="测试病",
        cards=[],
        cards_by_id={},
    )


def _make_workflow_pack() -> SkillPack:
    return SkillPack(
        skill_dir=ROOT,
        skill={
            "metadata": {"skill_id": "workflow_test_skill", "disease_name": "工作流测试病"},
            "routing_profile": {
                "positive_features": {
                    "symptoms": [
                        {
                            "name": "腹痛",
                            "hpo_code": "HP:0002027",
                            "hpo_term": "腹痛",
                            "similarity_score": 1.0,
                            "status": "mapped",
                            "weight": 0.2,
                        }
                    ]
                },
                "scoring": {
                    "method": "hybrid_weighted_semantic",
                    "normalization": "sum_max_1",
                    "thresholds": {"candidate": 0.1, "strong_candidate": 0.5},
                    "top_k_default": 1,
                    "safety_override": True,
                },
            },
            "workflow": {
                "entrypoint": "candidate_summary_output",
                "steps": [
                    {
                        "step_id": "candidate_summary_output",
                        "type": "terminal_output",
                        "config": {"output_template": "candidate_summary"},
                        "transitions": [],
                    }
                ],
            },
            "subskills": [{"subskill_id": "general_guideline_support"}],
            "output_templates": {"candidate_summary": {"audience": "clinician", "structure": []}},
        },
        skill_id="workflow_test_skill",
        disease_name="工作流测试病",
        cards=[],
        cards_by_id={},
    )
