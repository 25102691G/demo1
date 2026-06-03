from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from guideline_skill.schema import (
    DiseaseSkillPack,
    RecommendationCard,
    RoutingProfile,
    SubSkill,
    load_skill_pack,
)
from patient_case_extractor import extract_patient_case
from skill_router import DiseaseSkillRouter, RoutingResult, route_disease_skills


ROOT = Path(__file__).resolve().parents[1]
CROHN_SEED_PATH = ROOT / "data" / "skills" / "crohn_disease_2023_guangzhou.yaml"


def test_routes_typical_crohn_case_to_crohn_seed_skill() -> None:
    patient_case = extract_patient_case(
        "腹痛腹泻三个月，体重下降，肠镜提示回盲部溃疡和狭窄"
    )
    crohn_pack = load_skill_pack(CROHN_SEED_PATH)

    results = route_disease_skills(patient_case, [crohn_pack], top_k=5)

    assert len(results) == 1
    assert results[0].skill_name == "crohn_disease_2023_guangzhou"
    assert results[0].score > 0
    assert results[0].matched_symptoms == ["chronic diarrhea", "abdominal pain", "weight loss"]
    assert "colonoscopy" in results[0].matched_tests
    assert "stricture" in results[0].matched_findings
    assert "not diagnostic probability" in results[0].reason


def test_router_returns_top_k_across_multiple_skill_packs() -> None:
    patient_case = extract_patient_case(
        "腹痛腹泻三个月，体重下降，肠镜提示回盲部纵行溃疡和狭窄"
    )
    crohn_pack = load_skill_pack(CROHN_SEED_PATH)
    respiratory_pack = _minimal_pack(
        skill_name="asthma_seed",
        disease_name="Asthma",
        aliases=["Asthma", "哮喘"],
        key_symptoms=["wheezing", "cough"],
        key_tests=["spirometry"],
        key_findings=["reversible airflow limitation"],
        red_flags=["respiratory failure"],
    )

    results = DiseaseSkillRouter().route(
        patient_case,
        [respiratory_pack, crohn_pack],
        top_k=2,
    )

    assert [result.skill_name for result in results] == [
        "crohn_disease_2023_guangzhou",
        "asthma_seed",
    ]
    assert results[0].score > results[1].score
    assert results[1].score == 0


def test_router_scores_aliases_and_red_flags() -> None:
    patient_case = extract_patient_case("医生怀疑克罗恩病，患者高热、剧烈腹痛。")
    crohn_pack = load_skill_pack(CROHN_SEED_PATH)

    result = route_disease_skills(patient_case, [crohn_pack], top_k=1)[0]

    assert "克罗恩病" in result.matched_aliases
    assert "high fever with severe abdominal pain" in result.matched_red_flags
    assert result.score >= 9


def test_routing_result_schema_shape() -> None:
    result = RoutingResult(
        skill_name="demo",
        disease_name="Demo disease",
        score=0,
        reason="No routing profile terms matched; score is 0 and is not a diagnostic probability.",
    )

    assert result.matched_symptoms == []
    assert result.score == 0


def test_route_skill_cli_outputs_candidates() -> None:
    script = ROOT / "scripts" / "route_skill.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--skills",
            str(ROOT / "data" / "skills"),
            "--text",
            "腹痛腹泻三个月，体重下降，肠镜提示回盲部溃疡和狭窄",
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    payload = json.loads(completed.stdout)

    assert payload["candidates"][0]["skill_name"] == "crohn_disease_2023_guangzhou"
    assert payload["candidates"][0]["score"] > 0
    assert "reason" in payload["candidates"][0]


def _minimal_pack(
    *,
    skill_name: str,
    disease_name: str,
    aliases: list[str],
    key_symptoms: list[str],
    key_tests: list[str],
    key_findings: list[str],
    red_flags: list[str],
) -> DiseaseSkillPack:
    return DiseaseSkillPack(
        skill_name=skill_name,
        disease_name=disease_name,
        disease_aliases=aliases,
        guideline_name=f"{disease_name} demo guideline",
        guideline_version="seed",
        source_pdf="demo.pdf",
        target_users=["clinical agent"],
        scope="Demo pack for router tests.",
        routing_profile=RoutingProfile(
            body_system="demo",
            key_symptoms=key_symptoms,
            key_tests=key_tests,
            key_findings=key_findings,
            red_flags=red_flags,
            must_differentiate=["demo mimic"],
            disease_aliases=aliases,
        ),
        subskills=[
            SubSkill(
                subskill_id="demo_screening",
                name="Demo screening",
                description="Demo subskill.",
                recommendation_ids=["DEMO-001"],
            )
        ],
        recommendation_cards=[
            RecommendationCard(
                recommendation_id="DEMO-001",
                source_section="Demo",
                clinical_task="routing",
                population="Demo",
                condition="Demo",
                action="Demo",
                evidence_level="seed",
                recommendation_strength="seed",
                rationale="Demo",
                source_span="Demo",
            )
        ],
        safety_constraints=["Do not diagnose."],
    )
