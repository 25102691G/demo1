from __future__ import annotations

import pytest

from guideline_skill.schema import (
    DiseaseSkillPack,
    PatientCase,
    RecommendationCard,
    RoutingProfile,
    SkillExecutionResult,
    SubSkill,
    load_skill_pack,
    save_skill_pack,
    validate_skill_pack,
)


def minimal_crohn_disease_guideline_pack() -> DiseaseSkillPack:
    aliases = ["Crohn disease", "CD", "克罗恩病"]
    return DiseaseSkillPack(
        skill_name="crohn_disease_guideline_pack",
        disease_name="Crohn disease",
        disease_aliases=aliases,
        guideline_name="中国克罗恩病诊治指南",
        guideline_version="2023 Guangzhou",
        source_pdf="china-crohns-guideline-2023.pdf",
        target_users=["gastroenterologist", "general physician", "clinical agent"],
        scope="Support guideline-based suspicion assessment and next-step planning; not final diagnosis.",
        routing_profile=RoutingProfile(
            body_system="gastrointestinal",
            key_symptoms=["chronic diarrhea", "abdominal pain", "weight loss", "perianal disease"],
            key_tests=["fecal calprotectin", "colonoscopy", "CTE", "MRE", "pathology"],
            key_findings=["segmental inflammation", "longitudinal ulcer", "terminal ileum involvement"],
            red_flags=["acute abdomen", "massive gastrointestinal bleeding", "suspected perforation"],
            must_differentiate=["intestinal tuberculosis", "intestinal Behcet disease", "lymphoma"],
            disease_aliases=aliases,
        ),
        subskills=[
            SubSkill(
                subskill_id="diagnostic_assessment",
                name="Diagnostic assessment",
                description="Evaluate whether clinical, laboratory, endoscopic, imaging, and pathology features support Crohn disease suspicion.",
                clinical_tasks=["suspicion assessment", "missing information review"],
                required_inputs=["symptoms", "labs", "imaging", "endoscopy", "pathology"],
                recommendation_ids=["REC-CD-001"],
                output_fields=[
                    "suspicion_level",
                    "support_evidence",
                    "against_evidence",
                    "missing_information",
                ],
            )
        ],
        recommendation_cards=[
            RecommendationCard(
                recommendation_id="REC-CD-001",
                source_section="Diagnosis",
                clinical_task="diagnosis",
                population="Patients with suspected Crohn disease",
                condition="No single diagnostic gold standard is available",
                action="Use integrated assessment of clinical manifestations, laboratory tests, imaging, endoscopy, and histopathology.",
                evidence_level="BPS",
                recommendation_strength="BPS",
                rationale="Crohn disease diagnosis requires synthesis across multiple clinical evidence sources.",
                required_inputs=["symptoms", "labs", "imaging", "endoscopy", "pathology"],
                safety_notes=["Do not output a final diagnosis without clinician confirmation."],
                source_span="Recommendation 1, PDF page 2",
            )
        ],
        safety_constraints=[
            "Never output a final diagnosis.",
            "Escalate emergency red flags to urgent medical care advice.",
        ],
    )


def test_minimal_crohn_pack_validates() -> None:
    pack = minimal_crohn_disease_guideline_pack()

    validated = validate_skill_pack(pack)

    assert validated.skill_name == "crohn_disease_guideline_pack"
    assert validated.routing_profile.body_system == "gastrointestinal"
    assert validated.recommendation_cards[0].recommendation_id == "REC-CD-001"


@pytest.mark.parametrize("suffix", [".yaml", ".json"])
def test_skill_pack_round_trips_yaml_and_json(tmp_path, suffix: str) -> None:
    pack = minimal_crohn_disease_guideline_pack()
    path = tmp_path / f"crohn_pack{suffix}"

    save_skill_pack(pack, path)
    loaded = load_skill_pack(path)

    assert loaded == pack
    assert loaded.disease_aliases == ["Crohn disease", "CD", "克罗恩病"]


def test_validate_rejects_unknown_recommendation_reference() -> None:
    pack = minimal_crohn_disease_guideline_pack()
    pack.subskills[0].recommendation_ids.append("REC-CD-999")

    with pytest.raises(ValueError, match="unknown recommendations"):
        validate_skill_pack(pack)


def test_validate_requires_routing_aliases_to_cover_pack_aliases() -> None:
    pack = minimal_crohn_disease_guideline_pack()
    pack.routing_profile.disease_aliases = ["Crohn disease"]

    with pytest.raises(ValueError, match="routing_profile.disease_aliases"):
        validate_skill_pack(pack)


def test_patient_case_and_skill_result_models() -> None:
    patient_case = PatientCase(
        raw_text="腹痛腹泻半年，体重下降，肠镜提示回肠末端纵行溃疡。",
        symptoms=["腹痛", "腹泻", "体重下降"],
        endoscopy=["回肠末端纵行溃疡"],
        unknowns=["粪便钙卫蛋白", "病理活检"],
    )

    result = SkillExecutionResult(
        skill_name="crohn_disease_guideline_pack",
        disease_name="Crohn disease",
        suspicion_level="suspected",
        support_evidence=["chronic diarrhea", "terminal ileum ulcer"],
        against_evidence=[],
        missing_information=[
            {
                "information_key": "pathology",
                "question": "是否已完成多肠段活检及病理评估？",
                "reason": "Crohn disease suspicion assessment needs histopathology input.",
                "priority": "high",
            }
        ],
        recommended_next_steps=["Consider colonoscopy biopsy review and CTE/MRE if not completed."],
        differential_diagnoses=[
            {
                "disease_name": "intestinal tuberculosis",
                "rationale": "Can mimic ileocecal inflammation and ulceration.",
                "distinguishing_tests": ["TB infection workup", "pathology review"],
            }
        ],
        safety_warnings=["This is not a final diagnosis."],
        source_references=[
            {
                "source_name": "中国克罗恩病诊治指南",
                "recommendation_id": "REC-CD-001",
                "source_span": "Recommendation 1, PDF page 2",
            }
        ],
    )

    assert patient_case.raw_text.startswith("腹痛")
    assert result.suspicion_level == "suspected"
    assert result.missing_information[0].information_key == "pathology"
