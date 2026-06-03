from __future__ import annotations

from pathlib import Path

from agent_orchestrator import run_agent


ROOT = Path(__file__).resolve().parents[1]
SKILLS_DIR = ROOT / "data" / "skills"


def test_current_crohn_example_keeps_stage_ranked_consistent_output() -> None:
    response = run_agent(
        "腹痛腹泻三个月，体重下降，肠镜提示回盲部及回肠末端多发溃疡并狭窄",
        skills_path=SKILLS_DIR,
    )
    result = response.skill_results[0]

    assert result.suspicion_level == "suspected"
    assert set(response.clinical_stage.stages).issuperset(
        {
            "diagnostic_workup",
            "differential_diagnosis",
            "extent_and_complication_assessment",
        }
    )
    assert len(response.top_missing_information) <= 7
    assert result.recommended_next_steps == response.recommended_next_steps
    assert result.missing_information == response.top_missing_information
    assert result.raw_missing_information
    assert result.full_missing_information
    assert result.top_missing_information == response.top_missing_information

    steps = response.recommended_next_steps
    cte_index = next(index for index, step in enumerate(steps) if "CTE 或 MRE" in step)
    capsule_index = next(index for index, step in enumerate(steps) if "胶囊" in step)
    assert cte_index < capsule_index
    assert all(reference.page is not None for reference in result.source_references)


def test_visible_output_has_chinese_missing_and_no_direct_drug_plan() -> None:
    response = run_agent(
        "腹痛腹泻三个月，体重下降，肠镜提示回盲部及回肠末端多发溃疡并狭窄",
        skills_path=SKILLS_DIR,
    )
    visible_text = "\n".join(
        [
            response.readable_summary,
            *response.recommended_next_steps,
            *[item.question for item in response.full_missing_information],
            *[item.reason for item in response.full_missing_information],
        ]
    )

    assert "perianal disease" not in visible_text
    assert "pregnancy status if relevant" not in visible_text
    assert "renal function if contrast needed" not in visible_text
    assert "Base treatment suggestions" not in visible_text
    assert "英夫利昔" not in visible_text
    assert "阿达木" not in visible_text
    assert "确诊为克罗恩病" not in visible_text


def test_differential_diagnoses_have_case_specific_features() -> None:
    response = run_agent(
        "腹痛腹泻三个月，体重下降，肠镜提示回盲部及回肠末端多发溃疡并狭窄",
        skills_path=SKILLS_DIR,
    )
    diagnoses = {
        item.disease_name: item
        for item in response.skill_results[0].differential_diagnoses
    }

    tb = diagnoses["肠结核"]
    assert {"回盲部受累", "溃疡", "狭窄"}.issubset(set(tb.supporting_features))
    assert {"未提供结核感染证据", "未提供胸部影像", "未提供抗酸染色/分枝杆菌检测"}.issubset(
        set(tb.against_features)
    )
    assert tb.missing_tests
    assert diagnoses["溃疡性结肠炎"].missing_tests
    assert diagnoses["感染性肠炎"].missing_tests
