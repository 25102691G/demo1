from __future__ import annotations

from pathlib import Path

from agent_orchestrator import run_agent


ROOT = Path(__file__).resolve().parents[1]
SKILLS_DIR = ROOT / "data" / "skills"


def test_readable_summary_is_chinese_and_stage_aware() -> None:
    response = run_agent(
        "腹痛腹泻三个月，体重下降，肠镜提示回盲部及回肠末端多发溃疡并狭窄",
        skills_path=SKILLS_DIR,
    )

    summary = response.readable_summary

    assert "Agent Summary" in summary
    assert "当前判断" in summary
    assert "为什么还不能确诊" in summary
    assert "最优先补充信息" in summary
    assert "建议下一步" in summary
    assert "需要鉴别的疾病" in summary
    assert "诊断资料补全" in summary
    assert "不是最终诊断" in summary
    assert "Recommended next steps" not in summary


def test_agent_response_top_level_output_is_stage_ranked() -> None:
    response = run_agent(
        "腹痛腹泻三个月，体重下降，肠镜提示回盲部及回肠末端多发溃疡并狭窄",
        skills_path=SKILLS_DIR,
    )

    assert set(response.clinical_stage.stages).issuperset(
        {
            "diagnostic_workup",
            "differential_diagnosis",
            "extent_and_complication_assessment",
        }
    )
    assert response.skill_results[0].suspicion_level == "suspected"
    assert len(response.top_missing_information) <= 7

    steps = response.recommended_next_steps
    joined = "\n".join(steps)
    assert "confirmed" not in response.final_assessment.casefold()
    assert "活检" in steps[0] and "病理" in steps[0]
    assert any("CTE 或 MRE" in step for step in steps)
    assert any("CRP、ESR" in step for step in steps)
    assert any("肠结核" in step for step in steps)
    assert any("感染性肠炎" in step for step in steps)
    assert "Base treatment suggestions" not in joined

    cte_index = next(i for i, step in enumerate(steps) if "CTE 或 MRE" in step)
    capsule_index = next(i for i, step in enumerate(steps) if "胶囊" in step)
    assert cte_index < capsule_index
