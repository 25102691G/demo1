from __future__ import annotations

from datetime import datetime
import json
import subprocess
import sys
from pathlib import Path

from agent_orchestrator import (
    AgentResponse,
    MedicalGuidelineAgentOrchestrator,
    load_skill_packs,
    run_agent,
)


ROOT = Path(__file__).resolve().parents[1]
SKILLS_DIR = ROOT / "data" / "skills"


def test_orchestrator_runs_full_crohn_flow() -> None:
    response = run_agent(
        "腹痛腹泻三个月，体重下降，肠镜提示回盲部及回肠末端多发溃疡并狭窄",
        skills_path=SKILLS_DIR,
    )

    assert isinstance(response, AgentResponse)
    assert response.patient_case_summary["symptoms"] == ["腹痛", "腹泻", "体重下降"]
    assert response.candidate_diseases[0].skill_name == "crohn_disease_2023_guangzhou"
    assert response.skill_results[0].skill_name == "crohn_disease_2023_guangzhou"
    assert response.skill_results[0].suspicion_level == "suspected"
    assert "不是最终诊断" in response.final_assessment
    assert response.recommended_next_steps
    assert response.disclaimer
    assert "Agent Summary" in response.readable_summary


def test_orchestrator_prioritizes_red_flag_safety() -> None:
    response = MedicalGuidelineAgentOrchestrator(SKILLS_DIR).run(
        "腹痛腹泻三个月，出现肠梗阻、大量便血和高热。"
    )

    assert response.safety_warnings
    assert response.final_assessment.startswith("因输入包含红旗")
    assert response.recommended_next_steps[0].startswith("优先处理红旗")


def test_run_agent_cli_outputs_structured_json_with_readable_summary() -> None:
    script = ROOT / "scripts" / "run_agent.py"
    started_at = datetime.now()
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--skills",
            str(SKILLS_DIR),
            "--text",
            "腹痛腹泻三个月，体重下降，肠镜提示回盲部及回肠末端多发溃疡并狭窄",
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    payload = json.loads(completed.stdout)

    assert payload["candidate_diseases"][0]["skill_name"] == "crohn_disease_2023_guangzhou"
    assert payload["skill_results"][0]["suspicion_level"] == "suspected"
    assert payload["final_assessment"]
    assert payload["recommended_next_steps"]
    assert "Agent Summary" in payload["readable_summary"]

    output_dir = ROOT / "data" / "output"
    expected_names = {
        f"{started_at.strftime('%Y%m%d_%H_%M')}.json",
        f"{datetime.now().strftime('%Y%m%d_%H_%M')}.json",
    }
    output_files = [output_dir / name for name in expected_names]
    saved_files = [path for path in output_files if path.exists()]

    assert saved_files
    assert any(json.loads(path.read_text(encoding="utf-8")) == payload for path in saved_files)
