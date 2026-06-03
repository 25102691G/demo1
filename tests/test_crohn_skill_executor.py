from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from guideline_skill.schema import load_skill_pack
from patient_case_extractor import extract_patient_case
from skill_executor import CrohnDiseaseSkillExecutor, execute_crohn_skill


ROOT = Path(__file__).resolve().parents[1]
CROHN_SEED_PATH = ROOT / "data" / "skills" / "crohn_disease_2023_guangzhou.yaml"


def _run(text: str):
    return execute_crohn_skill(
        extract_patient_case(text),
        load_skill_pack(CROHN_SEED_PATH),
    )


def test_only_abdominal_pain_and_diarrhea_is_possible_with_next_steps() -> None:
    result = _run("只有腹痛腹泻两周。")

    assert result.suspicion_level == "possible"
    assert any("腹痛" in item for item in result.support_evidence)
    assert any("结肠镜" in step for step in result.recommended_next_steps)
    assert result.missing_information
    assert result.differential_diagnoses
    assert result.safety_warnings
    assert result.suspicion_level != "confirmed_by_doctor_only"


def test_endoscopic_ileocecal_ulcer_and_stricture_is_suspected() -> None:
    result = _run("腹痛腹泻三个月，体重下降，肠镜提示回盲部多发溃疡并狭窄。")

    assert result.suspicion_level == "suspected"
    assert any("回盲部" in item for item in result.support_evidence)
    assert any("多发溃疡" in item for item in result.support_evidence)
    assert any("狭窄" in item for item in result.support_evidence)
    assert any(item.information_key == "pathology" for item in result.missing_information)


def test_endoscopy_plus_pathology_is_probable_but_not_auto_confirmed() -> None:
    result = _run(
        "腹痛腹泻半年，MRE提示小肠受累，结肠镜提示回肠末端纵行溃疡，"
        "活检病理提示慢性炎症和肉芽肿。"
    )

    assert result.suspicion_level == "probable"
    assert result.suspicion_level != "confirmed_by_doctor_only"
    assert any("病理支持信息：慢性炎症" in item for item in result.support_evidence)
    assert any("病理支持信息：肉芽肿" in item for item in result.support_evidence)
    assert any("排除" in warning or "不能自动确诊" in warning for warning in result.safety_warnings)
    assert any(item.disease_name == "肠结核" for item in result.differential_diagnoses)


def test_red_flags_generate_safety_warnings() -> None:
    result = _run("腹痛腹泻三个月，出现肠梗阻、大量便血和高热。")

    assert result.suspicion_level in {"possible", "suspected"}
    assert any("肠梗阻" in warning for warning in result.safety_warnings)
    assert any("大量便血" in warning for warning in result.safety_warnings)
    assert any("高热" in warning for warning in result.safety_warnings)
    assert result.recommended_next_steps[0].startswith("因出现红旗征象")


def test_missing_information_is_required_when_information_is_incomplete() -> None:
    result = _run("腹痛腹泻，CRP升高。")

    keys = {item.information_key for item in result.missing_information}

    assert {"ileocolonoscopy", "cross_sectional_imaging", "pathology"}.issubset(keys)
    assert "intestinal_tuberculosis_exclusion" in keys
    assert "infectious_enteritis_exclusion" in keys


def test_doctor_confirmed_phrase_is_the_only_confirmed_state() -> None:
    result = _run("医生已确诊克罗恩病，目前复查肠镜。")

    assert result.suspicion_level == "confirmed_by_doctor_only"
    assert any("医生已确诊" in item for item in result.support_evidence)


def test_crohn_skill_executor_class_api() -> None:
    executor = CrohnDiseaseSkillExecutor()
    result = executor.execute(
        extract_patient_case("腹痛腹泻。"),
        load_skill_pack(CROHN_SEED_PATH),
    )

    assert result.skill_name == "crohn_disease_2023_guangzhou"
    assert result.source_references


def test_run_skill_cli_outputs_skill_execution_result_json() -> None:
    script = ROOT / "scripts" / "run_skill.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--skill",
            str(CROHN_SEED_PATH),
            "--text",
            "腹痛腹泻三个月，体重下降，肠镜提示回盲部及回肠末端多发溃疡并狭窄",
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    payload = json.loads(completed.stdout)

    assert payload["skill_name"] == "crohn_disease_2023_guangzhou"
    assert payload["suspicion_level"] == "suspected"
    assert payload["differential_diagnoses"]
    assert payload["recommended_next_steps"]
