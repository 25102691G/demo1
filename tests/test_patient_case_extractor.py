from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from patient_case_extractor import PatientCaseExtractor, extract_patient_case


ROOT = Path(__file__).resolve().parents[1]


def test_extracts_typical_crohn_related_symptoms_and_findings() -> None:
    patient_case = extract_patient_case(
        "我腹痛腹泻三个月，体重下降，CRP 30 mg/L，粪便钙卫蛋白升高，"
        "肠镜提示回盲部纵行溃疡和狭窄，活检提示慢性炎症，MRE提示小肠受累。"
    )

    assert patient_case.symptoms == ["腹痛", "腹泻", "体重下降"]
    assert "CRP" in patient_case.labs
    assert "30mg/L" in patient_case.labs["CRP"]["value_texts"]
    assert "粪便钙卫蛋白" in patient_case.labs
    assert patient_case.endoscopy == ["结肠镜", "回盲部", "纵行溃疡", "溃疡", "狭窄", "活检"]
    assert patient_case.pathology == ["慢性炎症"]
    assert patient_case.imaging == ["MRE"]
    assert patient_case.red_flags == []


def test_extracts_red_flags() -> None:
    patient_case = extract_patient_case(
        "患者高热，剧烈腹痛，疑似肠梗阻，出现严重脱水和意识不清。"
    )

    assert patient_case.symptoms == ["腹痛", "发热"]
    assert patient_case.red_flags == ["剧烈腹痛", "肠梗阻", "高热", "严重脱水", "意识障碍"]


def test_extracts_imaging_endoscopy_and_pathology_terms() -> None:
    patient_case = extract_patient_case(
        "肛瘘伴肛周脓肿，肛周 MRI 发现瘘管，CTE显示狭窄，"
        "结肠镜见铺路石样改变，病理提示肉芽肿和透壁性炎症。"
    )

    assert patient_case.symptoms == ["肛瘘", "肛周脓肿"]
    assert patient_case.imaging == ["CTE", "肛周 MRI", "MRI"]
    assert "瘘管" in patient_case.endoscopy
    assert "铺路石样改变" in patient_case.endoscopy
    assert patient_case.pathology == ["肉芽肿", "透壁性炎症"]


def test_unrecognized_clauses_go_to_unknowns() -> None:
    patient_case = extract_patient_case("最近睡眠不好，腹痛一周，家里比较担心。")

    assert patient_case.symptoms == ["腹痛"]
    assert patient_case.unknowns == ["最近睡眠不好", "家里比较担心"]


def test_empty_text_is_rejected() -> None:
    with pytest.raises(ValueError, match="raw_text"):
        PatientCaseExtractor().extract("  ")


def test_extract_patient_case_cli_outputs_json() -> None:
    script = ROOT / "scripts" / "extract_patient_case.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--text",
            "我腹痛腹泻三个月，体重下降，肠镜提示回盲部溃疡和狭窄",
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    payload = json.loads(completed.stdout)

    assert payload["raw_text"].startswith("我腹痛腹泻")
    assert payload["symptoms"] == ["腹痛", "腹泻", "体重下降"]
    assert payload["endoscopy"] == ["结肠镜", "回盲部", "溃疡", "狭窄"]
