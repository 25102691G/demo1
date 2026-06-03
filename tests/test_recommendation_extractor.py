from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

from guideline_skill.schema import RecommendationCard, load_skill_pack
from pdf_guideline_parser import parse_guideline_sections
from recommendation_extractor import (
    extract_evidence_level,
    extract_recommendation_cards_from_sections,
    extract_recommendation_statement,
    extract_recommendation_strength,
    extract_recommendations_from_pdf,
)


ROOT = Path(__file__).resolve().parents[1]


SAMPLE_GUIDELINE_TEXT = """## Page 1
一、诊断
推荐意见1: CD的诊断缺乏金标准，需要结合临床表现、实验室检查、影像学检查、内镜及病理组织学检查进行综合判断。(证据等级:2, 推荐强度:强)
推荐理由：单一检查不足以完成诊断，需要多源证据综合评估。

## Page 2
二、检查
推荐意见2: 结肠镜应作为常规检查方法用于CD诊断、疗效评估及疾病监测，建议尽量进入回肠末段，疑诊患者应进行多肠段活检。(证据等级:2, 推荐强度:强)
实施建议：记录病变分布、活检部位和病理结果。
推荐意见3: 胶囊内镜主要用于疑诊CD但结肠镜及小肠放射影像学检查未能明确诊断者，检查前要评估狭窄和胶囊滞留风险。(证据等级:1, 推荐强度:弱)
"""


def test_parse_sections_from_extracted_text_file(tmp_path: Path) -> None:
    text_path = _write_sample_text(tmp_path)

    sections = parse_guideline_sections(text_path)

    assert [section.title for section in sections] == ["一、诊断", "二、检查"]
    assert sections[0].page_start == 1
    assert sections[1].page_start == 2
    assert "推荐意见2" in sections[1].text


def test_extracts_recommendation_number_body_evidence_and_strength(tmp_path: Path) -> None:
    sections = parse_guideline_sections(_write_sample_text(tmp_path))

    cards = extract_recommendation_cards_from_sections(
        sections,
        disease_name="Crohn's disease",
        guideline_name="中国克罗恩病诊治指南",
        guideline_version="2023 Guangzhou",
    )

    assert [card.recommendation_id for card in cards] == [
        "DRAFT-REC-001",
        "DRAFT-REC-002",
        "DRAFT-REC-003",
    ]
    assert "CD的诊断缺乏金标准" in cards[0].action
    assert cards[0].evidence_level == "2"
    assert cards[0].recommendation_strength == "强"
    assert "单一检查不足" in cards[0].rationale
    assert "pages 1-1" in cards[0].source_span
    assert cards[0].page == 1
    assert cards[0].source_quote == cards[0].action
    assert cards[0].review_status == "needs_human_review"


def test_duplicate_recommendation_numbers_get_stable_unique_ids(tmp_path: Path) -> None:
    text_path = tmp_path / "duplicate_recommendations.txt"
    text_path.write_text(
        """## Page 1
一、诊断
推荐意见18: 第一条推荐。(证据等级:2, 推荐强度:强)

## Page 2
二、治疗
推荐意见18: 第二条重复编号推荐。(证据等级:1, 推荐强度:弱)
""",
        encoding="utf-8",
    )
    sections = parse_guideline_sections(text_path)

    cards = extract_recommendation_cards_from_sections(
        sections,
        disease_name="Crohn's disease",
        guideline_name="中国克罗恩病诊治指南",
        guideline_version="2023 Guangzhou",
    )

    assert [card.recommendation_id for card in cards] == [
        "DRAFT-REC-018",
        "DRAFT-REC-018-002",
    ]
    assert [card.page for card in cards] == [1, 2]


def test_extraction_helpers_parse_common_patterns() -> None:
    raw = "结肠镜应作为常规检查方法。( 证据等级:2, 推荐强度: 强 ) 推荐理由：可直接观察黏膜。"

    assert extract_recommendation_statement(raw) == "结肠镜应作为常规检查方法"
    assert extract_evidence_level(raw) == "2"
    assert extract_recommendation_strength(raw) == "强"


def test_extract_recommendations_generates_schema_loadable_draft_yaml(tmp_path: Path) -> None:
    text_path = _write_sample_text(tmp_path)
    output_path = tmp_path / "crohn_recommendation_cards_draft.yaml"
    sections_output = tmp_path / "extracted_sections.json"

    draft_pack = extract_recommendations_from_pdf(
        pdf_path=text_path,
        disease_name="Crohn's disease",
        guideline_name="中国克罗恩病诊治指南",
        guideline_version="2023 Guangzhou",
        output_path=output_path,
        sections_output_path=sections_output,
    )
    loaded = load_skill_pack(output_path)
    sections_payload = json.loads(sections_output.read_text(encoding="utf-8"))

    assert output_path.exists()
    assert sections_output.exists()
    assert len(sections_payload) == 2
    assert loaded.skill_name == draft_pack.skill_name
    assert len(loaded.recommendation_cards) == 3
    assert all(card.review_status == "needs_human_review" for card in loaded.recommendation_cards)
    assert RecommendationCard.model_validate(loaded.recommendation_cards[0].model_dump())


def test_extract_recommendations_cli_outputs_summary_and_files(tmp_path: Path) -> None:
    text_path = _write_sample_text(tmp_path)
    output_path = tmp_path / "crohn_recommendation_cards_draft.yaml"
    sections_output = tmp_path / "extracted_sections.json"
    script = ROOT / "scripts" / "extract_recommendations_from_pdf.py"

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--pdf",
            str(text_path),
            "--disease",
            "Crohn's disease",
            "--guideline-name",
            "中国克罗恩病诊治指南",
            "--guideline-version",
            "2023 Guangzhou",
            "--output",
            str(output_path),
            "--sections-output",
            str(sections_output),
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    summary = json.loads(completed.stdout)
    loaded = load_skill_pack(output_path)
    raw_yaml = yaml.safe_load(output_path.read_text(encoding="utf-8"))

    assert summary["recommendation_card_count"] == 3
    assert output_path.exists()
    assert sections_output.exists()
    assert len(loaded.recommendation_cards) == 3
    assert raw_yaml["recommendation_cards"][0]["review_status"] == "needs_human_review"


def _write_sample_text(tmp_path: Path) -> Path:
    path = tmp_path / "crohn_guideline_extracted.txt"
    path.write_text(SAMPLE_GUIDELINE_TEXT, encoding="utf-8")
    return path
