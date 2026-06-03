from __future__ import annotations

import json
from pathlib import Path

from guideline_skill.cli import (
    DEFAULT_OUTPUT_DIR,
    batch_extract,
    extract_document,
    output_paths_for_input,
    resolve_batch_inputs,
)


class MockDeepSeekClient:
    model = "deepseek-mock"

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def chat_json(self, system_prompt: str, user_prompt: str) -> dict[str, object]:
        self.calls.append((system_prompt, user_prompt))
        if "unit_type" in system_prompt:
            return {
                "unit_type": "knowledge",
                "clinical_topic": "diagnosis",
                "action": "完善相关检查",
                "condition": None,
                "indication": [],
                "contraindication": [],
                "diagnostic_criteria": [],
                "differential_diagnosis": [],
                "drug": None,
                "dose": None,
                "route": None,
                "frequency": None,
                "duration": None,
                "confidence": 0.86,
                "needs_human_review": False,
                "review_reasons": [],
            }
        return {
            "evidence_quality_normalized": "moderate",
            "strength_normalized": "strong",
            "confidence": 0.92,
            "needs_human_review": False,
            "review_reasons": [],
        }


def test_structured_guideline_uses_structured_pipeline_and_writes_default_outputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    input_path = tmp_path / "structured.txt"
    input_path.write_text(
        (
            "推荐意见1：建议完善内镜检查。证据等级：2，推荐强度：强。\n"
            "推荐意见2：建议结合影像检查。证据等级：2，推荐强度：强。"
        ),
        encoding="utf-8",
    )

    summary = extract_document(input_path, deepseek_client=MockDeepSeekClient())

    output_dir, result_path, summary_path = output_paths_for_input(input_path)
    records = _read_jsonl(result_path)
    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))

    assert summary["doc_type"] == "structured_guideline"
    assert records[0]["record_type"] == "statement_unit"
    assert result_path.exists()
    assert summary_path.exists()
    assert summary_payload["doc_type"] == "structured_guideline"
    assert summary_payload["total_units"] == 2
    assert summary_payload["human_review_count"] == 0
    assert summary_payload["llm_model"] == "deepseek-mock"
    assert summary_payload["output_dir"] == output_dir.as_posix()


def test_narrative_guideline_uses_narrative_pipeline_and_writes_summary(tmp_path: Path) -> None:
    input_path = tmp_path / "narrative.txt"
    output_path = tmp_path / "out" / "result.jsonl"
    summary_path = tmp_path / "out" / "summary.json"
    input_path.write_text(
        """一、诊断
诊断需要结合临床表现和检查结果。
1.1 实验室检查
可以完善血常规和炎症指标。
""",
        encoding="utf-8",
    )

    extract_document(
        input_path,
        output_path=output_path,
        summary_path=summary_path,
        deepseek_client=MockDeepSeekClient(),
    )

    records = _read_jsonl(output_path)
    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))

    assert records
    assert records[0]["record_type"] == "clinical_info_unit"
    assert summary_payload["doc_type"] == "narrative_guideline"
    assert summary_payload["total_units"] >= 1
    assert "human_review_count" in summary_payload
    assert summary_payload["llm_model"] == "deepseek-mock"
    assert summary_payload["output_dir"] == (tmp_path / "out").as_posix()


def test_batch_extract_writes_one_folder_per_input(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text(
        "推荐意见1：建议完善内镜检查。证据等级：2，推荐强度：强。\n"
        "推荐意见2：建议结合影像检查。证据等级：2，推荐强度：强。",
        encoding="utf-8",
    )
    second.write_text(
        """一、诊断
诊断需要结合临床表现和检查结果。
""",
        encoding="utf-8",
    )

    summary = batch_extract([first, second], deepseek_client=MockDeepSeekClient())

    first_dir, first_result, first_summary = output_paths_for_input(first)
    second_dir, second_result, second_summary = output_paths_for_input(second)

    assert first_result.exists()
    assert first_summary.exists()
    assert second_result.exists()
    assert second_summary.exists()
    assert json.loads(first_summary.read_text(encoding="utf-8"))["output_dir"] == first_dir.as_posix()
    assert json.loads(second_summary.read_text(encoding="utf-8"))["output_dir"] == second_dir.as_posix()
    assert summary["doc_type"] == "batch"
    assert summary["total_units"] >= 2


def test_resolve_batch_inputs_scans_directory_non_recursively(tmp_path: Path) -> None:
    (tmp_path / "b.pdf").write_text("b", encoding="utf-8")
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    (tmp_path / "ignored.csv").write_text("ignored", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "nested.pdf").write_text("nested", encoding="utf-8")

    inputs = resolve_batch_inputs(input_dir=tmp_path)

    assert [path.name for path in inputs] == ["a.txt", "b.pdf"]


def test_batch_extract_accepts_inputs_resolved_from_directory(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    input_dir = tmp_path / "guides"
    input_dir.mkdir()
    first = input_dir / "first.txt"
    second = input_dir / "second.txt"
    first.write_text(
        "推荐意见1：建议完善内镜检查。证据等级：2，推荐强度：强。\n"
        "推荐意见2：建议结合影像检查。证据等级：2，推荐强度：强。",
        encoding="utf-8",
    )
    second.write_text("一、诊断\n诊断需要综合判断。", encoding="utf-8")

    summary = batch_extract(
        resolve_batch_inputs(input_dir=input_dir),
        deepseek_client=MockDeepSeekClient(),
    )

    assert (DEFAULT_OUTPUT_DIR / "first" / "result.jsonl").exists()
    assert (DEFAULT_OUTPUT_DIR / "second" / "summary.json").exists()
    assert summary["total_units"] >= 2


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
