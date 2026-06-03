from __future__ import annotations

from guideline_skill.extractors.clinical_info_extractor import ClinicalInfoExtractor
from guideline_skill.pipelines.narrative_pipeline import NarrativeGuidelinePipeline


class MockDeepSeekClient:
    def __init__(self) -> None:
        self.calls = 0

    def chat_json(self, system_prompt: str, user_prompt: str) -> dict[str, object]:
        self.calls += 1
        return {
            "unit_type": "knowledge",
            "clinical_topic": "diagnosis",
            "action": None,
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
            "confidence": 0.82,
            "needs_human_review": False,
            "review_reasons": [],
        }


def test_narrative_pipeline_segments_by_heading_and_calls_llm_per_segment() -> None:
    client = MockDeepSeekClient()
    pipeline = NarrativeGuidelinePipeline(
        clinical_info_extractor=ClinicalInfoExtractor(client),
    )
    text = """## Page 1
一、诊断
诊断需要综合判断。
1.1 实验室检查
可完善血常规。
1.2 影像检查
可结合影像资料。
"""

    units = pipeline.run(text, title="叙述性指南")

    assert client.calls == len(units)
    assert len(units) == 3
    assert all(unit.record_type == "clinical_info_unit" for unit in units)
    assert units[1].unit.section_path == ["诊断", "实验室检查"]
