from __future__ import annotations

from clinical_stage_classifier import classify_clinical_stage
from guideline_skill.schema import MissingInformationItem
from missing_information_ranker import rank_missing_information
from patient_case_extractor import extract_patient_case


def _missing(key: str, priority: str = "medium") -> MissingInformationItem:
    return MissingInformationItem(
        information_key=key,
        question=f"{key}?",
        reason=f"{key} reason",
        priority=priority,  # type: ignore[arg-type]
    )


def test_ranker_merges_synonyms_and_limits_top_items() -> None:
    clinical_stage = classify_clinical_stage(
        extract_patient_case("腹痛腹泻三个月，肠镜提示回盲部溃疡并狭窄。")
    )
    ranking = rank_missing_information(
        [
            _missing("pathology", "high"),
            _missing("pathology_report", "critical"),
            _missing("labs", "high"),
            _missing("inflammatory_markers", "medium"),
            _missing("imaging", "high"),
            _missing("intestinal_tuberculosis_exclusion", "critical"),
            _missing("infection_workup", "high"),
            _missing("medication_history", "medium"),
            _missing("mental_health_status", "medium"),
            _missing("contraindications", "medium"),
        ],
        clinical_stage,
        top_k=7,
    )

    top_keys = [item.information_key for item in ranking.top_missing_information]

    assert len(ranking.top_missing_information) <= 7
    assert top_keys.count("pathology") == 1
    assert "inflammatory_labs" in top_keys
    assert "cross_sectional_imaging" in top_keys
    assert "mental_health_status" not in top_keys
    assert "contraindications" not in top_keys


def test_ranker_prioritizes_diagnostic_information() -> None:
    clinical_stage = classify_clinical_stage(
        extract_patient_case("腹痛腹泻，肠镜提示回肠末端溃疡并狭窄。")
    )
    ranking = rank_missing_information(
        [
            _missing("drug_induced_enteritis_review", "medium"),
            _missing("infectious_enteritis_exclusion", "high"),
            _missing("pathology", "critical"),
            _missing("cross_sectional_imaging", "high"),
            _missing("inflammatory_labs", "high"),
        ],
        clinical_stage,
    )

    assert [item.information_key for item in ranking.top_missing_information[:3]] == [
        "pathology",
        "cross_sectional_imaging",
        "inflammatory_labs",
    ]
