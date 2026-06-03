from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from pydantic import BaseModel

from .models import aliases_to_json, model_to_dict

# NODE_COLUMNS = [
#     "node_id",
#     "label",
#     "name",
#     "normalized_name",
#     "aliases",
#     "description",
#     "section",
#     "page_start",
#     "page_end",
#     "source_text",
#     "evidence_grade",
#     "recommendation_strength",
#     "confidence",
# ]

# EDGE_COLUMNS = [
#     "edge_id",
#     "source_id",
#     "target_id",
#     "relation_type",
#     "description",
#     "section",
#     "page_start",
#     "page_end",
#     "source_text",
#     "evidence_grade",
#     "recommendation_strength",
#     "confidence",
# ]

RECOMMENDATION_COLUMNS = [
    "recommendation_id",
    "number",
    "title",
    "text",
    "evidence_grade",
    "recommendation_strength",
    "is_bps",
    "reason",
    "implementation_advice",
    "section",
    "page_start",
    "page_end",
    "source_text",
]

# SECTION_COLUMNS = [
#     "section_id",
#     "parent_section_id",
#     "title",
#     "level",
#     "page_start",
#     "page_end",
#     "source_text",
# ]


def _records_to_rows(records: Iterable[BaseModel], columns: list[str]) -> list[dict]:
    rows: list[dict] = []
    for record in records:
        row = model_to_dict(record)
        if isinstance(row.get("aliases"), list):
            row["aliases"] = aliases_to_json(row["aliases"])
        rows.append({column: row.get(column, "") for column in columns})
    return rows


def write_csv(records: Iterable[BaseModel], path: str | Path, columns: list[str]) -> None:
    """Write pydantic records to CSV with pandas and UTF-8-SIG encoding."""

    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError(
            "缺少 pandas 依赖，无法写出 CSV。请先运行: python -m pip install -r requirements.txt"
        ) from exc

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = _records_to_rows(records, columns)
    dataframe = pd.DataFrame(rows, columns=columns)
    dataframe.to_csv(output_path, index=False, encoding="utf-8-sig")


# def write_report(report: dict, path: str | Path) -> None:
#     output_path = Path(path)
#     output_path.parent.mkdir(parents=True, exist_ok=True)
#     output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


# def export_all(
#     output_dir: str | Path,
#     nodes: Iterable[BaseModel],
#     edges: Iterable[BaseModel],
#     recommendations: Iterable[BaseModel],
#     sections: Iterable[BaseModel],
#     report: dict,
# ) -> None:
#     output_path = Path(output_dir)
#     output_path.mkdir(parents=True, exist_ok=True)
#     write_csv(nodes, output_path / "nodes.csv", NODE_COLUMNS)
#     write_csv(edges, output_path / "edges.csv", EDGE_COLUMNS)
#     write_csv(recommendations, output_path / "recommendations.csv", RECOMMENDATION_COLUMNS)
#     write_csv(sections, output_path / "sections.csv", SECTION_COLUMNS)
#     write_report(report, output_path / "extraction_report.json")