from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "ICD10" / "医保ICD10_v2.0_0122.xlsx"
DEFAULT_SHEET_NAME = "完整分类与代码"

NS = {
    "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "rel": "http://schemas.openxmlformats.org/package/2006/relationships",
    "office_rel": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}

HEADER_MAP = {
    "章": "chapter",
    "章代码范围": "chapter_code_range",
    "章的名称": "chapter_name",
    "节代码范围": "section_code_range",
    "节名称": "section_name",
    "类目代码": "category_code",
    "类目名称": "category_name",
    "亚目代码": "subcategory_code",
    "亚目名称": "subcategory_name",
    "诊断代码": "diagnosis_code",
    "诊断名称": "diagnosis_name",
}


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Extract ICD10 full classification sheet from xlsx to JSON.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Input xlsx path.")
    parser.add_argument("--output", default=None, help="Output JSON path. Defaults to input directory.")
    parser.add_argument("--sheet-name", default=DEFAULT_SHEET_NAME, help="Worksheet name to extract.")
    parser.add_argument("--indent", type=int, default=2, help="JSON indent. Use 0 for compact JSON.")
    args = parser.parse_args(argv)

    input_path = _resolve_path(args.input)
    output_path = _resolve_output_path(input_path, args.output)
    records = extract_sheet(input_path, args.sheet_name)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    indent = None if args.indent == 0 else args.indent
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(records, file, ensure_ascii=False, indent=indent)
        file.write("\n")

    print(f"Wrote {len(records)} records to {output_path}")
    return 0


def extract_sheet(input_path: Path, sheet_name: str) -> list[dict[str, str]]:
    with zipfile.ZipFile(input_path) as archive:
        shared_strings = _load_shared_strings(archive)
        worksheet_path = _get_worksheet_path(archive, sheet_name)
        worksheet_root = ET.fromstring(archive.read(worksheet_path))

    rows = _iter_rows(worksheet_root, shared_strings)
    try:
        header_row = next(rows)
    except StopIteration as exc:
        raise ValueError(f"Worksheet is empty: {sheet_name}") from exc

    headers = [_map_header(value) for value in header_row]
    records: list[dict[str, str]] = []
    for row in rows:
        record = {
            header: row[index] if index < len(row) else ""
            for index, header in enumerate(headers)
            if header
        }
        if any(value != "" for value in record.values()):
            records.append(record)

    return records


def _resolve_path(path: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    return candidate.resolve()


def _resolve_output_path(input_path: Path, output: str | None) -> Path:
    if output:
        return _resolve_path(output)
    return input_path.with_name("ICD10.json")


def _load_shared_strings(archive: zipfile.ZipFile) -> list[str]:
    try:
        root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    except KeyError:
        return []

    strings: list[str] = []
    for item in root.findall("main:si", NS):
        strings.append("".join(text.text or "" for text in item.findall(".//main:t", NS)))
    return strings


def _get_worksheet_path(archive: zipfile.ZipFile, sheet_name: str) -> str:
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    relationship_by_id = {
        rel.attrib["Id"]: rel.attrib["Target"]
        for rel in relationships.findall("rel:Relationship", NS)
    }

    sheets = workbook.find("main:sheets", NS)
    if sheets is None:
        raise ValueError("Workbook does not contain sheets.")

    for sheet in sheets.findall("main:sheet", NS):
        if sheet.attrib.get("name") != sheet_name:
            continue
        rel_id = sheet.attrib[f"{{{NS['office_rel']}}}id"]
        target = relationship_by_id[rel_id].lstrip("/")
        return target if target.startswith("xl/") else f"xl/{target}"

    available = ", ".join(sheet.attrib.get("name", "") for sheet in sheets.findall("main:sheet", NS))
    raise ValueError(f"Worksheet not found: {sheet_name}. Available sheets: {available}")


def _iter_rows(root: ET.Element, shared_strings: list[str]) -> Any:
    sheet_data = root.find("main:sheetData", NS)
    if sheet_data is None:
        return

    for row in sheet_data.findall("main:row", NS):
        values: list[str] = []
        for cell in row.findall("main:c", NS):
            index = _column_index(cell.attrib.get("r", ""))
            while len(values) <= index:
                values.append("")
            values[index] = _cell_value(cell, shared_strings)
        yield values


def _column_index(cell_ref: str) -> int:
    match = re.match(r"([A-Z]+)", cell_ref)
    if not match:
        return 0

    index = 0
    for char in match.group(1):
        index = index * 26 + ord(char) - ord("A") + 1
    return index - 1


def _cell_value(cell: ET.Element, shared_strings: list[str]) -> str:
    cell_type = cell.attrib.get("t")

    if cell_type == "inlineStr":
        return "".join(text.text or "" for text in cell.findall(".//main:t", NS))

    value = cell.find("main:v", NS)
    if value is None or value.text is None:
        return ""

    if cell_type == "s":
        return shared_strings[int(value.text)]

    return value.text


def _map_header(header: str) -> str:
    try:
        return HEADER_MAP[header]
    except KeyError as exc:
        raise ValueError(f"Unsupported header: {header}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
