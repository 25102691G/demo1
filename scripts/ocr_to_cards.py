from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


END_PUNCTUATION = "。！？；!?;”）)]"
NEW_NUMBERED_SECTION_RE = re.compile(r"^\s*\d+[.．、]\s*[^。；;]{1,40}[:：]")
ALLOWED_POPULATIONS = {"儿童", "青少年", "成人", "老年人", "孕妇", "没有明确人群"}
POPULATION_FROM_FILENAME_SYSTEM_PROMPT = """你是医学指南文件名解析助手。
请只根据输入文件名判断该指南适用患者人群。
只能返回 JSON 对象，且 population 必须是以下值之一：
儿童、青少年、成人、老年人、孕妇、没有明确人群。
如果文件名没有明确人群信息，返回 {"population": "没有明确人群"}。"""


@dataclass(slots=True)
class OcrLayout:
    layout_id: str
    layout_type: str
    text: str
    page_num: int
    position: tuple[float, float, float, float]
    parent: str | None
    children: list[str] = field(default_factory=list)
    section_path: list[str] = field(default_factory=list)
    is_metadata: bool = False
    is_reference: bool = False
    action_summary: str | None = None


@dataclass(slots=True)
class ClinicalTextUnit:
    unit_id: str
    disease: str
    population: str | None
    section_path: list[str]
    raw_text: str
    page_start: int
    page_end: int
    source_layout_ids: list[str]
    action_summary: str | None = None


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    input_paths = collect_input_paths(args.input, args.input_dir)
    client = build_llm_client()
    population_cache: dict[str, str] = {}
    total_inputs = len(input_paths)

    if args.output is not None:
        cards: list[dict[str, Any]] = []
        for index, input_path in enumerate(input_paths, 1):
            log_file_start(index, total_inputs, input_path)
            file_cards, summary = convert_input_file(input_path, client, population_cache, args.llm_workers)
            cards.extend(file_cards)
            summary_path = default_summary_path(input_path)
            write_json(summary_path, summary)
            log_file_done(index, total_inputs, input_path, len(file_cards), None, summary_path)
        write_jsonl(args.output, cards)
        log_batch_done(total_inputs, len(cards), args.output)
        return

    total_cards = 0
    for index, input_path in enumerate(input_paths, 1):
        log_file_start(index, total_inputs, input_path)
        cards, summary = convert_input_file(input_path, client, population_cache, args.llm_workers)
        output_path = default_output_path(input_path)
        summary_path = default_summary_path(input_path)
        write_jsonl(output_path, cards)
        write_json(summary_path, summary)
        total_cards += len(cards)
        log_file_done(index, total_inputs, input_path, len(cards), output_path, summary_path)
    log_batch_done(total_inputs, total_cards, None)


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert OCR parse_result.json to recommendation_card.jsonl.")
    parser.add_argument("--input", type=Path, default=None, help="Input OCR parse_result.json file.")
    parser.add_argument("--input-dir", type=Path, default=None, help="Directory containing *.parse_result.json files.")
    parser.add_argument("--output", type=Path, default=None, help="Output JSONL path.")
    parser.add_argument("--llm-workers", type=int, default=1, help="Concurrent LLM workers for text layout cleaning.")
    args = parser.parse_args(argv)
    if args.input is None and args.input_dir is None:
        parser.error("one of --input or --input-dir is required")
    if args.input is not None and args.input_dir is not None:
        parser.error("--input and --input-dir cannot be used together")
    if args.llm_workers < 1:
        parser.error("--llm-workers must be >= 1")
    return args


def collect_input_paths(input_path: Path | None, input_dir: Path | None) -> list[Path]:
    if input_path is not None:
        if not input_path.is_file():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        return [input_path]

    assert input_dir is not None
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory not found: {input_dir}")
    paths = sorted(input_dir.rglob("*.parse_result.json"))
    if not paths:
        raise FileNotFoundError(f"No *.parse_result.json files found in: {input_dir}")
    return paths


def convert_input_file(
    input_path: Path,
    client: Any,
    population_cache: dict[str, str],
    llm_workers: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = read_json(input_path)
    summary = create_summary(input_path, payload)
    population = extract_population_from_filename(payload, input_path=input_path, client=client, cache=population_cache)
    units = parse_ocr_payload(
        payload,
        input_path=input_path,
        client=client,
        population=population,
        summary=summary,
        llm_workers=llm_workers,
    )
    cards = [unit_to_card(unit, payload) for unit in units if clinical_stage(unit.section_path) != "reference"]
    summary["output_cards"] = len(cards)
    summary["discarded_layout_count"] = len(summary["discarded_layouts"])
    summary["discarded_by_reason"] = count_by_key(summary["discarded_layouts"], "reason")
    return cards, summary


def default_output_path(input_path: Path) -> Path:
    return input_path.parent / "recommendation_card.jsonl"


def default_summary_path(input_path: Path) -> Path:
    return input_path.parent / "summary.json"


def log_file_start(index: int, total: int, input_path: Path) -> None:
    remaining = max(total - index, 0)
    print(f"[{index}/{total}] 开始处理: {input_path}，剩余 {remaining} 个", flush=True)


def log_file_done(
    index: int,
    total: int,
    input_path: Path,
    card_count: int,
    output_path: Path | None,
    summary_path: Path,
) -> None:
    output_text = f"，output: {output_path}" if output_path is not None else ""
    print(
        f"[{index}/{total}] 完成: {input_path}，输出 {card_count} 张 card{output_text}，summary: {summary_path}",
        flush=True,
    )


def log_batch_done(total_inputs: int, total_cards: int, output_path: Path | None) -> None:
    output_text = f"，合并输出: {output_path}" if output_path is not None else ""
    print(f"批量处理完成: 输入 {total_inputs} 个文件，输出 {total_cards} 张 card{output_text}", flush=True)


def log_llm_progress(done: int, total: int) -> None:
    print(f"LLM 清洗 text layouts: {done}/{total}", flush=True)


def extract_population_from_filename(
    payload: Mapping[str, Any],
    *,
    input_path: Path,
    client: Any,
    cache: dict[str, str],
) -> str | None:
    filename = source_file_name(payload) or guideline_name(payload) or input_path.name
    cache_key = clean_text(filename) or "__unknown_filename__"
    if cache_key in cache:
        return normalize_population(cache[cache_key])

    llm_payload = client.chat_json(
        POPULATION_FROM_FILENAME_SYSTEM_PROMPT,
        json.dumps({"filename": filename}, ensure_ascii=False),
    )
    population = normalize_population(clean_text(llm_payload.get("population")))
    cache[cache_key] = population or "没有明确人群"
    return population


def build_llm_client() -> Any:
    missing = [
        name
        for name in ("DEEPSEEK_MODEL", "DEEPSEEK_BASE_URL", "DEEPSEEK_API_KEY")
        if not os.getenv(name)
    ]
    if missing:
        raise RuntimeError(f"Missing required LLM environment variables: {', '.join(missing)}")
    module_path = SOURCE_ROOT / "skill_engine" / "llm_client.py"
    spec = importlib.util.spec_from_file_location("_ocr_to_cards_llm_client", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load LLM client module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    config = module.load_deepseek_config_from_env()
    return module.OpenAICompatibleJsonChatClient(config)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            file.write("\n")


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")


def create_summary(input_path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    layouts_by_type: dict[str, int] = {}
    total_layouts = 0
    pages = payload.get("pages")
    if isinstance(pages, list):
        for page in pages:
            if not isinstance(page, Mapping):
                continue
            raw_layouts = page.get("layouts")
            if not isinstance(raw_layouts, list):
                continue
            for raw_layout in raw_layouts:
                if not isinstance(raw_layout, Mapping):
                    continue
                total_layouts += 1
                layout_type = clean_text(raw_layout.get("type")) or "unknown"
                layouts_by_type[layout_type] = layouts_by_type.get(layout_type, 0) + 1

    return {
        "input_file": str(input_path),
        "guideline_title": guideline_name(payload),
        "source_file": source_file_name(payload),
        "total_layouts": total_layouts,
        "layouts_by_type": layouts_by_type,
        "discarded_layout_count": 0,
        "discarded_by_reason": {},
        "discarded_layouts": [],
        "merged_layout_groups": [],
        "output_cards": 0,
    }


def add_discarded_layout(
    summary: dict[str, Any],
    *,
    layout_id: str,
    page_num: int,
    layout_type: str,
    reason: str,
    text: str,
) -> None:
    discarded = summary.get("discarded_layouts")
    if not isinstance(discarded, list):
        return
    discarded.append(
        {
            "layout_id": layout_id,
            "page_num": page_num,
            "type": layout_type,
            "reason": reason,
            "text_preview": preview_text(text),
        }
    )


def count_by_key(items: Sequence[Mapping[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        value = clean_text(item.get(key)) or "unknown"
        counts[value] = counts.get(value, 0) + 1
    return counts


def parse_ocr_payload(
    payload: Mapping[str, Any],
    *,
    input_path: Path,
    client: Any,
    population: str | None,
    summary: dict[str, Any],
    llm_workers: int,
) -> list[ClinicalTextUnit]:
    layouts = extract_layouts(payload, summary)
    apply_section_paths(layouts)
    ordered_layouts = sort_reading_order(layouts, payload)
    text_layouts = [layout for layout in ordered_layouts if layout.layout_type == "text" and layout.text]
    enrich_layouts_with_llm(text_layouts, client, llm_workers)
    for layout in text_layouts:
        if layout.is_metadata:
            add_discarded_layout(
                summary,
                layout_id=layout.layout_id,
                page_num=layout.page_num,
                layout_type=layout.layout_type,
                reason="llm_metadata",
                text=layout.text,
            )
        elif layout.is_reference:
            add_discarded_layout(
                summary,
                layout_id=layout.layout_id,
                page_num=layout.page_num,
                layout_type=layout.layout_type,
                reason="llm_reference",
                text=layout.text,
            )
        elif section_is_reference(layout.section_path):
            add_discarded_layout(
                summary,
                layout_id=layout.layout_id,
                page_num=layout.page_num,
                layout_type=layout.layout_type,
                reason="section_reference",
                text=layout.text,
            )

    disease = infer_disease(payload, input_path)
    units = build_units(text_layouts, disease=disease, population=population)
    merged_units = merge_units(units)
    summary["merged_layout_groups"] = merged_layout_groups(merged_units)
    return merged_units


def extract_layouts(payload: Mapping[str, Any], summary: dict[str, Any]) -> dict[str, OcrLayout]:
    result: dict[str, OcrLayout] = {}
    pages = payload.get("pages")
    if not isinstance(pages, list):
        return result

    for page_index, page in enumerate(pages):
        if not isinstance(page, Mapping):
            continue
        page_num = positive_page_num(page.get("page_num"), page_index)
        raw_layouts = page.get("layouts")
        if not isinstance(raw_layouts, list):
            continue
        for layout_index, raw_layout in enumerate(raw_layouts):
            if not isinstance(raw_layout, Mapping):
                continue
            layout_type = clean_text(raw_layout.get("type"))
            layout_id = clean_text(raw_layout.get("layout_id")) or f"page-{page_num}-layout-{layout_index + 1}"
            text = clean_text(raw_layout.get("text"))
            if layout_type not in {"title", "text"}:
                add_discarded_layout(
                    summary,
                    layout_id=layout_id,
                    page_num=page_num,
                    layout_type=layout_type or "unknown",
                    reason="type_not_supported",
                    text=text,
                )
                continue
            if layout_type == "text" and not text:
                add_discarded_layout(
                    summary,
                    layout_id=layout_id,
                    page_num=page_num,
                    layout_type=layout_type,
                    reason="empty_text",
                    text=text,
                )
                continue
            result[layout_id] = OcrLayout(
                layout_id=layout_id,
                layout_type=layout_type,
                text=text,
                page_num=page_num,
                position=parse_position(raw_layout.get("position")),
                parent=clean_text(raw_layout.get("parent")) or None,
                children=[clean_text(child) for child in raw_layout.get("children") or [] if clean_text(child)],
            )
    return result


def apply_section_paths(layouts: Mapping[str, OcrLayout]) -> None:
    for layout in layouts.values():
        if layout.layout_type != "text":
            continue
        titles: list[str] = []
        seen: set[str] = set()
        parent_id = layout.parent
        while parent_id and parent_id not in seen:
            seen.add(parent_id)
            parent = layouts.get(parent_id)
            if parent is None:
                break
            if parent.layout_type == "title" and parent.text:
                titles.append(parent.text)
            parent_id = parent.parent
        layout.section_path = list(reversed(titles))


def sort_reading_order(layouts: Mapping[str, OcrLayout], payload: Mapping[str, Any]) -> list[OcrLayout]:
    page_widths = infer_page_widths(payload, layouts.values())

    def key(layout: OcrLayout) -> tuple[int, int, float, float]:
        x, y, width, _height = layout.position
        page_width = page_widths.get(layout.page_num) or max(x + width, 1.0)
        center_x = x + width / 2
        column_index = 0 if center_x < page_width / 2 else 1
        return (layout.page_num, column_index, y, x)

    return sorted(layouts.values(), key=key)


def infer_page_widths(payload: Mapping[str, Any], layouts: Sequence[OcrLayout]) -> dict[int, float]:
    page_widths: dict[int, float] = {}
    pages = payload.get("pages")
    if isinstance(pages, list):
        for page_index, page in enumerate(pages):
            if not isinstance(page, Mapping):
                continue
            page_num = positive_page_num(page.get("page_num"), page_index)
            width = first_number(page.get("page_width"), page.get("width"), page.get("w"))
            if width:
                page_widths[page_num] = width

    for layout in layouts:
        x, _y, width, _height = layout.position
        page_widths[layout.page_num] = max(page_widths.get(layout.page_num, 0.0), x + width)
    return page_widths


def enrich_layouts_with_llm(layouts: Sequence[OcrLayout], client: Any, llm_workers: int) -> None:
    total = len(layouts)
    if total == 0:
        log_llm_progress(0, 0)
        return

    log_llm_progress(0, total)
    if llm_workers <= 1 or len(layouts) <= 1:
        for index, layout in enumerate(layouts, 1):
            enrich_layout_with_llm(layout, client)
            if should_log_llm_progress(index, total):
                log_llm_progress(index, total)
        return

    with ThreadPoolExecutor(max_workers=llm_workers) as executor:
        futures = [executor.submit(enrich_layout_with_llm, layout, client) for layout in layouts]
        for index, future in enumerate(as_completed(futures), 1):
            future.result()
            if should_log_llm_progress(index, total):
                log_llm_progress(index, total)


def should_log_llm_progress(done: int, total: int) -> bool:
    return done == total or done % 20 == 0


def enrich_layout_with_llm(layout: OcrLayout, client: Any) -> None:
    payload = client.chat_json(
        system_prompt=(
            "你是医学指南 OCR 文本清洗助手。判断输入文本是否为正文、明显元信息或参考文献内容，"
            "并在正文时给出一句简短 action 摘要。只输出 JSON。"
        ),
        user_prompt=json.dumps(
            {
                "section_path": layout.section_path,
                "text": layout.text,
                "output_schema": {
                    "is_metadata": "boolean",
                    "is_reference": "boolean",
                    "action_summary": "string or null",
                },
                "metadata_examples": [
                    "DOI",
                    "作者单位",
                    "通讯作者",
                    "收稿日期",
                    "本文编辑",
                    "页码",
                    "期刊页眉",
                ],
            },
            ensure_ascii=False,
        ),
    )
    layout.is_metadata = bool(payload.get("is_metadata"))
    layout.is_reference = bool(payload.get("is_reference"))
    action_summary = clean_text(payload.get("action_summary"))
    layout.action_summary = action_summary or None


def build_units(layouts: Sequence[OcrLayout], *, disease: str, population: str | None) -> list[ClinicalTextUnit]:
    units: list[ClinicalTextUnit] = []
    for layout in layouts:
        if layout.is_metadata or layout.is_reference or section_is_reference(layout.section_path):
            continue
        units.append(
            ClinicalTextUnit(
                unit_id=unit_id(layout.layout_id, layout.text),
                disease=disease,
                population=population,
                section_path=list(layout.section_path),
                raw_text=layout.text,
                page_start=layout.page_num,
                page_end=layout.page_num,
                source_layout_ids=[layout.layout_id],
                action_summary=layout.action_summary,
            )
        )
    return units


def merge_units(units: Sequence[ClinicalTextUnit]) -> list[ClinicalTextUnit]:
    merged: list[ClinicalTextUnit] = []
    for unit in units:
        if merged and should_merge(merged[-1], unit):
            previous = merged[-1]
            previous.raw_text = join_text(previous.raw_text, unit.raw_text)
            previous.source_layout_ids.extend(unit.source_layout_ids)
            previous.page_end = max(previous.page_end, unit.page_end)
            if not previous.action_summary and unit.action_summary:
                previous.action_summary = unit.action_summary
            if not previous.population and unit.population:
                previous.population = unit.population
            previous.unit_id = unit_id(previous.source_layout_ids[0], previous.raw_text)
            continue
        merged.append(unit)
    return merged


def merged_layout_groups(units: Sequence[ClinicalTextUnit]) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for unit in units:
        if len(unit.source_layout_ids) <= 1:
            continue
        groups.append(
            {
                "unit_id": unit.unit_id,
                "page_start": unit.page_start,
                "page_end": unit.page_end,
                "section_path": unit.section_path,
                "source_layout_ids": unit.source_layout_ids,
                "raw_text_preview": preview_text(unit.raw_text),
            }
        )
    return groups


def should_merge(previous: ClinicalTextUnit, current: ClinicalTextUnit) -> bool:
    if previous.section_path != current.section_path:
        return False
    if is_complete_sentence(previous.raw_text):
        return False
    if NEW_NUMBERED_SECTION_RE.match(current.raw_text):
        return False
    if section_is_reference(current.section_path):
        return False
    return True


def unit_to_card(unit: ClinicalTextUnit, payload: Mapping[str, Any]) -> dict[str, Any]:
    guideline_title = guideline_name(payload)
    source_file = source_file_name(payload)
    raw_text = unit.raw_text
    return {
        "record_type": "recommendation_card",
        "card_id": card_id(unit),
        "source_statement_id": unit.unit_id,
        "disease": unit.disease,
        "guideline": {
            "title": guideline_title,
            "source_file": source_file,
            "doc_type": "ocr_parse_result",
        },
        "clinical_stage": clinical_stage(unit.section_path),
        "clinical_task": None,
        "population": unit.population,
        "condition": None,
        "raw_chunk_text": raw_text,
        "action": unit.action_summary or raw_text,
        "required_inputs": [],
        "safety_notes": [],
        "evidence": {
            "evidence_quality_raw": None,
            "evidence_quality_normalized": 0.5,
            "recommendation_strength_raw": None,
            "recommendation_strength_normalized": 0.5,
            "consensus_level": None,
            "grading_system": None,
        },
        "source_location": {
            "pdf": source_file,
            "page_start": unit.page_start,
            "page_end": unit.page_end,
            "quote": raw_text,
            "source_span": ",".join(unit.source_layout_ids) or None,
        },
        "section_path": unit.section_path,
        "source_layout_ids": unit.source_layout_ids,
    }


def clinical_stage(section_path: Sequence[str]) -> str:
    section_text = " / ".join(section_path)
    if "参考文献" in section_text:
        return "reference"
    if "诊断" in section_text or "璇婃柇" in section_text:
        return "diagnosis"
    if "治疗" in section_text or "娌荤枟" in section_text:
        return "treatment"
    if "病因" in section_text or "鐥呭洜" in section_text:
        return "etiology"
    return "unknown"


def section_is_reference(section_path: Sequence[str]) -> bool:
    section_text = " / ".join(section_path)
    return "参考文献" in section_text


def is_complete_sentence(text: str) -> bool:
    stripped = text.rstrip()
    return bool(stripped and stripped[-1] in END_PUNCTUATION)


def join_text(left: str, right: str) -> str:
    if not left:
        return right
    if not right:
        return left
    if left.endswith("-"):
        return left[:-1] + right.lstrip()
    return left.rstrip() + right.lstrip()


def infer_disease(payload: Mapping[str, Any], input_path: Path) -> str:
    name = clean_text(payload.get("disease")) or guideline_name(payload) or input_path.stem
    name = re.sub(r"\.parse_result$", "", name)
    name = re.sub(r"\.(pdf|json)$", "", name, flags=re.IGNORECASE)
    name = re.sub(r"(诊治指南|指南|共识|专家共识|诊断与治疗|诊疗规范)$", "", name)
    return name.strip(" -_") or "unknown"


def guideline_name(payload: Mapping[str, Any]) -> str:
    return clean_text(payload.get("file_name")) or clean_text(payload.get("source_file")) or "unknown guideline"


def source_file_name(payload: Mapping[str, Any]) -> str | None:
    return clean_text(payload.get("source_file")) or clean_text(payload.get("file_name")) or None


def normalize_population(value: str | None) -> str | None:
    text = clean_text(value)
    if not text:
        return None
    if text in ALLOWED_POPULATIONS:
        return text
    return "没有明确人群"


def unit_id(layout_id: str, text: str) -> str:
    digest = hashlib.sha1(f"{layout_id}\n{text}".encode("utf-8")).hexdigest()[:12]
    return f"clinical_text_unit_{digest}"


def card_id(unit: ClinicalTextUnit) -> str:
    digest = hashlib.sha1(f"{unit.unit_id}\n{unit.raw_text}".encode("utf-8")).hexdigest()[:12]
    return f"ocr_card_{digest}"


def parse_position(value: Any) -> tuple[float, float, float, float]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        numbers = [float(item) for item in value[:4] if isinstance(item, (int, float))]
        if len(numbers) == 4:
            return (numbers[0], numbers[1], numbers[2], numbers[3])
    return (0.0, 0.0, 0.0, 0.0)


def positive_page_num(value: Any, fallback_index: int) -> int:
    if isinstance(value, int):
        return value + 1 if value >= 0 else fallback_index + 1
    return fallback_index + 1


def first_number(*values: Any) -> float | None:
    for value in values:
        if isinstance(value, (int, float)) and value > 0:
            return float(value)
    return None


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).replace("\u3000", " ").split())


def preview_text(value: Any, limit: int = 120) -> str:
    text = clean_text(value)
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


if __name__ == "__main__":
    main()
