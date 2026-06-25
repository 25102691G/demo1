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
ALLOWED_POPULATIONS = {"儿童", "青少年", "成人", "老年人", "孕妇", "普遍适用"}
DEFAULT_EVIDENCE_QUALITY_SCORE = 0.75
MIN_EVIDENCE_QUALITY_SCORE = 0.6
EVIDENCE_INFO_NO_RAW = "default_no_raw_evidence"
EVIDENCE_INFO_UNMAPPED_RAW = "default_unmapped_raw_evidence"
EVIDENCE_INFO_MISSING_LLM_SCORE = "default_missing_llm_score"
EVIDENCE_INFO_LLM_NORMALIZED = "llm_normalized"
EVIDENCE_INFO_BPS_NORMALIZED = "bps_normalized"
POPULATION_FROM_FILENAME_SYSTEM_PROMPT = """你是医学指南文件名解析助手。
请只根据输入文件名判断该指南适用患者人群。
只能返回 JSON 对象，且 population 必须是以下值之一：
儿童、青少年、成人、老年人、孕妇、普遍适用。
如果文件名没有明确人群信息，返回 {"population": "普遍适用"}。"""
DISEASE_FROM_TITLE_SYSTEM_PROMPT = """你是医学指南标题解析助手。
请只根据输入的指南标题或文件名，截取该指南适用的疾病名称。
规则：
1. 只能从输入原文中截取疾病名称，不能做 ICD 标准化、同义词改写或医学常识补全。
2. 去掉“诊断”“治疗”“诊治”“指南”“共识意见”“专家共识”“临床实践指南”“推荐意见”“版”“解读”等文档类型词。
3. 去掉文件后缀、年份、地区、发布机构、括号中的年份或版本信息，但保留疾病别名括号，例如“肠型贝赫切特综合征(肠白塞病)”。
4. 如果标题中有多个疾病名称，只返回指南核心适用疾病名称。
5. 只能返回 JSON 对象，格式为 {"disease": "疾病名称"}，不要返回解释。"""
ALLOWED_CLINICAL_STAGES = {"诊断评估流程", "治疗流程", "其他流程"}
CLINICAL_STAGE_SYSTEM_PROMPT = """你是医学指南章节分类助手。
请只根据输入的 section_path 判断该章节属于哪类临床流程。
只能返回 JSON 对象，且 clinical_stage 必须是以下值之一：诊断评估流程、治疗流程、其他流程。
判断规则：
1. 章节涉及诊断、鉴别诊断、检查、检测、评估、分型、分期、病情活动度、风险评估、筛查、监测等，返回诊断评估流程。
2. 章节涉及药物治疗、手术治疗、营养治疗、维持治疗、诱导缓解、治疗选择、治疗调整、并发症处理等，返回治疗流程。
3. 章节路径为空，或无法明确归入以上两类，返回其他流程。
不要返回解释。"""
DIAGNOSIS_CLINICAL_TASKS = {"初步筛查与临床表现评估", "实验室检查", "影像学检查", "内镜检查", "病理", "综合诊断"}
TREATMENT_CLINICAL_TASKS = {"一般治疗", "药物治疗", "手术治疗", "随访与监测"}
CLINICAL_TASK_SYSTEM_PROMPT = """你是医学指南推荐内容分类助手。
请根据输入的 clinical_stage 和 raw_text 判断该推荐对应的具体 clinical_task。
只能返回 JSON 对象，且 clinical_task 必须是允许值之一。
如果 clinical_stage 是诊断评估流程，clinical_task 只能是以下值之一：初步筛查与临床表现评估、实验室检查、影像学检查、内镜检查、病理、综合诊断。
如果 clinical_stage 是治疗流程，clinical_task 只能是以下值之一：一般治疗、药物治疗、手术治疗、随访与监测。
如果无法明确判断，返回未知。
不要返回解释。"""
EVIDENCE_QUALITY_RAW_SYSTEM_PROMPT = """你是医学指南证据等级提取助手。
只能从输入原文中提取内容，不能凭空生成、不能基于医学常识推断。
提取目标是能表示该片段可信程度的原文表述，包括但不限于证据等级、证据质量、推荐强度、推荐等级、共识等级、专家共识比例或投票比例。
如果原文没有这类内容，返回 {"evidence_quality_raw": null}。
如果有多处相关内容，合并为一个简短原文片段，保留原文关键词。
只能返回 JSON 对象，不要返回解释。"""
EVIDENCE_QUALITY_NORMALIZATION_SYSTEM_PROMPT = """你是医学指南证据加权系数标准化助手。
请只根据输入的 evidence_quality_raw 列表，在同一指南内部统一评估证据或推荐可信程度。
输出 0.6 到 1.0 的数字，1.0 表示该指南内最高可信程度，0.6 表示该指南内最低但仍可作为指南证据使用的可信程度。
不要输出低于 0.6 的分数。
相同或等价的原文等级必须给相同分数。
如果 evidence_quality_raw 中包含 BPS、bps、Bps、(BPS)、（BPS）等大小写或括号变体，视为 best practice statement，evidence_quality_normalized 必须返回 1.0，evidence_quality_normalized_info 返回 bps_normalized。
如果可以判断可信程度，evidence_quality_normalized_info 返回 llm_normalized。
如果无法判断可信程度，evidence_quality_normalized 返回 0.75，evidence_quality_normalized_info 返回 default_unmapped_raw_evidence。
只能返回 JSON 对象，格式为 {"scores":[{"card_id":"...","evidence_quality_normalized":0.85,"evidence_quality_normalized_info":"llm_normalized"}]}。
不要返回解释。"""


class OcrToCardsError(ValueError):
    """OCR 转 recommendation_card 失败。"""


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
    is_abstract: bool = False
    is_conclusion: bool = False
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
    clinical_stage_cache: dict[str, str] = {}
    total_inputs = len(input_paths)

    if args.output is not None:
        cards: list[dict[str, Any]] = []
        for index, input_path in enumerate(input_paths, 1):
            log_file_start(index, total_inputs, input_path)
            file_cards, summary = convert_input_file(
                input_path,
                client,
                population_cache,
                clinical_stage_cache,
                args.llm_workers,
            )
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
        cards, summary = convert_input_file(
            input_path,
            client,
            population_cache,
            clinical_stage_cache,
            args.llm_workers,
        )
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
    parser.add_argument("--llm-workers", type=int, default=20, help="Concurrent LLM workers for text layout cleaning.")
    parser.add_argument(
        "--model-path",
        default=None,
        help=argparse.SUPPRESS,
    )
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
    clinical_stage_cache: dict[str, str],
    llm_workers: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """读取单个 OCR parse_result.json 文件，转换为 recommendation_card 列表和处理 summary"""
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
    cards = build_cards(units, payload, client, clinical_stage_cache, summary, llm_workers)
    normalize_evidence_quality_scores(cards, client)
    summary["output_cards"] = len(cards)
    summary["discarded_layout_count"] = len(summary["discarded_layouts"])
    summary["discarded_by_reason"] = count_by_key(summary["discarded_layouts"], "reason")
    summary["discarded_unit_count"] = len(summary["discarded_units"])
    summary["discarded_units_by_reason"] = count_by_key(summary["discarded_units"], "reason")
    return cards, summary


def default_output_path(input_path: Path) -> Path:
    return input_path.parent / "recommendation_card.jsonl"


def default_summary_path(input_path: Path) -> Path:
    return input_path.parent / "recommendation_card_summary.json"


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


def log_card_progress(done: int, total: int) -> None:
    print(f"LLM 生成 recommendation cards: {done}/{total}", flush=True)


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
    cache[cache_key] = population or "普遍适用"
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
        "discarded_unit_count": 0,
        "discarded_units_by_reason": {},
        "discarded_units": [],
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


def add_discarded_unit(
    summary: dict[str, Any],
    *,
    unit: ClinicalTextUnit,
    reason: str,
    clinical_stage_value: str,
) -> None:
    discarded = summary.get("discarded_units")
    if not isinstance(discarded, list):
        return
    discarded.append(
        {
            "unit_id": unit.unit_id,
            "page_start": unit.page_start,
            "page_end": unit.page_end,
            "section_path": unit.section_path,
            "source_layout_ids": unit.source_layout_ids,
            "reason": reason,
            "clinical_stage": clinical_stage_value,
            "text_preview": preview_text(unit.raw_text),
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
    """从 OCR parse_result.json 中提取文本布局，清洗过滤后构建 ClinicalTextUnit 列表"""
    layouts = extract_layouts(payload, summary)
    apply_section_paths(layouts)
    ordered_layouts = sort_reading_order(layouts, payload)
    text_layouts = [layout for layout in ordered_layouts if layout.layout_type == "text" and layout.text]
    enrich_layouts_with_llm(text_layouts, client, llm_workers)
    discard_layouts(text_layouts, summary)

    disease = resolve_disease(payload, input_path, client)
    units = build_units(text_layouts, disease=disease, population=population)
    merged_units = merge_units(units)
    summary["merged_layout_groups"] = merged_layout_groups(merged_units)
    return merged_units


def discard_layouts(text_layouts: Sequence[OcrLayout], summary: dict[str, Any]) -> None:
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
        elif layout.is_abstract:
            add_discarded_layout(
                summary,
                layout_id=layout.layout_id,
                page_num=layout.page_num,
                layout_type=layout.layout_type,
                reason="llm_abstract",
                text=layout.text,
            )
        elif layout.is_conclusion:
            add_discarded_layout(
                summary,
                layout_id=layout.layout_id,
                page_num=layout.page_num,
                layout_type=layout.layout_type,
                reason="llm_conclusion",
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
            # TODO： 目前只支持 title 和 text 类型，其他类型暂时丢弃
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
            "你是医学指南 OCR 文本清洗助手。判断输入文本是否为正文、明显元信息、摘要、总结展望或参考文献内容，"
            "并在正文时给出一句简短 action 摘要。只输出 JSON。"
        ),
        user_prompt=json.dumps(
            {
                "section_path": layout.section_path,
                "text": layout.text,
                "output_schema": {
                    "is_metadata": "boolean",
                    "is_reference": "boolean",
                    "is_abstract": "boolean",
                    "is_conclusion": "boolean",
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
                "abstract_examples": [
                    "摘要",
                    "Abstract",
                    "关键词",
                    "Key words",
                ],
                "conclusion_examples": [
                    "总结",
                    "结语",
                    "展望",
                    "总结与展望",
                    "Conclusion",
                    "Future perspectives",
                ],
            },
            ensure_ascii=False,
        ),
    )
    layout.is_metadata = bool(payload.get("is_metadata"))
    layout.is_reference = bool(payload.get("is_reference"))
    layout.is_abstract = bool(payload.get("is_abstract"))
    layout.is_conclusion = bool(payload.get("is_conclusion"))
    action_summary = clean_text(payload.get("action_summary"))
    layout.action_summary = action_summary or None


def build_units(layouts: Sequence[OcrLayout], *, disease: str, population: str | None) -> list[ClinicalTextUnit]:
    units: list[ClinicalTextUnit] = []
    for layout in layouts:
        if (
            layout.is_metadata
            or layout.is_reference
            or layout.is_abstract
            or layout.is_conclusion
            or section_is_reference(layout.section_path)
        ):
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
    # layout merge 条件
    # 1、两段必须属于同一个 section_path
    if previous.section_path != current.section_path:
        return False
    # 2、前一段不能已经是完整句子，完整句子判断是末尾是否为句号、问号、冒号等结束标点
    if is_complete_sentence(previous.raw_text):
        return False
    # 3、当前段不能是新的编号小节
    if NEW_NUMBERED_SECTION_RE.match(current.raw_text):
        return False
    # 4、不能是参考文献章节
    if section_is_reference(current.section_path):
        return False
    return True


def build_cards(
    units: Sequence[ClinicalTextUnit],
    payload: Mapping[str, Any],
    client: Any,
    clinical_stage_cache: dict[str, str],
    summary: dict[str, Any],
    llm_workers: int,
) -> list[dict[str, Any]]:
    card_units = [unit for unit in units if not section_is_reference(unit.section_path)]
    if not card_units:
        log_card_progress(0, 0)
        return []

    staged_units = [
        (unit, clinical_stage(unit.section_path, client=client, cache=clinical_stage_cache))
        for unit in card_units
    ]
    staged_units = discard_units(staged_units, summary)
    total = len(staged_units)
    if total == 0:
        log_card_progress(0, 0)
        return []

    log_card_progress(0, total)
    if llm_workers <= 1 or total <= 1:
        cards: list[dict[str, Any]] = []
        for index, (unit, stage) in enumerate(staged_units, 1):
            cards.append(build_card(unit, payload, stage, client))
            if should_log_llm_progress(index, total):
                log_card_progress(index, total)
        return cards

    cards_by_index: list[dict[str, Any] | None] = [None] * total
    with ThreadPoolExecutor(max_workers=llm_workers) as executor:
        futures = {
            executor.submit(build_card, unit, payload, stage, client): index
            for index, (unit, stage) in enumerate(staged_units)
        }
        for done, future in enumerate(as_completed(futures), 1):
            index = futures[future]
            cards_by_index[index] = future.result()
            if should_log_llm_progress(done, total):
                log_card_progress(done, total)

    return [card for card in cards_by_index if card is not None]


def discard_units(
    staged_units: Sequence[tuple[ClinicalTextUnit, str]],
    summary: dict[str, Any],
) -> list[tuple[ClinicalTextUnit, str]]:
    kept_units: list[tuple[ClinicalTextUnit, str]] = []
    for unit, stage in staged_units:
        if stage == "其他流程":
            add_discarded_unit(
                summary,
                unit=unit,
                reason="clinical_stage_other",
                clinical_stage_value=stage,
            )
            continue
        kept_units.append((unit, stage))
    return kept_units


def build_card(
    unit: ClinicalTextUnit,
    payload: Mapping[str, Any],
    clinical_stage_value: str,
    client: Any,
) -> dict[str, Any]:
    task = clinical_task(clinical_stage_value, unit.raw_text, client=client)
    evidence_quality_raw = extract_evidence_quality_raw(unit.raw_text, client=client)
    return unit_to_card(unit, payload, clinical_stage_value, task, evidence_quality_raw)


def unit_to_card(
    unit: ClinicalTextUnit,
    payload: Mapping[str, Any],
    clinical_stage_value: str,
    clinical_task_value: str,
    evidence_quality_raw: str | None,
) -> dict[str, Any]:
    """将 ClinicalTextUnit 转换为 recommendation_card dict，guideline 信息来自 payload"""
    guideline_title = guideline_name(payload)
    source_file = source_file_name(payload)
    raw_text = unit.raw_text
    return {
        "card_id": card_id(unit),
        "source_statement_id": unit.unit_id,
        "disease": unit.disease,
        "guideline": {
            "title": guideline_title,
            "source_file": source_file,
            "doc_type": "ocr_parse_result",
        },
        "clinical_stage": clinical_stage_value,
        "clinical_task": clinical_task_value,
        "population": unit.population,
        "condition": None,
        "action": unit.action_summary or raw_text,
        "required_inputs": [],
        "safety_notes": [],
        "evidence": {
            "evidence_quality_raw": evidence_quality_raw,
            "evidence_quality_normalized": DEFAULT_EVIDENCE_QUALITY_SCORE,
            "evidence_quality_normalized_info": EVIDENCE_INFO_MISSING_LLM_SCORE
            if evidence_quality_raw
            else EVIDENCE_INFO_NO_RAW,
        },
        "source_location": {
            "pdf": source_file,
            "page_start": unit.page_start,
            "page_end": unit.page_end,
            "raw_chunk_text": raw_text,
            "source_span": ",".join(unit.source_layout_ids) or None,
        },
        "section_path": unit.section_path,
    }


def extract_evidence_quality_raw(raw_text: str, *, client: Any) -> str | None:
    payload = client.chat_json(
        EVIDENCE_QUALITY_RAW_SYSTEM_PROMPT,
        json.dumps(
            {
                "raw_text": raw_text,
                "output_schema": {"evidence_quality_raw": "string or null"},
            },
            ensure_ascii=False,
        ),
    )
    evidence_quality_raw = clean_text(payload.get("evidence_quality_raw"))
    return evidence_quality_raw or None


def normalize_evidence_quality_scores(cards: Sequence[dict[str, Any]], client: Any) -> None:
    items = []
    for card in cards:
        evidence = card.get("evidence")
        if not isinstance(evidence, dict):
            continue
        evidence["evidence_quality_normalized"] = DEFAULT_EVIDENCE_QUALITY_SCORE
        evidence_quality_raw = clean_text(evidence.get("evidence_quality_raw"))
        if evidence_quality_raw:
            evidence["evidence_quality_normalized_info"] = EVIDENCE_INFO_MISSING_LLM_SCORE
            items.append(
                {
                    "card_id": clean_text(card.get("card_id")),
                    "evidence_quality_raw": evidence_quality_raw,
                }
            )
        else:
            evidence["evidence_quality_normalized_info"] = EVIDENCE_INFO_NO_RAW

    if not items:
        return

    payload = client.chat_json(
        EVIDENCE_QUALITY_NORMALIZATION_SYSTEM_PROMPT,
        json.dumps({"items": items}, ensure_ascii=False),
    )
    scores = payload.get("scores")
    if not isinstance(scores, list):
        return

    normalized_by_card_id: dict[str, tuple[float, str]] = {}
    for item in scores:
        if not isinstance(item, Mapping):
            continue
        card_id_value = clean_text(item.get("card_id"))
        if not card_id_value:
            continue
        normalized_score = normalize_optional_score(item.get("evidence_quality_normalized"))
        normalized_info = clean_text(item.get("evidence_quality_normalized_info"))
        if normalized_score is None:
            normalized_by_card_id[card_id_value] = (
                DEFAULT_EVIDENCE_QUALITY_SCORE,
                EVIDENCE_INFO_UNMAPPED_RAW,
            )
        elif normalized_info == EVIDENCE_INFO_UNMAPPED_RAW:
            normalized_by_card_id[card_id_value] = (
                DEFAULT_EVIDENCE_QUALITY_SCORE,
                EVIDENCE_INFO_UNMAPPED_RAW,
            )
        elif normalized_info == EVIDENCE_INFO_BPS_NORMALIZED:
            normalized_by_card_id[card_id_value] = (normalized_score, EVIDENCE_INFO_BPS_NORMALIZED)
        else:
            normalized_by_card_id[card_id_value] = (normalized_score, EVIDENCE_INFO_LLM_NORMALIZED)

    for card in cards:
        card_id_value = clean_text(card.get("card_id"))
        if card_id_value not in normalized_by_card_id:
            continue
        evidence = card.get("evidence")
        if isinstance(evidence, dict):
            normalized_score, normalized_info = normalized_by_card_id[card_id_value]
            evidence["evidence_quality_normalized"] = normalized_score
            evidence["evidence_quality_normalized_info"] = normalized_info


def normalize_optional_score(value: Any) -> float | None:
    if value is None:
        return None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if score < MIN_EVIDENCE_QUALITY_SCORE:
        return MIN_EVIDENCE_QUALITY_SCORE
    if score > 1:
        return 1.0
    return score


def clinical_stage(section_path: Sequence[str], *, client: Any, cache: dict[str, str]) -> str:
    cache_key = json.dumps([clean_text(part) for part in section_path], ensure_ascii=False, separators=(",", ":"))
    if cache_key in cache:
        return normalize_clinical_stage(cache[cache_key])

    payload = client.chat_json(
        CLINICAL_STAGE_SYSTEM_PROMPT,
        json.dumps({"section_path": list(section_path)}, ensure_ascii=False),
    )
    stage = normalize_clinical_stage(clean_text(payload.get("clinical_stage")))
    cache[cache_key] = stage
    return stage


def normalize_clinical_stage(value: Any) -> str:
    text = clean_text(value)
    return text if text in ALLOWED_CLINICAL_STAGES else "其他流程"


def clinical_task(clinical_stage_value: str, raw_text: str, *, client: Any) -> str:
    stage = normalize_clinical_stage(clinical_stage_value)
    if stage not in {"诊断评估流程", "治疗流程"}:
        return "未知"

    payload = client.chat_json(
        CLINICAL_TASK_SYSTEM_PROMPT,
        json.dumps({"clinical_stage": stage, "raw_text": raw_text}, ensure_ascii=False),
    )
    return normalize_clinical_task(stage, clean_text(payload.get("clinical_task")))


def normalize_clinical_task(clinical_stage_value: str, value: Any) -> str:
    text = clean_text(value)
    if clinical_stage_value == "诊断评估流程" and text in DIAGNOSIS_CLINICAL_TASKS:
        return text
    if clinical_stage_value == "治疗流程" and text in TREATMENT_CLINICAL_TASKS:
        return text
    return "未知"


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


def resolve_disease(
    payload: Mapping[str, Any],
    input_path: Path,
    client: Any,
) -> str:
    query = metadata_disease_query(payload, input_path)
    llm_payload = client.chat_json(
        DISEASE_FROM_TITLE_SYSTEM_PROMPT,
        json.dumps({"title_or_filename": query}, ensure_ascii=False),
    )
    disease = clean_text(llm_payload.get("disease"))
    if not disease:
        raise OcrToCardsError(f"{input_path}: LLM disease extraction returned empty disease for {query!r}")
    return disease


def metadata_disease_query(payload: Mapping[str, Any], input_path: Path) -> str:
    name = guideline_name(payload) or input_path.stem
    name = re.sub(r"\.parse_result$", "", name)
    name = re.sub(r"\.(pdf|json)$", "", name, flags=re.IGNORECASE)
    name = name.strip(" -_")
    if not name:
        raise OcrToCardsError(f"{input_path}: cannot infer metadata disease query")
    return name


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
    return "普遍适用"


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
