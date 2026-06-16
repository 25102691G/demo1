from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

DEFAULT_INPUT = ROOT / "data" / "ontology" / "hp-zh.babelon.tsv"
DEFAULT_OUTPUT = ROOT / "data" / "ontology" / "definition2id.json"
LLM_CLIENT_PATH = SRC / "skill_engine" / "llm_client.py"
_LLM_CLIENT_MODULE: Any | None = None
BODY_SITES = {
    "未知",
    "不适用",
    "全身",
    "腹部",
    "口腔",
    "食管",
    "胃",
    "十二指肠",
    "小肠",
    "空肠",
    "回肠",
    "回盲部",
    "结肠",
    "直肠",
    "肛门肛周",
    "肠道",
    "胃肠道",
    "肝",
    "胆道",
    "胰腺",
    "脾",
    "腹膜",
    "肠系膜",
    "门静脉系统",
}
BODY_SITE_SYSTEM_PROMPT = """你是一名消化内科医学术语标注助手。
请为输入的每个中文症状或表型术语选择一个最相关的解剖部位 body_site。
body_site 必须且只能从以下列表中选择：
未知, 不适用, 全身, 腹部, 口腔, 食管, 胃, 十二指肠, 小肠, 空肠, 回肠, 回盲部, 结肠, 直肠, 肛门肛周, 肠道, 胃肠道, 肝, 胆道, 胰腺, 脾, 腹膜, 肠系膜, 门静脉系统。
如果术语没有明确解剖部位，选择“未知”；如果术语不是症状/体征/异常表现或不适合标注部位，选择“不适用”。
只输出 JSON 对象，格式为 {"items":[{"term":"原术语","body_site":"枚举值"}]}。"""


def build_definition2id(input_path: Path) -> dict[str, list[str]]:
    definition2id: dict[str, list[str]] = {}

    with input_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file, delimiter="\t")
        required_columns = {"subject_id", "translation_value"}
        missing_columns = required_columns.difference(reader.fieldnames or [])
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Missing required TSV column(s): {missing}")

        for line_no, row in enumerate(reader, start=2):
            subject_id = (row.get("subject_id") or "").strip()
            translation_value = (row.get("translation_value") or "").strip()
            if not subject_id or not translation_value:
                continue

            subject_ids = definition2id.setdefault(translation_value, [])
            if subject_id not in subject_ids:
                subject_ids.append(subject_id)

    return definition2id


def build_definition2id_with_body_site(
    definition2id: dict[str, list[str]],
    *,
    llm_batch_size: int,
    llm_workers: int,
) -> dict[str, dict[str, Any]]:
    terms = list(definition2id.keys())
    batch_size = max(1, llm_batch_size)
    batches = [terms[start : start + batch_size] for start in range(0, len(terms), batch_size)]
    body_sites = _build_body_sites_serial(batches) if llm_workers <= 1 else _build_body_sites_parallel(batches, llm_workers)

    return {
        term: {
            "hpo_ids": hpo_ids,
            "body_site": body_sites.get(term, "未知"),
        }
        for term, hpo_ids in definition2id.items()
    }


def _build_body_sites_serial(batches: list[list[str]]) -> dict[str, str]:
    client = _build_deepseek_client()
    body_sites: dict[str, str] = {}
    total = sum(len(batch) for batch in batches)
    completed = 0
    for batch in batches:
        body_sites.update(_request_body_sites(client, batch))
        completed += len(batch)
        _log_progress(completed, total)
    return body_sites


def _build_body_sites_parallel(batches: list[list[str]], llm_workers: int) -> dict[str, str]:
    worker_count = max(1, llm_workers)
    body_sites: dict[str, str] = {}
    total = sum(len(batch) for batch in batches)
    completed = 0

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(_request_body_sites_for_worker, batch) for batch in batches]
        for future in as_completed(futures):
            batch_body_sites = future.result()
            body_sites.update(batch_body_sites)
            completed += len(batch_body_sites)
            _log_progress(completed, total)
    return body_sites


def _request_body_sites_for_worker(batch: list[str]) -> dict[str, str]:
    client = _build_deepseek_client()
    return _request_body_sites(client, batch)


def _request_body_sites(client: Any, batch: list[str]) -> dict[str, str]:
    payload = client.chat_json(
        BODY_SITE_SYSTEM_PROMPT,
        json.dumps({"terms": batch}, ensure_ascii=False),
    )
    return _parse_body_site_payload(payload, batch)


def _build_deepseek_client() -> Any:
    module = _load_llm_client_module()
    config = module.load_llm_config_from_env()
    return module.OpenAICompatibleJsonChatClient(config)


def _load_llm_client_module() -> Any:
    global _LLM_CLIENT_MODULE
    if _LLM_CLIENT_MODULE is not None:
        return _LLM_CLIENT_MODULE

    spec = importlib.util.spec_from_file_location("skill_engine_llm_client", LLM_CLIENT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load LLM client module from {LLM_CLIENT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _LLM_CLIENT_MODULE = module
    return module


def _log_progress(completed: int, total: int) -> None:
    print(f"Body site progress: {min(completed, total)}/{total}", flush=True)


def _parse_body_site_payload(payload: dict[str, Any], expected_terms: list[str]) -> dict[str, str]:
    items = payload.get("items")
    if not isinstance(items, list):
        return {term: "未知" for term in expected_terms}

    body_sites: dict[str, str] = {}
    expected = set(expected_terms)
    for item in items:
        if not isinstance(item, dict):
            continue
        term = str(item.get("term") or "").strip()
        body_site = str(item.get("body_site") or "").strip()
        if term in expected:
            body_sites[term] = body_site if body_site in BODY_SITES else "未知"

    for term in expected_terms:
        body_sites.setdefault(term, "未知")
    return body_sites


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Build definition2id.json from HPO Babelon Chinese translations."
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Input Babelon TSV path.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output JSON path.")
    parser.add_argument("--llm-batch-size", type=int, default=50, help="Terms per DeepSeek body_site request.")
    parser.add_argument("--llm-workers", type=int, default=1, help="Concurrent DeepSeek body_site requests.")
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    output_path = Path(args.output)
    definition2id = build_definition2id(input_path)
    definition2id_with_body_site = build_definition2id_with_body_site(
        definition2id,
        llm_batch_size=args.llm_batch_size,
        llm_workers=args.llm_workers,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(definition2id_with_body_site, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(definition2id)} definitions to {output_path}")
    multi_id_count = sum(1 for subject_ids in definition2id.values() if len(subject_ids) > 1)
    print(f"Found {multi_id_count} definitions with multiple IDs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
