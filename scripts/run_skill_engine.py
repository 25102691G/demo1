from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from skill_engine.case_normalizer import load_case_json, normalize_case, normalize_case_from_json
from skill_engine.hpo_extractor import (
    DEFAULT_DEFINITION2ID_PATH,
    DEFAULT_DEFINITION_EMBEDDINGS_PATH,
    DEFAULT_MODEL_PATH,
    HpoExtractor,
)
from skill_engine.llm_client import OpenAICompatibleJsonChatClient, load_llm_config_from_env
from skill_engine.output_builder import build_error_output, build_workflow_output
from skill_engine.router import route_skills
from skill_engine.skill_loader import load_skill_packs
from skill_engine.workflow_engine import WorkflowEngine


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Run a generic guideline SkillEngine workflow.")
    parser.add_argument("--skills-dir", default="data/skills", help="Directory containing */skill.yaml packs.")

    # Input options:text/file/json 三选一。
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input-text", help="Raw case text.")
    input_group.add_argument("--input-file", help="Raw case text file.")
    input_group.add_argument("--case-json", help="Partially or fully structured canonical case JSON.")
    
    parser.add_argument("--skill-schema", default="schema/skill_pack.schema.json")
    parser.add_argument("--case-schema", default="schema/canonical_case.schema.json")
    parser.add_argument("--output-schema", default="schema/workflow_output.schema.json")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-score", type=float)
    parser.add_argument(
        "--endoscopy",
        action="store_true",
        help="Use HPO extraction for canonical.endoscopy.items instead of rule keywords.",
    )
    parser.add_argument("--hpo-model-path", default=str(DEFAULT_MODEL_PATH), help="Local BGE model path.")
    parser.add_argument("--hpo-definition2id-path", default=str(DEFAULT_DEFINITION2ID_PATH))
    parser.add_argument("--hpo-definition-embeddings-path", default=str(DEFAULT_DEFINITION_EMBEDDINGS_PATH))
    parser.add_argument("--hpo-similarity-threshold", type=float, default=0.8)
    parser.add_argument("--hpo-batch-size", type=int, default=30)
    parser.add_argument("--hpo-max-length", type=int, default=128)
    parser.add_argument("--llm-temperature", type=float, default=0.0)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--print-canonical-case", action="store_true")
    args = parser.parse_args(argv)

    # TODO：病人case提取需要改为LLM + HPO数据库形式
    try:
        raw_input = _read_raw_input(args)
        endoscopy_items_extractor = _build_hpo_endoscopy_extractor(args) if args.endoscopy else None
        if args.case_json:
            canonical_case = normalize_case_from_json(
                load_case_json(_resolve(args.case_json)),
                raw_input,
                _resolve(args.case_schema),
                endoscopy_items_extractor=endoscopy_items_extractor,
            )
        else:
            canonical_case = normalize_case(
                raw_input,
                _resolve(args.case_schema),
                endoscopy_items_extractor=endoscopy_items_extractor,
            )

        packs, load_errors = load_skill_packs(_resolve(args.skills_dir), _resolve(args.skill_schema))
        if not packs:
            output = build_error_output(
                canonical_case=canonical_case,
                output_schema_path=_resolve(args.output_schema),
                errors=load_errors,
                debug=True,
            )
        else:
            top_candidates = route_skills(
                canonical_case,
                packs,
                top_k=args.top_k,
                min_score=args.min_score,
            )
            packs_by_id = {pack.skill_id: pack for pack in packs}
            engine = WorkflowEngine()
            selected_outputs = [
                engine.run(packs_by_id[candidate["skill_id"]], canonical_case, candidate)
                for candidate in top_candidates
                if candidate["skill_id"] in packs_by_id
            ]
            output = build_workflow_output(
                canonical_case=canonical_case,
                top_candidates=top_candidates,
                selected_skill_outputs=selected_outputs,
                skill_packs=packs,
                output_schema_path=_resolve(args.output_schema),
                debug=args.debug,
                errors=load_errors,
            )
    except Exception as exc:
        print(f"run_skill_engine: error: {exc}", file=sys.stderr)
        return 1

    if args.print_canonical_case:
        print(json.dumps(canonical_case, ensure_ascii=False, indent=2))
    out_path = _default_output_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"workflow_output written to {out_path}")
    return 0


def _read_raw_input(args: argparse.Namespace) -> str:
    if args.input_text:
        return args.input_text
    if args.input_file:
        return _resolve(args.input_file).read_text(encoding="utf-8-sig")
    if args.case_json:
        data: dict[str, Any] = load_case_json(_resolve(args.case_json))
        return str(data.get("raw_input") or "")
    raise ValueError("provide --input-text, --input-file, or --case-json")


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _build_hpo_endoscopy_extractor(args: argparse.Namespace):
    hpo_extractor = HpoExtractor.from_paths(
        model_path=_resolve(args.hpo_model_path),
        definition2id_path=_resolve(args.hpo_definition2id_path),
        definition_embeddings_path=_resolve(args.hpo_definition_embeddings_path),
        similarity_threshold=args.hpo_similarity_threshold,
        batch_size=args.hpo_batch_size,
        max_length=args.hpo_max_length,
    )
    llm_client = OpenAICompatibleJsonChatClient(
        load_llm_config_from_env(temperature=args.llm_temperature)
    )

    def extract(text: str) -> list[dict[str, Any]]:
        result = hpo_extractor.extract_from_text(text, llm_client)
        return _hpo_result_to_endoscopy_items(result)

    return extract


def _hpo_result_to_endoscopy_items(result: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    mappings = result.get("hpo_mappings") or []
    for mapping in mappings:
        if not isinstance(mapping, dict):
            continue
        phenotype = str(mapping.get("original_phenotype") or "").strip()
        hpo_term = str(mapping.get("hpo_term") or "").strip()
        finding = hpo_term or phenotype
        if not finding:
            continue
        items.append(
            {
                "type": "hpo_extraction",
                "findings": [finding],
                "biopsy_taken": "unknown",
                "date": None,
                "source_text": phenotype or finding,
                "hpo_code": mapping.get("hpo_code"),
                "hpo_term": hpo_term or None,
                "similarity_score": mapping.get("similarity_score"),
                "status": mapping.get("status"),
            }
        )
    return items


def _default_output_path() -> Path:
    filename = datetime.now().strftime("%Y%m%d_%H_%M.json")
    return ROOT / "data" / "runs" / filename


if __name__ == "__main__":
    raise SystemExit(main())
