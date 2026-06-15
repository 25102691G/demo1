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
from skill_engine.llm_client import OpenAICompatibleJsonChatClient, load_deepseek_config_from_env
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

    # 输入来源：文本、文件、JSON 三选一。
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input-text", help="Raw case text.")
    input_group.add_argument("--input-file", help="Raw case text file.")
    input_group.add_argument("--case-json", help="Partially or fully structured canonical case JSON.")

    parser.add_argument("--skill-schema", default="schema/skill_pack.schema.json")
    parser.add_argument("--case-schema", default="schema/canonical_case.schema.json")
    parser.add_argument("--output-schema", default="schema/workflow_output.schema.json")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-score", type=float)
    parser.add_argument("--hpo-similarity-threshold", type=_similarity_threshold)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--print-canonical-case", action="store_true")
    args = parser.parse_args(argv)

    try:
        raw_input = _read_raw_input(args)
        hpo_extractor, deepseek_client = _build_hpo_dependencies(args.hpo_similarity_threshold)

        if args.case_json:
            canonical_case = normalize_case_from_json(
                load_case_json(_resolve(args.case_json)),
                raw_input,
                _resolve(args.case_schema),
                hpo_extractor=hpo_extractor,
                deepseek_client=deepseek_client,
            )
        else:
            canonical_case = normalize_case(
                raw_input,
                _resolve(args.case_schema),
                hpo_extractor=hpo_extractor,
                deepseek_client=deepseek_client,
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


def _default_output_path() -> Path:
    filename = datetime.now().strftime("%Y%m%d_%H_%M.json")
    return ROOT / "data" / "runs" / filename


def _similarity_threshold(value: str) -> float:
    try:
        threshold = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number between 0 and 1") from exc
    if not 0 <= threshold <= 1:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return threshold


def _build_hpo_dependencies(
    hpo_similarity_threshold: float | None = None,
) -> tuple[HpoExtractor, OpenAICompatibleJsonChatClient]:
    hpo_kwargs: dict[str, Any] = {}
    if hpo_similarity_threshold is not None:
        hpo_kwargs["similarity_threshold"] = hpo_similarity_threshold
    hpo_extractor = HpoExtractor.from_paths(
        model_path=DEFAULT_MODEL_PATH,
        definition2id_path=DEFAULT_DEFINITION2ID_PATH,
        definition_embeddings_path=DEFAULT_DEFINITION_EMBEDDINGS_PATH,
        **hpo_kwargs,
    )
    deepseek_client = OpenAICompatibleJsonChatClient(load_deepseek_config_from_env())
    return hpo_extractor, deepseek_client


if __name__ == "__main__":
    raise SystemExit(main())
