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

from skill_engine.case_normalizer import normalize_case
from skill_engine.hpo_extractor import (
    DEFAULT_DEFINITION2ID_PATH,
    DEFAULT_DEFINITION_EMBEDDINGS_PATH,
    DEFAULT_MODEL_PATH as DEFAULT_HPO_MODEL_PATH,
    HpoExtractor,
)
from skill_engine.icd_extractor import (
    DEFAULT_ICD10_EMBEDDINGS_PATH,
    DEFAULT_ICD10_PATH,
    DEFAULT_MODEL_PATH as DEFAULT_ICD_MODEL_PATH,
    IcdExtractor,
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
    parser.add_argument(
        "--skills-dir",
        default="data/skills",
        help="Directory containing */skill_hpo.yaml or */skill_icd10.yaml packs.",
    )

    # 病人基本信息：必填。
    parser.add_argument("--name", required=True, help="姓名。")
    parser.add_argument(
        "--sex",
        choices=("male", "female", "unknown"),
        required=True,
        help="性别。",
    )
    parser.add_argument("--age", type=int, required=True, help="年龄。")

    # 病人病例信息：五类病例文本至少填写一类。
    parser.add_argument("--clinical-presentation", help="临床表现。")
    parser.add_argument("--lab-tests", help="实验室检查。")
    parser.add_argument("--imaging-tests", help="影像学检查。")
    parser.add_argument("--endoscopy", help="内镜检查。")
    parser.add_argument("--pathology", help="病理。")

    # 标准化数据库来源（必填，互斥）：HPO / ICD10
    feature_group = parser.add_mutually_exclusive_group(required=True)
    feature_group.add_argument("--hpo", action="store_true", help="Use HPO feature extraction.")
    feature_group.add_argument("--icd10", action="store_true", help="Use ICD10 feature extraction.")

    parser.add_argument("--skill-schema", default="schema/skill_pack.schema.json")
    parser.add_argument("--case-schema", default="schema/canonical_case.schema.json")
    parser.add_argument("--output-schema", default="schema/workflow_output.schema.json")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-score", type=float)
    parser.add_argument("--similarity-threshold", type=_similarity_threshold)
    parser.add_argument(
        "--model-path",
        default=None,
        help="Optional ICD10 embedding model path. Use the same model that built ICD10_embeddings.pt.",
    )
    parser.add_argument(
        "--hpo-summary-output",
        default=None,
        help="Optional extra feature summary JSON path. By default, feature summary is embedded in workflow output.",
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--print-canonical-case", action="store_true")
    args = parser.parse_args(argv)

    try:
        structured_input = _read_structured_input(args)
        raw_input = _build_raw_input(structured_input)
        feature_mode = _feature_mode_from_args(args)
        skill_filename = _skill_filename_for_mode(feature_mode)
        feature_extractor, deepseek_client = _build_feature_dependencies(
            feature_mode,
            args.similarity_threshold,
            model_path=args.model_path,
        )

        canonical_case = normalize_case(
            raw_input,
            _resolve(args.case_schema),
            deepseek_client=deepseek_client,
            feature_extractor=feature_extractor,
            feature_mode=feature_mode,
        )
        canonical_case["raw_input"] = structured_input
        _apply_basic_case_fields(canonical_case, args)

        packs, load_errors = load_skill_packs(
            _resolve(args.skills_dir),
            _resolve(args.skill_schema),
            skill_filename=skill_filename,
        )
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
                feature_mode=feature_mode,
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
    summary_key = "hpo_summary" if feature_mode == "hpo" else "icd10_summary"
    output[summary_key] = feature_extractor.get_last_summary()
    out_path = _default_output_path(feature_mode)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.hpo_summary_output:
        feature_summary_path = _hpo_summary_output_path(args)
        feature_extractor.write_last_summary(feature_summary_path)
        print(f"{summary_key} written to {feature_summary_path}")
    print(f"workflow_output written to {out_path}")
    return 0


def _read_structured_input(args: argparse.Namespace) -> dict[str, str]:
    """读取病人病例结构化输入，至少包含五类病例文本中的一类。"""
    structured_input = {
        "clinical_presentation": _clean_arg_text(args.clinical_presentation),
        "lab_tests": _clean_arg_text(args.lab_tests),
        "imaging_tests": _clean_arg_text(args.imaging_tests),
        "endoscopy": _clean_arg_text(args.endoscopy),
        "pathology": _clean_arg_text(args.pathology),
    }
    if not any(structured_input.values()):
        raise ValueError(
            "请至少填写一个病例信息参数：--clinical-presentation、--lab-tests、"
            "--imaging-tests、--endoscopy 或 --pathology"
        )
    return structured_input


def _build_raw_input(structured_input: dict[str, str]) -> str:
    labels = {
        "clinical_presentation": "临床表现",
        "lab_tests": "实验室检查",
        "imaging_tests": "影像学检查",
        "endoscopy": "内镜检查",
        "pathology": "病理",
    }
    return "\n".join(
        f"{labels[key]}：{value}" for key, value in structured_input.items() if value
    )


def _clean_arg_text(value: str | None) -> str:
    return str(value or "").strip()


def _apply_basic_case_fields(canonical_case: dict[str, Any], args: argparse.Namespace) -> None:
    canonical_case["name"] = _clean_arg_text(args.name)
    canonical_case["sex"] = args.sex
    canonical_case["age"] = args.age


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _default_output_path(feature_mode: str) -> Path:
    filename = datetime.now().strftime(f"%Y%m%d_%H_%M_{feature_mode}.json")
    return ROOT / "data" / "runs" / filename


def _hpo_summary_output_path(args: argparse.Namespace) -> Path:
    if args.hpo_summary_output:
        return _resolve(args.hpo_summary_output)
    raise ValueError("--hpo-summary-output is required")


def _similarity_threshold(value: str) -> float:
    try:
        threshold = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number between 0 and 1") from exc
    if not 0 <= threshold <= 1:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return threshold


def _feature_mode_from_args(args: argparse.Namespace) -> str:
    if args.hpo:
        return "hpo"
    if args.icd10:
        return "icd10"
    raise ValueError("provide exactly one of --hpo or --icd10")


def _skill_filename_for_mode(feature_mode: str) -> str:
    if feature_mode == "hpo":
        return "skill_hpo.yaml"
    if feature_mode == "icd10":
        return "skill_icd10.yaml"
    raise ValueError(f"unsupported feature mode: {feature_mode}")


def _build_feature_dependencies(
    feature_mode: str,
    similarity_threshold: float | None = None,
    model_path: str | Path | None = None,
) -> tuple[Any, OpenAICompatibleJsonChatClient]:
    if feature_mode == "hpo":
        return _build_hpo_dependencies(similarity_threshold)
    if feature_mode == "icd10":
        return _build_icd10_dependencies(similarity_threshold, model_path=model_path)
    raise ValueError(f"unsupported feature mode: {feature_mode}")


def _build_hpo_dependencies(
    similarity_threshold: float | None = None,
) -> tuple[HpoExtractor, OpenAICompatibleJsonChatClient]:
    hpo_kwargs: dict[str, Any] = {}
    if similarity_threshold is not None:
        hpo_kwargs["similarity_threshold"] = similarity_threshold
    hpo_extractor = HpoExtractor.from_paths(
        model_path=DEFAULT_HPO_MODEL_PATH,
        definition2id_path=DEFAULT_DEFINITION2ID_PATH,
        definition_embeddings_path=DEFAULT_DEFINITION_EMBEDDINGS_PATH,
        **hpo_kwargs,
    )
    deepseek_client = OpenAICompatibleJsonChatClient(load_deepseek_config_from_env())
    return hpo_extractor, deepseek_client


def _build_icd10_dependencies(
    similarity_threshold: float | None = None,
    model_path: str | Path | None = None,
) -> tuple[IcdExtractor, OpenAICompatibleJsonChatClient]:
    icd_kwargs: dict[str, Any] = {}
    if similarity_threshold is not None:
        icd_kwargs["similarity_threshold"] = similarity_threshold
    icd_extractor = IcdExtractor.from_paths(
        model_path=model_path or DEFAULT_ICD_MODEL_PATH,
        icd10_path=DEFAULT_ICD10_PATH,
        icd10_embeddings_path=DEFAULT_ICD10_EMBEDDINGS_PATH,
        **icd_kwargs,
    )
    deepseek_client = OpenAICompatibleJsonChatClient(load_deepseek_config_from_env())
    return icd_extractor, deepseek_client


if __name__ == "__main__":
    raise SystemExit(main())
