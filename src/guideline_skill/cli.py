from __future__ import annotations

import argparse
import json
import logging
import os
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from extractor.models import TextPage
from extractor.pdf_loader import load_pdf_pages
from extractor.text_cleaner import clean_pages

from .anchors import AnchorRegistry
from .classifier import ClassificationResult, GuidelineClassifier
from .extractors import ClinicalInfoExtractor
from .llm import DeepSeekClient
from .normalizer import LLMNormalizer
from .pipelines import NarrativeGuidelinePipeline, StructuredGuidelinePipeline
from .schemas import StatementUnit


logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path("data/skills")
RESULT_FILENAME = "result.jsonl"
SUMMARY_FILENAME = "summary.json"
SUPPORTED_INPUT_SUFFIXES = {".pdf", ".txt", ".md"}


ExtractedUnit = StatementUnit


def extract_document(
    input_path: str | Path,
    *,
    output_path: str | Path | None = None,
    summary_path: str | Path | None = None,
    output_root: str | Path = DEFAULT_OUTPUT_DIR,
    deepseek_client: Any | None = None,
    title: str | None = None,
) -> dict[str, Any]:
    """Extract one guideline document and write JSONL plus summary files."""

    source_path = Path(input_path)
    default_output_dir, default_output, default_summary = output_paths_for_input(
        source_path,
        output_root=output_root,
    )
    output = Path(output_path) if output_path is not None else default_output
    summary = Path(summary_path) if summary_path is not None else default_summary
    client = deepseek_client or DeepSeekClient()

    logger.info("Loading guideline document: %s", source_path)
    pages = load_document_pages(source_path)
    cleaned_pages = clean_pages(pages)
    text = pages_to_text(cleaned_pages)

    units, classification = extract_units_from_text(
        text,
        source_file=str(source_path),
        title=title or source_path.stem,
        deepseek_client=client,
    )

    write_jsonl(units, output)
    summary_payload = build_summary(
        source_file=str(source_path),
        classification=classification,
        units=units,
        output_dir=output.parent,
        llm_model=getattr(client, "model", None) or os.getenv("DEEPSEEK_MODEL"),
    )
    write_json(summary_payload, summary)

    logger.info("Wrote %s extracted units to %s", len(units), output)
    logger.info("Wrote extraction summary to %s", summary)
    return summary_payload


def batch_extract(
    inputs: Sequence[str | Path],
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    deepseek_client: Any | None = None,
) -> dict[str, Any]:
    """Extract multiple guideline documents, one output folder per input."""

    output_directory = Path(output_dir)
    client = deepseek_client or DeepSeekClient()

    document_summaries: list[dict[str, Any]] = []
    for input_path in inputs:
        source_path = Path(input_path)
        logger.info("Batch extracting guideline document: %s", source_path)
        document_summaries.append(
            extract_document(
                source_path,
                output_root=output_directory,
                deepseek_client=client,
                title=source_path.stem,
            )
        )

    return build_batch_summary(
        document_summaries,
        output_dir=output_directory,
        llm_model=getattr(client, "model", None) or os.getenv("DEEPSEEK_MODEL"),
    )


def resolve_batch_inputs(
    inputs: Sequence[str | Path] | None = None,
    *,
    input_dir: str | Path | None = None,
) -> list[Path]:
    resolved: list[Path] = []
    seen: set[str] = set()

    for value in inputs or []:
        path = Path(value)
        key = _stable_path_key(path)
        if key in seen:
            continue
        seen.add(key)
        resolved.append(path)

    if input_dir is not None:
        directory = Path(input_dir)
        if not directory.exists():
            raise FileNotFoundError(f"Input directory does not exist: {directory}")
        if not directory.is_dir():
            raise NotADirectoryError(f"Input path is not a directory: {directory}")
        for path in sorted(directory.iterdir(), key=lambda item: item.name.casefold()):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_INPUT_SUFFIXES:
                continue
            key = _stable_path_key(path)
            if key in seen:
                continue
            seen.add(key)
            resolved.append(path)

    if not resolved:
        raise ValueError("batch requires --inputs and/or --input-dir with at least one supported file.")
    return resolved


def _stable_path_key(path: Path) -> str:
    try:
        return str(path.resolve()).casefold()
    except OSError:
        return str(path).casefold()


def output_paths_for_input(
    input_path: str | Path,
    *,
    output_root: str | Path = DEFAULT_OUTPUT_DIR,
) -> tuple[Path, Path, Path]:
    source_path = Path(input_path)
    output_dir = Path(output_root) / source_path.stem
    return output_dir, output_dir / RESULT_FILENAME, output_dir / SUMMARY_FILENAME


def _units_from_summary(summary: dict[str, Any]) -> int:
    total_units = summary.get("total_units", 0)
    return int(total_units) if isinstance(total_units, int) else 0


def _human_review_from_summary(summary: dict[str, Any]) -> int:
    count = summary.get("human_review_count", 0)
    return int(count) if isinstance(count, int) else 0


def _counter_from_summaries(
    summaries: Sequence[dict[str, Any]],
    key: str,
) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for summary in summaries:
        payload = summary.get(key, {})
        if not isinstance(payload, dict):
            continue
        for item_key, item_value in payload.items():
            if isinstance(item_value, int):
                counter[str(item_key)] += item_value
    return dict(counter)


def extract_units_from_text(
    text: str,
    *,
    source_file: str | None,
    title: str | None,
    deepseek_client: Any,
) -> tuple[list[ExtractedUnit], ClassificationResult]:
    anchor_registry = AnchorRegistry()
    classifier = GuidelineClassifier(anchor_registry)
    classification = classifier.classify(text)

    logger.info(
        "Classified document as %s (total_score=%s, unit_anchors=%s)",
        classification.doc_type,
        classification.total_score,
        classification.unit_anchor_count,
    )

    if classification.doc_type == "structured_guideline":
        pipeline = StructuredGuidelinePipeline(
            anchor_registry=anchor_registry,
            normalizer=LLMNormalizer(deepseek_client),
        )
        return pipeline.run(
            text,
            classification,
            title=title,
            source_file=source_file,
        ), classification

    pipeline = NarrativeGuidelinePipeline(
        clinical_info_extractor=ClinicalInfoExtractor(deepseek_client),
    )
    return pipeline.run(
        text,
        title=title,
        source_file=source_file,
    ), classification


def load_document_pages(path: str | Path) -> list[TextPage]:
    source_path = Path(path)
    if source_path.suffix.lower() in {".txt", ".md"}:
        return [TextPage(page_number=1, text=source_path.read_text(encoding="utf-8"))]
    return load_pdf_pages(source_path)


def pages_to_text(pages: Sequence[TextPage]) -> str:
    return "\n".join(f"## Page {page.page_number}\n{page.text}" for page in pages)


def write_jsonl(units: Sequence[ExtractedUnit], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for unit in units:
            handle.write(json.dumps(unit.model_dump(mode="json"), ensure_ascii=False))
            handle.write("\n")


def write_json(payload: dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_summary(
    *,
    source_file: str,
    classification: ClassificationResult,
    units: Sequence[ExtractedUnit],
    output_dir: Path,
    llm_model: str | None,
) -> dict[str, Any]:
    record_type_counts = Counter(unit.record_type for unit in units)
    unit_type_counts: Counter[str] = Counter()
    human_review_count = 0

    for extracted_unit in units:
        unit_body = extracted_unit
        if getattr(unit_body, "needs_human_review", False):
            human_review_count += 1
        unit_type = getattr(unit_body, "unit_type", None) or getattr(unit_body, "statement_type", None)
        if unit_type:
            unit_type_counts[str(unit_type)] += 1

    return {
        "source_file": source_file,
        "doc_type": classification.doc_type,
        "total_score": classification.total_score,
        "unit_score": classification.unit_score,
        "field_score": classification.field_score,
        "unit_anchor_count": classification.unit_anchor_count,
        "field_anchor_count": classification.field_anchor_count,
        "primary_unit_anchor": classification.primary_unit_anchor,
        "total_units": len(units),
        "record_type_counts": dict(record_type_counts),
        "unit_type_counts": dict(unit_type_counts),
        "human_review_count": human_review_count,
        "llm_enabled": True,
        "llm_model": llm_model,
        "output_dir": output_dir.as_posix(),
    }


def build_batch_summary(
    document_summaries: Sequence[dict[str, Any]],
    *,
    output_dir: Path,
    llm_model: str | None,
) -> dict[str, Any]:
    return {
        "source_file": [summary["source_file"] for summary in document_summaries],
        "doc_type": "batch",
        "total_score": None,
        "unit_score": None,
        "field_score": None,
        "unit_anchor_count": None,
        "field_anchor_count": None,
        "primary_unit_anchor": None,
        "total_units": sum(_units_from_summary(summary) for summary in document_summaries),
        "record_type_counts": _counter_from_summaries(document_summaries, "record_type_counts"),
        "unit_type_counts": _counter_from_summaries(document_summaries, "unit_type_counts"),
        "human_review_count": sum(_human_review_from_summary(summary) for summary in document_summaries),
        "llm_enabled": True,
        "llm_model": llm_model,
        "output_dir": output_dir.as_posix(),
        "documents": list(document_summaries),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Classify and extract medical guideline documents.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract_parser = subparsers.add_parser("extract", help="Extract one guideline document.")
    extract_parser.add_argument("--input", required=True, help="Input PDF/TXT/MD file path.")
    extract_parser.add_argument(
        "--output",
        default=None,
        help="Optional JSONL override. Defaults to data/skills/<input-stem>/result.jsonl.",
    )
    extract_parser.add_argument(
        "--summary",
        default=None,
        help="Optional summary override. Defaults to data/skills/<input-stem>/summary.json.",
    )
    extract_parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output root used when --output/--summary are omitted.",
    )
    extract_parser.add_argument("--title", default=None, help="Optional guideline title.")

    batch_parser = subparsers.add_parser("batch", help="Extract multiple guideline documents.")
    batch_parser.add_argument("--inputs", nargs="+", default=None, help="Input PDF/TXT/MD file paths.")
    batch_parser.add_argument(
        "--input-dir",
        default=None,
        help="Directory containing PDF/TXT/MD files. Files are scanned non-recursively.",
    )
    batch_parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Optional output root. Each input gets data/skills/<input-stem>/ by default.",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = build_arg_parser().parse_args(argv)

    if args.command == "extract":
        extract_document(
            args.input,
            output_path=args.output,
            summary_path=args.summary,
            output_root=args.output_root,
            title=args.title,
        )
        return 0

    if args.command == "batch":
        batch_extract(
            resolve_batch_inputs(args.inputs, input_dir=args.input_dir),
            output_dir=args.output_dir,
        )
        return 0

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
