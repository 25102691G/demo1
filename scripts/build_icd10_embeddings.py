from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "ICD10" / "ICD10.json"
DEFAULT_OUTPUT = ROOT / "data" / "ICD10" / "ICD10_embeddings.pt"
DEFAULT_MODEL_PATH = ROOT / "data" / "qwen3-embedding-8b"


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Build ICD10 diagnosis_name embeddings.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Input ICD10 JSON path.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output torch tensor path.")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH), help="Local embedding model path.")
    parser.add_argument("--batch-size", type=int, default=64, help="Embedding batch size.")
    parser.add_argument("--max-length", type=int, default=128, help="Tokenizer max length.")
    args = parser.parse_args(argv)

    input_path = _resolve_path(args.input)
    output_path = _resolve_path(args.output)
    model_path = _resolve_path(args.model_path)

    records = _load_records(input_path)
    diagnosis_names = _extract_diagnosis_names(records)
    embeddings = build_embeddings(
        diagnosis_names,
        model_path=model_path,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch = _load_torch()
    torch.save(embeddings.cpu().float(), str(output_path))

    print(f"Wrote embeddings for {len(diagnosis_names)} ICD10 items to {output_path}")
    return 0


def build_embeddings(
    texts: list[str],
    *,
    model_path: Path,
    batch_size: int,
    max_length: int,
) -> Any:
    if batch_size <= 0:
        raise ValueError("--batch-size must be greater than 0")
    if max_length <= 0:
        raise ValueError("--max-length must be greater than 0")

    torch = _load_torch()
    AutoTokenizer, AutoModel = _load_transformers()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pooling_mode = _embedding_pooling_mode(model_path)

    tokenizer_kwargs: dict[str, Any] = {"local_files_only": True}
    if pooling_mode == "last_token":
        tokenizer_kwargs["padding_side"] = "left"
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), **tokenizer_kwargs)
    model = AutoModel.from_pretrained(str(model_path), local_files_only=True).to(device)
    model.eval()

    batches = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        batch_embeddings = _pool_embeddings(
            outputs.last_hidden_state,
            inputs["attention_mask"],
            pooling_mode=pooling_mode,
            torch=torch,
        )
        batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1).float()
        batches.append(batch_embeddings.cpu())
        print(f"Embedded {min(start + batch_size, len(texts))}/{len(texts)}")

    if not batches:
        return torch.empty((0, 0))
    return torch.cat(batches, dim=0)


def _pool_embeddings(
    last_hidden_state: Any,
    attention_mask: Any,
    *,
    pooling_mode: str,
    torch: Any,
) -> Any:
    if pooling_mode != "last_token":
        return last_hidden_state[:, 0, :]
    left_padding = bool((attention_mask[:, -1].sum() == attention_mask.shape[0]).item())
    if left_padding:
        return last_hidden_state[:, -1, :]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_state.shape[0]
    return last_hidden_state[torch.arange(batch_size, device=last_hidden_state.device), sequence_lengths]


def _embedding_pooling_mode(model_path: Path) -> str:
    pooling_config_path = model_path / "1_Pooling" / "config.json"
    if pooling_config_path.exists():
        with pooling_config_path.open("r", encoding="utf-8") as handle:
            pooling_config = json.load(handle)
        if pooling_config.get("pooling_mode_lasttoken") is True:
            return "last_token"
    config_path = model_path / "config.json"
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as handle:
            model_config = json.load(handle)
        if str(model_config.get("model_type") or "").lower() == "qwen3":
            return "last_token"
    return "cls"


def _resolve_path(path: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    return candidate.resolve()


def _load_records(input_path: Path) -> list[dict[str, Any]]:
    with input_path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"ICD10 JSON must be a list: {input_path}")
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"ICD10 item must be an object at index {index}")
    return data


def _extract_diagnosis_names(records: list[dict[str, Any]]) -> list[str]:
    diagnosis_names: list[str] = []
    skipped = 0
    for index, record in enumerate(records):
        value = record.get("diagnosis_name")
        if not isinstance(value, str) or not value.strip():
            skipped += 1
            continue
        diagnosis_names.append(value)
    if skipped:
        print(f"Skipped {skipped} ICD10 items with empty diagnosis_name")
    return diagnosis_names


def _load_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "Building ICD10 embeddings requires the optional dependency 'torch'. "
            "Run this script in an environment where torch is installed."
        ) from exc
    return torch


def _load_transformers() -> tuple[Any, Any]:
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Building ICD10 embeddings requires the optional dependency 'transformers'. "
            "Run this script in an environment where transformers is installed."
        ) from exc
    return AutoTokenizer, AutoModel


if __name__ == "__main__":
    raise SystemExit(main())
