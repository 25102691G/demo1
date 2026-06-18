from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pdf_ocr import BaiduDocParserClient, BaiduDocParserConfig


def preview_pdf_ocr_file(
    input_path: str | Path,
    *,
    client: BaiduDocParserClient,
) -> int:
    pdf_path = Path(input_path)
    artifacts = client.parse_pdf_artifacts(pdf_path)
    if artifacts.parse_result_text:
        pdf_path.with_suffix(".parse_result.json").write_text(
            artifacts.parse_result_text,
            encoding="utf-8",
            newline="\n",
        )
    if artifacts.markdown_text:
        pdf_path.with_suffix(".markdown.md").write_text(
            artifacts.markdown_text,
            encoding="utf-8",
            newline="\n",
        )

    return 0


def preview_pdf_ocr(
    input_paths: Sequence[str | Path],
    *,
    baidu_api_key_env: str = "BAIDU_API_KEY",
    baidu_secret_key_env: str = "BAIDU_SECRET_KEY",
) -> int:
    client = BaiduDocParserClient(
        BaiduDocParserConfig(
            api_key_env=baidu_api_key_env,
            secret_key_env=baidu_secret_key_env,
        )
    )
    for input_path in input_paths:
        preview_pdf_ocr_file(input_path, client=client)
    return 0


def resolve_pdf_inputs(
    input_path: str | Path | None,
    input_dir: str | Path | None,
) -> list[Path]:
    if bool(input_path) == bool(input_dir):
        raise ValueError("必须且只能指定 --input 或 --input-dir 其中一个。")

    if input_path is not None:
        path = Path(input_path)
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"--input 仅支持 PDF 文件: {path}")
        return [path]

    directory = Path(input_dir or "")
    if not directory.exists():
        raise FileNotFoundError(f"Input directory does not exist: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {directory}")
    return sorted(
        (path for path in directory.rglob("*.pdf") if path.is_file()),
        key=lambda item: str(item).casefold(),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preview Baidu OCR text extracted from a PDF.")
    parser.add_argument("--input", default=None, help="Input PDF file path.")
    parser.add_argument("--input-dir", default=None, help="Directory containing PDF files.")
    parser.add_argument(
        "--baidu-api-key-env",
        default="BAIDU_API_KEY",
        help="Environment variable name for Baidu API Key. Defaults to BAIDU_API_KEY.",
    )
    parser.add_argument(
        "--baidu-secret-key-env",
        default="BAIDU_SECRET_KEY",
        help="Environment variable name for Baidu Secret Key. Defaults to BAIDU_SECRET_KEY.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    return preview_pdf_ocr(
        resolve_pdf_inputs(args.input, args.input_dir),
        baidu_api_key_env=args.baidu_api_key_env,
        baidu_secret_key_env=args.baidu_secret_key_env,
    )


if __name__ == "__main__":
    raise SystemExit(main())
