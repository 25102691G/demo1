from pathlib import Path

import argparse
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from extractor.csv_exporter import RECOMMENDATION_COLUMNS, write_csv
    from extractor.pdf_loader import load_pdf_pages
    from extractor.recommendation_parser import parse_recommendations
    from extractor.section_parser import parse_sections
    from extractor.text_cleaner import clean_pages
else:
    from .csv_exporter import RECOMMENDATION_COLUMNS, write_csv
    from .pdf_loader import load_pdf_pages
    from .recommendation_parser import parse_recommendations
    from .section_parser import parse_sections
    from .text_cleaner import clean_pages


def export_recommendations_csv(pdf_path: str, output_csv: str) -> None:
    raw_pages = load_pdf_pages(pdf_path)
    pages = clean_pages(raw_pages)
    sections = parse_sections(pages)
    recommendations = parse_recommendations(pages, sections)
    write_csv(recommendations, Path(output_csv), RECOMMENDATION_COLUMNS)

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract medical guideline PDF into graph CSV files.")
    parser.add_argument("--pdf", required=True, help="输入 PDF 路径")
    parser.add_argument("--output", default="data/csv/recommendations.csv", help="输出 CSV 文件路径")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    export_recommendations_csv(args.pdf, args.output)


if __name__ == "__main__":
    main()
