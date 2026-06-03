from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from recommendation_extractor import extract_recommendations_from_pdf  # noqa: E402


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Extract draft RecommendationCards from a guideline PDF.")
    parser.add_argument("--pdf", required=True, help="Path to source PDF, or extracted .txt/.md text for testing.")
    parser.add_argument("--disease", required=True, help="Disease name for the draft skill pack.")
    parser.add_argument("--guideline-name", required=True, help="Guideline name.")
    parser.add_argument("--guideline-version", required=True, help="Guideline version.")
    parser.add_argument("--output", required=True, help="Output draft skill pack YAML path.")
    parser.add_argument("--sections-output", help="Optional extracted sections JSON path.")
    args = parser.parse_args()

    draft_pack = extract_recommendations_from_pdf(
        pdf_path=args.pdf,
        disease_name=args.disease,
        guideline_name=args.guideline_name,
        guideline_version=args.guideline_version,
        output_path=args.output,
        sections_output_path=args.sections_output,
    )
    summary = {
        "output": str(Path(args.output)),
        "sections_output": args.sections_output or str(Path(args.output).with_name("extracted_sections.json")),
        "skill_name": draft_pack.skill_name,
        "recommendation_card_count": len(draft_pack.recommendation_cards),
        "review_status": "needs_human_review",
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
