from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from guideline_skill.schema import load_skill_pack, validate_skill_pack  # noqa: E402


def count_routing_keywords(skill_pack) -> int:
    routing = skill_pack.routing_profile
    return sum(
        len(items)
        for items in [
            routing.key_symptoms,
            routing.key_tests,
            routing.key_findings,
            routing.red_flags,
            routing.must_differentiate,
            routing.disease_aliases,
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a guideline skill pack YAML/JSON file.")
    parser.add_argument("--input", required=True, help="Path to a skill pack .yaml/.yml/.json file.")
    args = parser.parse_args()

    try:
        skill_pack = validate_skill_pack(load_skill_pack(args.input))
    except Exception as exc:
        print("validation: fail")
        print(f"error: {exc}")
        return 1

    print(f"skill_name: {skill_pack.skill_name}")
    print(f"disease_name: {skill_pack.disease_name}")
    print(f"subskill_count: {len(skill_pack.subskills)}")
    print(f"recommendation_card_count: {len(skill_pack.recommendation_cards)}")
    print(f"routing_keywords_count: {count_routing_keywords(skill_pack)}")
    print("validation: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
