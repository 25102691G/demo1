from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from guideline_skill.schema import load_skill_pack  # noqa: E402
from patient_case_extractor import extract_patient_case  # noqa: E402
from skill_executor import execute_crohn_skill  # noqa: E402


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Run a disease skill executor against patient text.")
    parser.add_argument("--skill", required=True, help="Path to a disease skill pack YAML/JSON file.")
    parser.add_argument("--text", required=True, help="Patient-entered clinical text.")
    args = parser.parse_args()

    skill_pack = load_skill_pack(args.skill)
    patient_case = extract_patient_case(args.text)
    result = execute_crohn_skill(patient_case, skill_pack)

    print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
