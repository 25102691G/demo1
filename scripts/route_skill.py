from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from guideline_skill.schema import DiseaseSkillPack, load_skill_pack  # noqa: E402
from patient_case_extractor import extract_patient_case  # noqa: E402
from skill_router import route_disease_skills  # noqa: E402


def load_skill_packs(path: str | Path) -> list[DiseaseSkillPack]:
    skill_path = Path(path)
    if skill_path.is_file():
        return [load_skill_pack(skill_path)]
    if not skill_path.is_dir():
        raise ValueError(f"Skill path does not exist: {skill_path}")

    candidates = sorted(
        [
            *skill_path.rglob("*.yaml"),
            *skill_path.rglob("*.yml"),
            *skill_path.rglob("*.json"),
        ]
    )
    return [load_skill_pack(candidate) for candidate in candidates]


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Route a patient case to candidate disease skill packs.")
    parser.add_argument("--skills", required=True, help="Skill pack file or directory containing YAML/JSON packs.")
    parser.add_argument("--text", required=True, help="Patient-entered clinical text.")
    parser.add_argument("--top-k", type=int, default=5, help="Maximum number of candidate skills to return.")
    args = parser.parse_args()

    patient_case = extract_patient_case(args.text)
    skill_packs = load_skill_packs(args.skills)
    results = route_disease_skills(patient_case, skill_packs, top_k=args.top_k)

    payload = {
        "patient_case": patient_case.model_dump(mode="json"),
        "candidates": [result.model_dump(mode="json") for result in results],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
