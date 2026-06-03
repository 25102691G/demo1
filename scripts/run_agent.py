from __future__ import annotations

import argparse
from datetime import datetime
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agent_orchestrator import run_agent  # noqa: E402


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Run the medical guideline agent orchestrator.")
    parser.add_argument("--skills", required=True, help="Skill pack file or directory containing YAML/JSON packs.")
    parser.add_argument("--text", required=True, help="Patient-entered clinical text.")
    parser.add_argument("--top-k", type=int, default=5, help="Maximum number of candidate skills to route.")
    args = parser.parse_args()

    response = run_agent(args.text, skills_path=args.skills, top_k=args.top_k)
    payload = json.dumps(response.model_dump(mode="json"), ensure_ascii=False, indent=2)
    output_dir = ROOT / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{datetime.now().strftime('%Y%m%d_%H_%M')}.json"
    output_file.write_text(payload, encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
