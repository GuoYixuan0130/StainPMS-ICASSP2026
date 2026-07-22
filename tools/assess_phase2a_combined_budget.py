"""Apply the frozen combined Phase 2A GPU-hour stop rule."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stainpms.phase2a_budget import assess_combined_budget


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--dataset-report", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    recipe_path = Path(args.recipe).resolve()
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    reports = []
    for value in args.dataset_report:
        path = Path(value).resolve()
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("recipe_sha256") != sha256_file(recipe_path):
            raise ValueError(f"budget report recipe mismatch: {path}")
        reports.append(report)

    result = assess_combined_budget(recipe, reports)
    result["recipe_path"] = str(recipe_path)
    result["recipe_sha256"] = sha256_file(recipe_path)
    result["dataset_report_paths"] = [str(Path(value).resolve()) for value in args.dataset_report]
    output = Path(args.output).resolve()
    if output.exists():
        raise ValueError(f"combined budget output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": result["status"],
                "combined_gpu_hours": result["estimated_combined_gpu_hours"],
                "output": str(output),
            }
        )
    )
    return 0 if result["status"] == "gate_pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
