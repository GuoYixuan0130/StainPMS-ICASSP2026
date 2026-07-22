"""Apply the frozen Phase 2A GPU-hour gate to synchronized timing reports."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stainpms.phase2a_budget import estimate_dataset_budget


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--dataset", required=True, choices=["tnbc", "monuseg"])
    parser.add_argument("--base-timing", required=True)
    parser.add_argument("--active-timing", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    recipe_path = Path(args.recipe).resolve()
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    base = json.loads(Path(args.base_timing).read_text(encoding="utf-8"))
    active = json.loads(Path(args.active_timing).read_text(encoding="utf-8"))
    report = estimate_dataset_budget(recipe, args.dataset, base, active)
    report["recipe_path"] = str(recipe_path)
    report["recipe_sha256"] = sha256_file(recipe_path)
    output = Path(args.output).resolve()
    if output.exists():
        raise ValueError(f"budget output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "gpu_hours": report["estimated_total_gpu_hours"], "output": str(output)}))
    return 0 if report["status"] == "gate_pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
