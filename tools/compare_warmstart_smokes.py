#!/usr/bin/env python3
"""Compare Phase 2A legacy/C0/C1 one-update smoke artifacts."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stainpms.warmstart_equivalence import compare_c0_reference, summarize_c1_scale


def read_report(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def write_atomic(path: Path, payload: dict) -> None:
    if path.exists():
        raise ValueError(f"refusing to overwrite comparison report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy", required=True, type=Path)
    parser.add_argument("--c0", required=True, type=Path)
    parser.add_argument("--c1", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--absolute-tolerance", default=1e-6, type=float)
    parser.add_argument("--relative-tolerance", default=1e-5, type=float)
    args = parser.parse_args()

    legacy = read_report(args.legacy)
    c0 = read_report(args.c0)
    c1 = read_report(args.c1)
    if legacy["training_configuration"]["arm"] != "legacy":
        raise ValueError("--legacy artifact does not declare legacy arm")
    if c0["training_configuration"]["arm"] != "c0":
        raise ValueError("--c0 artifact does not declare c0 arm")
    if c1["training_configuration"]["arm"] != "c1":
        raise ValueError("--c1 artifact does not declare c1 arm")

    equivalence = compare_c0_reference(
        legacy,
        c0,
        absolute_tolerance=args.absolute_tolerance,
        relative_tolerance=args.relative_tolerance,
    )
    c1_scale = summarize_c1_scale(c0, c1)
    passed = equivalence["status"] == "pass" and c1_scale["status"] == "complete"
    payload = {
        "schema_version": 1,
        "phase": "2A-warmstart-feasibility",
        "status": "pass" if passed else "fail",
        "inputs": {
            "legacy": str(args.legacy.resolve()),
            "c0": str(args.c0.resolve()),
            "c1": str(args.c1.resolve()),
        },
        "c0_numerical_equivalence": equivalence,
        "c1_scale_and_forward_audit": c1_scale,
    }
    write_atomic(args.output.resolve(), payload)
    print(json.dumps({"status": payload["status"], "output": str(args.output.resolve())}))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
