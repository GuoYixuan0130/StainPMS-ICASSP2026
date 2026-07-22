#!/usr/bin/env python3
"""Estimate approved C0/C1 5/10-epoch cost from measured update timings."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


PLANS = {
    "tnbc": {"updates_per_epoch": 270, "screen_updates": 1350, "full_updates": 2700},
    "monuseg": {
        "updates_per_epoch": 1332,
        "screen_updates": 6660,
        "full_updates": 13320,
    },
}


def read_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def timing_seconds(path: Path, dataset: str, arm: str) -> tuple[float, dict]:
    payload = read_json(path)
    if payload.get("status") != "complete" or payload.get("stage") != "timing":
        raise ValueError(f"incomplete timing artifact: {path}")
    if payload.get("training_configuration", {}).get("arm") != arm:
        raise ValueError(f"timing arm mismatch: {path}")
    isolation = payload.get("timing_audit_isolation", {})
    if isolation.get("warmup", {}).get("status") != "pass" or isolation.get(
        "timed", {}
    ).get("status") != "pass":
        raise ValueError(f"timing includes or does not attest diagnostic isolation: {path}")
    if payload.get("data", {}).get("coverage", {}).get("record_count") != (
        30 if dataset == "tnbc" else 37
    ):
        raise ValueError(f"timing dataset/coverage mismatch: {path}")
    return float(payload["timed"]["seconds_per_optimizer_update"]), payload


def coverage_seconds(path: Path, dataset: str) -> float:
    payload = read_json(path)
    if payload.get("status") != "complete" or payload.get("dataset") != dataset:
        raise ValueError(f"coverage manifest mismatch: {path}")
    return float(payload["runtime"]["wall_seconds"])


def main() -> int:
    parser = argparse.ArgumentParser()
    for dataset in PLANS:
        parser.add_argument(f"--{dataset}-c0", required=True, type=Path)
        parser.add_argument(f"--{dataset}-c1", required=True, type=Path)
        parser.add_argument(f"--{dataset}-coverage", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f"refusing to overwrite budget report: {args.output}")

    result = {}
    for dataset, plan in PLANS.items():
        c0_seconds, c0_payload = timing_seconds(
            getattr(args, f"{dataset}_c0"), dataset, "c0"
        )
        c1_seconds, c1_payload = timing_seconds(
            getattr(args, f"{dataset}_c1"), dataset, "c1"
        )
        shared_coverage_seconds = coverage_seconds(
            getattr(args, f"{dataset}_coverage"), dataset
        )
        arms = {}
        for arm, seconds in (("c0", c0_seconds), ("c1", c1_seconds)):
            arms[arm] = {
                "measured_seconds_per_optimizer_update": seconds,
                "five_epoch_train_gpu_hours": plan["screen_updates"] * seconds / 3600.0,
                "ten_epoch_train_gpu_hours": plan["full_updates"] * seconds / 3600.0,
            }
        result[dataset] = {
            **plan,
            "shared_initial_coverage_gpu_hours": shared_coverage_seconds / 3600.0,
            "arms": arms,
            "C1_over_C0_seconds_per_update_ratio": c1_seconds / c0_seconds,
            "both_arms_five_epoch_gpu_hours_including_one_shared_coverage": (
                shared_coverage_seconds
                + plan["screen_updates"] * (c0_seconds + c1_seconds)
            )
            / 3600.0,
            "both_arms_ten_epoch_gpu_hours_including_one_shared_coverage": (
                shared_coverage_seconds
                + plan["full_updates"] * (c0_seconds + c1_seconds)
            )
            / 3600.0,
            "identity": {
                "C0_checkpoint_sha256": c0_payload["initialization"]["checkpoint_sha256"],
                "C1_checkpoint_sha256": c1_payload["initialization"]["checkpoint_sha256"],
                "C0_coverage_manifest_sha256": c0_payload["data"]["coverage"]["sha256"],
                "C1_coverage_manifest_sha256": c1_payload["data"]["coverage"]["sha256"],
            },
        }

    payload = {
        "schema_version": 1,
        "phase": "2A-warmstart-feasibility",
        "status": "estimate_only_formal_5epoch_not_authorized",
        "datasets": result,
        "scope": (
            "GPU-resident train updates plus one shared initial train-only coverage refresh. "
            "Development evaluation, checkpoint I/O, and failure recovery are excluded."
        ),
        "scratch_200_epoch_gate_reused": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_name(args.output.name + ".tmp")
    temp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, args.output)
    print(json.dumps({"status": payload["status"], "output": str(args.output.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
