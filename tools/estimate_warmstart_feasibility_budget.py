"""Estimate C0 warm-start cost from synchronized PMS-active timing reports.

C1 remains explicitly pending until its approved loss is implemented and timed
with its own 10 warm-up + 100 synchronized optimizer updates.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"expected DATASET=PATH, received {value!r}")
    dataset, raw_path = value.split("=", 1)
    return dataset, Path(raw_path).expanduser().resolve()


def estimate_c0_stages(dataset_spec: dict, timing: dict) -> dict:
    """Return the provisional C0 cost for the fixed five/ten-epoch stages."""
    if timing.get("status") != "complete" or timing.get("profile") != "pms_active":
        raise ValueError("a complete pms_active timing report is required")
    if int(timing["data"]["record_count"]) != int(dataset_spec["train_images"]):
        raise ValueError("timing record count mismatch")
    seconds_per_update = float(timing["timed"]["seconds_per_optimizer_update"])
    refresh_seconds = float(timing["coverage_refresh"]["wall_seconds"])
    stages = {}
    for stage in ("screen", "full"):
        updates = int(dataset_spec[f"{stage}_planned_updates"])
        train_seconds = updates * seconds_per_update
        total_seconds = train_seconds + refresh_seconds
        stages[stage] = {
            "epochs": int(dataset_spec[f"{stage}_epochs"]),
            "optimizer_updates": updates,
            "C0_train_seconds": train_seconds,
            "initial_coverage_refresh_seconds": refresh_seconds,
            "C0_total_seconds": total_seconds,
            "C0_estimated_gpu_hours": total_seconds / 3600.0,
            "C1_estimated_gpu_hours": None,
            "C1_status": "pending_owner_approval_implementation_and_dedicated_timing",
        }
    return {
        "seconds_per_optimizer_update": seconds_per_update,
        "initial_coverage_refresh_seconds": refresh_seconds,
        "stages": stages,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal", required=True)
    parser.add_argument("--active-timing", action="append", required=True, help="DATASET=PATH")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    proposal_path = Path(args.proposal).resolve()
    proposal = json.loads(proposal_path.read_text(encoding="utf-8"))
    timing_paths = dict(parse_named_path(value) for value in args.active_timing)
    if set(timing_paths) != set(proposal["datasets"]):
        raise ValueError("timing datasets must exactly match proposal datasets")

    estimates = {}
    for dataset, dataset_spec in proposal["datasets"].items():
        timing = json.loads(timing_paths[dataset].read_text(encoding="utf-8"))
        try:
            estimate = estimate_c0_stages(dataset_spec, timing)
        except ValueError as error:
            raise ValueError(f"{dataset}: {error}") from error
        estimates[dataset] = {
            "timing_path": str(timing_paths[dataset]),
            "timing_repository": timing.get("repository"),
            "proxy_limitation": "measured with the PMS-active objective before warm-start C0/C1 implementation; must be replaced by arm-specific timing",
            **estimate,
        }

    report = {
        "schema_version": 1,
        "phase": "2A-warmstart-feasibility",
        "status": "C0_proxy_complete_C1_pending",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "proposal_path": str(proposal_path),
        "old_200_epoch_gate_used": False,
        "estimates": estimates,
        "formal_timing_requirement": {
            "arms": ["C0", "C1"],
            "warmup_optimizer_updates": 10,
            "timed_optimizer_updates": 100,
            "cuda_synchronized": True
        }
    }
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise ValueError(f"refusing to overwrite existing report: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
