"""Pure budget accounting for Phase 2A clean baselines."""

from __future__ import annotations

from typing import Any


def estimate_dataset_budget(
    recipe: dict[str, Any],
    dataset: str,
    base_timing: dict[str, Any],
    active_timing: dict[str, Any],
) -> dict[str, Any]:
    if base_timing.get("status") != "complete" or active_timing.get("status") != "complete":
        raise ValueError("both Phase 2A timing profiles must be complete")
    if base_timing.get("profile") != "base" or active_timing.get("profile") != "pms_active":
        raise ValueError("expected base and pms_active timing profiles")
    for field in ("manifest_sha256", "protocol_id"):
        if base_timing["data"].get(field) != active_timing["data"].get(field):
            raise ValueError(f"timing profiles differ in data {field}")
    if (
        base_timing["initialization"]["checkpoint_sha256"]
        != active_timing["initialization"]["checkpoint_sha256"]
    ):
        raise ValueError("timing profiles use different initialization checkpoints")

    optimization = recipe["optimization"]
    pms = recipe["stainpms"]
    dataset_recipe = recipe["datasets"][dataset]
    epochs = int(optimization["epochs"])
    start_epoch = int(pms["start_epoch"])
    updates_per_epoch = int(dataset_recipe["optimizer_updates_per_epoch"])
    base_updates = start_epoch * updates_per_epoch
    active_updates = (epochs - start_epoch) * updates_per_epoch
    if base_updates + active_updates != int(dataset_recipe["planned_optimizer_updates"]):
        raise ValueError("recipe optimizer-update accounting is inconsistent")

    base_seconds_per_update = float(base_timing["timed"]["seconds_per_optimizer_update"])
    active_seconds_per_update = float(active_timing["timed"]["seconds_per_optimizer_update"])
    refresh_count = int(pms["expected_refresh_count"])
    refresh_seconds_each = float(active_timing["coverage_refresh"]["wall_seconds"])
    checkpoint_count = int(dataset_recipe["checkpoint_count"])
    eval_seconds_each = float(dataset_recipe.get("evaluation_seconds_per_checkpoint_proxy", 0.0))
    components = {
        "base_objective_train_seconds": base_updates * base_seconds_per_update,
        "pms_active_train_seconds": active_updates * active_seconds_per_update,
        "coverage_refresh_seconds": refresh_count * refresh_seconds_each,
        "checkpoint_evaluation_seconds": checkpoint_count * eval_seconds_each,
    }
    total_seconds = sum(components.values())
    limit_hours = float(recipe["timing"]["single_dataset_stop_gpu_hours"])
    return {
        "dataset": dataset,
        "status": "gate_pass" if total_seconds / 3600 <= limit_hours else "gate_stop",
        "planned": {
            "epochs": epochs,
            "base_optimizer_updates": base_updates,
            "pms_active_optimizer_updates": active_updates,
            "total_optimizer_updates": base_updates + active_updates,
            "actual_crops_target": int(dataset_recipe["planned_optimizer_updates"])
            * int(optimization["crop_batch_size"]),
            "coverage_refresh_count": refresh_count,
            "checkpoint_count": checkpoint_count,
        },
        "measured": {
            "base_seconds_per_optimizer_update": base_seconds_per_update,
            "pms_active_seconds_per_optimizer_update": active_seconds_per_update,
            "coverage_refresh_seconds_each": refresh_seconds_each,
            "evaluation_seconds_per_checkpoint_proxy": eval_seconds_each,
            "base_peak_memory_allocated_mib": base_timing["timed"][
                "peak_memory_allocated_mib"
            ],
            "pms_active_peak_memory_allocated_mib": active_timing["timed"][
                "peak_memory_allocated_mib"
            ],
        },
        "estimated_components_seconds": components,
        "estimated_total_seconds": total_seconds,
        "estimated_total_gpu_hours": total_seconds / 3600,
        "single_dataset_stop_gpu_hours": limit_hours,
        "gate_basis": "10 warm-up + 100 CUDA-synchronized updates for each objective profile",
    }
