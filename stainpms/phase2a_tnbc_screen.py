"""Fixed reporting and decision logic for the approved TNBC C0/C1 screen."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


TASK_METRICS = ("dice1", "dice2", "aji", "dq", "sq", "pq")
MECHANISM_METRICS = (
    "best_candidate_iou",
    "best_candidate_ccr_at_0_5",
    "selected_candidate_iou",
    "selected_candidate_ccr_at_0_5",
    "selection_regret",
)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _threshold_value(items: list[dict[str, Any]], threshold: float) -> float:
    matches = [item for item in items if float(item["threshold"]) == float(threshold)]
    if len(matches) != 1 or matches[0].get("value") is None:
        raise ValueError(f"expected exactly one finite CCR value at threshold {threshold}")
    return float(matches[0]["value"])


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty value list")
    return float(np.mean(values))


def patient_report(summary: dict[str, Any], images: list[dict[str, Any]], patient: int) -> dict[str, Any]:
    grouped = [record for record in images if int(record.get("patient", -1)) == patient]
    if not grouped:
        raise ValueError(f"diagnosis has no images for TNBC patient {patient}")
    task_records = []
    for record in grouped:
        strict = record.get("final_task_metrics")
        if not isinstance(strict, dict):
            raise ValueError(
                "formal screen diagnosis must use --include-final-task-metrics "
                f"(missing for {record.get('sample_id')})"
            )
        if not strict.get("included_in_macro"):
            raise ValueError(f"development image unexpectedly excluded from strict macro: {record.get('sample_id')}")
        metrics = strict.get("metrics")
        if not isinstance(metrics, dict) or any(metrics.get(name) is None for name in TASK_METRICS):
            raise ValueError(f"missing strict task metric for {record.get('sample_id')}")
        task_records.append(metrics)
    group = summary.get("groups", {}).get(f"patient:{patient}")
    if not isinstance(group, dict):
        raise ValueError(f"summary has no mechanism group for TNBC patient {patient}")
    candidate = group.get("candidate_iou", {})
    selected_ccr = _threshold_value(group["ccr_auto_e2e"], 0.5)
    # The common decoder's native quality-head selection is the selected
    # candidate.  The end-to-end automatic point denominator is retained for
    # both best and selected coverage so point misses cannot be hidden.
    selected_iou = candidate.get("selected_standard_candidate_mean")
    best_iou = candidate.get("best_mean")
    regret = candidate.get("selection_regret_mean")
    if any(value is None for value in (best_iou, selected_iou, regret)):
        raise ValueError(f"incomplete candidate statistics for TNBC patient {patient}")
    # Phase 1 only stores the selected candidate IoU; CCR at 0.5 for the
    # selected mask is recomputed directly from per-GT rows by the caller.
    return {
        "patient": int(patient),
        "image_count": len(grouped),
        "gt_instance_count": int(group.get("gt_instance_count", 0)),
        "task_metrics_image_macro": {name: _mean([float(row[name]) for row in task_records]) for name in TASK_METRICS},
        "mechanism": {
            "best_candidate_iou": float(best_iou),
            "best_candidate_ccr_at_0_5": selected_ccr,
            "selected_candidate_iou": float(selected_iou),
            "selection_regret": float(regret),
        },
    }


def selected_candidate_ccr_at_0_5(gt_rows: list[dict[str, Any]], patient: int) -> float:
    rows = [row for row in gt_rows if int(row.get("patient", -1)) == patient]
    if not rows:
        raise ValueError(f"no GT rows for TNBC patient {patient}")
    values = [row.get("auto_selected_candidate_iou") for row in rows]
    return float(sum(value is not None and float(value) >= 0.5 for value in values) / len(rows))


def build_epoch_record(
    *,
    arm: str,
    epoch: int,
    summary: dict[str, Any],
    images: list[dict[str, Any]],
    gt_rows: list[dict[str, Any]],
    source_dir: Path,
) -> dict[str, Any]:
    patients = {}
    for patient in (7, 8):
        value = patient_report(summary, images, patient)
        value["mechanism"]["selected_candidate_ccr_at_0_5"] = selected_candidate_ccr_at_0_5(gt_rows, patient)
        patients[str(patient)] = value
    task_macro = {
        name: _mean([patients[str(patient)]["task_metrics_image_macro"][name] for patient in (7, 8)])
        for name in TASK_METRICS
    }
    mechanism_macro = {
        name: _mean([patients[str(patient)]["mechanism"][name] for patient in (7, 8)])
        for name in MECHANISM_METRICS
    }
    return {
        "arm": arm,
        "epoch": int(epoch),
        "source_dir": str(source_dir),
        "patients": patients,
        "patient_macro": {
            "aggregation": "equal_weight_mean_of_patient_7_and_patient_8",
            "task_metrics_image_macro": task_macro,
            "mechanism": mechanism_macro,
        },
    }


def metric_deltas(c0: dict[str, Any], c1: dict[str, Any]) -> dict[str, Any]:
    if c0["epoch"] != c1["epoch"]:
        raise ValueError("C0/C1 epoch mismatch")
    result: dict[str, Any] = {"epoch": c0["epoch"], "direction": "C1_minus_C0", "patients": {}, "patient_macro": {}}
    for key in ("7", "8"):
        result["patients"][key] = {
            "task_metrics_image_macro": {
                name: c1["patients"][key]["task_metrics_image_macro"][name] - c0["patients"][key]["task_metrics_image_macro"][name]
                for name in TASK_METRICS
            },
            "mechanism": {
                name: c1["patients"][key]["mechanism"][name] - c0["patients"][key]["mechanism"][name]
                for name in MECHANISM_METRICS
            },
        }
    result["patient_macro"] = {
        "task_metrics_image_macro": {
            name: c1["patient_macro"]["task_metrics_image_macro"][name] - c0["patient_macro"]["task_metrics_image_macro"][name]
            for name in TASK_METRICS
        },
        "mechanism": {
            name: c1["patient_macro"]["mechanism"][name] - c0["patient_macro"]["mechanism"][name]
            for name in MECHANISM_METRICS
        },
    }
    return result


def assess_epoch5(c0: dict[str, Any], c1: dict[str, Any]) -> dict[str, Any]:
    """Apply the owner-frozen decision rule only to epoch five."""
    if c0["epoch"] != 5 or c1["epoch"] != 5:
        raise ValueError("the formal promotion decision is defined only for epoch 5")
    delta = metric_deltas(c0, c1)
    ccr = "best_candidate_ccr_at_0_5"
    patient_non_decrease = {
        key: c1["patients"][key]["mechanism"][ccr] >= c0["patients"][key]["mechanism"][ccr]
        for key in ("7", "8")
    }
    patient_strict_increase = {
        key: c1["patients"][key]["mechanism"][ccr] > c0["patients"][key]["mechanism"][ccr]
        for key in ("7", "8")
    }
    macro_ccr_delta = delta["patient_macro"]["mechanism"][ccr]
    aji_delta = delta["patient_macro"]["task_metrics_image_macro"]["aji"]
    pq_delta = delta["patient_macro"]["task_metrics_image_macro"]["pq"]
    checks = {
        "best_ccr_patient_7_non_decrease": patient_non_decrease["7"],
        "best_ccr_patient_8_non_decrease": patient_non_decrease["8"],
        "best_ccr_at_least_one_patient_strict_increase": any(patient_strict_increase.values()),
        "best_ccr_patient_macro_strict_increase": macro_ccr_delta > 0.0,
        "patient_macro_aji_delta_at_least_0_005": aji_delta >= 0.005,
        "patient_macro_pq_delta_at_least_minus_0_002": pq_delta >= -0.002,
    }
    return {
        "primary_epoch": 5,
        "decision": "pass_all_promotion_rules" if all(checks.values()) else "do_not_auto_promote",
        "checks": checks,
        "epoch5_c1_minus_c0": delta,
        "thresholds": {"aji_delta": 0.005, "pq_delta": -0.002, "candidate_ccr_threshold": 0.5},
        "interpretation_boundary": (
            "Single-seed exploratory warm-start evidence only. A pass is a stable "
            "exploratory signal, not final validation."
        ),
    }
