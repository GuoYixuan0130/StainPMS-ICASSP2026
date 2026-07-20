"""Versioned strict and legacy instance-segmentation evaluation policies.

Metric formulae and the 0.5 matching threshold remain delegated to the existing
StainPMS/CA-SAM2 implementation.  This module changes only explicit empty-case
handling, records inclusion decisions, and emits auditable per-image outputs.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import Any, Iterable

import numpy as np


EVALUATOR_POLICY_IDS = {
    "strict": "strict_empty_handling_v1",
    "legacy_skip": "legacy_skip_empty_handling_v1",
}


def _load_stats_utils():
    """Load the tracked metric file without importing SAM2/Hydra package setup."""

    path = Path(__file__).resolve().parents[1] / "sam2_train" / "modeling" / "stats_utils.py"
    spec = importlib.util.spec_from_file_location("stainpms_strict_stats_utils", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load evaluator metrics from {path}")
    module = importlib.util.module_from_spec(spec)
    inserted_cv2_stub = False
    try:
        import cv2  # noqa: F401
    except ModuleNotFoundError:
        sys.modules["cv2"] = types.ModuleType("cv2")
        inserted_cv2_stub = True
    try:
        spec.loader.exec_module(module)
    finally:
        if inserted_cv2_stub:
            sys.modules.pop("cv2", None)
    return module


_STATS = _load_stats_utils()
get_dice_1 = _STATS.get_dice_1
get_fast_aji = _STATS.get_fast_aji
get_fast_aji_plus = _STATS.get_fast_aji_plus
get_fast_dice_2 = _STATS.get_fast_dice_2
get_fast_pq = _STATS.get_fast_pq
remap_label = _STATS.remap_label


EVALUATOR_MODES = {"strict", "legacy_skip"}
METRIC_NAMES = ("dice1", "dice2", "aji", "aji_p", "dq", "sq", "pq")


def _validate_map(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value)
    array = np.squeeze(array)
    if array.ndim != 2:
        raise ValueError(f"{name} must be 2-D after squeeze, got {array.shape}")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must be numeric, got {array.dtype}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    if (array < 0).any() or not np.equal(array, np.floor(array)).all():
        raise ValueError(f"{name} must contain non-negative integer instance IDs")
    return array.astype(np.int32, copy=False)


def _zero_metrics() -> dict[str, float]:
    return {name: 0.0 for name in METRIC_NAMES}


def _safe_metric(value: float) -> float:
    numeric = float(value)
    return numeric if np.isfinite(numeric) else 0.0


def _safe_fast_dice_2(gt: np.ndarray, pred: np.ndarray) -> float:
    """Map the legacy formula's zero denominator for disjoint maps to Dice 0."""

    try:
        return _safe_metric(get_fast_dice_2(gt, pred))
    except ZeroDivisionError:
        return 0.0


def evaluate_instance_pair(
    gt_map: np.ndarray,
    pred_map: np.ndarray,
    *,
    mode: str = "strict",
    match_iou: float = 0.5,
    sample_id: str | None = None,
) -> dict[str, Any]:
    """Evaluate one complete image under an explicit inclusion policy."""

    if mode not in EVALUATOR_MODES:
        raise ValueError(f"unknown evaluator mode {mode!r}; choose {sorted(EVALUATOR_MODES)}")
    gt_raw = _validate_map(gt_map, "gt_map")
    pred_raw = _validate_map(pred_map, "pred_map")
    if gt_raw.shape != pred_raw.shape:
        raise ValueError(f"GT/pred shape mismatch: {gt_raw.shape} vs {pred_raw.shape}")
    gt_count = int(np.count_nonzero(np.unique(gt_raw)))
    pred_count = int(np.count_nonzero(np.unique(pred_raw)))
    empty_gt = gt_count == 0
    empty_pred = pred_count == 0
    both_empty = empty_gt and empty_pred
    base: dict[str, Any] = {
        "sample_id": sample_id,
        "mode": mode,
        "evaluator_policy_id": EVALUATOR_POLICY_IDS[mode],
        "match_iou": float(match_iou),
        "shape": list(gt_raw.shape),
        "gt_instance_count": gt_count,
        "pred_instance_count": pred_count,
        "empty_gt": empty_gt,
        "empty_prediction": empty_pred,
        "both_empty": both_empty,
    }

    if mode == "legacy_skip" and (empty_gt or empty_pred):
        base.update(
            {
                "included_in_macro": False,
                "skip_reason": "legacy_skip_empty_gt_or_prediction",
                "no_match": True,
                "metrics": {name: None for name in METRIC_NAMES},
                "pairing": None,
            }
        )
        return base

    if both_empty:
        base.update(
            {
                "included_in_macro": False,
                "skip_reason": "strict_both_empty_excluded_from_benchmark_macro",
                "no_match": True,
                "metrics": {name: None for name in METRIC_NAMES},
                "pairing": {
                    "tp": 0,
                    "fp": 0,
                    "fn": 0,
                    "paired_true": [],
                    "paired_pred": [],
                    "unpaired_true": [],
                    "unpaired_pred": [],
                },
            }
        )
        return base

    if empty_gt or empty_pred:
        pairing = {
            "tp": 0,
            "fp": pred_count,
            "fn": gt_count,
            "paired_true": [],
            "paired_pred": [],
            "unpaired_true": list(range(1, gt_count + 1)),
            "unpaired_pred": list(range(1, pred_count + 1)),
        }
        base.update(
            {
                "included_in_macro": True,
                "skip_reason": None,
                "no_match": True,
                "metrics": _zero_metrics(),
                "pairing": pairing,
            }
        )
        return base

    gt = remap_label(gt_raw)
    pred = remap_label(pred_raw)
    (dq, sq, pq), pair_info = get_fast_pq(gt, pred, match_iou=match_iou)
    paired_true, paired_pred, unpaired_true, unpaired_pred = pair_info
    metrics = {
        "dice1": _safe_metric(get_dice_1(gt, pred)),
        "dice2": _safe_fast_dice_2(gt, pred),
        "aji": _safe_metric(get_fast_aji(gt, pred)),
        "aji_p": _safe_metric(get_fast_aji_plus(gt, pred)),
        "dq": _safe_metric(dq),
        "sq": _safe_metric(sq),
        "pq": _safe_metric(pq),
    }
    pairing = {
        "tp": len(paired_true),
        "fp": len(unpaired_pred),
        "fn": len(unpaired_true),
        "paired_true": [int(value) for value in paired_true],
        "paired_pred": [int(value) for value in paired_pred],
        "unpaired_true": [int(value) for value in unpaired_true],
        "unpaired_pred": [int(value) for value in unpaired_pred],
    }
    base.update(
        {
            "included_in_macro": True,
            "skip_reason": None,
            "no_match": pairing["tp"] == 0,
            "metrics": metrics,
            "pairing": pairing,
        }
    )
    return base


def aggregate_image_metrics(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(records)
    included = [row for row in rows if row.get("included_in_macro")]
    summary_metrics: dict[str, float | None] = {}
    for name in METRIC_NAMES:
        values = [row["metrics"][name] for row in included]
        numeric = [float(value) for value in values if value is not None]
        summary_metrics[name] = float(np.mean(numeric)) if numeric else None
    return {
        "image_count": len(rows),
        "included_image_count": len(included),
        "excluded_image_count": len(rows) - len(included),
        "empty_gt_count": sum(bool(row.get("empty_gt")) for row in rows),
        "empty_prediction_count": sum(bool(row.get("empty_prediction")) for row in rows),
        "both_empty_count": sum(bool(row.get("both_empty")) for row in rows),
        "no_match_count": sum(bool(row.get("no_match")) for row in rows),
        "metrics_macro": summary_metrics,
    }


def write_evaluation_outputs(
    records: Iterable[dict[str, Any]],
    output_dir: str | Path,
    *,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rows = list(records)
    summary = aggregate_image_metrics(rows)
    payload = {
        "schema_version": 1,
        "context": context or {},
        "summary": summary,
        "images": rows,
    }
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    with (root / "metrics_per_image.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    with (root / "metrics_summary.json").open("w", encoding="utf-8") as handle:
        json.dump({"schema_version": 1, "context": context or {}, "summary": summary}, handle, indent=2)
        handle.write("\n")
    fieldnames = [
        "sample_id",
        "mode",
        "included_in_macro",
        "skip_reason",
        "gt_instance_count",
        "pred_instance_count",
        "empty_gt",
        "empty_prediction",
        "both_empty",
        "no_match",
        *METRIC_NAMES,
    ]
    with (root / "metrics_per_image.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            flat = {key: row.get(key) for key in fieldnames if key not in METRIC_NAMES}
            flat.update(row.get("metrics") or {})
            writer.writerow(flat)
    return payload
