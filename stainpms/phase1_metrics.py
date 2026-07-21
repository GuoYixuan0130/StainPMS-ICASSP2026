"""Pure, CPU-testable accounting for the Phase 1 candidate-failure diagnosis.

The module owns no model state and never reads images. The GPU diagnostic runner
supplies candidate IoUs and final instance maps; this module freezes the
definitions used for coverage, stage loss, and mutually exclusive GT errors.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

import numpy as np
from scipy import ndimage as ndi

from stainpms.evaluator import evaluate_instance_pair


ERROR_CLASSES = (
    "final_matched_tp",
    "point_miss",
    "candidate_generation_miss",
    "selection_ranking_miss",
    "assembly_nms_conflict_miss",
)


def instance_ids(inst_map: np.ndarray) -> list[int]:
    return [int(value) for value in np.unique(inst_map) if int(value) != 0]


def choose_edt_interior_points(inst_map: np.ndarray) -> dict[int, tuple[int, int]]:
    """Return deterministic (x, y) EDT-max interior points for every instance."""

    output: dict[int, tuple[int, int]] = {}
    for instance_id in instance_ids(inst_map):
        mask = np.asarray(inst_map) == instance_id
        distances = ndi.distance_transform_edt(mask)
        peak = float(distances.max())
        if peak <= 0:
            raise ValueError(f"instance {instance_id} has no positive pixels")
        ys, xs = np.where(distances == peak)
        # np.where is row-major; the first element is smallest y then x.
        output[instance_id] = (int(xs[0]), int(ys[0]))
    return output


def mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=bool)
    right = np.asarray(right, dtype=bool)
    union = int(np.logical_or(left, right).sum())
    if union == 0:
        return 0.0
    return float(np.logical_and(left, right).sum() / union)


def iou_against_label(mask: np.ndarray, inst_map: np.ndarray, instance_id: int) -> float:
    return mask_iou(mask, np.asarray(inst_map) == int(instance_id))


def max_iou_with_final_prediction(gt_mask: np.ndarray, pred_map: np.ndarray) -> tuple[float, int | None]:
    best_iou = 0.0
    best_id: int | None = None
    for pred_id in instance_ids(pred_map):
        score = mask_iou(gt_mask, np.asarray(pred_map) == pred_id)
        if score > best_iou or (score == best_iou and best_id is not None and pred_id < best_id):
            best_iou = score
            best_id = pred_id
    return best_iou, best_id


def strict_final_pairing(gt_map: np.ndarray, pred_map: np.ndarray, match_iou: float) -> dict[str, Any]:
    """Return the existing strict evaluator record plus unambiguous GT matches."""

    record = evaluate_instance_pair(gt_map, pred_map, mode="strict", match_iou=match_iou)
    pairing = record["pairing"] or {}
    pairs = {
        int(gt_id): int(pred_id)
        for gt_id, pred_id in zip(pairing.get("paired_true", []), pairing.get("paired_pred", []), strict=True)
    }
    return {"evaluator": record, "pairs": pairs}


def classify_gt_error(
    *,
    point_count: int,
    best_candidate_iou: float | None,
    selected_candidate_iou: float | None,
    final_matched: bool,
    match_iou: float,
) -> str:
    """Return exactly one owner-approved Phase 1 causal class."""

    if final_matched:
        return "final_matched_tp"
    if point_count <= 0:
        return "point_miss"
    best = float(best_candidate_iou or 0.0)
    selected = float(selected_candidate_iou or 0.0)
    if best < match_iou:
        return "candidate_generation_miss"
    if selected < match_iou:
        return "selection_ranking_miss"
    return "assembly_nms_conflict_miss"


def attach_gt_error_classes(rows: Iterable[dict[str, Any]], match_iou: float) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for raw in rows:
        row = dict(raw)
        row["error_class"] = classify_gt_error(
            point_count=int(row.get("auto_point_count", 0)),
            best_candidate_iou=row.get("auto_best_candidate_iou"),
            selected_candidate_iou=row.get("auto_selected_candidate_iou"),
            final_matched=bool(row.get("final_matched", False)),
            match_iou=match_iou,
        )
        row["selection_regret"] = (
            None
            if row.get("auto_best_candidate_iou") is None or row.get("auto_selected_candidate_iou") is None
            else float(row["auto_best_candidate_iou"] - row["auto_selected_candidate_iou"])
        )
        output.append(row)
    return output


def ccr_fraction(values: Iterable[float | None], threshold: float, denominator: int | None = None) -> dict[str, float | int | None]:
    numeric = [float(value) for value in values if value is not None]
    denom = len(numeric) if denominator is None else int(denominator)
    hits = sum(value >= threshold for value in numeric)
    return {
        "threshold": float(threshold),
        "numerator": int(hits),
        "denominator": denom,
        "value": float(hits / denom) if denom else None,
    }


def summarize_gt_rows(rows: Iterable[dict[str, Any]], *, thresholds: Iterable[float], match_iou: float) -> dict[str, Any]:
    rows = list(rows)
    gt_point_values = [row.get("gt_point_best_candidate_iou") for row in rows]
    auto_values = [row.get("auto_best_candidate_iou") for row in rows]
    recalled_rows = [row for row in rows if int(row.get("auto_point_count", 0)) > 0]
    errors = Counter(row.get("error_class") for row in rows)
    best_values = [float(value) for value in auto_values if value is not None]
    selected_values = [float(row["auto_selected_candidate_iou"]) for row in rows if row.get("auto_selected_candidate_iou") is not None]
    regret_values = [float(row["selection_regret"]) for row in rows if row.get("selection_regret") is not None]
    qualified = [row for row in rows if (row.get("auto_best_candidate_iou") or 0.0) >= match_iou]
    return {
        "gt_instance_count": len(rows),
        "auto_point_recall": {
            "numerator": len(recalled_rows),
            "denominator": len(rows),
            "value": float(len(recalled_rows) / len(rows)) if rows else None,
        },
        "ccr_gt_point": [ccr_fraction(gt_point_values, threshold) for threshold in thresholds],
        "ccr_auto_given_point": [
            ccr_fraction([row.get("auto_best_candidate_iou") for row in recalled_rows], threshold)
            for threshold in thresholds
        ],
        "ccr_auto_e2e": [ccr_fraction(auto_values, threshold, denominator=len(rows)) for threshold in thresholds],
        "candidate_iou": {
            "best_mean": float(np.mean(best_values)) if best_values else None,
            "selected_standard_candidate_mean": float(np.mean(selected_values)) if selected_values else None,
            "selection_regret_mean": float(np.mean(regret_values)) if regret_values else None,
            "qualified_candidate_count": len(qualified),
            "qualified_but_not_final_count": sum(not bool(row.get("final_matched")) for row in qualified),
            "qualified_but_not_final_fraction": (
                float(sum(not bool(row.get("final_matched")) for row in qualified) / len(qualified))
                if qualified
                else None
            ),
        },
        "error_classes": {name: int(errors.get(name, 0)) for name in ERROR_CLASSES},
    }


def structural_errors(gt_map: np.ndarray, pred_map: np.ndarray, match_iou: float) -> dict[str, Any]:
    """Supplementary final-map FP/FN, split, merge and boundary accounting."""

    pairing_info = strict_final_pairing(gt_map, pred_map, match_iou)
    pairing = pairing_info["evaluator"]["pairing"] or {}
    unmatched_gt = [int(value) for value in pairing.get("unpaired_true", [])]
    unmatched_pred = [int(value) for value in pairing.get("unpaired_pred", [])]
    boundary = 0
    split = 0
    for gt_id in unmatched_gt:
        gt_mask = np.asarray(gt_map) == gt_id
        max_iou, _ = max_iou_with_final_prediction(gt_mask, pred_map)
        boundary += int(0.0 < max_iou < match_iou)
        overlaps = sum(bool(np.logical_and(gt_mask, np.asarray(pred_map) == pred_id).any()) for pred_id in instance_ids(pred_map))
        split += int(overlaps >= 2)
    merge = 0
    for pred_id in unmatched_pred:
        pred_mask = np.asarray(pred_map) == pred_id
        overlaps = sum(bool(np.logical_and(pred_mask, np.asarray(gt_map) == gt_id).any()) for gt_id in instance_ids(gt_map))
        merge += int(overlaps >= 2)
    return {
        "tp": int(pairing.get("tp", 0)),
        "fp": int(pairing.get("fp", 0)),
        "fn": int(pairing.get("fn", 0)),
        "split_unmatched_gt_count": split,
        "merge_unmatched_pred_count": merge,
        "boundary_localization_unmatched_gt_count": boundary,
        "strict_evaluator": pairing_info["evaluator"],
    }
