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
from scipy.optimize import linear_sum_assignment


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


def final_instance_overlap_table(gt_map: np.ndarray, pred_map: np.ndarray) -> dict[str, Any]:
    """Build exact GT/pred intersection counts with one image-wide pass.

    The Phase 1 runner used to materialize a complete boolean prediction map
    for every GT instance, then scan all of them one at a time.  For dense
    MoNuSeg images that is O(|G|*|P|*H*W).  This table has the identical
    intersections and areas, but is built once in O(H*W) plus a compact label
    table.  Input labels are the frozen StainPMS prepared maps, whose IDs are
    non-negative and continuous; the implementation nevertheless tolerates
    unused IDs.
    """

    gt = np.asarray(gt_map, dtype=np.int64)
    pred = np.asarray(pred_map, dtype=np.int64)
    if gt.shape != pred.shape or gt.ndim != 2:
        raise ValueError(f"expected same-shape 2-D maps, received {gt.shape} and {pred.shape}")
    if (gt < 0).any() or (pred < 0).any():
        raise ValueError("instance maps must have non-negative IDs")
    max_gt = int(gt.max()) if gt.size else 0
    max_pred = int(pred.max()) if pred.size else 0
    width = max_pred + 1
    encoded = gt.reshape(-1) * width + pred.reshape(-1)
    table = np.bincount(encoded, minlength=(max_gt + 1) * width).reshape(max_gt + 1, width)
    gt_ids = np.flatnonzero(table.sum(axis=1))
    gt_ids = gt_ids[gt_ids != 0].astype(np.int64, copy=False)
    pred_ids = np.flatnonzero(table.sum(axis=0))
    pred_ids = pred_ids[pred_ids != 0].astype(np.int64, copy=False)
    return {
        "intersections": table,
        "gt_areas": table.sum(axis=1),
        "pred_areas": table.sum(axis=0),
        "gt_ids": gt_ids,
        "pred_ids": pred_ids,
    }


def final_max_iou_by_gt(overlap: dict[str, Any]) -> dict[int, tuple[float, int | None]]:
    """Return the exact best final-prediction IoU for every GT instance."""

    table = np.asarray(overlap["intersections"])
    gt_areas = np.asarray(overlap["gt_areas"])
    pred_areas = np.asarray(overlap["pred_areas"])
    pred_ids = np.asarray(overlap["pred_ids"], dtype=np.int64)
    output: dict[int, tuple[float, int | None]] = {}
    for raw_gt_id in np.asarray(overlap["gt_ids"], dtype=np.int64):
        gt_id = int(raw_gt_id)
        if pred_ids.size == 0:
            output[gt_id] = (0.0, None)
            continue
        intersections = table[gt_id, pred_ids].astype(np.float64, copy=False)
        unions = float(gt_areas[gt_id]) + pred_areas[pred_ids] - intersections
        ious = np.divide(intersections, unions, out=np.zeros_like(intersections), where=unions > 0)
        best_index = int(np.argmax(ious))
        best_iou = float(ious[best_index])
        # Match max_iou_with_final_prediction: no overlapping prediction has
        # a score of zero and a None prediction ID.
        output[gt_id] = (best_iou, int(pred_ids[best_index]) if best_iou > 0.0 else None)
    return output


def strict_final_pairing(gt_map: np.ndarray, pred_map: np.ndarray, match_iou: float) -> dict[str, Any]:
    """Compute the strict evaluator's pairing rule without unrelated dense metrics.

    ``evaluate_instance_pair`` also calculates Dice/AJI/AJI+ for every image.
    Its legacy implementations repeatedly materialise every GT/pred mask and
    are not tractable for Phase 1's dense MoNuSeg bookkeeping.  Phase 1 needs
    only the frozen strict matching definition (TP/FP/FN and pair identity),
    so this is an exact vectorised implementation of ``get_fast_pq`` at the
    configured matching threshold.  It preserves its special inclusive rule
    when any IoU is exactly the threshold.
    """

    if not 0.0 <= float(match_iou) <= 1.0:
        raise ValueError("match_iou must lie in [0, 1]")
    overlap = final_instance_overlap_table(gt_map, pred_map)
    table = np.asarray(overlap["intersections"], dtype=np.float64)
    gt_ids = np.asarray(overlap["gt_ids"], dtype=np.int64)
    pred_ids = np.asarray(overlap["pred_ids"], dtype=np.int64)
    gt_count = int(gt_ids.size)
    pred_count = int(pred_ids.size)
    base: dict[str, Any] = {
        "mode": "strict_pairing_only",
        "evaluator_policy_id": "strict_empty_handling_v1",
        "match_iou": float(match_iou),
        "shape": list(np.asarray(gt_map).shape),
        "gt_instance_count": gt_count,
        "pred_instance_count": pred_count,
        "empty_gt": gt_count == 0,
        "empty_prediction": pred_count == 0,
        "both_empty": gt_count == 0 and pred_count == 0,
        "metrics_computed": ["dq", "sq", "pq"],
    }
    if base["both_empty"]:
        pairing = {"tp": 0, "fp": 0, "fn": 0, "paired_true": [], "paired_pred": [], "unpaired_true": [], "unpaired_pred": []}
        base.update({"included_in_macro": False, "skip_reason": "strict_both_empty_excluded_from_benchmark_macro", "no_match": True, "metrics": {"dq": None, "sq": None, "pq": None}, "pairing": pairing})
    elif gt_count == 0 or pred_count == 0:
        pairing = {
            "tp": 0,
            "fp": pred_count,
            "fn": gt_count,
            "paired_true": [],
            "paired_pred": [],
            "unpaired_true": [int(value) for value in gt_ids],
            "unpaired_pred": [int(value) for value in pred_ids],
        }
        base.update({"included_in_macro": True, "skip_reason": None, "no_match": True, "metrics": {"dq": 0.0, "sq": 0.0, "pq": 0.0}, "pairing": pairing})
    else:
        intersections = table[np.ix_(gt_ids, pred_ids)]
        unions = (
            np.asarray(overlap["gt_areas"], dtype=np.float64)[gt_ids, None]
            + np.asarray(overlap["pred_areas"], dtype=np.float64)[None, pred_ids]
            - intersections
        )
        pairwise_iou = np.divide(intersections, unions, out=np.zeros_like(intersections), where=unions > 0)
        if float(match_iou) >= 0.5:
            if np.any(pairwise_iou == float(match_iou)):
                eligible = pairwise_iou >= float(match_iou)
                cardinality_bonus = float(min(pairwise_iou.shape) + 1)
                weights = np.where(eligible, cardinality_bonus + pairwise_iou, 0.0)
                paired_rows, paired_cols = linear_sum_assignment(-weights)
                keep = eligible[paired_rows, paired_cols]
                paired_rows, paired_cols = paired_rows[keep], paired_cols[keep]
            else:
                paired_rows, paired_cols = np.nonzero(pairwise_iou > float(match_iou))
        else:
            paired_rows, paired_cols = linear_sum_assignment(-pairwise_iou)
            keep = pairwise_iou[paired_rows, paired_cols] > float(match_iou)
            paired_rows, paired_cols = paired_rows[keep], paired_cols[keep]
        paired_ious = pairwise_iou[paired_rows, paired_cols]
        paired_true = [int(value) for value in gt_ids[paired_rows]]
        paired_pred = [int(value) for value in pred_ids[paired_cols]]
        paired_true_set = set(paired_true)
        paired_pred_set = set(paired_pred)
        unpaired_true = [int(value) for value in gt_ids if int(value) not in paired_true_set]
        unpaired_pred = [int(value) for value in pred_ids if int(value) not in paired_pred_set]
        tp = len(paired_true)
        fp = len(unpaired_pred)
        fn = len(unpaired_true)
        dq = float(tp / (tp + 0.5 * fp + 0.5 * fn)) if tp or fp or fn else 0.0
        sq = float(paired_ious.sum() / (tp + 1.0e-6))
        pairing = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "paired_true": paired_true,
            "paired_pred": paired_pred,
            "unpaired_true": unpaired_true,
            "unpaired_pred": unpaired_pred,
        }
        base.update({"included_in_macro": True, "skip_reason": None, "no_match": tp == 0, "metrics": {"dq": dq, "sq": sq, "pq": dq * sq}, "pairing": pairing})
    record = base
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


def structural_errors(
    gt_map: np.ndarray,
    pred_map: np.ndarray,
    match_iou: float,
    *,
    pairing_info: dict[str, Any] | None = None,
    overlap: dict[str, Any] | None = None,
    best_iou_by_gt: dict[int, tuple[float, int | None]] | None = None,
) -> dict[str, Any]:
    """Supplementary final-map FP/FN, split, merge and boundary accounting."""

    pairing_info = pairing_info or strict_final_pairing(gt_map, pred_map, match_iou)
    overlap = overlap or final_instance_overlap_table(gt_map, pred_map)
    best_iou_by_gt = best_iou_by_gt or final_max_iou_by_gt(overlap)
    pairing = pairing_info["evaluator"]["pairing"] or {}
    unmatched_gt = [int(value) for value in pairing.get("unpaired_true", [])]
    unmatched_pred = [int(value) for value in pairing.get("unpaired_pred", [])]
    table = np.asarray(overlap["intersections"])
    pred_ids = np.asarray(overlap["pred_ids"], dtype=np.int64)
    gt_ids = np.asarray(overlap["gt_ids"], dtype=np.int64)
    boundary = 0
    split = 0
    for gt_id in unmatched_gt:
        max_iou, _ = best_iou_by_gt.get(gt_id, (0.0, None))
        boundary += int(0.0 < max_iou < match_iou)
        if gt_id < table.shape[0] and pred_ids.size:
            split += int(np.count_nonzero(table[gt_id, pred_ids]) >= 2)
    merge = 0
    for pred_id in unmatched_pred:
        if pred_id < table.shape[1] and gt_ids.size:
            merge += int(np.count_nonzero(table[gt_ids, pred_id]) >= 2)
    return {
        "tp": int(pairing.get("tp", 0)),
        "fp": int(pairing.get("fp", 0)),
        "fn": int(pairing.get("fn", 0)),
        "split_unmatched_gt_count": split,
        "merge_unmatched_pred_count": merge,
        "boundary_localization_unmatched_gt_count": boundary,
        "strict_evaluator": pairing_info["evaluator"],
    }
