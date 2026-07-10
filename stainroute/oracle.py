"""Ground-truth-only oracle metrics for StainRoute.

These functions are valid for Stage 0 reconciliation and for training-time
action labels only.  They must never be imported by pre-decode inference
features or selection code.
"""

from __future__ import annotations

import numpy as np


def _instance_ids(label_map: np.ndarray) -> np.ndarray:
    """Return foreground instance IDs without requiring contiguous labels."""

    return np.asarray([value for value in np.unique(label_map) if int(value) != 0])


def _pairwise_iou(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Compute foreground instance IoU without allocating all masks at once."""

    gt_ids = _instance_ids(gt)
    pred_ids = _instance_ids(pred)
    intersections = np.zeros((len(gt_ids), len(pred_ids)), dtype=np.float64)
    gt_areas = np.zeros(len(gt_ids), dtype=np.float64)
    pred_areas = np.zeros(len(pred_ids), dtype=np.float64)
    pred_index = {int(instance_id): index for index, instance_id in enumerate(pred_ids)}

    for gt_index, gt_id in enumerate(gt_ids):
        gt_mask = gt == gt_id
        gt_areas[gt_index] = float(gt_mask.sum())
        overlapping_ids, counts = np.unique(pred[gt_mask], return_counts=True)
        for pred_id, count in zip(overlapping_ids, counts):
            pred_index_value = pred_index.get(int(pred_id))
            if pred_index_value is not None:
                intersections[gt_index, pred_index_value] = float(count)

    for pred_index_value, pred_id in enumerate(pred_ids):
        pred_areas[pred_index_value] = float((pred == pred_id).sum())

    unions = gt_areas[:, None] + pred_areas[None, :] - intersections
    return np.divide(
        intersections,
        unions,
        out=np.zeros_like(intersections),
        where=unions > 0,
    )


def _matched_pair_ious(
    gt: np.ndarray,
    pred: np.ndarray,
    match_iou: float,
) -> np.ndarray:
    """Return IoUs from a maximum-weight one-to-one thresholded matching.

    At the standard 0.5 threshold, matches strictly above the threshold are
    mathematically unique.  The explicit assignment also handles the stated
    ``>= 0.5`` convention safely when an exact 0.5 tie is present.
    """

    if not 0.0 <= float(match_iou) <= 1.0:
        raise ValueError("match_iou must be in [0, 1]")
    if gt.shape != pred.shape:
        raise ValueError(f"gt and pred shapes differ: {gt.shape} != {pred.shape}")

    iou = _pairwise_iou(gt, pred)
    if iou.size == 0:
        return np.empty(0, dtype=np.float64)

    # scipy is already required by the repository's analysis tooling.  Import
    # lazily so the lightweight package itself remains importable without it.
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as exc:  # pragma: no cover - exercised only in a broken env
        raise RuntimeError("SciPy is required for StainRoute PQ matching") from exc

    # Invalid edges receive a large cost.  A cardinality bonus makes matching
    # every eligible edge preferable to leaving it unmatched, then maximises
    # IoU among matchings of that cardinality.
    eligible = iou >= float(match_iou)
    if not np.any(eligible):
        return np.empty(0, dtype=np.float64)
    cardinality_bonus = 2.0
    cost = np.where(eligible, -(cardinality_bonus + iou), 0.0)
    row_indices, col_indices = linear_sum_assignment(cost)
    selected = iou[row_indices, col_indices]
    return selected[selected >= float(match_iou)]


def matched_iou_sum(
    gt: np.ndarray,
    pred: np.ndarray,
    match_iou: float = 0.5,
) -> float:
    """Sum IoUs of one-to-one PQ matches at ``IoU >= match_iou``.

    This is a ground-truth-derived oracle quantity.  It is permitted for
    training labels, calibration, and evaluation, but never for inference
    feature construction or action selection.
    """

    gt = np.asarray(gt)
    pred = np.asarray(pred)
    return float(_matched_pair_ious(gt, pred, match_iou).sum())


def pq_factorized(
    gt: np.ndarray,
    pred: np.ndarray,
    match_iou: float = 0.5,
) -> float:
    """Compute PQ as ``2 * matched_iou_sum / (|pred| + |gt|)``.

    The all-background case follows the existing repository convention and
    returns 1.0.
    """

    gt = np.asarray(gt)
    pred = np.asarray(pred)
    if gt.shape != pred.shape:
        raise ValueError(f"gt and pred shapes differ: {gt.shape} != {pred.shape}")
    denominator = len(_instance_ids(gt)) + len(_instance_ids(pred))
    if denominator == 0:
        return 1.0
    return float(2.0 * matched_iou_sum(gt, pred, match_iou) / denominator)
