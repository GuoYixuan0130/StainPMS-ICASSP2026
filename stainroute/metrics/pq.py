"""Inclusive one-to-one PQ evaluation for StainRoute oracle utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _ids(label_map: np.ndarray) -> np.ndarray:
    return np.asarray([value for value in np.unique(label_map) if int(value) != 0])


def _pairwise_iou(gt: np.ndarray, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gt_ids = _ids(gt)
    pred_ids = _ids(pred)
    intersections = np.zeros((len(gt_ids), len(pred_ids)), dtype=np.float64)
    gt_areas = np.zeros(len(gt_ids), dtype=np.float64)
    pred_areas = np.zeros(len(pred_ids), dtype=np.float64)
    pred_index = {int(instance_id): index for index, instance_id in enumerate(pred_ids)}

    for gt_index, gt_id in enumerate(gt_ids):
        mask = gt == gt_id
        gt_areas[gt_index] = float(mask.sum())
        values, counts = np.unique(pred[mask], return_counts=True)
        for pred_id, count in zip(values, counts):
            pred_index_value = pred_index.get(int(pred_id))
            if pred_index_value is not None:
                intersections[gt_index, pred_index_value] = float(count)
    for pred_index_value, pred_id in enumerate(pred_ids):
        pred_areas[pred_index_value] = float((pred == pred_id).sum())

    union = gt_areas[:, None] + pred_areas[None, :] - intersections
    iou = np.divide(intersections, union, out=np.zeros_like(intersections), where=union > 0)
    return gt_ids, pred_ids, iou


def _match_indices(iou: np.ndarray, match_iou: float) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 <= float(match_iou) <= 1.0:
        raise ValueError("match_iou must be in [0, 1]")
    if iou.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as exc:  # pragma: no cover - environment requirement
        raise RuntimeError("SciPy is required for one-to-one PQ matching") from exc

    # The bonus first maximizes the number of threshold-eligible pairs, then
    # maximizes their IoU sum. This is required for inclusive exact-0.5 edges.
    eligible = iou >= float(match_iou)
    if not np.any(eligible):
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    cardinality_bonus = float(min(iou.shape) + 1)
    weights = np.where(eligible, cardinality_bonus + iou, 0.0)
    rows, cols = linear_sum_assignment(-weights)
    keep = eligible[rows, cols]
    return rows[keep].astype(np.int64), cols[keep].astype(np.int64)


@dataclass(frozen=True)
class PQEvaluation:
    """Full global PQ state, suitable for action-utility deltas."""

    matched_iou_sum: float
    tp: int
    fp: int
    fn: int
    dq: float
    sq: float
    pq: float
    matched_pairs: tuple[tuple[int, int, float], ...]


def evaluate_pq(gt: np.ndarray, pred: np.ndarray, match_iou: float = 0.5) -> PQEvaluation:
    """Evaluate global inclusive-IoU PQ with one-to-one matching."""

    gt = np.asarray(gt)
    pred = np.asarray(pred)
    if gt.shape != pred.shape:
        raise ValueError(f"gt and pred shapes differ: {gt.shape} != {pred.shape}")
    gt_ids, pred_ids, iou = _pairwise_iou(gt, pred)
    if len(gt_ids) == 0 and len(pred_ids) == 0:
        return PQEvaluation(0.0, 0, 0, 0, 1.0, 1.0, 1.0, ())

    rows, cols = _match_indices(iou, match_iou)
    matched_iou = iou[rows, cols] if len(rows) else np.empty(0, dtype=np.float64)
    tp = int(len(rows))
    fp = int(len(pred_ids) - tp)
    fn = int(len(gt_ids) - tp)
    denominator = tp + 0.5 * fp + 0.5 * fn
    dq = float(tp / denominator) if denominator > 0 else 1.0
    iou_sum = float(matched_iou.sum())
    sq = float(iou_sum / tp) if tp > 0 else 0.0
    pq = float(2.0 * iou_sum / (len(gt_ids) + len(pred_ids))) if (len(gt_ids) + len(pred_ids)) else 1.0
    pairs = tuple(
        (int(gt_ids[row]), int(pred_ids[col]), float(iou[row, col]))
        for row, col in zip(rows, cols)
    )
    return PQEvaluation(iou_sum, tp, fp, fn, dq, sq, pq, pairs)


def matched_iou_sum(gt: np.ndarray, pred: np.ndarray, match_iou: float = 0.5) -> float:
    return evaluate_pq(gt, pred, match_iou).matched_iou_sum


def pq_factorized(gt: np.ndarray, pred: np.ndarray, match_iou: float = 0.5) -> float:
    return evaluate_pq(gt, pred, match_iou).pq
