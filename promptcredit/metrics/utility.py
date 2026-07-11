"""Dependency-light, explicit score-versus-mask-utility metrics."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
from scipy.stats import pearsonr, spearmanr


def _one_iou(intersection: np.ndarray, union: np.ndarray) -> np.ndarray:
    return np.divide(intersection, union, out=np.ones_like(intersection, dtype=np.float64), where=union > 0)


def binary_iou(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    pred = np.asarray(prediction, dtype=bool)
    gt = np.asarray(target, dtype=bool)
    if pred.shape != gt.shape or pred.ndim < 2:
        raise ValueError("prediction and target must have identical [N, H, W] shapes")
    axes = tuple(range(1, pred.ndim))
    return _one_iou(np.logical_and(pred, gt).sum(axis=axes), np.logical_or(pred, gt).sum(axis=axes))


def soft_iou(probability: np.ndarray, target: np.ndarray) -> np.ndarray:
    prob = np.asarray(probability, dtype=np.float64)
    gt = np.asarray(target, dtype=np.float64)
    if prob.shape != gt.shape or prob.ndim < 2:
        raise ValueError("probability and target must have identical [N, H, W] shapes")
    axes = tuple(range(1, prob.ndim))
    intersection = (prob * gt).sum(axis=axes)
    union = prob.sum(axis=axes) + gt.sum(axis=axes) - intersection
    return _one_iou(intersection, union)


def _correlation(fn: Any, scores: np.ndarray, utility: np.ndarray) -> float | None:
    if len(scores) < 2 or np.all(scores == scores[0]) or np.all(utility == utility[0]):
        return None
    value = float(fn(scores, utility).statistic)
    return value if np.isfinite(value) else None


def _auroc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]
    start = 0
    while start < len(scores):
        stop = start + 1
        while stop < len(scores) and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2.0
        start = stop
    return float((ranks[labels].sum() - positives * (positives + 1) / 2.0) / (positives * negatives))


def _auprc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    positives = int(labels.sum())
    if positives == 0:
        return None
    order = np.argsort(-scores, kind="mergesort")
    ordered_labels = labels[order].astype(np.float64)
    precision = np.cumsum(ordered_labels) / np.arange(1, len(labels) + 1)
    return float(precision[ordered_labels == 1].sum() / positives)


def reliability_bins(scores: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> list[dict[str, float | int]]:
    if n_bins != 10:
        raise ValueError("Stage 0 ECE is frozen to 10 equal-frequency bins")
    if len(scores) == 0:
        return []
    order = np.argsort(scores, kind="mergesort")
    bins: list[dict[str, float | int]] = []
    for bin_index, indices in enumerate(np.array_split(order, n_bins)):
        if len(indices) == 0:
            continue
        confidence = float(scores[indices].mean())
        accuracy = float(labels[indices].mean())
        bins.append(
            {
                "bin": bin_index,
                "count": int(len(indices)),
                "mean_score": confidence,
                "empirical_matchability": accuracy,
                "absolute_gap": abs(confidence - accuracy),
            }
        )
    return bins


def score_utility_summary(
    foreground_probability: Iterable[float], hard_mask_iou: Iterable[float]
) -> dict[str, Any]:
    """Compute the pre-registered utility calibration statistics for Audit B."""
    scores = np.asarray(list(foreground_probability), dtype=np.float64)
    iou = np.asarray(list(hard_mask_iou), dtype=np.float64)
    if scores.shape != iou.shape or scores.ndim != 1:
        raise ValueError("scores and hard_mask_iou must be same-length vectors")
    if not (np.isfinite(scores).all() and np.isfinite(iou).all()):
        raise ValueError("score/IoU metrics require finite values")
    labels = iou >= 0.5
    bins = reliability_bins(scores, labels.astype(np.float64), n_bins=10)
    ece = float(sum(item["count"] * item["absolute_gap"] for item in bins) / len(scores)) if len(scores) else None
    return {
        "n_prompts": int(len(scores)),
        "spearman_point_score_vs_hard_iou": _correlation(spearmanr, scores, iou),
        "pearson_point_score_vs_hard_iou": _correlation(pearsonr, scores, iou),
        "auroc_iou_ge_0_5": _auroc(scores, labels),
        "auprc_iou_ge_0_5": _auprc(scores, labels),
        "brier_iou_ge_0_5": float(np.mean((scores - labels.astype(np.float64)) ** 2)) if len(scores) else None,
        "ece_10_equal_frequency": ece,
        "reliability_diagram": bins,
    }

