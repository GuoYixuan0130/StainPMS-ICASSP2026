"""Ranking and calibration diagnostics for fixed four-token prompt groups."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import pearsonr, spearmanr


def _finite_stat(function, first: np.ndarray, second: np.ndarray) -> float | None:
    if len(first) < 2 or np.all(first == first[0]) or np.all(second == second[0]):
        return None
    result = float(function(first, second).statistic)
    return result if np.isfinite(result) else None


def _brier_ece(score: np.ndarray, target: np.ndarray, bins: int = 10) -> tuple[float, float]:
    binary = (target >= 0.5).astype(np.float64)
    brier = float(np.mean((score - binary) ** 2))
    if not len(score):
        return brier, 0.0
    order = np.argsort(score, kind="mergesort")
    ece = 0.0
    for indices in np.array_split(order, bins):
        if len(indices):
            ece += len(indices) / len(score) * abs(float(score[indices].mean()) - float(binary[indices].mean()))
    return brier, float(ece)


def _mrr(scores: np.ndarray, oracle: np.ndarray) -> float:
    ranks = np.empty_like(oracle, dtype=np.int64)
    for index, (row, best) in enumerate(zip(scores, oracle)):
        # Stable sort fixes deterministic ties to lower token index, matching argmax.
        ranks[index] = int(np.flatnonzero(np.argsort(-row, kind="stable") == best)[0]) + 1
    return float(np.mean(1.0 / ranks)) if len(ranks) else 0.0


def ranking_metrics(scores: np.ndarray, true_iou: np.ndarray, matched: np.ndarray | None = None) -> dict[str, Any]:
    """Metrics for groupwise selection and pointwise calibration; no GT enters scores."""
    score = np.asarray(scores, dtype=np.float64)
    truth = np.asarray(true_iou, dtype=np.float64)
    if score.shape != truth.shape or score.ndim != 2 or score.shape[1] != 4:
        raise ValueError("NuRank metrics require [groups,4]")
    selected = score.argmax(axis=1)
    oracle = truth.argmax(axis=1)
    selected_truth = truth[np.arange(len(truth)), selected]
    oracle_truth = truth[np.arange(len(truth)), oracle]
    regret = oracle_truth - selected_truth
    brier, ece = _brier_ece(score.reshape(-1), truth.reshape(-1))
    result: dict[str, Any] = {
        "group_count": int(len(score)), "top1_accuracy": float(np.mean(selected == oracle)) if len(score) else 0.0,
        "mean_selection_regret": float(np.mean(regret)) if len(regret) else 0.0,
        "median_selection_regret": float(np.median(regret)) if len(regret) else 0.0,
        "mrr": _mrr(score, oracle), "spearman": _finite_stat(spearmanr, score.reshape(-1), truth.reshape(-1)),
        "pearson": _finite_stat(pearsonr, score.reshape(-1), truth.reshape(-1)), "brier": brier, "ece": ece,
        "non_token0_selection_rate": float(np.mean(selected != 0)) if len(score) else 0.0,
        "token_selection_histogram": np.bincount(selected, minlength=4).astype(int).tolist(),
        "selected_indices": selected, "oracle_indices": oracle, "selected_true_iou": selected_truth,
        "oracle_true_iou": oracle_truth, "selection_regret": regret,
    }
    if matched is not None:
        matched = np.asarray(matched, dtype=bool)
        result["matched_group_count"] = int(matched.sum())
        result["unmatched_group_count"] = int((~matched).sum())
    return result
