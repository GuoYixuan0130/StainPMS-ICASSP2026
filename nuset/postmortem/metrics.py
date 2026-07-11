"""Pure postmortem ranking, oracle-gain and preregistered verdict calculations."""

from __future__ import annotations

from typing import Any

import numpy as np


def selector_metrics(selection: np.ndarray, true_iou: np.ndarray) -> dict[str, Any]:
    """Groupwise metrics for an already-selected one-of-four token path."""
    truth = np.asarray(true_iou, dtype=np.float64)
    selected = np.asarray(selection, dtype=np.int64)
    oracle = truth.argmax(axis=1)
    selected_iou = truth[np.arange(len(truth)), selected]
    oracle_iou = truth[np.arange(len(truth)), oracle]
    regret = oracle_iou - selected_iou
    ranks = []
    for row, target in zip(truth, selected):
        order = np.argsort(-row, kind="stable")
        ranks.append(int(np.flatnonzero(order == target)[0]) + 1)
    return {
        "prompt_count": int(len(truth)),
        "selected_mask_mean_iou": float(selected_iou.mean()) if len(truth) else 0.0,
        "top1_accuracy": float(np.mean(selected == oracle)) if len(truth) else 0.0,
        "mean_oracle_regret": float(regret.mean()) if len(truth) else 0.0,
        "median_oracle_regret": float(np.median(regret)) if len(truth) else 0.0,
        "mrr": float(np.mean(1.0 / np.asarray(ranks))) if ranks else 0.0,
        "selected_token_histogram": np.bincount(selected, minlength=4).astype(int).tolist(),
        "selected_iou": selected_iou,
        "oracle_iou": oracle_iou,
        "regret": regret,
        "oracle_token": oracle,
    }


def changed_winner_metrics(existing: np.ndarray, candidate: np.ndarray, true_iou: np.ndarray) -> dict[str, Any]:
    existing, candidate, truth = np.asarray(existing), np.asarray(candidate), np.asarray(true_iou)
    changed = existing != candidate
    before = truth[np.arange(len(truth)), existing]
    after = truth[np.arange(len(truth)), candidate]
    delta = after - before
    return {
        "changed_winner_count": int(changed.sum()),
        "changed_winner_improved_fraction": float(np.mean(delta[changed] > 0)) if changed.any() else 0.0,
        "changed_winner_unchanged_fraction": float(np.mean(delta[changed] == 0)) if changed.any() else 0.0,
        "changed_winner_decreased_fraction": float(np.mean(delta[changed] < 0)) if changed.any() else 0.0,
        "changed_winner_mean_iou_delta": float(delta[changed].mean()) if changed.any() else 0.0,
    }


def score_and_truth_gap(scores: np.ndarray, true_iou: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    score = np.asarray(scores, dtype=np.float64)
    truth = np.asarray(true_iou, dtype=np.float64)
    score_sorted = np.sort(score, axis=1)
    truth_sorted = np.sort(truth, axis=1)
    return score_sorted[:, -1] - score_sorted[:, -2], truth_sorted[:, -1] - truth_sorted[:, -2]


def failure_mode(*, train_existing: dict[str, Any], train_nurank: dict[str, Any], development_existing: dict[str, Any], development_nurank: dict[str, Any], development_single: dict[str, Any], development_pq_delta: float) -> dict[str, Any]:
    train_top1_pp = (train_nurank["top1_accuracy"] - train_existing["top1_accuracy"]) * 100
    train_regret_reduction = (train_existing["mean_oracle_regret"] - train_nurank["mean_oracle_regret"]) / train_existing["mean_oracle_regret"] if train_existing["mean_oracle_regret"] > 0 else 0.0
    dev_top1_pp = (development_nurank["top1_accuracy"] - development_existing["top1_accuracy"]) * 100
    dev_regret_reduction = (development_existing["mean_oracle_regret"] - development_nurank["mean_oracle_regret"]) / development_existing["mean_oracle_regret"] if development_existing["mean_oracle_regret"] > 0 else 0.0
    flags = []
    if train_top1_pp < 5 or train_regret_reduction < .10:
        flags.append("representation_or_objective_failure")
    if train_top1_pp >= 5 and train_regret_reduction >= .10 and (dev_top1_pp <= 0 or dev_regret_reduction < 0):
        flags.append("cross_patient_generalization_failure")
    if development_nurank["selected_mask_mean_iou"] - development_single["selected_mask_mean_iou"] >= .005 and development_pq_delta <= .001:
        flags.append("assembly_mismatch")
    mode = "mixed" if len(flags) > 1 else flags[0] if flags else "no_preregistered_failure_pattern"
    return {"failure_mode": mode, "flags": flags, "train_top1_improvement_points": train_top1_pp, "train_regret_reduction_fraction": train_regret_reduction, "development_top1_improvement_points": dev_top1_pp, "development_regret_reduction_fraction": dev_regret_reduction}


def oracle_gain_summary(true_iou: np.ndarray) -> dict[str, Any]:
    truth = np.asarray(true_iou, dtype=np.float64)
    best = truth.max(axis=1); token0 = truth[:, 0]
    order = np.sort(truth, axis=1)
    gain, gap = best - token0, order[:, -1] - order[:, -2]
    positive = gain[gain > 0]
    contribution = {}
    for percentage in (1, 5, 10):
        count = max(1, int(np.ceil(len(gain) * percentage / 100)))
        contribution[f"top_{percentage}_percent_contribution"] = float(np.sort(gain)[-count:].sum() / gain.sum()) if gain.sum() > 0 else 0.0
    return {
        "prompt_count": int(len(truth)), "oracle_best_token_histogram": np.bincount(truth.argmax(axis=1), minlength=4).astype(int).tolist(),
        "token0_not_best_fraction": float(np.mean(truth.argmax(axis=1) != 0)), "best_vs_token0_mean_delta_iou": float(gain.mean()), "best_vs_token0_median_delta_iou": float(np.median(gain)),
        **{f"gain_ge_{threshold:0.3f}_fraction": float(np.mean(gain >= threshold)) for threshold in (.005, .01, .02, .05)},
        **{f"near_tie_lt_{threshold:0.3f}_fraction": float(np.mean(gap < threshold)) for threshold in (.002, .005, .01)},
        **contribution,
    }
