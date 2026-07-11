"""Fixed four-token mask and ranking metrics for NuSet Stage 0."""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
from scipy.stats import pearsonr, spearmanr
import torch
import torch.nn.functional as F

from sam2_train.modeling.stats_utils import (
    get_dice_1,
    get_fast_aji,
    get_fast_aji_plus,
    get_fast_pq,
    remap_label,
)


TOKEN_COUNT = 4


def _safe_correlation(function: Any, left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 2 or np.all(left == left[0]) or np.all(right == right[0]):
        return None
    value = float(function(left, right).statistic)
    return value if np.isfinite(value) else None


def hard_iou(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """IoU for [N, 4, H, W] masks versus [N, H, W] targets."""
    if logits.ndim != 4 or target.ndim != 3 or logits.shape[0] != target.shape[0]:
        raise ValueError("NuSet hard IoU requires [N,4,H,W] logits and [N,H,W] target")
    prediction, truth = logits > 0, target.bool().unsqueeze(1)
    intersection = (prediction & truth).sum(dim=(-1, -2)).float()
    union = (prediction | truth).sum(dim=(-1, -2)).float()
    return torch.where(union > 0, intersection / union, torch.ones_like(union))


def soft_iou(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probability, truth = torch.sigmoid(logits), target.float().unsqueeze(1)
    intersection = (probability * truth).sum(dim=(-1, -2))
    union = probability.sum(dim=(-1, -2)) + truth.sum(dim=(-1, -2)) - intersection
    return torch.where(union > 0, intersection / union, torch.ones_like(union))


def focal_and_dice_loss(logits: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-prompt/per-token focal and Dice values for analysis only."""
    truth = target.float().unsqueeze(1).expand_as(logits)
    probability = torch.sigmoid(logits)
    cross_entropy = F.binary_cross_entropy_with_logits(logits, truth, reduction="none")
    pt = probability * truth + (1 - probability) * (1 - truth)
    focal = ((1 - pt).pow(2) * cross_entropy).mean(dim=(-1, -2))
    numerator = 2 * (probability * truth).sum(dim=(-1, -2)) + 1.0
    denominator = probability.sum(dim=(-1, -2)) + truth.sum(dim=(-1, -2)) + 1.0
    return focal, 1 - numerator / denominator


def pairwise_mask_iou(logits: torch.Tensor) -> dict[str, torch.Tensor]:
    masks = logits > 0
    result: dict[str, torch.Tensor] = {}
    for left, right in combinations(range(TOKEN_COUNT), 2):
        intersection = (masks[:, left] & masks[:, right]).sum(dim=(-1, -2)).float()
        union = (masks[:, left] | masks[:, right]).sum(dim=(-1, -2)).float()
        result[f"{left}_{right}"] = torch.where(union > 0, intersection / union, torch.ones_like(union))
    return result


def _boundary(mask: np.ndarray) -> np.ndarray:
    value = np.asarray(mask, dtype=bool)
    return value ^ np.roll(value, 1, 0) | value ^ np.roll(value, -1, 0) | value ^ np.roll(value, 1, 1) | value ^ np.roll(value, -1, 1)


def pairwise_boundary_disagreement(logits: torch.Tensor) -> dict[str, np.ndarray]:
    masks = (logits > 0).detach().cpu().numpy()
    result: dict[str, np.ndarray] = {}
    for left, right in combinations(range(TOKEN_COUNT), 2):
        first = np.stack([_boundary(mask) for mask in masks[:, left]])
        second = np.stack([_boundary(mask) for mask in masks[:, right]])
        result[f"{left}_{right}"] = np.logical_xor(first, second).mean(axis=(1, 2))
    return result


def selector_indices(predicted_iou: torch.Tensor, true_iou: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    if predicted_iou.ndim != 2 or predicted_iou.size(1) != TOKEN_COUNT:
        raise ValueError("NuSet requires four predicted-IoU token scores")
    result = {
        "single": torch.zeros(predicted_iou.size(0), dtype=torch.long, device=predicted_iou.device),
        "multi_pred": predicted_iou[:, 1:].argmax(dim=1) + 1,
        "all_pred": predicted_iou.argmax(dim=1),
    }
    if true_iou is not None:
        result["multi_oracle"] = true_iou[:, 1:].argmax(dim=1) + 1
        result["all_oracle"] = true_iou.argmax(dim=1)
    return result


def token_record_rows(
    *,
    scope: str,
    image_id: str,
    crop_id: int,
    prompt_xy: np.ndarray,
    logits: torch.Tensor,
    predicted_iou: torch.Tensor,
    target_masks: torch.Tensor | None,
    target_instance_ids: np.ndarray | None,
) -> list[dict[str, Any]]:
    """One serializable row per prompt, preserving all four token measurements."""
    if logits.ndim != 4 or logits.size(1) != TOKEN_COUNT:
        raise ValueError("NuSet record builder requires four token logits")
    hard = soft = focal = dice = None
    selected: dict[str, torch.Tensor] = selector_indices(predicted_iou)
    if target_masks is not None:
        hard, soft = hard_iou(logits, target_masks), soft_iou(logits, target_masks)
        focal, dice = focal_and_dice_loss(logits, target_masks)
        selected = selector_indices(predicted_iou, hard)
    pairwise = pairwise_mask_iou(logits)
    disagreement = pairwise_boundary_disagreement(logits)
    binary = logits > 0
    rows: list[dict[str, Any]] = []
    for index in range(logits.size(0)):
        point = prompt_xy[index]
        x, y = int(np.clip(np.trunc(point[0]), 0, logits.shape[-1] - 1)), int(np.clip(np.trunc(point[1]), 0, logits.shape[-2] - 1))
        row: dict[str, Any] = {
            "scope": scope,
            "image_id": image_id,
            "crop_id": int(crop_id),
            "prompt_x": float(point[0]),
            "prompt_y": float(point[1]),
            "target_instance_id": int(target_instance_ids[index]) if target_instance_ids is not None else None,
            "matched": target_masks is not None,
            "predicted_iou_tokens": predicted_iou[index].detach().cpu().tolist(),
            "hard_iou_tokens": hard[index].detach().cpu().tolist() if hard is not None else None,
            "soft_iou_tokens": soft[index].detach().cpu().tolist() if soft is not None else None,
            "focal_loss_tokens": focal[index].detach().cpu().tolist() if focal is not None else None,
            "dice_loss_tokens": dice[index].detach().cpu().tolist() if dice is not None else None,
            "mask_area_tokens": binary[index].sum(dim=(-1, -2)).detach().cpu().tolist(),
            "contains_positive_point_tokens": binary[index, :, y, x].detach().cpu().tolist(),
            "pairwise_mask_iou": {name: float(value[index].detach().cpu()) for name, value in pairwise.items()},
            "pairwise_boundary_disagreement": {name: float(value[index]) for name, value in disagreement.items()},
            "selected_token": {name: int(value[index].detach().cpu()) for name, value in selected.items()},
        }
        if hard is not None:
            token0 = float(hard[index, 0])
            row["delta_iou_vs_token0"] = {
                name: float(hard[index, token] - token0) for name, token in row["selected_token"].items()
            }
            row["token0_below_half_other_above_half"] = bool(hard[index, 0] < 0.5 and torch.any(hard[index, 1:] >= 0.5))
        rows.append(row)
    return rows


def _flatten_rows(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    predicted, true = [], []
    for row in rows:
        if row.get("hard_iou_tokens") is None:
            continue
        predicted.extend(row["predicted_iou_tokens"])
        true.extend(row["hard_iou_tokens"])
    return np.asarray(predicted, dtype=np.float64), np.asarray(true, dtype=np.float64)


def _ece(scores: np.ndarray, labels: np.ndarray) -> tuple[float | None, list[dict[str, Any]]]:
    if not len(scores):
        return None, []
    order = np.argsort(scores, kind="mergesort")
    bins = []
    for number, indices in enumerate(np.array_split(order, 10)):
        if not len(indices):
            continue
        confidence, accuracy = float(scores[indices].mean()), float(labels[indices].mean())
        bins.append({"bin": number, "count": int(len(indices)), "mean_predicted_iou": confidence, "empirical_iou_ge_0_5": accuracy, "absolute_gap": abs(confidence - accuracy)})
    return float(sum(item["count"] * item["absolute_gap"] for item in bins) / len(scores)), bins


def ranking_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Predicted-IoU ranking diagnostics; unmatched prompts have no GT metric."""
    matched = [row for row in rows if row.get("hard_iou_tokens") is not None]
    predicted, true = _flatten_rows(matched)
    top1, regrets, reciprocal_ranks = [], [], []
    for row in matched:
        prediction, truth = np.asarray(row["predicted_iou_tokens"]), np.asarray(row["hard_iou_tokens"])
        selected, oracle = int(prediction.argmax()), int(truth.argmax())
        top1.append(selected == oracle)
        regrets.append(float(truth[oracle] - truth[selected]))
        order = np.argsort(-prediction, kind="mergesort")
        reciprocal_ranks.append(1.0 / (int(np.flatnonzero(order == oracle)[0]) + 1))
    clipped = np.clip(predicted, 0.0, 1.0)
    ece, bins = _ece(clipped, true >= 0.5)
    return {
        "matched_prompt_count": len(matched),
        "unmatched_prompt_count": len(rows) - len(matched),
        "spearman_predicted_vs_true_iou": _safe_correlation(spearmanr, predicted, true),
        "pearson_predicted_vs_true_iou": _safe_correlation(pearsonr, predicted, true),
        "top1_token_selection_accuracy": float(np.mean(top1)) if top1 else None,
        "mean_oracle_regret": float(np.mean(regrets)) if regrets else None,
        "mean_reciprocal_rank": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else None,
        "brier_iou_ge_0_5": float(np.mean((clipped - (true >= 0.5)) ** 2)) if len(clipped) else None,
        "ece_10_equal_frequency": ece,
        "token_wise_calibration": {
            str(token): {
                "n": len(matched),
                "brier_iou_ge_0_5": float(np.mean((np.clip(np.asarray([row["predicted_iou_tokens"][token] for row in matched]), 0, 1) - np.asarray([row["hard_iou_tokens"][token] >= 0.5 for row in matched])) ** 2)) if matched else None,
            }
            for token in range(TOKEN_COUNT)
        },
        "reliability_diagram": bins,
    }


def headroom_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    matched = [row for row in rows if row.get("hard_iou_tokens") is not None]
    if not matched:
        return {"n_prompts": 0}
    hard = np.asarray([row["hard_iou_tokens"] for row in matched], dtype=np.float64)
    selectors = {name: np.asarray([row["selected_token"][name] for row in matched], dtype=np.int64) for name in ("single", "multi_pred", "all_pred", "multi_oracle", "all_oracle")}
    values = {name: hard[np.arange(len(hard)), token] for name, token in selectors.items()}
    token0 = values["single"]
    all_oracle = values["all_oracle"]
    areas = np.asarray([row["mask_area_tokens"] for row in matched], dtype=np.float64)
    return {
        "n_prompts": len(matched),
        "selectors": {
            name: {
                "mean_hard_iou": float(value.mean()),
                "median_hard_iou": float(np.median(value)),
                "mean_delta_vs_token0": float((value - token0).mean()),
                "median_delta_vs_token0": float(np.median(value - token0)),
            }
            for name, value in values.items()
        },
        "all_oracle_non_token0_fraction": float((selectors["all_oracle"] != 0).mean()),
        "all_oracle_delta_ge_0_01_fraction": float((all_oracle - token0 >= 0.01).mean()),
        "all_oracle_delta_ge_0_02_fraction": float((all_oracle - token0 >= 0.02).mean()),
        "all_oracle_delta_ge_0_05_fraction": float((all_oracle - token0 >= 0.05).mean()),
        "token0_below_half_other_above_half_fraction": float(np.mean([row["token0_below_half_other_above_half"] for row in matched])),
        "pairwise_mask_iou": {name: float(np.mean([row["pairwise_mask_iou"][name] for row in matched])) for name in matched[0]["pairwise_mask_iou"]},
        "pairwise_boundary_disagreement": {name: float(np.mean([row["pairwise_boundary_disagreement"][name] for row in matched])) for name in matched[0]["pairwise_boundary_disagreement"]},
        "mask_area_ratio_to_token0": {
            str(token): {
                "mean": float(np.mean(areas[:, token] / np.maximum(areas[:, 0], 1))),
                "median": float(np.median(areas[:, token] / np.maximum(areas[:, 0], 1))),
            }
            for token in range(TOKEN_COUNT)
        },
        "token_collapse_fraction": float(np.mean([
            all(value >= 0.999999 for value in row["pairwise_mask_iou"].values()) for row in matched
        ])),
    }


def assembly_metrics(instance_map: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    """Use the repository's inclusive IoU>=0.5 evaluator and expose TP/FP/FN."""
    truth, pred = remap_label(instance_map), remap_label(prediction)
    truth_count, prediction_count = int(len(np.unique(truth)) - 1), int(len(np.unique(pred)) - 1)
    if prediction_count == 0:
        return {
            "dice": 0.0 if truth_count else 1.0,
            "aji": 0.0 if truth_count else 1.0,
            "aji_plus": 0.0 if truth_count else 1.0,
            "dq": 0.0 if truth_count else 1.0,
            "sq": 0.0 if truth_count else 1.0,
            "pq": 0.0 if truth_count else 1.0,
            "tp": 0,
            "fp": 0,
            "fn": truth_count,
            "matched_iou_sum": 0.0,
            "instance_count": 0,
        }
    if truth_count == 0:
        return {
            "dice": 0.0,
            "aji": 0.0,
            "aji_plus": 0.0,
            "dq": 0.0,
            "sq": 0.0,
            "pq": 0.0,
            "tp": 0,
            "fp": prediction_count,
            "fn": 0,
            "matched_iou_sum": 0.0,
            "instance_count": prediction_count,
        }
    (dq, sq, pq), (paired_true, paired_pred, unpaired_true, unpaired_pred) = get_fast_pq(truth, pred, match_iou=0.5)
    return {
        "dice": float(get_dice_1(truth, pred)),
        "aji": float(get_fast_aji(truth, pred)),
        "aji_plus": float(get_fast_aji_plus(truth, pred)),
        "dq": float(dq),
        "sq": float(sq),
        "pq": float(pq),
        "tp": int(len(paired_true)),
        "fp": int(len(unpaired_pred)),
        "fn": int(len(unpaired_true)),
        "matched_iou_sum": float(sq * len(paired_true)),
        "instance_count": prediction_count,
    }
