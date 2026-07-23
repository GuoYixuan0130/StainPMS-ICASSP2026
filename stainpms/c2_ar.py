"""C2-AR training-only losses for assembly-robust selected predictions.

The deployed inference path is deliberately untouched.  This module consumes
the same native token-0 mask logits and IoU-head score that C1 already emits.
Discrete matching is detached; gradients flow only to the selected mask logits
and the existing IoU/quality score.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


STRICT_MATCH_IOU = 0.5


def _validate_inputs(
    selected_logits: torch.Tensor,
    selected_quality: torch.Tensor,
    own_gt_masks: torch.Tensor,
    image_prompt_counts: Iterable[int],
) -> list[int]:
    if selected_logits.ndim != 3:
        raise ValueError("selected_logits must be [N,H,W]")
    if selected_quality.shape != (selected_logits.shape[0],):
        raise ValueError("selected_quality must be [N]")
    if own_gt_masks.shape != selected_logits.shape:
        raise ValueError("own_gt_masks must have the same [N,H,W] shape as logits")
    counts = [int(value) for value in image_prompt_counts]
    if any(value < 0 for value in counts) or sum(counts) != int(selected_logits.shape[0]):
        raise ValueError("image_prompt_counts must be non-negative and sum to N")
    return counts


def _graph_zero(reference: torch.Tensor) -> torch.Tensor:
    return reference.float().sum() * 0.0


def _same_instance_matrix(gt_masks: torch.Tensor) -> torch.Tensor:
    """Return whether two binary masks describe the same non-empty instance."""

    flat = gt_masks.bool().reshape(gt_masks.shape[0], -1)
    areas = flat.sum(dim=1)
    intersection = flat.float() @ flat.float().T
    same = intersection == torch.minimum(areas[:, None], areas[None, :]).float()
    return same & (areas[:, None] > 0) & (areas[None, :] > 0)


def selected_mask_exclusivity_loss(
    selected_logits: torch.Tensor,
    own_gt_masks: torch.Tensor,
    image_prompt_counts: Iterable[int],
    *,
    neighbor_radius: int = 2,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Penalize foreign-nucleus leakage and local soft selected-mask overlap.

    The loss intentionally never supervises background pixels a second time.
    Leakage is measured only on other annotated nuclei.  The conflict term is
    evaluated only for distinct GT instances whose radius-dilated supports
    touch, focusing the penalty on crowded/contact regions.
    """

    counts = _validate_inputs(
        selected_logits,
        selected_logits.new_zeros((selected_logits.shape[0],)),
        own_gt_masks,
        image_prompt_counts,
    )
    if int(neighbor_radius) < 0:
        raise ValueError("neighbor_radius must be non-negative")

    if selected_logits.shape[0] == 0:
        zero = _graph_zero(selected_logits)
        return zero, {
            "valid_prompt_count": 0,
            "foreign_valid_prompt_count": 0,
            "neighbor_pair_count": 0,
            "foreign_leakage": 0.0,
            "conflict": 0.0,
        }

    probabilities = torch.sigmoid(selected_logits.float())
    target_masks = own_gt_masks.bool()
    leakage_values: list[torch.Tensor] = []
    conflict_values: list[torch.Tensor] = []
    neighbor_pair_count = 0
    offset = 0
    kernel = 2 * int(neighbor_radius) + 1

    for count in counts:
        if count == 0:
            continue
        next_offset = offset + count
        prob = probabilities[offset:next_offset]
        target = target_masks[offset:next_offset]
        foreign = torch.zeros_like(target)
        if count >= 2:
            if int(neighbor_radius) == 0:
                dilated = target
            else:
                dilated = F.max_pool2d(
                    target.float().unsqueeze(1),
                    kernel_size=kernel,
                    stride=1,
                    padding=int(neighbor_radius),
                )[:, 0].bool()
            flat_dilated = dilated.reshape(count, -1).float()
            flat_target = target.reshape(count, -1).float()
            close = (flat_dilated @ flat_target.T) > 0
            same = _same_instance_matrix(target)
            # Only close, distinct GT nuclei contribute to leakage. This
            # keeps the extra supervision local to crowding/contact regions
            # instead of repeating ordinary background supervision.
            for row in range(count):
                neighbors = close[row] & ~same[row]
                if bool(neighbors.any()):
                    foreign[row] = target[neighbors].any(dim=0)
            valid_pair = torch.triu(close & ~same, diagonal=1)
            rows, cols = valid_pair.nonzero(as_tuple=True)
            for row, col in zip(rows.tolist(), cols.tolist(), strict=True):
                local_region = dilated[row] | dilated[col]
                first = prob[row] * local_region.float()
                second = prob[col] * local_region.float()
                intersection = (first * second).sum()
                union_soft = (first + second - first * second).sum().clamp_min(1e-7)
                conflict_values.append(intersection / union_soft)
            neighbor_pair_count += int(rows.numel())
        foreign_area = foreign.reshape(count, -1).sum(dim=1)
        valid_foreign = foreign_area > 0
        if bool(valid_foreign.any()):
            leakage = (prob * foreign.float()).reshape(count, -1).sum(dim=1)
            leakage_values.extend(
                (leakage[valid_foreign] / foreign_area[valid_foreign].float()).unbind()
            )
        offset = next_offset

    foreign_loss = (
        torch.stack(leakage_values).mean() if leakage_values else _graph_zero(selected_logits)
    )
    conflict_loss = (
        torch.stack(conflict_values).mean() if conflict_values else _graph_zero(selected_logits)
    )
    return foreign_loss + conflict_loss, {
        "valid_prompt_count": int(selected_logits.shape[0]),
        "foreign_valid_prompt_count": len(leakage_values),
        "neighbor_pair_count": neighbor_pair_count,
        "foreign_leakage": float(foreign_loss.detach().cpu()),
        "conflict": float(conflict_loss.detach().cpu()),
    }


def _pairwise_hard_iou(predictions: torch.Tensor, targets: torch.Tensor) -> np.ndarray:
    pred = predictions.bool().reshape(predictions.shape[0], -1).float()
    target = targets.bool().reshape(targets.shape[0], -1).float()
    intersection = pred @ target.T
    pred_area = pred.sum(dim=1, keepdim=True)
    target_area = target.sum(dim=1, keepdim=True).T
    union = pred_area + target_area - intersection
    return (intersection / union.clamp_min(1.0)).detach().cpu().numpy().astype(np.float64)


def _maximum_cardinality_match(iou: np.ndarray, *, threshold: float) -> list[tuple[int, int]]:
    if iou.size == 0:
        return []
    eligible = iou > float(threshold)
    cardinality_bonus = float(min(iou.shape) + 1)
    weights = np.where(eligible, cardinality_bonus + iou, 0.0)
    rows, cols = linear_sum_assignment(-weights)
    return [
        (int(row), int(col))
        for row, col in zip(rows.tolist(), cols.tolist(), strict=True)
        if bool(eligible[row, col])
    ]


def unique_tp_utility_loss(
    selected_logits: torch.Tensor,
    selected_quality: torch.Tensor,
    own_gt_masks: torch.Tensor,
    image_prompt_counts: Iterable[int],
    *,
    match_iou: float = STRICT_MATCH_IOU,
    merge_risk_overlap_fraction: float = 0.1,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Calibrate the existing assembly score to detached unique-TP utility.

    A selected prediction receives a positive target only if the same strict
    one-to-one, maximum-cardinality-then-IoU matching used by the evaluator
    retains it. Unmatched predictions are target zero; those that also match
    a GT already won by another prediction are explicitly counted as
    duplicates. A matched prediction leaking into another GT gets a reduced
    target, so merge risk cannot receive full assembly utility.
    """

    counts = _validate_inputs(
        selected_logits, selected_quality, own_gt_masks, image_prompt_counts
    )
    if not math.isfinite(float(match_iou)) or not 0.0 < float(match_iou) < 1.0:
        raise ValueError("match_iou must be finite and in (0,1)")
    if not math.isfinite(float(merge_risk_overlap_fraction)) or not 0.0 <= float(merge_risk_overlap_fraction) <= 1.0:
        raise ValueError("merge_risk_overlap_fraction must be in [0,1]")

    if selected_logits.shape[0] == 0:
        zero = _graph_zero(selected_quality)
        return zero, {
            "valid_prompt_count": 0,
            "unique_tp_count": 0,
            "unmatched_fp_count": 0,
            "duplicate_count": 0,
            "merge_risk_count": 0,
            "utility_target_mean": 0.0,
            "matched_iou_mean": None,
        }

    with torch.no_grad():
        predicted = selected_logits.detach().float() > 0.0
        targets = own_gt_masks.detach().bool()
        utility_target = torch.zeros(
            (selected_logits.shape[0],), device=selected_logits.device, dtype=torch.float32
        )
        unique_tp_count = 0
        unmatched_fp_count = 0
        duplicate_count = 0
        merge_risk_count = 0
        matched_ious: list[float] = []
        offset = 0
        for count in counts:
            if count == 0:
                continue
            next_offset = offset + count
            pred_chunk = predicted[offset:next_offset]
            gt_chunk = targets[offset:next_offset]
            iou = _pairwise_hard_iou(pred_chunk, gt_chunk)
            pairs = _maximum_cardinality_match(iou, threshold=float(match_iou))
            matched_predictions = {row for row, _ in pairs}
            matched_gt = {col for _, col in pairs}
            for row, col in pairs:
                foreign_fraction = 0.0
                if count > 1:
                    other = torch.arange(count, device=gt_chunk.device) != int(col)
                    if bool(other.any()):
                        foreign = gt_chunk[other].any(dim=0)
                        foreign_fraction = float(
                            (pred_chunk[row] & foreign).sum().float()
                            / foreign.sum().float().clamp_min(1.0)
                        )
                target = max(0.0, 1.0 - foreign_fraction)
                utility_target[offset + row] = target
                unique_tp_count += 1
                matched_ious.append(float(iou[row, col]))
                if foreign_fraction > float(merge_risk_overlap_fraction):
                    merge_risk_count += 1
            for row in range(count):
                if row in matched_predictions:
                    continue
                eligible_gt = set(np.flatnonzero(iou[row] > float(match_iou)).tolist())
                if eligible_gt & matched_gt:
                    duplicate_count += 1
                else:
                    unmatched_fp_count += 1
            offset = next_offset
        utility_target = utility_target.detach()

    loss = (selected_quality.float() - utility_target).pow(2).mean()
    return loss, {
        "valid_prompt_count": int(selected_logits.shape[0]),
        "unique_tp_count": unique_tp_count,
        "unmatched_fp_count": unmatched_fp_count,
        "duplicate_count": duplicate_count,
        "merge_risk_count": merge_risk_count,
        "utility_target_mean": float(utility_target.mean().cpu()),
        "matched_iou_mean": (
            float(np.mean(matched_ious)) if matched_ious else None
        ),
    }


def c2_ar_losses(
    selected_logits: torch.Tensor,
    selected_quality: torch.Tensor,
    own_gt_masks: torch.Tensor,
    image_prompt_counts: Iterable[int],
    *,
    neighbor_radius: int = 2,
    match_iou: float = STRICT_MATCH_IOU,
    merge_risk_overlap_fraction: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Compute the two independent C2-AR raw loss terms."""

    exclusivity, exclusivity_audit = selected_mask_exclusivity_loss(
        selected_logits,
        own_gt_masks,
        image_prompt_counts,
        neighbor_radius=neighbor_radius,
    )
    utility, utility_audit = unique_tp_utility_loss(
        selected_logits,
        selected_quality,
        own_gt_masks,
        image_prompt_counts,
        match_iou=match_iou,
        merge_risk_overlap_fraction=merge_risk_overlap_fraction,
    )
    return exclusivity, utility, {
        "exclusivity": exclusivity_audit,
        "utility": utility_audit,
    }


def compose_c2_ar_total_loss(
    c1_loss: torch.Tensor,
    exclusivity_loss: torch.Tensor,
    utility_loss: torch.Tensor,
    *,
    exclusivity_coefficient: float,
    utility_coefficient: float,
) -> torch.Tensor:
    """Add C2-AR to C1; zero coefficients are exact C1 regression fallback."""

    return (
        c1_loss
        + float(exclusivity_coefficient) * exclusivity_loss
        + float(utility_coefficient) * utility_loss
    )
