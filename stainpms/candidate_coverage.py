"""Frozen native-candidate losses for Phase 2A warm-start feasibility.

This module is deliberately independent of model and dataset code.  It
implements the approved per-prompt, within-view four-candidate soft minimum
and the detached hard-IoU quality target.  All reductions that are sensitive
to AMP are performed in FP32.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch
import torch.nn.functional as F


def stable_softmin(losses: torch.Tensor, temperature: float) -> torch.Tensor:
    """Return a normalized soft minimum over the last dimension in FP32."""
    if losses.ndim < 1 or losses.shape[-1] < 1:
        raise ValueError("softmin requires at least one candidate")
    if not math.isfinite(float(temperature)) or float(temperature) <= 0:
        raise ValueError("softmin temperature must be finite and positive")
    values = losses.float()
    tau = float(temperature)
    return -tau * (
        torch.logsumexp(-values / tau, dim=-1)
        - math.log(int(values.shape[-1]))
    )


def softmin_gradient_weights(
    losses: torch.Tensor, temperature: float
) -> torch.Tensor:
    """Analytic d softmin / d candidate-loss weights, useful for audits."""
    if losses.ndim < 1 or losses.shape[-1] < 1:
        raise ValueError("softmin requires at least one candidate")
    if not math.isfinite(float(temperature)) or float(temperature) <= 0:
        raise ValueError("softmin temperature must be finite and positive")
    return torch.softmax(-losses.float() / float(temperature), dim=-1)


def _per_prompt_dice_and_focal(
    candidate_logits: torch.Tensor,
    gt_masks: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized single-prompt equivalents of the existing toolbelt losses.

    Args:
        candidate_logits: ``[N,K,H,W]`` raw mask logits.
        gt_masks: ``[N,H,W]`` binary masks.

    Returns:
        Dice and binary focal losses with shape ``[N,K]``.  Each entry is
        exactly the scalar obtained by applying the existing losses to one
        prompt/candidate pair (DiceLoss binary defaults; BinaryFocalLoss
        alpha=None, gamma=2, reduction=mean defaults).
    """
    logits = candidate_logits.float()
    targets = gt_masks.float().unsqueeze(1).expand_as(logits)

    probabilities = F.logsigmoid(logits).exp()
    intersection = (probabilities * targets).sum(dim=(-2, -1))
    cardinality = (probabilities + targets).sum(dim=(-2, -1))
    dice_score = (2.0 * intersection) / cardinality.clamp_min(1e-7)
    dice_loss = 1.0 - dice_score
    nonempty = targets.sum(dim=(-2, -1)) > 0
    dice_loss = dice_loss * nonempty.to(dice_loss.dtype)

    probabilities_focal = torch.sigmoid(logits)
    cross_entropy = F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    )
    pt = probabilities_focal * targets + (1.0 - probabilities_focal) * (
        1.0 - targets
    )
    focal_loss = ((1.0 - pt).pow(2.0) * cross_entropy).mean(dim=(-2, -1))
    return dice_loss, focal_loss


def candidate_prompt_losses(
    candidate_logits: torch.Tensor,
    quality_predictions: torch.Tensor,
    gt_masks: torch.Tensor,
    *,
    temperature: float = 0.1,
    collect_softmin_audit: bool = True,
) -> dict[str, torch.Tensor]:
    """Compute frozen per-prompt candidate coverage and quality losses."""
    if candidate_logits.ndim != 4:
        raise ValueError("candidate logits must have shape [N,K,H,W]")
    n_prompts, n_candidates, height, width = candidate_logits.shape
    if quality_predictions.shape != (n_prompts, n_candidates):
        raise ValueError("quality predictions must have shape [N,K]")
    if gt_masks.shape != (n_prompts, height, width):
        raise ValueError("GT masks must have shape [N,H,W]")

    device_type = candidate_logits.device.type
    with torch.amp.autocast(device_type=device_type, enabled=False):
        dice, focal = _per_prompt_dice_and_focal(candidate_logits, gt_masks)
        segmentation = (20.0 * dice + focal) / 21.0
        coverage = stable_softmin(segmentation, temperature)
        gradient_weights = (
            softmin_gradient_weights(segmentation, temperature)
            if collect_softmin_audit
            else None
        )

        with torch.no_grad():
            predicted = candidate_logits.detach().float() > 0.0
            target = gt_masks.detach().bool().unsqueeze(1).expand_as(predicted)
            intersection = (predicted & target).sum(dim=(-2, -1)).float()
            union = (predicted | target).sum(dim=(-2, -1)).float()
            hard_iou_target = intersection / union.clamp_min(1.0)
        hard_iou_target = hard_iou_target.detach()
        quality_error = (
            quality_predictions.float() - hard_iou_target
        ).pow(2.0)
        quality = quality_error.mean(dim=-1)

    result = {
        "dice_per_candidate": dice,
        "focal_per_candidate": focal,
        "segmentation_per_candidate": segmentation,
        "coverage_per_prompt": coverage,
        "hard_iou_target": hard_iou_target,
        "quality_per_prompt": quality,
    }
    if gradient_weights is not None:
        result["softmin_gradient_weights"] = gradient_weights
    return result


def aggregate_candidate_prompt_groups(
    groups: Mapping[str, Mapping[str, Any]],
    *,
    temperature: float = 0.1,
    collect_audit: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, dict[str, Any]]]:
    """Average valid prompts within each group, then apply frozen weights.

    Each group must contain ``candidate_logits``, ``quality_predictions``,
    ``gt_masks``, and scalar ``alpha``.  Empty groups contribute exact graph-
    connected zeros and never enter another group's denominator.
    """
    coverage_total = None
    quality_total = None
    audit: dict[str, dict[str, Any]] = {}
    for name, group in groups.items():
        logits = group["candidate_logits"]
        quality_predictions = group["quality_predictions"]
        gt_masks = group["gt_masks"]
        alpha = float(group["alpha"])
        if logits.ndim != 4:
            raise ValueError(f"{name}: candidate logits must be [N,K,H,W]")
        count = int(logits.shape[0])
        if count == 0:
            zero = logits.sum().float() * 0.0 + quality_predictions.sum().float() * 0.0
            coverage_mean = zero
            quality_mean = zero
            best_weight_mean = None
            effective_candidate_count_mean = None
        else:
            target_area = gt_masks.reshape(count, -1).sum(dim=1)
            if not bool(torch.all(target_area > 0)):
                raise ValueError(f"{name}: positive prompt has empty GT assignment")
            result = candidate_prompt_losses(
                logits,
                quality_predictions,
                gt_masks,
                temperature=temperature,
                collect_softmin_audit=collect_audit,
            )
            coverage_mean = result["coverage_per_prompt"].mean()
            quality_mean = result["quality_per_prompt"].mean()
            if collect_audit:
                weights = result["softmin_gradient_weights"].detach()
                best_weight_mean = float(weights.max(dim=1).values.mean().cpu())
                effective_candidate_count_mean = float(
                    (1.0 / weights.pow(2).sum(dim=1)).mean().cpu()
                )
            else:
                best_weight_mean = None
                effective_candidate_count_mean = None
        weighted_coverage = coverage_mean * alpha
        weighted_quality = quality_mean * alpha
        coverage_total = (
            weighted_coverage
            if coverage_total is None
            else coverage_total + weighted_coverage
        )
        quality_total = (
            weighted_quality
            if quality_total is None
            else quality_total + weighted_quality
        )
        if collect_audit:
            audit[name] = {
                "valid_prompt_count": count,
                "alpha": alpha,
                "coverage_mean": float(coverage_mean.detach().cpu()),
                "quality_mean": float(quality_mean.detach().cpu()),
                "weighted_coverage": float(weighted_coverage.detach().cpu()),
                "weighted_quality": float(weighted_quality.detach().cpu()),
                "best_softmin_gradient_weight_mean": best_weight_mean,
                "effective_candidate_count_mean": effective_candidate_count_mean,
            }
    if coverage_total is None or quality_total is None:
        raise ValueError("at least one declared prompt group is required")
    return coverage_total, quality_total, audit


def compose_candidate_total_loss(
    stainpms_loss: torch.Tensor,
    coverage_loss: torch.Tensor,
    quality_loss: torch.Tensor,
    *,
    coverage_coefficient: float,
    quality_coefficient: float,
) -> torch.Tensor:
    """Compose C1; zero coefficients are an exact algebraic C0 fallback."""
    return (
        stainpms_loss
        + float(coverage_coefficient) * coverage_loss
        + float(quality_coefficient) * quality_loss
    )
