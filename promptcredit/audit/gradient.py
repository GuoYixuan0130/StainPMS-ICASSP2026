"""Independent coordinate-gradient diagnostic used only by PromptCredit Audit C."""

from __future__ import annotations

from typing import Any, Callable, Iterable

import torch
import torch.nn.functional as F


def freeze_parameters_and_clear_gradients(modules: Iterable[torch.nn.Module]) -> list[torch.nn.Parameter]:
    """Freeze diagnostic modules and clear stale gradients before autograd."""
    parameters: list[torch.nn.Parameter] = []
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
            parameters.append(parameter)
    return parameters


def focal_dice_per_prompt(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-prompt binary focal (gamma=2) plus soft Dice loss."""
    if logits.ndim == 4 and logits.shape[1] == 1:
        logits = logits[:, 0]
    if target.ndim == 4 and target.shape[1] == 1:
        target = target[:, 0]
    if logits.shape != target.shape or logits.ndim != 3:
        raise ValueError("logits and target must have matching [N, H, W] shapes")
    target = target.float()
    probability = torch.sigmoid(logits)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    pt = probability * target + (1.0 - probability) * (1.0 - target)
    focal = ((1.0 - pt).pow(2.0) * bce).mean(dim=(1, 2))
    intersection = (probability * target).sum(dim=(1, 2))
    dice = 1.0 - (2.0 * intersection + 1.0) / (
        probability.sum(dim=(1, 2)) + target.sum(dim=(1, 2)) + 1.0
    )
    return focal + dice


def _hard_iou(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    hard = logits > 0
    truth = target.bool()
    intersection = (hard & truth).sum(dim=(1, 2)).float()
    union = (hard | truth).sum(dim=(1, 2)).float()
    return torch.where(union > 0, intersection / union, torch.ones_like(union))


def coordinate_gradient_probe(
    decode_logits: Callable[[torch.Tensor], torch.Tensor],
    coordinates: torch.Tensor,
    target_masks: torch.Tensor,
    *,
    width: int,
    height: int,
    frozen_parameters: Iterable[torch.nn.Parameter],
    eta_pixels: float = 1.0,
) -> dict[str, Any]:
    """Probe a frozen decoder's gradient with respect to prompt coordinates.

    The callback must contain no ``torch.no_grad`` scope.  This function does
    not update a network parameter; the second decode is the Stage 0-only
    one-pixel local-actionability diagnostic.
    """
    if eta_pixels != 1.0:
        raise ValueError("Stage 0 eta is frozen to one pixel")
    if coordinates.ndim != 3 or coordinates.shape[1:] != (1, 2):
        raise ValueError("coordinates must have shape [N, 1, 2]")
    params = list(frozen_parameters)
    for parameter in params:
        parameter.grad = None
    coord = coordinates.detach().clone().requires_grad_(True)
    original_logits = decode_logits(coord)
    original_loss = focal_dice_per_prompt(original_logits, target_masks)
    gradient = torch.autograd.grad(original_loss.sum(), coord, allow_unused=False)[0]
    gradient_norm = gradient.flatten(1).norm(dim=1)
    finite = torch.isfinite(gradient).all(dim=(1, 2)) & torch.isfinite(gradient_norm)
    nonzero = finite & (gradient_norm > 0)
    normalized = gradient / gradient_norm.clamp_min(torch.finfo(gradient.dtype).eps).view(-1, 1, 1)
    moved = coord.detach() - eta_pixels * normalized.detach()
    moved[..., 0].clamp_(0.0, float(width - 1))
    moved[..., 1].clamp_(0.0, float(height - 1))
    with torch.no_grad():
        moved_logits = decode_logits(moved)
        moved_loss = focal_dice_per_prompt(moved_logits, target_masks)
        original_iou = _hard_iou(original_logits.detach(), target_masks)
        moved_iou = _hard_iou(moved_logits, target_masks)
    return {
        "coordinate_gradient": gradient.detach(),
        "gradient_norm": gradient_norm.detach(),
        "finite": finite.detach(),
        "nonzero": nonzero.detach(),
        "original_loss": original_loss.detach(),
        "moved_loss": moved_loss.detach(),
        "original_iou": original_iou.detach(),
        "moved_iou": moved_iou.detach(),
        "moved_coordinates": moved.detach(),
        "frozen_parameter_grads_none": all(parameter.grad is None for parameter in params),
    }

