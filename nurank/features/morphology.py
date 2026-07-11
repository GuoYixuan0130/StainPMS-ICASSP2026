"""Token morphology features derived from already-produced mask logits."""

from __future__ import annotations

import torch


MORPHOLOGY_NAMES = (
    "original_predicted_iou",
    "soft_foreground_area_fraction",
    "hard_foreground_area_fraction",
    "stability_score",
    "mean_sigmoid_probability",
    "mean_absolute_logit",
    "boundary_band_entropy",
    "positive_point_inside_hard_mask",
)


def morphology_features(logits: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
    """Return seven non-GT scalars for [N,4,H,W] logits and [N,1,2] points."""
    if logits.ndim != 4 or logits.size(1) != 4 or coordinates.shape != (logits.size(0), 1, 2):
        raise ValueError("NuRank morphology requires [N,4,H,W] logits and [N,1,2] coordinates")
    probability = torch.sigmoid(logits)
    soft_area = probability.mean(dim=(-1, -2))
    hard = logits > 0
    hard_area = hard.float().mean(dim=(-1, -2))
    stable_inner = (logits > 1.0).float().sum(dim=(-1, -2))
    stable_outer = (logits > -1.0).float().sum(dim=(-1, -2))
    stability = torch.where(stable_outer > 0, stable_inner / stable_outer, torch.ones_like(stable_outer))
    mean_probability = probability.mean(dim=(-1, -2))
    mean_abs_logit = logits.abs().mean(dim=(-1, -2))
    entropy = -(probability.clamp(1e-6, 1 - 1e-6) * probability.clamp(1e-6, 1 - 1e-6).log() + (1 - probability).clamp(1e-6, 1 - 1e-6) * (1 - probability).clamp(1e-6, 1 - 1e-6).log())
    band = logits.abs() <= 1.0
    boundary_entropy = torch.where(
        band.any(dim=(-1, -2)),
        (entropy * band).sum(dim=(-1, -2)) / band.sum(dim=(-1, -2)).clamp_min(1),
        torch.zeros_like(mean_probability),
    )
    x = coordinates[:, 0, 0].trunc().long().clamp(0, logits.shape[-1] - 1)
    y = coordinates[:, 0, 1].trunc().long().clamp(0, logits.shape[-2] - 1)
    point_inside = hard[torch.arange(logits.size(0), device=logits.device)[:, None], torch.arange(4, device=logits.device)[None, :], y[:, None], x[:, None]].float()
    return torch.stack((soft_area, hard_area, stability, mean_probability, mean_abs_logit, boundary_entropy, point_inside), dim=-1)
