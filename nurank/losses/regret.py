"""Fixed calibration plus pairwise regret-margin loss for NuRank."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def regret_aware_loss(scores: torch.Tensor, target_iou: torch.Tensor, *, tie_epsilon: float = 1e-3) -> dict[str, torch.Tensor]:
    """Return fixed 1:1 SmoothL1 calibration and groupwise regret-margin losses."""
    if scores.shape != target_iou.shape or scores.ndim != 2 or scores.size(1) != 4:
        raise ValueError("NuRank loss requires [groups,4] scores and true IoUs")
    calibration = F.smooth_l1_loss(scores, target_iou, reduction="mean")
    pair_losses: list[torch.Tensor] = []
    for left in range(4):
        for right in range(left + 1, 4):
            difference = target_iou[:, left] - target_iou[:, right]
            valid = difference.abs() >= tie_epsilon
            if valid.any():
                predicted_difference = scores[:, left] - scores[:, right]
                pair_losses.append(F.relu(difference[valid].abs() - difference[valid].sign() * predicted_difference[valid]))
    ranking = torch.cat(pair_losses).mean() if pair_losses else scores.sum() * 0.0
    return {"total": calibration + ranking, "calibration": calibration, "ranking": ranking, "valid_pair_count": torch.as_tensor(sum(loss.numel() for loss in pair_losses), device=scores.device)}
