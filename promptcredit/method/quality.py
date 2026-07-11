"""Scalar mask-utility credit and inference ranking scores."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class QualityTargets:
    values: torch.Tensor
    matched_proposals: torch.Tensor
    matched_count: int
    duplicate_source_events: int


def utility_target_from_hard_iou(hard_iou: torch.Tensor) -> torch.Tensor:
    """Threshold-aware detached utility target fixed by the project lead."""
    values = hard_iou.detach().to(torch.float32)
    return values * torch.sigmoid((values - 0.5) / 0.1)


def build_quality_targets(
    quality_logits: torch.Tensor,
    source_indices: Sequence[torch.Tensor],
    selected_hard_iou: torch.Tensor,
) -> QualityTargets:
    """Map selected prompts to proposal targets; duplicate sources receive max utility."""
    if quality_logits.ndim != 2:
        raise ValueError("quality_logits must have shape [batch, proposals]")
    targets = torch.zeros_like(quality_logits)
    matched = torch.zeros_like(quality_logits, dtype=torch.bool)
    offset = 0
    duplicate_events = 0
    for batch_index, indices in enumerate(source_indices):
        count = int(indices.numel())
        utilities = utility_target_from_hard_iou(selected_hard_iou[offset:offset + count])
        offset += count
        seen: dict[int, float] = {}
        for local_index, source_index in enumerate(indices.detach().cpu().tolist()):
            utility = float(utilities[local_index].detach().cpu())
            if source_index in seen:
                duplicate_events += 1
            seen[source_index] = max(seen.get(source_index, 0.0), utility)
        for source_index, utility in seen.items():
            targets[batch_index, source_index] = utility
            matched[batch_index, source_index] = True
    if offset != int(selected_hard_iou.numel()):
        raise ValueError("selected_hard_iou count does not match source indices")
    return QualityTargets(
        values=targets,
        matched_proposals=matched,
        matched_count=int(matched.sum().item()),
        duplicate_source_events=duplicate_events,
    )


def quality_focal_loss(logits: torch.Tensor, targets: QualityTargets, gamma: float = 2.0) -> torch.Tensor:
    """Quality Focal Loss with balanced unmatched-proposal contribution."""
    if gamma != 2.0:
        raise ValueError("PromptCredit v1 Quality Focal Loss gamma is frozen to 2")
    probability = torch.sigmoid(logits)
    cross_entropy = F.binary_cross_entropy_with_logits(logits, targets.values, reduction="none")
    losses = (targets.values - probability).abs().pow(gamma) * cross_entropy
    positive = targets.matched_proposals
    negative = ~positive
    positive_count = max(int(positive.sum().item()), 1)
    negative_count = int(negative.sum().item())
    positive_loss = losses[positive].sum()
    # Scale the large unmatched set to one positive-set equivalent before division.
    negative_loss = losses[negative].sum() * (positive_count / max(negative_count, 1))
    return (positive_loss + negative_loss) / positive_count


def prompt_ranking_scores(
    foreground_probability: torch.Tensor,
    quality_logits: torch.Tensor | None,
    mode: str,
) -> torch.Tensor:
    """Return only the ranking score; class decisions and mask assembly are untouched."""
    if mode == "objectness":
        return foreground_probability
    if quality_logits is None:
        raise ValueError(f"prompt_score_mode={mode} requires a PromptCredit quality head")
    quality_probability = torch.sigmoid(quality_logits)
    if mode == "objectness_x_quality":
        return foreground_probability * quality_probability
    if mode == "quality":
        return quality_probability
    raise ValueError(f"Unknown prompt score mode: {mode}")

