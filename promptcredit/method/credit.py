"""Detached-index nearest selection and scaled coordinate credit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass(frozen=True)
class NearestCoordinateSelection:
    coordinates: torch.Tensor
    source_indices: list[torch.Tensor]


def legacy_nearest_indices(predicted_coordinates: torch.Tensor, gt_points: Sequence[torch.Tensor]) -> list[torch.Tensor]:
    """Reference implementation of the historical CPU nearest-index calculation."""
    indices: list[torch.Tensor] = []
    for crop_index, points in enumerate(gt_points):
        source = predicted_coordinates[crop_index].detach().cpu().float()
        targets = points.detach().cpu().reshape(-1, 2).float()
        if len(targets) == 0:
            indices.append(torch.empty(0, dtype=torch.long))
        else:
            indices.append(torch.cdist(source.unsqueeze(0), targets.unsqueeze(0)).squeeze(0).argmin(dim=0))
    return indices


def gather_nearest_coordinates(predicted_coordinates: torch.Tensor, gt_points: Sequence[torch.Tensor]) -> NearestCoordinateSelection:
    """Select nearest indices from detached coordinates and gather from the live tensor.

    The forward coordinate values exactly equal the legacy gathered values.  The
    discrete nearest index is detached, while the gathered coordinates retain
    the original point-head autograd path.
    """
    selected: list[torch.Tensor] = []
    source_indices: list[torch.Tensor] = []
    for crop_index, points in enumerate(gt_points):
        targets = points.reshape(-1, 2).to(predicted_coordinates.device, dtype=torch.float32)
        source = predicted_coordinates[crop_index]
        if len(targets) == 0:
            source_indices.append(torch.empty(0, dtype=torch.long, device=source.device))
            continue
        indices = torch.cdist(source.detach().float().unsqueeze(0), targets.unsqueeze(0)).squeeze(0).argmin(dim=0)
        source_indices.append(indices)
        selected.append(source.index_select(0, indices).unsqueeze(1))
    if selected:
        coordinates = torch.cat(selected, dim=0)
    else:
        coordinates = predicted_coordinates.new_empty((0, 1, 2))
    return NearestCoordinateSelection(coordinates=coordinates, source_indices=source_indices)


def directional_credit(coordinates: torch.Tensor, alpha: float) -> torch.Tensor:
    """Keep coordinate values unchanged while scaling only mask-to-coordinate gradient."""
    if not 0.0 <= float(alpha) <= 0.10:
        raise ValueError("PromptCredit v1 directional alpha must be within [0, 0.10]")
    detached = coordinates.detach()
    return detached + float(alpha) * (coordinates - detached)
