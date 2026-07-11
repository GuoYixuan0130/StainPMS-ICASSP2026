"""Pure assignment functions used by PromptCredit Audit A.

`nearest_assignment` intentionally duplicates the independent nearest-proposal
selection in ``run.run_on_epoch.find_nearest_points``.  `hungarian_assignment`
duplicates the cost direction in ``HungarianMatcher`` without altering that
production matcher.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class AssignmentResult:
    source_for_gt: np.ndarray
    distance_for_gt: np.ndarray


def _as_coords(coords: np.ndarray | Iterable[Iterable[float]], name: str) -> np.ndarray:
    array = np.asarray(coords, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"{name} must have shape [N, 2]")
    return array


def _pairwise_distance(proposals: np.ndarray, gt_centroids: np.ndarray) -> np.ndarray:
    return np.linalg.norm(proposals[:, None, :] - gt_centroids[None, :, :], axis=-1)


def nearest_assignment(
    proposal_coords: np.ndarray | Iterable[Iterable[float]],
    gt_centroids: np.ndarray | Iterable[Iterable[float]],
) -> AssignmentResult:
    """Assign each GT independently to its nearest proposal (collisions allowed)."""
    proposals = _as_coords(proposal_coords, "proposal_coords")
    centroids = _as_coords(gt_centroids, "gt_centroids")
    if len(proposals) == 0:
        return AssignmentResult(
            source_for_gt=np.full(len(centroids), -1, dtype=np.int64),
            distance_for_gt=np.full(len(centroids), np.inf, dtype=np.float64),
        )
    distances = _pairwise_distance(proposals, centroids)
    sources = np.argmin(distances, axis=0).astype(np.int64)
    return AssignmentResult(sources, distances[sources, np.arange(len(centroids))])


def hungarian_assignment(
    proposal_coords: np.ndarray | Iterable[Iterable[float]],
    gt_centroids: np.ndarray | Iterable[Iterable[float]],
    foreground_probability: np.ndarray | Iterable[float],
    *,
    cost_point: float = 0.1,
    cost_class: float = 1.0,
) -> AssignmentResult:
    """Reproduce ``HungarianMatcher``'s proposal-by-GT cost and orientation."""
    proposals = _as_coords(proposal_coords, "proposal_coords")
    centroids = _as_coords(gt_centroids, "gt_centroids")
    probabilities = np.asarray(foreground_probability, dtype=np.float64).reshape(-1)
    if len(probabilities) != len(proposals):
        raise ValueError("foreground_probability must have one value per proposal")
    sources = np.full(len(centroids), -1, dtype=np.int64)
    assigned_distances = np.full(len(centroids), np.inf, dtype=np.float64)
    if len(proposals) == 0 or len(centroids) == 0:
        return AssignmentResult(sources, assigned_distances)
    distances = _pairwise_distance(proposals, centroids)
    cost = float(cost_point) * distances - float(cost_class) * probabilities[:, None]
    source_indices, gt_indices = linear_sum_assignment(cost)
    sources[gt_indices] = source_indices
    assigned_distances[gt_indices] = distances[source_indices, gt_indices]
    return AssignmentResult(sources, assigned_distances)


def collision_groups(source_for_gt: np.ndarray | Iterable[int]) -> dict[int, list[int]]:
    """Return only sources independently selected by two or more GT instances."""
    groups: dict[int, list[int]] = {}
    for gt_index, source_index in enumerate(np.asarray(source_for_gt, dtype=np.int64)):
        if source_index >= 0:
            groups.setdefault(int(source_index), []).append(int(gt_index))
    return {source: gt_indices for source, gt_indices in groups.items() if len(gt_indices) > 1}


def point_inside_mask(mask: np.ndarray, point_xy: np.ndarray | Iterable[float]) -> bool:
    """Use nearest-pixel coordinates and return false for an out-of-crop point."""
    instance_mask = np.asarray(mask, dtype=bool)
    point = np.asarray(point_xy, dtype=np.float64).reshape(2)
    x, y = np.rint(point).astype(np.int64)
    return bool(0 <= y < instance_mask.shape[0] and 0 <= x < instance_mask.shape[1] and instance_mask[y, x])

