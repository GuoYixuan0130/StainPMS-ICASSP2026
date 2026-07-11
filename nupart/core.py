"""Pure array operations for the NuPart Stage-0 ownership protocol.

These helpers intentionally do not import a model, optimizer, or any of the
terminated NuSet/NuRank methods.  They are also used by the synthetic tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class ConflictEdge:
    left: int
    right: int
    overlap_pixels: int


def distinct_gt_conflicts(masks: np.ndarray, associations: np.ndarray) -> list[ConflictEdge]:
    """Return only hard-mask overlaps between two different non-background GT IDs."""
    hard = np.asarray(masks, dtype=bool)
    target = np.asarray(associations, dtype=np.int64)
    if hard.ndim != 3 or len(hard) != len(target):
        raise ValueError("masks must be [prompt,height,width] with one association each")
    result: list[ConflictEdge] = []
    for left, right in combinations(range(len(hard)), 2):
        if target[left] == 0 or target[right] == 0 or target[left] == target[right]:
            continue
        overlap = int(np.logical_and(hard[left], hard[right]).sum())
        if overlap:
            result.append(ConflictEdge(left, right, overlap))
    return result


def connected_components(node_count: int, edges: Iterable[ConflictEdge]) -> list[list[int]]:
    """Connected components containing at least one distinct-GT conflict edge."""
    parent = list(range(node_count))

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def join(left: int, right: int) -> None:
        left, right = find(left), find(right)
        if left != right:
            parent[right] = left

    materialized = list(edges)
    active: set[int] = set()
    for edge in materialized:
        join(edge.left, edge.right)
        active.update((edge.left, edge.right))
    grouped: dict[int, list[int]] = {}
    for node in sorted(active):
        grouped.setdefault(find(node), []).append(node)
    return sorted((sorted(nodes) for nodes in grouped.values()), key=lambda nodes: (nodes[0], len(nodes)))


def overlap_pixels(masks: np.ndarray) -> np.ndarray:
    return np.asarray(masks, dtype=bool).sum(axis=0) >= 2


def _tie_winner(candidates: np.ndarray, standard_owner: int) -> int:
    if standard_owner in candidates:
        return int(standard_owner)
    return int(candidates.min())


def logit_wta(masks: np.ndarray, logits: np.ndarray, standard_owner: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Resolve only overlap pixels; exact score ties use the Standard owner."""
    hard = np.asarray(masks, dtype=bool)
    score = np.asarray(logits, dtype=np.float32)
    if hard.shape != score.shape:
        raise ValueError("hard masks and logits must have the same shape")
    result, changed = hard.copy(), np.zeros(hard.shape[1:], dtype=bool)
    for y, x in np.argwhere(overlap_pixels(hard)):
        candidates = np.flatnonzero(hard[:, y, x])
        values = score[candidates, y, x]
        maximum = values.max()
        winner = _tie_winner(candidates[values == maximum], int(standard_owner[y, x]))
        result[candidates, y, x] = False
        result[winner, y, x] = True
        changed[y, x] = not np.array_equal(result[:, y, x], hard[:, y, x])
    return result, changed


def nearest_prompt_wta(
    masks: np.ndarray, points_xy: np.ndarray, standard_owner: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve only overlap pixels by Euclidean prompt distance."""
    hard = np.asarray(masks, dtype=bool)
    points = np.asarray(points_xy, dtype=np.float32)
    if points.shape != (len(hard), 2):
        raise ValueError("points_xy must be [prompt,2]")
    result, changed = hard.copy(), np.zeros(hard.shape[1:], dtype=bool)
    for y, x in np.argwhere(overlap_pixels(hard)):
        candidates = np.flatnonzero(hard[:, y, x])
        distances = ((points[candidates] - np.asarray((x, y), dtype=np.float32)) ** 2).sum(axis=1)
        winner = _tie_winner(candidates[distances == distances.min()], int(standard_owner[y, x]))
        result[candidates, y, x] = False
        result[winner, y, x] = True
        changed[y, x] = not np.array_equal(result[:, y, x], hard[:, y, x])
    return result, changed


def gt_ownership_oracle(
    masks: np.ndarray,
    associations: np.ndarray,
    instance_map: np.ndarray,
    standard_owner: np.ndarray,
    *,
    allowed_gt_ids: set[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the preregistered oracle only at authorized distinct-GT overlaps.

    ``allowed_gt_ids`` is solely for the required touching/non-touching
    decomposition.  Passing ``None`` is the full oracle.
    """
    hard = np.asarray(masks, dtype=bool)
    target = np.asarray(associations, dtype=np.int64)
    truth = np.asarray(instance_map, dtype=np.int64)
    if hard.shape[1:] != truth.shape:
        raise ValueError("oracle masks and GT map must share a spatial shape")
    result, changed = hard.copy(), np.zeros(truth.shape, dtype=bool)
    authorized = np.zeros(truth.shape, dtype=bool)
    for y, x in np.argwhere(overlap_pixels(hard)):
        candidates = np.flatnonzero(hard[:, y, x])
        distinct = {int(target[item]) for item in candidates if target[item] != 0}
        if len(distinct) < 2:
            continue
        gt_id = int(truth[y, x])
        if gt_id == 0 or gt_id not in distinct or (allowed_gt_ids is not None and gt_id not in allowed_gt_ids):
            continue
        correct = candidates[target[candidates] == gt_id]
        if not len(correct):
            continue
        # Same-GT duplicates are never independently "improved".  If their
        # Standard winner is correct retain it; otherwise take a fixed lowest
        # candidate index only to select the already-authorized GT owner.
        winner = _tie_winner(correct, int(standard_owner[y, x]))
        result[candidates, y, x] = False
        result[winner, y, x] = True
        changed[y, x] = not np.array_equal(result[:, y, x], hard[:, y, x])
        authorized[y, x] = True
    return result, changed, authorized


def foreground_dice(truth: np.ndarray, prediction: np.ndarray) -> float:
    left, right = np.asarray(truth) > 0, np.asarray(prediction) > 0
    denominator = int(left.sum() + right.sum())
    return 1.0 if denominator == 0 else float(2 * np.logical_and(left, right).sum() / denominator)
