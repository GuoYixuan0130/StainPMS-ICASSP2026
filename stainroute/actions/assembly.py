"""GT-free ADD/SPLIT instance-map assembly with explicit rejection reasons."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .schema import Point


@dataclass(frozen=True)
class AssemblyResult:
    prediction: np.ndarray
    applied: bool
    reason: str
    details: dict[str, float | int]


@dataclass(frozen=True)
class SplitAssemblyConfig:
    min_child_area: int = 8
    min_parent_coverage: float = 0.20
    max_raw_child_iou: float = 0.90


def _instance_count(prediction: np.ndarray) -> int:
    return int(np.count_nonzero(np.unique(prediction)))


def _mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    intersection = int(np.count_nonzero(first & second))
    union = int(np.count_nonzero(first | second))
    return float(intersection / union) if union else 0.0


def apply_add_action(
    prediction: np.ndarray,
    decoded_mask: np.ndarray,
    *,
    min_added_area: int = 8,
) -> AssemblyResult:
    """Insert only previously uncovered decoded pixels as one new instance."""

    prediction = np.asarray(prediction)
    decoded_mask = np.asarray(decoded_mask, dtype=bool)
    if prediction.shape != decoded_mask.shape:
        raise ValueError(f"prediction/mask shape mismatch: {prediction.shape} != {decoded_mask.shape}")
    candidate_area = int(decoded_mask.sum())
    added = decoded_mask & (prediction == 0)
    added_area = int(added.sum())
    overlap_area = candidate_area - added_area
    details = {
        "decoded_area": candidate_area,
        "added_area": added_area,
        "overlap_area": overlap_area,
        "overlap_fraction": float(overlap_area / max(1, candidate_area)),
    }
    if added_area < min_added_area:
        return AssemblyResult(prediction.copy(), False, "min_added_area", details)
    output = prediction.copy()
    output[added] = int(output.max()) + 1
    return AssemblyResult(output, True, "applied", details)


def _resolve_child_overlap(
    first: np.ndarray,
    second: np.ndarray,
    first_point: Point,
    second_point: Point,
) -> tuple[np.ndarray, np.ndarray]:
    overlap = first & second
    if not np.any(overlap):
        return first, second
    ys, xs = np.nonzero(overlap)
    first_distance = (xs - first_point.x) ** 2 + (ys - first_point.y) ** 2
    second_distance = (xs - second_point.x) ** 2 + (ys - second_point.y) ** 2
    first_wins = first_distance <= second_distance
    first = first.copy()
    second = second.copy()
    first[ys[~first_wins], xs[~first_wins]] = False
    second[ys[first_wins], xs[first_wins]] = False
    return first, second


def apply_split_action(
    prediction: np.ndarray,
    *,
    parent_id: int,
    child_first: np.ndarray,
    child_second: np.ndarray,
    first_point: Point,
    second_point: Point,
    config: SplitAssemblyConfig = SplitAssemblyConfig(),
) -> AssemblyResult:
    """Replace one parent with two mutually exclusive child instances."""

    prediction = np.asarray(prediction)
    child_first = np.asarray(child_first, dtype=bool)
    child_second = np.asarray(child_second, dtype=bool)
    if prediction.shape != child_first.shape or prediction.shape != child_second.shape:
        raise ValueError("prediction and child masks must have equal shape")
    parent = prediction == int(parent_id)
    parent_area = int(parent.sum())
    if parent_area == 0:
        return AssemblyResult(prediction.copy(), False, "missing_parent", {"parent_id": int(parent_id)})

    first = child_first & parent
    second = child_second & parent
    raw_iou = _mask_iou(first, second)
    raw_first_area = int(first.sum())
    raw_second_area = int(second.sum())
    details: dict[str, float | int] = {
        "parent_id": int(parent_id),
        "parent_area": parent_area,
        "raw_child_first_area": raw_first_area,
        "raw_child_second_area": raw_second_area,
        "raw_child_iou": raw_iou,
    }
    if raw_iou >= config.max_raw_child_iou:
        return AssemblyResult(prediction.copy(), False, "near_identical_children", details)

    first, second = _resolve_child_overlap(first, second, first_point, second_point)
    first_area = int(first.sum())
    second_area = int(second.sum())
    details["child_first_area"] = first_area
    details["child_second_area"] = second_area
    details["overlap_after_assignment"] = int((first & second).sum())
    if first_area < config.min_child_area or second_area < config.min_child_area:
        return AssemblyResult(prediction.copy(), False, "min_child_area", details)
    covered = first | second
    coverage = float(covered.sum() / parent_area)
    details["parent_coverage"] = coverage
    if coverage < config.min_parent_coverage:
        return AssemblyResult(prediction.copy(), False, "insufficient_parent_coverage", details)

    before_count = _instance_count(prediction)
    output = prediction.copy()
    output[parent] = 0
    next_id = int(output.max()) + 1
    output[first] = next_id
    output[second] = next_id + 1
    after_count = _instance_count(output)
    details["instance_count_before"] = before_count
    details["instance_count_after"] = after_count
    if after_count != before_count + 1:
        return AssemblyResult(prediction.copy(), False, "invalid_instance_count_delta", details)
    return AssemblyResult(output, True, "applied", details)
