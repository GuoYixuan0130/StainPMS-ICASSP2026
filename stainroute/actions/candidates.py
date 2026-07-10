"""Deterministic ADD/SPLIT candidates generated without ground truth."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations

import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.color import rgb2hed
from skimage.feature import peak_local_max
from skimage.filters import gaussian
from skimage.measure import label, regionprops
from skimage.morphology import binary_dilation, disk
from skimage.segmentation import watershed

from stainroute.utils import canonical_json_sha256

from .schema import ActionCandidate, ActionType, Point


def _robust_normalize(values: np.ndarray, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    lo, hi = np.percentile(values, [low, high])
    if hi <= lo:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - lo) / (hi - lo)).clip(0.0, 1.0).astype(np.float32)


def hematoxylin_evidence(image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Return a deterministic H-channel evidence map without GT input."""

    rgb = np.asarray(image)[..., :3].astype(np.float32)
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"Expected HxWx3 image, got {rgb.shape}")
    if rgb.max(initial=0.0) > 1.5:
        rgb = rgb / 255.0
    evidence = _robust_normalize(rgb2hed(rgb.clip(0.0, 1.0))[..., 0])
    if sigma > 0:
        evidence = gaussian(evidence, sigma=float(sigma), preserve_range=True)
    return np.asarray(evidence, dtype=np.float32).clip(0.0, 1.0)


def _box(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _local_stats(values: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    selected = np.asarray(values)[mask]
    if selected.size == 0:
        return {"mean": 0.0, "std": 0.0, "max": 0.0}
    return {"mean": float(selected.mean()), "std": float(selected.std()), "max": float(selected.max())}


@dataclass(frozen=True)
class AddCandidateConfig:
    h_sigma: float = 1.0
    coverage_dilation_radius: int = 5
    residual_percentile: float = 80.0
    min_component_area: int = 8
    min_peak_distance: int = 12
    max_candidates: int = 20
    generator_version: str = "stainroute-add-v1"


@dataclass(frozen=True)
class SplitCandidateConfig:
    h_sigma: float = 1.0
    min_parent_area: int = 64
    min_peak_distance: int = 12
    min_normalized_peak_distance: float = 0.20
    max_peaks_per_parent: int = 4
    max_pairs_per_parent: int = 3
    min_peak_height_ratio: float = 0.50
    min_valley_depth: float = 0.20
    generator_version: str = "stainroute-split-v1"


def _action_hash(config: object) -> str:
    return canonical_json_sha256(asdict(config))


def _line_values(image: np.ndarray, p1: Point, p2: Point) -> np.ndarray:
    steps = max(abs(p1.x - p2.x), abs(p1.y - p2.y)) + 1
    xs = np.rint(np.linspace(p1.x, p2.x, steps)).astype(int)
    ys = np.rint(np.linspace(p1.y, p2.y, steps)).astype(int)
    return image[ys, xs]


def _distance_basin_features(mask: np.ndarray, p1: Point, p2: Point) -> dict[str, float]:
    """Compute deterministic distance-transform basins from the two H peaks."""

    distance = distance_transform_edt(mask)
    markers = np.zeros(mask.shape, dtype=np.int32)
    markers[p1.y, p1.x] = 1
    markers[p2.y, p2.x] = 2
    basins = watershed(-distance, markers=markers, mask=mask)
    first = int(np.count_nonzero(basins == 1))
    second = int(np.count_nonzero(basins == 2))
    return {
        "distance_basin_area_ratio": float(min(first, second) / max(1, max(first, second))),
        "distance_transform_at_peak_1": float(distance[p1.y, p1.x]),
        "distance_transform_at_peak_2": float(distance[p2.y, p2.x]),
        "distance_transform_max": float(distance.max()) if distance.size else 0.0,
    }


def generate_add_candidates(
    image: np.ndarray,
    prediction: np.ndarray,
    *,
    image_id: str,
    config: AddCandidateConfig = AddCandidateConfig(),
) -> list[ActionCandidate]:
    """Generate residual H-component ADD candidates without GT access."""

    prediction = np.asarray(prediction)
    evidence = hematoxylin_evidence(image, config.h_sigma)
    if prediction.shape != evidence.shape:
        raise ValueError(f"prediction/image shape mismatch: {prediction.shape} != {evidence.shape}")
    covered = prediction > 0
    if config.coverage_dilation_radius > 0:
        covered = binary_dilation(covered, footprint=disk(config.coverage_dilation_radius))
    residual = evidence.copy()
    residual[covered] = 0.0
    positive_values = residual[residual > 0]
    if positive_values.size == 0:
        return []
    threshold = float(np.percentile(positive_values, config.residual_percentile))
    components = label(residual >= threshold, connectivity=2)
    candidates: list[tuple[float, int, int, np.ndarray]] = []
    for component in regionprops(components):
        component_mask = components == component.label
        area = int(component.area)
        if area < config.min_component_area:
            continue
        ys, xs = np.nonzero(component_mask)
        scores = residual[ys, xs]
        best = int(np.argmax(scores))
        candidates.append((float(scores[best]), int(ys[best]), int(xs[best]), component_mask))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))

    selected: list[tuple[float, int, int, np.ndarray]] = []
    for candidate in candidates:
        _, y, x, _ = candidate
        if all((x - old_x) ** 2 + (y - old_y) ** 2 >= config.min_peak_distance**2 for _, old_y, old_x, _ in selected):
            selected.append(candidate)
        if len(selected) >= config.max_candidates:
            break

    config_hash = _action_hash(config)
    output: list[ActionCandidate] = []
    for rank, (score, y, x, component_mask) in enumerate(selected):
        support_box = _box(component_mask)
        output.append(
            ActionCandidate(
                action_id=f"{image_id}:ADD:{rank:03d}",
                image_id=image_id,
                action_type=ActionType.ADD,
                affected_instance_ids=(),
                positive_points=(Point(x=x, y=y),),
                negative_points=(),
                action_cost=1,
                generation_features={
                    "candidate_rank": rank,
                    "h_evidence": score,
                    "residual_threshold": threshold,
                    "residual_component_area": int(component_mask.sum()),
                    "residual_h": _local_stats(evidence, component_mask),
                    "distance_to_predicted_coverage": float(distance_transform_edt(~covered)[y, x]),
                },
                generator_version=config.generator_version,
                config_hash=config_hash,
                support_box=support_box,
            )
        )
    return output


def generate_split_candidates(
    image: np.ndarray,
    prediction: np.ndarray,
    *,
    image_id: str,
    config: SplitCandidateConfig = SplitCandidateConfig(),
) -> list[ActionCandidate]:
    """Generate mutual-negative split proposals using prediction/image signals only."""

    prediction = np.asarray(prediction)
    evidence = hematoxylin_evidence(image, config.h_sigma)
    if prediction.shape != evidence.shape:
        raise ValueError(f"prediction/image shape mismatch: {prediction.shape} != {evidence.shape}")
    config_hash = _action_hash(config)
    output: list[ActionCandidate] = []

    for parent_id in sorted(int(item) for item in np.unique(prediction) if int(item) != 0):
        parent = prediction == parent_id
        area = int(parent.sum())
        if area < config.min_parent_area:
            continue
        masked_h = np.where(parent, evidence, 0.0)
        peaks_yx = peak_local_max(
            masked_h,
            min_distance=config.min_peak_distance,
            threshold_abs=0.0,
            exclude_border=False,
            labels=parent.astype(np.uint8),
            num_peaks=config.max_peaks_per_parent,
        )
        if len(peaks_yx) < 2:
            continue
        # Stable score ordering makes the proposal independent of library tie order.
        peaks = sorted(
            (Point(x=int(x), y=int(y)) for y, x in peaks_yx),
            key=lambda point: (-float(masked_h[point.y, point.x]), point.y, point.x),
        )
        props = regionprops(parent.astype(np.uint8))[0]
        parent_box = _box(parent)
        proposals: list[tuple[float, Point, Point, dict]] = []
        for p1, p2 in combinations(peaks, 2):
            distance = float(np.hypot(p1.x - p2.x, p1.y - p2.y))
            normalized_distance = distance / max(1.0, float(np.sqrt(area)))
            if normalized_distance < config.min_normalized_peak_distance:
                continue
            height1 = float(masked_h[p1.y, p1.x])
            height2 = float(masked_h[p2.y, p2.x])
            height_ratio = min(height1, height2) / max(1.0e-12, max(height1, height2))
            if height_ratio < config.min_peak_height_ratio:
                continue
            line = _line_values(masked_h, p1, p2)
            valley = float(line.min()) if line.size else 0.0
            valley_depth = 1.0 - valley / max(1.0e-12, min(height1, height2))
            if valley_depth < config.min_valley_depth:
                continue
            basin_features = _distance_basin_features(parent, p1, p2)
            basin_ratio = basin_features["distance_basin_area_ratio"]
            proposal_score = height1 + height2 + normalized_distance + basin_ratio - valley_depth
            features = {
                "parent_pred_id": parent_id,
                "parent_area": area,
                "parent_perimeter": float(props.perimeter),
                "parent_solidity": float(props.solidity),
                "parent_eccentricity": float(props.eccentricity),
                "peak_height_1": height1,
                "peak_height_2": height2,
                "peak_height_ratio": height_ratio,
                "peak_distance": distance,
                "normalized_peak_distance": normalized_distance,
                "peak_valley_depth": valley_depth,
                **basin_features,
                "h_parent": _local_stats(evidence, parent),
            }
            proposals.append((proposal_score, p1, p2, features))
        proposals.sort(key=lambda item: (-item[0], item[1].y, item[1].x, item[2].y, item[2].x))
        for rank, (_, p1, p2, features) in enumerate(proposals[: config.max_pairs_per_parent]):
            output.append(
                ActionCandidate(
                    action_id=f"{image_id}:SPLIT:{parent_id}:{rank:03d}",
                    image_id=image_id,
                    action_type=ActionType.SPLIT,
                    affected_instance_ids=(parent_id,),
                    positive_points=(p1, p2),
                    negative_points=(p2, p1),
                    action_cost=2,
                    generation_features={"candidate_rank": rank, **features},
                    generator_version=config.generator_version,
                    config_hash=config_hash,
                    support_box=parent_box,
                )
            )
    return output
