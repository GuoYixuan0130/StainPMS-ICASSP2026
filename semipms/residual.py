"""GT-free stain-residual proposals and frozen cross-view acceptance features."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, shift as nd_shift
from skimage.color import rgb2hed
from skimage.feature import peak_local_max
from skimage.filters import gaussian
from skimage.measure import label, regionprops
from skimage.morphology import disk


DEFAULT_BUDGETS = (1, 2, 4, 8, 16, 32, 64)


def compute_h_evidence(image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Pure NumPy H optical-density evidence, avoiding legacy method imports."""
    rgb = np.asarray(image, dtype=np.float32)[..., :3]
    if rgb.max() > 1.5:
        rgb = rgb / 255.0
    h = rgb2hed(np.clip(rgb, 0.0, 1.0))[..., 0]
    low, high = np.percentile(h, [1, 99])
    if high <= low:
        return np.zeros_like(h, dtype=np.float32)
    evidence = np.clip((h - low) / (high - low), 0.0, 1.0)
    return gaussian(evidence, sigma=sigma, preserve_range=True).astype(np.float32) if sigma > 0 else evidence.astype(np.float32)


@dataclass(frozen=True)
class ResidualCandidate:
    x: float
    y: float
    evidence: float
    source: str


def residual_evidence(image: np.ndarray, teacher_instances: np.ndarray, dilation_radius: int = 5) -> np.ndarray:
    """H optical-density evidence outside the dilated teacher instance coverage."""
    h = compute_h_evidence(image, sigma=1.0)
    coverage = np.asarray(teacher_instances) > 0
    dilated = binary_dilation(coverage, footprint=disk(dilation_radius)) if dilation_radius > 0 else coverage
    return (h * (~dilated)).astype(np.float32)


def h_channel_evidence(image: np.ndarray) -> np.ndarray:
    """Unmasked H optical-density evidence used for the occupancy feature."""
    return compute_h_evidence(image, sigma=1.0).astype(np.float32)


def inverse_stain_mask(mask: np.ndarray) -> np.ndarray:
    """Color perturbation has no geometry, so its inverse mask map is identity."""
    return np.asarray(mask, dtype=bool).copy()


def propose_residual_points(
    residual: np.ndarray,
    *,
    max_candidates: int = 64,
    scales: tuple[tuple[int, int], ...] = ((6, 1), (12, 1), (20, 2)),
) -> list[ResidualCandidate]:
    """Multi-scale local maxima plus connected-component maxima, deduplicated by score."""
    residual = np.asarray(residual, dtype=np.float32)
    candidates: list[ResidualCandidate] = []
    for min_distance, threshold_percentile in scales:
        threshold = float(np.percentile(residual, max(0, 75 - threshold_percentile * 5)))
        points = peak_local_max(
            residual,
            min_distance=min_distance,
            threshold_abs=threshold,
            exclude_border=False,
        )
        for y, x in points:
            candidates.append(ResidualCandidate(float(x), float(y), float(residual[y, x]), f"peak_d{min_distance}"))
    components = label(residual >= np.percentile(residual, 90), connectivity=1)
    for region in regionprops(components, intensity_image=residual):
        if region.area < 4:
            continue
        y, x = region.weighted_centroid
        iy, ix = int(round(y)), int(round(x))
        iy = min(max(iy, 0), residual.shape[0] - 1)
        ix = min(max(ix, 0), residual.shape[1] - 1)
        candidates.append(ResidualCandidate(float(ix), float(iy), float(residual[iy, ix]), "component"))
    deduplicated: list[ResidualCandidate] = []
    for candidate in sorted(candidates, key=lambda row: (-row.evidence, row.y, row.x, row.source)):
        if not any((candidate.x - kept.x) ** 2 + (candidate.y - kept.y) ** 2 < 16 for kept in deduplicated):
            deduplicated.append(candidate)
        if len(deduplicated) >= max_candidates:
            break
    return deduplicated


def stain_perturbation(image: np.ndarray) -> np.ndarray:
    """Deterministic mild H&E-like color perturbation; no coordinate transform."""
    rgb = np.asarray(image, dtype=np.float32)
    if rgb.max() <= 1.5:
        rgb = rgb * 255.0
    matrix = np.asarray(
        [[1.04, -0.02, -0.01], [-0.01, 0.98, 0.02], [0.01, -0.02, 1.03]], dtype=np.float32
    )
    return np.clip(rgb @ matrix.T + np.asarray([1.0, -1.5, 0.5], dtype=np.float32), 0, 255).astype(np.uint8)


def geometric_view(image: np.ndarray, dx: int = 3, dy: int = -2) -> np.ndarray:
    return nd_shift(np.asarray(image), shift=(dy, dx, 0), order=1, mode="constant", cval=0, prefilter=False).astype(image.dtype)


def inverse_geometric_mask(mask: np.ndarray, dx: int = 3, dy: int = -2) -> np.ndarray:
    return nd_shift(np.asarray(mask, dtype=np.float32), shift=(-dy, -dx), order=0, mode="constant", cval=0, prefilter=False) > 0.5


def transform_points_xy(points: np.ndarray, dx: int = 3, dy: int = -2) -> np.ndarray:
    out = np.asarray(points, dtype=np.float32).copy()
    out[:, 0] += dx
    out[:, 1] += dy
    return out


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    union = int(np.logical_or(a, b).sum())
    return float(np.logical_and(a, b).sum() / union) if union else 1.0


def _centroid(mask: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.where(mask)
    if not len(xs):
        return None
    return float(xs.mean()), float(ys.mean())


def _boundary(mask: np.ndarray) -> np.ndarray:
    return binary_dilation(np.logical_xor(mask, binary_erosion(mask)), iterations=2)


def acceptance_features(
    original_mask: np.ndarray,
    stain_mask: np.ndarray,
    geometric_mask_inverse: np.ndarray,
    h_evidence: np.ndarray,
    pseudo_instances: np.ndarray,
) -> dict[str, float]:
    original = np.asarray(original_mask, dtype=bool)
    stain = np.asarray(stain_mask, dtype=bool)
    geometric = np.asarray(geometric_mask_inverse, dtype=bool)
    center_a, center_b = _centroid(original), _centroid(geometric)
    if center_a is None or center_b is None:
        displacement = float("inf")
    else:
        displacement = float(np.hypot(center_a[0] - center_b[0], center_a[1] - center_b[1]))
    area = max(int(original.sum()), 1)
    area_stability = min(int(stain.sum()), int(geometric.sum())) / max(area, int(stain.sum()), int(geometric.sum()), 1)
    pseudo = np.asarray(pseudo_instances) > 0
    conflict = float(np.logical_and(original, pseudo).sum() / max(int(original.sum()), 1))
    return {
        "stain_inverse_iou": _iou(original, stain),
        "geometric_inverse_iou": _iou(original, geometric),
        "centroid_displacement": displacement,
        "area_stability": float(area_stability),
        "h_occupancy": float(np.asarray(h_evidence)[original].mean()) if original.any() else 0.0,
        "boundary_stability": _iou(_boundary(original), _boundary(geometric)),
        "pseudo_conflict": conflict,
    }


def frozen_accept(features: dict[str, float], rule: dict[str, float]) -> bool:
    """The only non-oracle candidate filter used on the 24 hidden-GT images."""
    return (
        min(features["stain_inverse_iou"], features["geometric_inverse_iou"]) >= rule["min_view_iou"]
        and features["centroid_displacement"] <= rule["max_centroid_displacement"]
        and features["area_stability"] >= rule["min_area_stability"]
        and features["h_occupancy"] >= rule["min_h_occupancy"]
        and features["boundary_stability"] >= rule["min_boundary_stability"]
        and features["pseudo_conflict"] <= rule["max_pseudo_conflict"]
    )


def candidate_rows(candidates: Iterable[ResidualCandidate]) -> list[dict[str, object]]:
    return [asdict(candidate) for candidate in candidates]
