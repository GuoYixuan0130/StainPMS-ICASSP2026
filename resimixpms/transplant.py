"""Pure NumPy/SciPy primitives for deterministic ResiMix-PMS transplantation.

This module deliberately has no model, dataset, or filesystem dependency.  It
keeps the image-editing part of ResiMix testable with synthetic arrays before
it is connected to the StainPMS dataloader.  Coordinates are always ``(y, x)``
and RGB arrays use ``H x W x 3`` values in the 0--255 range (uint8 or float).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import blake2b
import json
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage as ndi


FORMAL_DONOR_RATIOS: Mapping[str, float] = {
    "Missed": 0.50,
    "IoU-Cliff": 0.30,
    "Low-Quality Matched": 0.20,
}

CONTEXT_FEATURE_NAMES: Tuple[str, ...] = (
    "h_mean",
    "h_std",
    "e_mean",
    "e_std",
    "total_od_mean",
    "total_od_std",
    "h_gradient_energy",
    "e_gradient_energy",
    "tissue_foreground_fraction",
)

_WHITE = 255.0
_OD_EPS = 1.0
# Rows are H, E, and residual stain OD directions.  The first two estimated
# concentrations are used only for lightweight, deterministic context features.
_HE_STAIN_DIRECTIONS = np.asarray(
    (
        (0.650, 0.704, 0.286),
        (0.072, 0.990, 0.105),
        (0.268, 0.570, 0.776),
    ),
    dtype=np.float64,
)
_HE_STAIN_PINV = np.linalg.pinv(_HE_STAIN_DIRECTIONS)


def _as_rgb255(image: np.ndarray) -> np.ndarray:
    """Return an HxWx3 float RGB array, accepting common [0, 1] input too."""

    rgb = np.asarray(image, dtype=np.float64)
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError("RGB input must have shape (H, W, 3)")
    finite = rgb[np.isfinite(rgb)]
    if finite.size and finite.min() >= -1e-8 and finite.max() <= 1.0 + 1e-8:
        rgb = rgb * _WHITE
    return rgb


def _as_mask(mask: np.ndarray, shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
    result = np.asarray(mask, dtype=bool)
    if result.ndim != 2:
        raise ValueError("mask must be a two-dimensional array")
    if shape is not None and result.shape != tuple(shape):
        raise ValueError(f"mask shape {result.shape} does not match expected {shape}")
    return result


def _canonical_bytes(value: Any) -> bytes:
    """Stable serialization for RNG keys; never uses Python's randomized hash."""

    try:
        text = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    except (TypeError, ValueError):
        text = repr(value)
    return text.encode("utf-8")


def stable_seed(seed: int, *parts: Any) -> int:
    """Derive a platform-stable child seed from the fixed experiment seed."""

    digest = blake2b(digest_size=16)
    digest.update(str(int(seed)).encode("ascii"))
    for part in parts:
        payload = _canonical_bytes(part)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return int.from_bytes(digest.digest()[:8], "big", signed=False)


def stable_rng(seed: int, *parts: Any) -> np.random.Generator:
    """Return a reproducible generator scoped to an operation and sample key."""

    return np.random.default_rng(stable_seed(seed, *parts))


def rgb_to_od(rgb: np.ndarray, epsilon: float = _OD_EPS) -> np.ndarray:
    """Convert RGB to optical density without creating invalid logarithms."""

    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    rgb255 = _as_rgb255(rgb)
    safe_rgb = np.clip(rgb255, 0.0, _WHITE)
    return -np.log((safe_rgb + epsilon) / (_WHITE + epsilon))


def od_to_rgb(od: np.ndarray, epsilon: float = _OD_EPS) -> Tuple[np.ndarray, np.ndarray, float]:
    """Convert OD to RGB and report the pre-clipping RGB and clipped-pixel rate."""

    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    od_array = np.asarray(od, dtype=np.float64)
    if od_array.ndim != 3 or od_array.shape[-1] != 3:
        raise ValueError("OD input must have shape (H, W, 3)")
    raw_rgb = (_WHITE + epsilon) * np.exp(-od_array) - epsilon
    clipped_pixels = np.any((raw_rgb < 0.0) | (raw_rgb > _WHITE), axis=-1)
    clip_fraction = float(np.mean(clipped_pixels)) if clipped_pixels.size else 0.0
    return np.clip(raw_rgb, 0.0, _WHITE), raw_rgb, clip_fraction


@dataclass(frozen=True)
class StainMatchResult:
    """OD affine stain matching output and its frozen numerical diagnostics."""

    rgb: np.ndarray
    raw_rgb: np.ndarray
    od: np.ndarray
    channel_scale: np.ndarray
    donor_ring_mean: np.ndarray
    donor_ring_std: np.ndarray
    host_ring_mean: np.ndarray
    host_ring_std: np.ndarray
    clip_fraction: float


def od_affine_stain_match(
    donor_rgb: np.ndarray,
    donor_annulus: np.ndarray,
    host_rgb: np.ndarray,
    host_annulus: np.ndarray,
    transplant_mask: Optional[np.ndarray] = None,
) -> StainMatchResult:
    """Match a donor to a host using only their annulus OD statistics.

    The affine transform follows the route specification exactly: channelwise
    donor-ring centering, a scale clipped to [0.75, 1.33], then host-ring
    recentering.  ``transplant_mask`` only controls the reported clipping
    fraction; the returned matched RGB remains a full donor patch so callers
    can use its original geometry and alpha taper.
    """

    donor = _as_rgb255(donor_rgb)
    host = _as_rgb255(host_rgb)
    donor_ring = _as_mask(donor_annulus, donor.shape[:2])
    host_ring = _as_mask(host_annulus, host.shape[:2])
    if not donor_ring.any() or not host_ring.any():
        raise ValueError("donor and host annuli must each contain at least one pixel")
    if not np.isfinite(donor).all() or not np.isfinite(host).all():
        raise ValueError("RGB values must be finite before OD matching")

    donor_od = rgb_to_od(donor)
    host_od = rgb_to_od(host)
    donor_values = donor_od[donor_ring]
    host_values = host_od[host_ring]
    donor_mean = donor_values.mean(axis=0)
    donor_std = donor_values.std(axis=0)
    host_mean = host_values.mean(axis=0)
    host_std = host_values.std(axis=0)
    safe_donor_std = np.maximum(donor_std, 1e-8)
    channel_scale = np.clip(host_std / safe_donor_std, 0.75, 1.33)
    matched_od = (donor_od - donor_mean) * channel_scale + host_mean
    matched_rgb, raw_rgb, _ = od_to_rgb(matched_od)

    if transplant_mask is None:
        clip_region = np.ones(donor.shape[:2], dtype=bool)
    else:
        clip_region = _as_mask(transplant_mask, donor.shape[:2])
    clipped_pixels = np.any((raw_rgb < 0.0) | (raw_rgb > _WHITE), axis=-1)
    clip_fraction = (
        float(np.mean(clipped_pixels[clip_region])) if clip_region.any() else 0.0
    )
    return StainMatchResult(
        rgb=matched_rgb,
        raw_rgb=raw_rgb,
        od=matched_od,
        channel_scale=channel_scale,
        donor_ring_mean=donor_mean,
        donor_ring_std=donor_std,
        host_ring_mean=host_mean,
        host_ring_std=host_std,
        clip_fraction=clip_fraction,
    )


@dataclass(frozen=True)
class DonorGeometry:
    """The only allowed donor geometry operations for the formal route."""

    rotation_deg: int = 0
    flip: Optional[str] = None
    scale: float = 1.0

    def validate(self) -> "DonorGeometry":
        if self.rotation_deg not in (0, 90, 180, 270):
            raise ValueError("rotation_deg must be one of 0, 90, 180, or 270")
        if self.flip not in (None, "horizontal", "vertical"):
            raise ValueError("flip must be None, 'horizontal', or 'vertical'")
        if not 0.9 <= float(self.scale) <= 1.1:
            raise ValueError("isotropic scale must lie in [0.9, 1.1]")
        return self


@dataclass(frozen=True)
class TransformedDonor:
    rgb: np.ndarray
    mask: np.ndarray
    annulus: np.ndarray
    geometry: DonorGeometry


def _transform_spatial(array: np.ndarray, geometry: DonorGeometry, order: int) -> np.ndarray:
    result = np.asarray(array)
    if geometry.rotation_deg:
        result = np.rot90(result, k=geometry.rotation_deg // 90, axes=(0, 1))
    if geometry.flip == "horizontal":
        result = np.flip(result, axis=1)
    elif geometry.flip == "vertical":
        result = np.flip(result, axis=0)
    if not np.isclose(geometry.scale, 1.0):
        zoom = (geometry.scale, geometry.scale) + ((1.0,) if result.ndim == 3 else ())
        result = ndi.zoom(result, zoom=zoom, order=order, mode="nearest", prefilter=False)
    return result


def transform_donor(
    donor_rgb: np.ndarray,
    donor_mask: np.ndarray,
    donor_annulus: np.ndarray,
    geometry: DonorGeometry,
) -> TransformedDonor:
    """Apply an allowed rotation/flip/isotropic scale to all donor channels."""

    geometry = geometry.validate()
    rgb = _as_rgb255(donor_rgb)
    mask = _as_mask(donor_mask, rgb.shape[:2])
    annulus = _as_mask(donor_annulus, rgb.shape[:2])
    if not mask.any():
        raise ValueError("donor_mask must contain a nucleus")
    out_rgb = _transform_spatial(rgb, geometry, order=1)
    out_mask = _transform_spatial(mask.astype(np.uint8), geometry, order=0).astype(bool)
    out_annulus = _transform_spatial(annulus.astype(np.uint8), geometry, order=0).astype(bool)
    out_annulus &= ~out_mask
    if not out_mask.any() or not out_annulus.any():
        raise ValueError("geometry produced an empty donor mask or annulus")
    return TransformedDonor(
        rgb=np.ascontiguousarray(out_rgb),
        mask=np.ascontiguousarray(out_mask),
        annulus=np.ascontiguousarray(out_annulus),
        geometry=geometry,
    )


def deterministic_geometry(seed: int, sample_key: Any) -> DonorGeometry:
    """Choose one formally allowed geometry from a sample-scoped stable RNG."""

    rng = stable_rng(seed, "resimix_geometry", sample_key)
    rotation = (0, 90, 180, 270)[int(rng.integers(4))]
    flip = (None, "horizontal", "vertical")[int(rng.integers(3))]
    scale = float(rng.uniform(0.9, 1.1))
    return DonorGeometry(rotation_deg=rotation, flip=flip, scale=scale)


@dataclass(frozen=True)
class Placement:
    """Cropping-aware placement of a source patch around a host center."""

    center_yx: Tuple[int, int]
    source_slice: Tuple[slice, slice]
    destination_slice: Tuple[slice, slice]
    mask_fully_inside: bool


def placement_for_mask(
    donor_mask: np.ndarray,
    center_yx: Tuple[int, int],
    canvas_shape: Tuple[int, int],
) -> Placement:
    """Locate a donor patch while separately checking that all mask pixels fit."""

    mask = _as_mask(donor_mask)
    if not mask.any():
        raise ValueError("donor_mask must contain at least one pixel")
    h, w = (int(canvas_shape[0]), int(canvas_shape[1]))
    if h <= 0 or w <= 0:
        raise ValueError("canvas_shape must be positive")
    cy, cx = int(center_yx[0]), int(center_yx[1])
    source_h, source_w = mask.shape
    y0, x0 = cy - source_h // 2, cx - source_w // 2
    ys, xs = np.nonzero(mask)
    mapped_y, mapped_x = y0 + ys, x0 + xs
    inside = bool(
        (mapped_y >= 0).all()
        and (mapped_y < h).all()
        and (mapped_x >= 0).all()
        and (mapped_x < w).all()
    )
    dst_y0, dst_x0 = max(0, y0), max(0, x0)
    dst_y1, dst_x1 = min(h, y0 + source_h), min(w, x0 + source_w)
    src_y0, src_x0 = dst_y0 - y0, dst_x0 - x0
    src_y1, src_x1 = src_y0 + max(0, dst_y1 - dst_y0), src_x0 + max(0, dst_x1 - dst_x0)
    return Placement(
        center_yx=(cy, cx),
        source_slice=(slice(src_y0, src_y1), slice(src_x0, src_x1)),
        destination_slice=(slice(dst_y0, dst_y1), slice(dst_x0, dst_x1)),
        mask_fully_inside=inside,
    )


def render_placed_mask(
    donor_mask: np.ndarray, placement: Placement, canvas_shape: Tuple[int, int]
) -> np.ndarray:
    """Render the in-canvas part of a donor mask without mutating the source."""

    mask = _as_mask(donor_mask)
    result = np.zeros(tuple(canvas_shape), dtype=bool)
    source = mask[placement.source_slice]
    destination = placement.destination_slice
    if source.size:
        result[destination] = source
    return result


def cosine_alpha_taper(mask: np.ndarray, width: int = 2) -> np.ndarray:
    """Create the specified inward cosine taper; deep mask pixels have alpha=1."""

    binary = _as_mask(mask)
    if width <= 0:
        raise ValueError("width must be a positive integer")
    distance = ndi.distance_transform_edt(binary)
    fraction = np.clip(distance / float(width), 0.0, 1.0)
    alpha = 0.5 - 0.5 * np.cos(np.pi * fraction)
    alpha[~binary] = 0.0
    return alpha.astype(np.float64, copy=False)


@dataclass(frozen=True)
class CompositeResult:
    rgb: np.ndarray
    placed_mask: np.ndarray
    alpha: np.ndarray
    placement: Placement


def composite_transplant(
    host_rgb: np.ndarray,
    donor_rgb: np.ndarray,
    donor_mask: np.ndarray,
    center_yx: Tuple[int, int],
    taper_width: int = 2,
) -> CompositeResult:
    """Composite only donor-mask pixels into a host using the cosine taper."""

    host = _as_rgb255(host_rgb)
    donor = _as_rgb255(donor_rgb)
    mask = _as_mask(donor_mask, donor.shape[:2])
    placement = placement_for_mask(mask, center_yx, host.shape[:2])
    if not placement.mask_fully_inside:
        raise ValueError("donor mask must be fully inside the host crop")
    placed_mask = render_placed_mask(mask, placement, host.shape[:2])
    source_alpha = cosine_alpha_taper(mask, width=taper_width)
    alpha = np.zeros(host.shape[:2], dtype=np.float64)
    alpha[placement.destination_slice] = source_alpha[placement.source_slice]
    result = host.copy()
    source_rgb = donor[placement.source_slice]
    target_rgb = result[placement.destination_slice]
    source_a = source_alpha[placement.source_slice][..., None]
    result[placement.destination_slice] = source_a * source_rgb + (1.0 - source_a) * target_rgb
    return CompositeResult(rgb=result, placed_mask=placed_mask, alpha=alpha, placement=placement)


def annulus_mask(mask: np.ndarray, width: int = 8) -> np.ndarray:
    """Return a Euclidean exterior annulus used by donor/host comparisons.

    The donor audit uses the same radius-8 disk convention.  Keeping this
    definition shared avoids silently comparing a disk donor ring with a
    4-connected host ring.
    """

    binary = _as_mask(mask)
    if width <= 0:
        raise ValueError("annulus width must be positive")
    distance_outside = ndi.distance_transform_edt(~binary)
    return (~binary) & (distance_outside <= float(width))


def he_concentrations(rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate lightweight non-negative H/E concentration maps from RGB OD."""

    od = rgb_to_od(rgb)
    flat = od.reshape(-1, 3)
    concentrations = flat @ _HE_STAIN_PINV
    concentrations = np.maximum(concentrations, 0.0).reshape(od.shape)
    return concentrations[..., 0], concentrations[..., 1]


def _gradient_energy(field: np.ndarray) -> np.ndarray:
    gy = ndi.sobel(field, axis=0, mode="nearest")
    gx = ndi.sobel(field, axis=1, mode="nearest")
    return np.hypot(gx, gy)


def context_features(
    rgb: np.ndarray, annulus: np.ndarray, tissue_mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """Compute the frozen H/E, OD, gradient, and tissue context feature vector."""

    image = _as_rgb255(rgb)
    ring = _as_mask(annulus, image.shape[:2])
    if not ring.any():
        raise ValueError("context annulus must not be empty")
    if not np.isfinite(image).all():
        raise ValueError("context RGB must be finite")
    h_concentration, e_concentration = he_concentrations(image)
    total_od = rgb_to_od(image).sum(axis=-1)
    if tissue_mask is None:
        tissue = np.ones(image.shape[:2], dtype=bool)
    else:
        tissue = _as_mask(tissue_mask, image.shape[:2])
    features = np.asarray(
        (
            h_concentration[ring].mean(),
            h_concentration[ring].std(),
            e_concentration[ring].mean(),
            e_concentration[ring].std(),
            total_od[ring].mean(),
            total_od[ring].std(),
            _gradient_energy(h_concentration)[ring].mean(),
            _gradient_energy(e_concentration)[ring].mean(),
            tissue[ring].mean(),
        ),
        dtype=np.float64,
    )
    if not np.isfinite(features).all():
        raise ValueError("context features are not finite")
    return features


def standardized_context_distance(
    donor_features: np.ndarray, host_features: np.ndarray, train_mean: np.ndarray, train_std: np.ndarray
) -> float:
    """L2 distance after normalization by statistics fixed from training data."""

    donor = np.asarray(donor_features, dtype=np.float64).reshape(-1)
    host = np.asarray(host_features, dtype=np.float64).reshape(-1)
    mean = np.asarray(train_mean, dtype=np.float64).reshape(-1)
    std = np.asarray(train_std, dtype=np.float64).reshape(-1)
    if not (donor.shape == host.shape == mean.shape == std.shape):
        raise ValueError("all context vectors and normalizer arrays must share a shape")
    if not np.isfinite(np.concatenate((donor, host, mean, std))).all():
        raise ValueError("context vectors and normalizer arrays must be finite")
    scale = np.maximum(np.abs(std), 1e-8)
    return float(np.linalg.norm((donor - mean) / scale - (host - mean) / scale))


def deterministic_candidate_centers(
    tissue_mask: np.ndarray, seed: int, sample_key: Any, max_candidates: int = 32
) -> Tuple[Tuple[int, int], ...]:
    """Draw at most 32 unique tissue centers deterministically, without retries."""

    tissue = _as_mask(tissue_mask)
    if max_candidates <= 0:
        return ()
    points = np.argwhere(tissue)
    if not len(points):
        return ()
    rng = stable_rng(seed, "resimix_host_centers", sample_key)
    order = rng.permutation(len(points))[: min(int(max_candidates), len(points))]
    return tuple((int(points[i, 0]), int(points[i, 1])) for i in order)


def _nearest_gt_distance(mask: np.ndarray, occupied: np.ndarray) -> float:
    if not occupied.any():
        return float("inf")
    distance_to_gt = ndi.distance_transform_edt(~occupied)
    return float(distance_to_gt[mask].min())


@dataclass(frozen=True)
class HostCandidate:
    center_yx: Tuple[int, int]
    placement: Placement
    context_distance: float
    nearest_gt_distance: float
    tissue_fraction: float
    mode: str
    host_context: np.ndarray


def enumerate_legal_hosts(
    host_rgb: np.ndarray,
    donor: TransformedDonor,
    instance_map: np.ndarray,
    static_coverage: np.ndarray,
    tissue_mask: np.ndarray,
    train_context_mean: np.ndarray,
    train_context_std: np.ndarray,
    *,
    centers: Optional[Sequence[Tuple[int, int]]] = None,
    seed: int = 3407,
    sample_key: Any = "",
    max_candidates: int = 32,
    clearance: int = 3,
    coverage_threshold: float = 0.5,
    min_tissue_fraction: float = 0.95,
) -> Tuple[HostCandidate, ...]:
    """Return legal adjacent/isolated host placements from <=32 fixed centers.

    A candidate is legal only when the entire donor mask fits, tissue covers the
    donor, the mask plus a 3-pixel clearance misses all occupied instances, and
    all donor-mask pixels are absent from the immutable static coverage map.
    The returned candidates retain their train-standardized context distance for
    deterministic top-five selection and later frozen p95 quality rejection.
    """

    host = _as_rgb255(host_rgb)
    shape = host.shape[:2]
    occupied = np.asarray(instance_map) > 0
    coverage = np.asarray(static_coverage)
    tissue = _as_mask(tissue_mask, shape)
    if occupied.shape != shape or coverage.shape != shape:
        raise ValueError("instance_map and static_coverage must match host spatial shape")
    if clearance < 0:
        raise ValueError("clearance must be non-negative")
    if not 0.0 < min_tissue_fraction <= 1.0:
        raise ValueError("min_tissue_fraction must lie in (0, 1]")

    if centers is None:
        candidate_centers = deterministic_candidate_centers(
            tissue, seed, sample_key, max_candidates=max_candidates
        )
    else:
        normalized = {(int(y), int(x)) for y, x in centers}
        candidate_centers = tuple(sorted(normalized)[: int(max_candidates)])
    donor_context = context_features(donor.rgb, donor.annulus)
    covered = coverage >= float(coverage_threshold)
    candidates = []
    for center in candidate_centers:
        cy, cx = center
        if not (0 <= cy < shape[0] and 0 <= cx < shape[1]) or not tissue[cy, cx]:
            continue
        placement = placement_for_mask(donor.mask, center, shape)
        if not placement.mask_fully_inside:
            continue
        placed = render_placed_mask(donor.mask, placement, shape)
        if occupied[placed].any() or covered[placed].any():
            continue
        clearance_mask = ndi.binary_dilation(placed, iterations=int(clearance))
        if occupied[clearance_mask].any():
            continue
        tissue_fraction = float(tissue[placed].mean())
        if tissue_fraction < min_tissue_fraction:
            continue
        nearest = _nearest_gt_distance(placed, occupied)
        if 2.0 <= nearest <= 8.0:
            mode = "adjacent"
        elif nearest >= 8.0:
            mode = "isolated"
        else:
            continue
        host_ring = annulus_mask(placed, width=8)
        if not host_ring.any():
            continue
        host_context = context_features(host, host_ring, tissue)
        distance = standardized_context_distance(
            donor_context, host_context, train_context_mean, train_context_std
        )
        candidates.append(
            HostCandidate(
                center_yx=center,
                placement=placement,
                context_distance=distance,
                nearest_gt_distance=nearest,
                tissue_fraction=tissue_fraction,
                mode=mode,
                host_context=host_context,
            )
        )
    return tuple(candidates)


def deterministic_host_mode(seed: int, sample_key: Any) -> str:
    """Give each attempted augmentation a fixed 50/50 adjacent/isolated mode."""

    return ("adjacent", "isolated")[
        int(stable_rng(seed, "resimix_host_mode", sample_key).integers(2))
    ]


@dataclass(frozen=True)
class HostSelection:
    candidate: HostCandidate
    requested_mode: str
    used_mode: str
    used_fallback: bool
    ranked_count: int


def choose_host_candidate(
    candidates: Sequence[HostCandidate],
    requested_mode: str,
    seed: int,
    sample_key: Any,
    top_k: int = 5,
) -> Optional[HostSelection]:
    """Sample deterministically from the five lowest-context-distance legal hosts."""

    if requested_mode not in ("adjacent", "isolated"):
        raise ValueError("requested_mode must be 'adjacent' or 'isolated'")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    preferred = [candidate for candidate in candidates if candidate.mode == requested_mode]
    fallback_mode = "isolated" if requested_mode == "adjacent" else "adjacent"
    pool = preferred or [candidate for candidate in candidates if candidate.mode == fallback_mode]
    if not pool:
        return None
    ranked = sorted(pool, key=lambda c: (c.context_distance, c.center_yx[0], c.center_yx[1]))
    top = ranked[: int(top_k)]
    rng = stable_rng(seed, "resimix_host_top5", sample_key, requested_mode)
    choice = top[int(rng.integers(len(top)))]
    return HostSelection(
        candidate=choice,
        requested_mode=requested_mode,
        used_mode=choice.mode,
        used_fallback=not bool(preferred),
        ranked_count=len(ranked),
    )


def normalized_donor_ratios(
    donors_by_category: Mapping[str, Sequence[Any]],
    ratios: Mapping[str, float] = FORMAL_DONOR_RATIOS,
) -> Dict[str, float]:
    """Renormalize only across nonempty formal categories, never using dev data."""

    available = {
        name: float(weight)
        for name, weight in ratios.items()
        if float(weight) > 0.0 and donors_by_category.get(name)
    }
    total = sum(available.values())
    return {name: weight / total for name, weight in available.items()} if total else {}


def _donor_sort_key(item: Any) -> str:
    if isinstance(item, Mapping):
        for key in ("donor_id", "id", "source_id"):
            if key in item:
                return str(item[key])
        return json.dumps(item, sort_keys=True, default=str)
    return str(item)


def deterministic_donor_choice(
    donors_by_category: Mapping[str, Sequence[Any]],
    seed: int,
    sample_key: Any,
    ratios: Mapping[str, float] = FORMAL_DONOR_RATIOS,
) -> Optional[Tuple[str, Any]]:
    """Draw one donor with fixed 50/30/20 ratios and formal shortage redistribution."""

    weights = normalized_donor_ratios(donors_by_category, ratios)
    if not weights:
        return None
    rng = stable_rng(seed, "resimix_donor", sample_key)
    draw = float(rng.random())
    running = 0.0
    category = next(iter(weights))
    for name in ratios:
        if name not in weights:
            continue
        running += weights[name]
        if draw < running:
            category = name
            break
    pool = sorted(donors_by_category[category], key=_donor_sort_key)
    return category, pool[int(rng.integers(len(pool)))]


def boundary_gradient_energy(rgb: np.ndarray, mask: np.ndarray) -> float:
    """Mean total-OD gradient on a one-pixel band around an instance boundary."""

    image = _as_rgb255(rgb)
    binary = _as_mask(mask, image.shape[:2])
    if not binary.any():
        return 0.0
    boundary = ndi.binary_dilation(binary, iterations=1) ^ ndi.binary_erosion(binary, iterations=1)
    if not boundary.any():
        return 0.0
    total_od = rgb_to_od(image).sum(axis=-1)
    return float(_gradient_energy(total_od)[boundary].mean())


@dataclass(frozen=True)
class QualityDecision:
    accepted: bool
    reasons: Tuple[str, ...]
    diagnostics: Mapping[str, float]


def quality_reject(
    source_mask: np.ndarray,
    transplanted_mask: np.ndarray,
    *,
    occupied_instances: Optional[np.ndarray] = None,
    composited_rgb: Optional[np.ndarray] = None,
    stain_match: Optional[StainMatchResult] = None,
    seam_gradient: Optional[float] = None,
    natural_boundary_p95: Optional[float] = None,
    context_distance: Optional[float] = None,
    legal_context_p95: Optional[float] = None,
    max_area_change: float = 0.25,
    max_clip_fraction: float = 0.01,
) -> QualityDecision:
    """Apply only the route's frozen mechanical/numerical quality rejections."""

    source = _as_mask(source_mask)
    placed = _as_mask(transplanted_mask)
    reasons = []
    source_area = int(source.sum())
    placed_area = int(placed.sum())
    if source_area == 0:
        reasons.append("invalid_source_area")
        area_ratio = float("nan")
    else:
        area_ratio = placed_area / float(source_area)
        if abs(area_ratio - 1.0) > float(max_area_change):
            reasons.append("area_change")
    if occupied_instances is not None:
        occupied = np.asarray(occupied_instances) > 0
        if occupied.shape != placed.shape:
            raise ValueError("occupied_instances must match transplanted_mask")
        if np.any(occupied & placed):
            reasons.append("instance_overlap")
    if composited_rgb is not None:
        image = _as_rgb255(composited_rgb)
        if image.shape[:2] != placed.shape:
            raise ValueError("composited_rgb and transplanted_mask must share spatial shape")
        if not np.isfinite(image).all():
            reasons.append("nonfinite_rgb")
    if stain_match is not None:
        if not (
            np.isfinite(stain_match.rgb).all()
            and np.isfinite(stain_match.raw_rgb).all()
            and np.isfinite(stain_match.od).all()
        ):
            reasons.append("nonfinite_od_or_rgb")
        clip_fraction = float(stain_match.clip_fraction)
    else:
        clip_fraction = 0.0
    if not np.isfinite(clip_fraction):
        reasons.append("nonfinite_clip_fraction")
    elif clip_fraction > float(max_clip_fraction):
        reasons.append("clip_fraction")
    if seam_gradient is None and composited_rgb is not None:
        seam_gradient = boundary_gradient_energy(composited_rgb, placed)
    if seam_gradient is not None:
        seam_gradient = float(seam_gradient)
        if not np.isfinite(seam_gradient):
            reasons.append("nonfinite_seam_gradient")
        elif natural_boundary_p95 is not None and seam_gradient > float(natural_boundary_p95):
            reasons.append("seam_gradient")
    if context_distance is not None:
        context_distance = float(context_distance)
        if not np.isfinite(context_distance):
            reasons.append("nonfinite_context_distance")
        elif legal_context_p95 is not None and context_distance > float(legal_context_p95):
            reasons.append("context_distance")
    diagnostics = {
        "source_area": float(source_area),
        "placed_area": float(placed_area),
        "area_ratio": float(area_ratio),
        "clip_fraction": float(clip_fraction),
        "seam_gradient": float(seam_gradient) if seam_gradient is not None else float("nan"),
        "context_distance": float(context_distance) if context_distance is not None else float("nan"),
    }
    return QualityDecision(accepted=not reasons, reasons=tuple(reasons), diagnostics=diagnostics)


def mask_medoid(mask: np.ndarray) -> Tuple[int, int]:
    """Return a deterministic interior medoid proxy: the deepest EDT pixel."""

    binary = _as_mask(mask)
    if not binary.any():
        raise ValueError("cannot compute a medoid for an empty mask")
    distance = ndi.distance_transform_edt(binary)
    # np.argmax's row-major tie-break is deliberately deterministic.
    y, x = np.unravel_index(int(np.argmax(distance)), distance.shape)
    return int(y), int(x)


@dataclass
class TransplantStats:
    """Small in-memory accounting object for the required training summaries."""

    attempted_crops: int = 0
    proposal_attempts: int = 0
    accepted_transplants: int = 0
    synthetic_prompts_added: int = 0
    donor_categories: Dict[str, int] = field(default_factory=dict)
    host_modes: Dict[str, int] = field(default_factory=dict)
    rejection_reasons: Dict[str, int] = field(default_factory=dict)

    def record(
        self,
        decision: QualityDecision,
        *,
        donor_category: Optional[str] = None,
        host_mode: Optional[str] = None,
        synthetic_prompt_added: bool = False,
        proposal_attempted: bool = True,
    ) -> None:
        self.attempted_crops += 1
        self.proposal_attempts += int(bool(proposal_attempted))
        if donor_category is not None:
            self.donor_categories[donor_category] = self.donor_categories.get(donor_category, 0) + 1
        if host_mode is not None:
            self.host_modes[host_mode] = self.host_modes.get(host_mode, 0) + 1
        if decision.accepted:
            self.accepted_transplants += 1
            self.synthetic_prompts_added += int(bool(synthetic_prompt_added))
        else:
            for reason in decision.reasons:
                self.rejection_reasons[reason] = self.rejection_reasons.get(reason, 0) + 1

    def as_dict(self) -> Dict[str, Any]:
        proposal_rate = (
            self.accepted_transplants / self.proposal_attempts if self.proposal_attempts else 0.0
        )
        prompt_rate = (
            self.synthetic_prompts_added / self.accepted_transplants if self.accepted_transplants else 0.0
        )
        return {
            "attempted_crops": self.attempted_crops,
            "proposal_attempts": self.proposal_attempts,
            "accepted_transplants": self.accepted_transplants,
            "proposal_acceptance_rate": proposal_rate,
            "synthetic_prompts_added": self.synthetic_prompts_added,
            "synthetic_prompt_entry_rate": prompt_rate,
            "donor_categories": dict(sorted(self.donor_categories.items())),
            "host_modes": dict(sorted(self.host_modes.items())),
            "rejection_reasons": dict(sorted(self.rejection_reasons.items())),
        }
