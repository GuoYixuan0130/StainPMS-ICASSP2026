"""Deterministic optical-density H&E counterfactuals used by Phase 0."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


EPSILON = 1.0e-6


@dataclass(frozen=True)
class StainDecomposition:
    matrix: np.ndarray  # [H, E] x RGB, each row unit length
    concentration: np.ndarray  # H x W x [H, E]
    i0: np.ndarray  # RGB illumination estimate in [0, 255]
    fallback_used: bool = False
    fallback_reason: str = ""


def _unit_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), EPSILON)


def rgb_to_od(rgb: np.ndarray, i0: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float64)
    return -np.log(np.clip((rgb + EPSILON) / i0.reshape(1, 1, 3), EPSILON, None))


def od_to_rgb(od: np.ndarray, i0: np.ndarray) -> np.ndarray:
    rgb = i0.reshape(1, 1, 3) * np.exp(-np.asarray(od, dtype=np.float64)) - EPSILON
    return np.clip(np.rint(rgb), 0, 255).astype(np.uint8)


def _order_he_rows(matrix: np.ndarray) -> np.ndarray:
    """Order two Macenko vectors as H then E using a deterministic blue/red rule."""
    matrix = _unit_rows(matrix)
    blue_over_red = matrix[:, 2] - matrix[:, 0]
    h_index = int(np.argmax(blue_over_red))
    return matrix[[h_index, 1 - h_index]]


def estimate_stain_matrix(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A compact Macenko-style non-negative H&E factorisation.

    The 1st/99th angular percentiles are fixed.  No model output enters this
    procedure, so it is safe for calibration and frozen-audit preparation.
    """
    rgb = np.asarray(rgb, dtype=np.uint8)[..., :3]
    i0 = np.maximum(np.percentile(rgb.reshape(-1, 3), 99.9, axis=0), 1.0)
    od = rgb_to_od(rgb, i0)
    tissue = od.sum(axis=-1) > 0.15
    pixels = od[tissue]
    if pixels.shape[0] < 64 or np.linalg.matrix_rank(pixels) < 2:
        raise ValueError("insufficient non-background optical-density pixels")
    _, _, vh = np.linalg.svd(pixels, full_matrices=False)
    plane = vh[:2].T
    projected = pixels @ plane
    angles = np.arctan2(projected[:, 1], projected[:, 0])
    low, high = np.percentile(angles, [1.0, 99.0])
    vectors = np.stack(
        [plane @ np.array([np.cos(low), np.sin(low)]), plane @ np.array([np.cos(high), np.sin(high)])]
    )
    # Stain OD directions must be non-negative. Flip signs before ordering.
    vectors *= np.where(vectors.sum(axis=1, keepdims=True) < 0.0, -1.0, 1.0)
    vectors = np.clip(vectors, 0.0, None)
    if np.any(np.linalg.norm(vectors, axis=1) < EPSILON):
        raise ValueError("degenerate Macenko stain vector")
    return _order_he_rows(vectors), i0


def decompose(rgb: np.ndarray, fallback_matrix: np.ndarray | None = None) -> StainDecomposition:
    try:
        matrix, i0 = estimate_stain_matrix(rgb)
        fallback_used = False
        fallback_reason = ""
    except ValueError as exc:
        if fallback_matrix is None:
            raise
        matrix = _order_he_rows(np.asarray(fallback_matrix, dtype=np.float64))
        i0 = np.maximum(np.percentile(np.asarray(rgb).reshape(-1, 3), 99.9, axis=0), 1.0)
        fallback_used = True
        fallback_reason = str(exc)
    od = rgb_to_od(np.asarray(rgb)[..., :3], i0)
    concentration = np.linalg.lstsq(matrix.T, od.reshape(-1, 3).T, rcond=None)[0].T
    concentration = np.clip(concentration, 0.0, None).reshape(*od.shape[:2], 2)
    return StainDecomposition(matrix, concentration, i0, fallback_used, fallback_reason)


def reconstruct(decomposition: StainDecomposition, matrix: np.ndarray | None = None, concentration: np.ndarray | None = None) -> np.ndarray:
    matrix = decomposition.matrix if matrix is None else _order_he_rows(matrix)
    concentration = decomposition.concentration if concentration is None else concentration
    return od_to_rgb(np.asarray(concentration) @ matrix, decomposition.i0)


def counterfactual_views(
    rgb: np.ndarray,
    own_fallback_matrix: np.ndarray,
    within_reference_matrix: np.ndarray,
    cross_reference_matrix: np.ndarray,
) -> tuple[dict[str, np.ndarray], StainDecomposition]:
    decomposition = decompose(rgb, fallback_matrix=own_fallback_matrix)
    concentrations = decomposition.concentration
    h_weak = concentrations.copy()
    h_weak[..., 0] *= 0.8
    h_strong = concentrations.copy()
    h_strong[..., 0] *= 1.2
    return {
        "V0": np.asarray(rgb, dtype=np.uint8)[..., :3].copy(),
        "V1": reconstruct(decomposition),
        "V2": reconstruct(decomposition, concentration=h_weak),
        "V3": reconstruct(decomposition, concentration=h_strong),
        "V4": reconstruct(decomposition, matrix=within_reference_matrix),
        "V5": reconstruct(decomposition, matrix=cross_reference_matrix),
    }, decomposition


def concentration_summary(concentration: np.ndarray) -> dict[str, float]:
    return {
        "h_mean": float(np.mean(concentration[..., 0])),
        "h_median": float(np.median(concentration[..., 0])),
        "e_mean": float(np.mean(concentration[..., 1])),
        "e_median": float(np.median(concentration[..., 1])),
    }


def stain_record(decomposition: StainDecomposition) -> dict[str, Any]:
    return {
        "matrix": np.asarray(decomposition.matrix).round(10).tolist(),
        "i0": np.asarray(decomposition.i0).round(6).tolist(),
        "concentration": concentration_summary(decomposition.concentration),
        "fallback_used": decomposition.fallback_used,
        "fallback_reason": decomposition.fallback_reason,
    }
