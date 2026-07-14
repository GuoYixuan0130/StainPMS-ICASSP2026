"""Immutable PromptQ-v2 protocol primitives.

This module contains no model or dataset discovery.  The explicit constants
and pure helpers make the audit's causal boundary unit-testable.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F


CANONICAL_BASELINE_SHA = "2a1348cb7a1158a6f77aae2f92c168f9552d8068"
CHECKPOINT_SHA256 = "44a3cb3e93051301d789e44f93769588abfa727d7a174b0270f55305ef023781"
SEED = 3407
NMS_RADIUS = 12
INCLUSIVE_IOU_THRESHOLD = 0.5
TTA_ENABLED = False
BATCH_SIZE = 1
QUALITY_HEAD_PARAMETER_COUNT = 66_049
QUALITY_EPOCH_LIMIT = 20
QUALITY_LR = 1e-4
QUALITY_WEIGHT_DECAY = 1e-4


def json_dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_sha256(module: torch.nn.Module, *, exclude_prefixes: tuple[str, ...] = ()) -> str:
    """Hash immutable state deterministically, excluding only the new head."""
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        if name.startswith(exclude_prefixes):
            continue
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def set_determinism() -> None:
    import random

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def utility_target(hard_iou: torch.Tensor) -> torch.Tensor:
    """Frozen PromptQ scalar target: r * sigmoid((r - .5) / .1)."""
    value = hard_iou.detach().to(torch.float32)
    return value * torch.sigmoid((value - 0.5) / 0.1)


def quality_focal_loss(logits: torch.Tensor, targets: torch.Tensor, matched: torch.Tensor) -> torch.Tensor:
    """Original QFL with one-positive-set-equivalent unmatched weighting."""
    probability = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    losses = (targets - probability).abs().pow(2.0) * ce
    positive_count = max(int(matched.sum().item()), 1)
    positives = losses[matched].sum()
    negatives = losses[~matched].sum() * (positive_count / max(int((~matched).sum().item()), 1))
    return (positives + negatives) / positive_count


def product_score(objectness: np.ndarray, quality_logit: np.ndarray) -> np.ndarray:
    quality = 1.0 / (1.0 + np.exp(-np.asarray(quality_logit, dtype=np.float64)))
    return np.asarray(objectness, dtype=np.float64) * quality


def quality_only_score(quality_logit: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(quality_logit, dtype=np.float64)))


def inclusive_iou(left: np.ndarray, right: np.ndarray) -> float:
    intersection = int(np.logical_and(left, right).sum())
    union = int(np.logical_or(left, right).sum())
    return float(intersection / union) if union else 1.0


def point_nms_indices(points: np.ndarray, scores: np.ndarray, radius: float = NMS_RADIUS) -> np.ndarray:
    """Exact ordering semantics of canonical ``sam2_train...point_nms``."""
    points = np.asarray(points, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    if not len(points):
        return np.empty(0, dtype=np.int64)
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    np.fill_diagonal(distances, np.inf)
    reserved = np.ones(len(points), dtype=bool)
    for index in np.argsort(-scores):
        if reserved[index]:
            reserved[distances[index] <= radius] = False
    return np.flatnonzero(reserved).astype(np.int64)


def paired_bootstrap(rows: list[dict[str, float]], metric: str, *, resamples: int = 2000) -> dict[str, float | int]:
    """Image-level paired bootstrap.  It is explanatory, never a gate."""
    if not rows:
        raise ValueError("bootstrap needs at least one image row")
    values = np.asarray([float(row[f"promptq_{metric}"]) - float(row[f"baseline_{metric}"]) for row in rows])
    generator = np.random.default_rng(SEED)
    sampled = np.asarray([generator.choice(values, len(values), replace=True).mean() for _ in range(resamples)])
    return {
        "n_images": int(len(values)),
        "paired_mean_delta": float(values.mean()),
        "ci95_low": float(np.quantile(sampled, 0.025)),
        "ci95_high": float(np.quantile(sampled, 0.975)),
        "resamples": int(resamples),
        "seed": SEED,
    }


def finite_arrays(arrays: Iterable[np.ndarray | torch.Tensor]) -> bool:
    for item in arrays:
        data = item.detach().cpu().numpy() if torch.is_tensor(item) else np.asarray(item)
        if not np.isfinite(data).all():
            return False
    return True


def verdict(delta_aji: float, delta_pq: float, patient_deltas: list[tuple[float, float]]) -> str:
    """Report-only project-lead rubric; never invokes a proxy-metric gate."""
    both_patients_nonnegative = all(aji >= 0.0 and pq >= 0.0 for aji, pq in patient_deltas)
    if both_patients_nonnegative and (
        (delta_aji >= 0.020 and delta_pq > 0.0)
        or (delta_pq >= 0.020 and delta_aji >= 0.0)
        or (delta_aji >= 0.010 and delta_pq >= 0.010)
    ):
        return "STRONG_GO"
    if (0.005 <= delta_aji < 0.020 and delta_pq >= -0.003) or (0.005 <= delta_pq < 0.020 and delta_aji >= -0.003):
        return "CONDITIONAL_PROMISING"
    if abs(delta_aji) <= 0.003 and abs(delta_pq) <= 0.003:
        return "NO_GO"
    return "PROJECT_LEAD_REVIEW"


def logit_prior(probability: float = 0.01) -> float:
    return math.log(probability / (1.0 - probability))
