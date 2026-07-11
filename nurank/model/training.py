"""Deterministic offline NuRank training on immutable four-token caches."""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from nurank.analysis.metrics import ranking_metrics
from nurank.cache.io import group_feature_matrix, iter_groups, load_manifest
from nurank.losses import regret_aware_loss
from nurank.model.ranker import NuRankSharedRanker, build_ranker


SEED, EPOCHS, BATCH_SIZE, LR, WEIGHT_DECAY = 3407, 30, 256, 1e-3, 1e-4


@dataclass(frozen=True)
class TrainingResult:
    ranker: NuRankSharedRanker
    checkpoint_path: Path
    normalization_path: Path
    curves_path: Path


def _set_seed() -> None:
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def _load_groups(cache_dir: Path, expected_role: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    manifest = load_manifest(cache_dir)
    if manifest["role"] != expected_role:
        raise ValueError(f"Expected {expected_role} cache, got {manifest['role']}")
    feature, target, matched = [], [], []
    for group in iter_groups(cache_dir):
        feature.append(group_feature_matrix(group))
        target.append(np.asarray(group["true_hard_iou"], dtype=np.float32))
        matched.append(np.asarray(group["matched"], dtype=np.bool_))
    if not feature:
        raise RuntimeError(f"NuRank {expected_role} cache has no automatic prompt groups")
    return torch.from_numpy(np.concatenate(feature)), torch.from_numpy(np.concatenate(target)), torch.from_numpy(np.concatenate(matched))


def _metrics(model: NuRankSharedRanker, features: torch.Tensor, targets: torch.Tensor, matched: torch.Tensor, device: torch.device) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        scores = model(features.to(device)).cpu().numpy()
        loss = regret_aware_loss(torch.from_numpy(scores), targets).copy()
    summary = ranking_metrics(scores, targets.numpy(), matched.numpy())
    for key in ("total", "calibration", "ranking"):
        summary[f"{key}_loss"] = float(loss[key])
    for name, selection in (("matched", matched.numpy()), ("unmatched", ~matched.numpy())):
        if selection.any():
            scoped = ranking_metrics(scores[selection], targets.numpy()[selection])
            for key, value in scoped.items():
                if not isinstance(value, np.ndarray):
                    summary[f"{name}_{key}"] = value
    return summary


def _serializable(row: dict[str, Any]) -> dict[str, Any]:
    return {key: (value.item() if isinstance(value, np.generic) else value) for key, value in row.items() if not isinstance(value, np.ndarray)}


def train_nurank(*, train_cache_dir: Path, development_cache_dir: Path, out_dir: Path, device: torch.device, batch_size: int = BATCH_SIZE) -> TrainingResult:
    """Train only the new ranker for fixed 30 epochs; development is diagnostics only."""
    if out_dir.exists():
        raise FileExistsError(f"NuRank training destination must be new: {out_dir}")
    out_dir.mkdir(parents=True)
    _set_seed()
    train_x, train_y, train_matched = _load_groups(train_cache_dir, "train")
    development_x, development_y, development_matched = _load_groups(development_cache_dir, "development")
    if set(load_manifest(train_cache_dir)["image_ids"]) & set(load_manifest(development_cache_dir)["image_ids"]):
        raise RuntimeError("NuRank train/development cache image leakage")
    mean, std = train_x[..., 256:].reshape(-1, 8).mean(dim=0), train_x[..., 256:].reshape(-1, 8).std(dim=0, unbiased=False).clamp_min(1e-6)
    model = build_ranker(scalar_mean=mean, scalar_std=std, seed=SEED).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    dataset = TensorDataset(train_x, train_y)
    generator = torch.Generator().manual_seed(SEED)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator, num_workers=0)
    curves: list[dict[str, Any]] = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        totals = {"total": 0.0, "calibration": 0.0, "ranking": 0.0}; count = 0
        for feature, target in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = regret_aware_loss(model(feature.to(device)), target.to(device))
            loss["total"].backward()
            optimizer.step()
            size = len(feature); count += size
            for key in totals: totals[key] += float(loss[key].detach()) * size
        train_metrics = _metrics(model, train_x, train_y, train_matched, device)
        development_metrics = _metrics(model, development_x, development_y, development_matched, device)
        curves.append({"epoch": epoch, **{f"optimization_{key}_loss": value / max(count, 1) for key, value in totals.items()}, **{f"train_{key}": value for key, value in _serializable(train_metrics).items()}, **{f"development_{key}": value for key, value in _serializable(development_metrics).items()}})
    normalization = {"scalar_feature_order": ["original_predicted_iou", "soft_area", "hard_area", "stability", "mean_probability", "mean_abs_logit", "boundary_entropy", "point_inside"], "mean": mean.tolist(), "std": std.tolist(), "train_cache": str(train_cache_dir), "development_cache": str(development_cache_dir)}
    normalization_path = out_dir / "feature_normalization.json"; normalization_path.write_text(json.dumps(normalization, indent=2) + "\n", encoding="utf-8")
    checkpoint_path = out_dir / "nurank_epoch_030.pt"
    torch.save({"schema": "nurank_ranker_v1", "epoch": EPOCHS, "seed": SEED, "model_state": model.cpu().state_dict(), "normalization": normalization, "parameter_count": model.parameter_count()}, checkpoint_path)
    model.to(device).eval()
    curves_path = out_dir / "training_curves.csv"
    fields = sorted({key for row in curves for key in row})
    with curves_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(curves)
    return TrainingResult(ranker=model, checkpoint_path=checkpoint_path, normalization_path=normalization_path, curves_path=curves_path)
