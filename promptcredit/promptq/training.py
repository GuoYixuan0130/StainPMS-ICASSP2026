"""Offline PromptQ quality-head-only cache training with fixed epoch 20."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr
import torch

from promptcredit.method import QualityTargets, quality_focal_loss
from promptcredit.metrics.utility import score_utility_summary
from promptcredit.promptq.cache import iter_cache_arrays


SEED = 3407
EPOCHS = 20
LR = 1e-4
WEIGHT_DECAY = 1e-4
PREFERRED_BATCH_SIZE = 4096


def _load_cache(manifest_path: Path) -> dict[str, torch.Tensor]:
    parts: dict[str, list[np.ndarray]] = {
        "features": [], "utility_target": [], "matched": [], "hard_mask_iou": [], "foreground_probability": []
    }
    for arrays in iter_cache_arrays(manifest_path):
        for name in parts:
            parts[name].append(arrays[name])
    if not parts["features"]:
        raise ValueError(f"PromptQ cache is empty: {manifest_path}")
    return {
        "features": torch.from_numpy(np.concatenate(parts["features"], axis=0).astype(np.float16)),
        "utility_target": torch.from_numpy(np.concatenate(parts["utility_target"], axis=0).astype(np.float32)),
        "matched": torch.from_numpy(np.concatenate(parts["matched"], axis=0).astype(np.bool_)),
        "hard_mask_iou": torch.from_numpy(np.concatenate(parts["hard_mask_iou"], axis=0).astype(np.float32)),
        "foreground_probability": torch.from_numpy(np.concatenate(parts["foreground_probability"], axis=0).astype(np.float32)),
    }


def _quality_targets(values: torch.Tensor, matched: torch.Tensor) -> QualityTargets:
    return QualityTargets(
        values=values.unsqueeze(0),
        matched_proposals=matched.unsqueeze(0),
        matched_count=int(matched.sum().item()),
        duplicate_source_events=0,
    )


def _spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 2 or np.all(left == left[0]) or np.all(right == right[0]):
        return None
    value = float(spearmanr(left, right).statistic)
    return value if np.isfinite(value) else None


@torch.no_grad()
def evaluate_quality_head(quality_head: torch.nn.Module, cache: dict[str, torch.Tensor], device: torch.device) -> dict[str, Any]:
    quality_head.eval()
    features = cache["features"].to(device=device, dtype=torch.float32)
    logits = quality_head(features).squeeze(-1)
    targets = _quality_targets(cache["utility_target"].to(device), cache["matched"].to(device))
    loss = quality_focal_loss(logits.unsqueeze(0), targets)
    quality_probability = torch.sigmoid(logits).cpu().numpy()
    hard_iou = cache["hard_mask_iou"].cpu().numpy()
    utility = cache["utility_target"].cpu().numpy()
    foreground = cache["foreground_probability"].cpu().numpy()
    positive = cache["matched"].cpu().numpy().astype(bool)
    # Unmatched sources deliberately have no decoder call and hence no decoded
    # IoU observation.  They still take target zero in QFL, but must not be
    # misreported as observed zero-IoU samples in calibration statistics.
    observed_quality = quality_probability[positive]
    observed_hard_iou = hard_iou[positive]
    observed_foreground = foreground[positive]
    observed_product = observed_foreground * observed_quality
    return {
        "quality_loss": float(loss.cpu()),
        "quality_target_spearman": _spearman(quality_probability[positive], utility[positive]),
        "quality_score_metrics": score_utility_summary(observed_quality, observed_hard_iou),
        "raw_objectness_metrics": score_utility_summary(observed_foreground, observed_hard_iou),
        "product_score_metrics": score_utility_summary(observed_product, observed_hard_iou),
        "positive_prediction_distribution": {
            "count": int(positive.sum()),
            "mean": float(quality_probability[positive].mean()) if positive.any() else None,
            "std": float(quality_probability[positive].std()) if positive.any() else None,
        },
        "negative_prediction_distribution": {
            "count": int((~positive).sum()),
            "mean": float(quality_probability[~positive].mean()) if (~positive).any() else None,
            "std": float(quality_probability[~positive].std()) if (~positive).any() else None,
        },
    }


def _try_epoch(
    quality_head: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train: dict[str, torch.Tensor],
    device: torch.device,
    batch_size: int,
    generator: torch.Generator,
) -> float:
    quality_head.train()
    indices = torch.randperm(train["features"].shape[0], generator=generator)
    losses: list[float] = []
    for start in range(0, len(indices), batch_size):
        index = indices[start:start + batch_size]
        feature = train["features"][index].to(device=device, dtype=torch.float32, non_blocking=True)
        target = train["utility_target"][index].to(device, non_blocking=True)
        matched = train["matched"][index].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = quality_head(feature).squeeze(-1)
        loss = quality_focal_loss(logits.unsqueeze(0), _quality_targets(target, matched))
        if not torch.isfinite(loss):
            raise FloatingPointError("PromptQ quality loss is non-finite")
        loss.backward()
        if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all() for parameter in quality_head.parameters()):
            raise FloatingPointError("PromptQ quality-head gradient is non-finite")
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def train_quality_head(
    *,
    quality_head: torch.nn.Module,
    train_manifest_path: Path,
    development_manifest_path: Path,
    out_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    """Train only the supplied quality head for exactly 20 epochs."""
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite PromptQ training artifacts: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(SEED)
    train = _load_cache(train_manifest_path)
    development = _load_cache(development_manifest_path)
    quality_head.to(device)
    optimizer = torch.optim.AdamW(quality_head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    batch_size = PREFERRED_BATCH_SIZE
    curves: list[dict[str, Any]] = []
    for epoch in range(1, EPOCHS + 1):
        while True:
            try:
                train_loss = _try_epoch(quality_head, optimizer, train, device, batch_size, generator)
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if batch_size <= 1:
                    raise
                batch_size //= 2
        train_metrics = evaluate_quality_head(quality_head, train, device)
        development_metrics = evaluate_quality_head(quality_head, development, device)
        curves.append(
            {
                "epoch": epoch,
                "batch_size": batch_size,
                "train_step_quality_loss": train_loss,
                "train_quality_loss": train_metrics["quality_loss"],
                "train_quality_target_spearman": train_metrics["quality_target_spearman"],
                "development_quality_loss": development_metrics["quality_loss"],
                "development_quality_target_spearman": development_metrics["quality_target_spearman"],
                "development_quality_auroc": development_metrics["quality_score_metrics"]["auroc_iou_ge_0_5"],
                "development_quality_auprc": development_metrics["quality_score_metrics"]["auprc_iou_ge_0_5"],
                "development_quality_brier": development_metrics["quality_score_metrics"]["brier_iou_ge_0_5"],
                "development_quality_ece": development_metrics["quality_score_metrics"]["ece_10_equal_frequency"],
                "development_positive_prediction_mean": development_metrics["positive_prediction_distribution"]["mean"],
                "development_negative_prediction_mean": development_metrics["negative_prediction_distribution"]["mean"],
            }
        )
    with (out_dir / "quality_training_curves.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in curves for key in row}))
        writer.writeheader()
        writer.writerows(curves)
    final_train = evaluate_quality_head(quality_head, train, device)
    final_development = evaluate_quality_head(quality_head, development, device)
    torch.save(quality_head.state_dict(), out_dir / "quality_head_epoch20.pt")
    report = {
        "epochs": EPOCHS,
        "seed": SEED,
        "optimizer": "AdamW",
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "batch_size": batch_size,
        "train_cache_rows": int(train["features"].shape[0]),
        "development_cache_rows": int(development["features"].shape[0]),
        "fixed_epoch": 20,
        "train": final_train,
        "development": final_development,
    }
    (out_dir / "calibration_metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
