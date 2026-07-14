"""The single authorized PromptQ-v2 Primary-Metric Audit runner."""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from .cache import create_quality_targets, extract_role_cache
from .data import materialize_labels
from .evaluate import evaluate_development
from .model import configure_quality_only, frozen_checksums, load_frozen_models
from .protocol import CANONICAL_BASELINE_SHA, CHECKPOINT_SHA256, NMS_RADIUS, SEED, json_dump, set_determinism, sha256_file
from .training import cached_quality_logits, choose_lopo_epoch, train_final_quality_head


def _git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _environment() -> dict:
    try:
        gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], text=True).strip()
    except Exception:
        gpu = "unavailable"
    return {"platform": platform.platform(), "python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": gpu}


def _artifact_checksums(out_dir: Path) -> None:
    paths = sorted(path for path in out_dir.rglob("*") if path.is_file() and path.name != "SHA256SUMS")
    (out_dir / "SHA256SUMS").write_text("".join(f"{sha256_file(path)}  {path.relative_to(out_dir).as_posix()}\n" for path in paths), encoding="utf-8")


def _assert_no_gt_in_deployment_cache(cache_dir: Path) -> None:
    forbidden = ("gt", "label", "target", "instance")
    for path in cache_dir.glob("*.npz"):
        with np.load(path, allow_pickle=False) as payload:
            if any(any(token in key.lower() for token in forbidden) for key in payload.files):
                raise RuntimeError(f"GT leakage into deployment cache: {path}")


def _write_lead_markdown(path: Path, report: dict) -> None:
    primary = report["primary_metrics"]
    metrics = primary["metrics"]
    delta = primary["paired_delta"]
    lines = [
        "# PromptQ-v2 Primary-Metric Audit",
        "",
        f"Recommendation: **{report['recommendation']}**",
        "",
        "| Path | AJI | PQ | DQ | TP | FP | FN |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode, label in (("baseline", "Baseline"), ("product", "PromptQ-v2 product"), ("quality_only", "Quality-only diagnostic"), ("oracle", "GT-IoU oracle diagnostic")):
        value = metrics[mode]
        lines.append(f"| {label} | {value['aji']:.6f} | {value['pq']:.6f} | {value['dq']:.6f} | {value['tp']} | {value['fp']} | {value['fn']} |")
    lines.extend([
        "",
        f"Paired deltas (PromptQ-v2 − Baseline): AJI {delta['aji']:+.6f}; PQ {delta['pq']:+.6f}; DQ {delta['dq']:+.6f}.",
        "",
        "Patients 7--8 only; patients 9--11 and MoNuSeg were not opened. Spearman, ECE, and quality loss are mechanism diagnostics, not stopping gates.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_primary_metric_audit(*, data_root: Path, checkpoint: Path, manifest_path: Path, out_dir: Path, sam_config: str = "sam2_hiera_l") -> dict:
    """Cache once, train only quality head, and evaluate 7--8 exactly once."""
    if _git_sha() != CANONICAL_BASELINE_SHA:
        raise RuntimeError(f"PromptQ-v2 must run at {CANONICAL_BASELINE_SHA}; refusing {_git_sha()}")
    if sha256_file(checkpoint) != CHECKPOINT_SHA256:
        raise RuntimeError("PromptQ-v2 checkpoint SHA256 mismatch")
    if not torch.cuda.is_available():
        raise RuntimeError("PromptQ-v2 Primary-Metric Audit requires the authorized AutoDL GPU")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite immutable audit directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=False)
    set_determinism()
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    try:
        run_manifest = {
            "method": "PromptQ-v2 Primary-Metric Audit",
            "git_sha": _git_sha(), "canonical_baseline_sha": CANONICAL_BASELINE_SHA,
            "checkpoint": str(checkpoint), "checkpoint_sha256": CHECKPOINT_SHA256,
            "data_manifest": str(manifest_path), "data_manifest_sha256": sha256_file(manifest_path),
            "fixed_protocol": {"tta": False, "batch_size": 1, "seed": SEED, "nms_threshold": NMS_RADIUS, "inclusive_iou": ">= 0.5", "train_patients": [1, 2, 3, 4, 5, 6], "development_patients": [7, 8], "closed": ["TNBC patients 9-11", "MoNuSeg"]},
            "reference_tnbc_nms12": {"aji": 0.647064, "dq": 0.830578, "sq": 0.803866, "pq": 0.668077},
            "environment": _environment(),
        }
        json_dump(out_dir / "run_manifest.json", run_manifest)
        # Labels are materialized independently.  Cache extraction never
        # imports this directory and its NPZ files contain no GT keys.
        train_labels = materialize_labels(data_root, manifest_path, "train", out_dir / "labels_train")
        development_labels = materialize_labels(data_root, manifest_path, "development", out_dir / "labels_development")
        device = torch.device("cuda")
        train_bundle = load_frozen_models(checkpoint, sam_config, device)
        trainable_manifest = configure_quality_only(train_bundle)
        train_before = frozen_checksums(train_bundle)
        train_cache = extract_role_cache(train_bundle, data_root, manifest_path, "train", out_dir / "deployment_cache_train")
        _assert_no_gt_in_deployment_cache(out_dir / "deployment_cache_train")
        if frozen_checksums(train_bundle) != train_before:
            raise RuntimeError("train cache changed frozen point model or SAM2")
        # Fresh model/bank makes 7--8 cache extraction independent of the 1--6
        # cache and exactly preserves the canonical inference initialization.
        development_bundle = load_frozen_models(checkpoint, sam_config, device)
        configure_quality_only(development_bundle)
        development_before = frozen_checksums(development_bundle)
        development_cache = extract_role_cache(development_bundle, data_root, manifest_path, "development", out_dir / "deployment_cache_development")
        _assert_no_gt_in_deployment_cache(out_dir / "deployment_cache_development")
        if frozen_checksums(development_bundle) != development_before:
            raise RuntimeError("development cache changed frozen point model or SAM2")
        train_targets = create_quality_targets(out_dir / "deployment_cache_train", out_dir / "labels_train", out_dir / "quality_targets_train", role="train")
        development_targets = create_quality_targets(out_dir / "deployment_cache_development", out_dir / "labels_development", out_dir / "quality_targets_development", role="development")
        lopo = choose_lopo_epoch(out_dir / "deployment_cache_train", out_dir / "quality_targets_train", out_dir / "lopo", device)
        final_training = train_final_quality_head(train_bundle, out_dir / "deployment_cache_train", out_dir / "quality_targets_train", lopo["chosen_epoch"], out_dir / "quality_training")
        train_after = frozen_checksums(train_bundle)
        if train_after != train_before:
            raise RuntimeError("quality-head training changed frozen point model or SAM2")
        quality_logits = cached_quality_logits(train_bundle, out_dir / "deployment_cache_development")
        evaluation = evaluate_development(out_dir / "deployment_cache_development", out_dir / "quality_targets_development", out_dir / "labels_development", quality_logits, out_dir / "development_audit")
        report = {
            "recommendation": evaluation["verdict"], "primary_metrics": evaluation,
            "trainable_manifest": trainable_manifest,
            "frozen_checksums": {"train_before": train_before, "train_after": train_after, "development_before": development_before, "development_after_cache": frozen_checksums(development_bundle)},
            "cache": {"train": train_cache, "development": development_cache, "labels": {"train": train_labels, "development": development_labels}, "targets": {"train": train_targets, "development": development_targets}},
            "lopo": lopo, "final_training": final_training,
            "call_counts": {"train": train_cache["call_counts"], "development": development_cache["call_counts"], "offline_paths_model_calls": 0},
            "counterfactual_invariance": {"pred_coords": "one immutable deployment cache shared by Baseline/Product/Quality-only/Oracle", "decoded_masks": "one immutable deployment cache shared by all four paths", "gt_leakage": "deployment cache key audit passed"},
            "runtime_seconds": time.perf_counter() - started,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "development_access": "patients 7-8 only; exactly seven committed IDs",
            "closed_access": "patients 9-11 and MoNuSeg were never enumerated or opened",
        }
        json_dump(out_dir / "PROJECT_LEAD_REPORT.json", report)
        _write_lead_markdown(out_dir / "PROJECT_LEAD_REPORT.md", report)
        return report
    finally:
        _artifact_checksums(out_dir)
