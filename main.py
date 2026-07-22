import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from mmengine.config import Config
from torch.utils.data import DataLoader

import cfg
from conf import settings
from run.dataset.monuseg import MONUSEG
from run.dataset.tnbc import TNBC
from run.run_on_epoch import train_on_epoch, validation_on_epoch
from run.utils import create_logger, get_network, set_log_dir
from sam2_train.modeling.criterion import build_criterion
from sam2_train.modeling.dpa_p2pnet import build_model
from sam2_train.modeling.utils import collate_fn, set_seed
from stainpms.warmstart_protocol import (
    build_coverage_manifest,
    finalize_runtime_audits,
    new_timing_runtime_stats,
    timing_audit_isolation,
    validate_train_manifest_identity,
    verify_coverage_manifest,
)


def count_trainable_params(*modules):
    return sum(
        param.numel()
        for module in modules
        for param in module.parameters()
        if param.requires_grad
    )


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_ca_sam2_point_head_checkpoint(cfgs, model1):
    ckpt = torch.load(cfgs.sam_ckpt, map_location="cpu")
    if "model1" not in ckpt:
        print(f"[checkpoint] no CA-SAM2 point-head weights found in {cfgs.sam_ckpt}")
        return

    missing_keys, unexpected_keys = model1.load_state_dict(ckpt["model1"], strict=False)
    print(f"[checkpoint] loaded CA-SAM2 point head from {cfgs.sam_ckpt}")
    print(f"[checkpoint] model1 missing keys: {len(missing_keys)}")
    if missing_keys:
        print(f"[checkpoint] model1 missing sample: {missing_keys[:8]}")
    print(f"[checkpoint] model1 unexpected keys: {len(unexpected_keys)}")
    if unexpected_keys:
        print(f"[checkpoint] model1 unexpected sample: {unexpected_keys[:8]}")


def load_ca_sam2_texture_bank(cfgs):
    ckpt = torch.load(cfgs.sam_ckpt, map_location="cpu")
    bank = ckpt.get("texture_memory_bank_list", []) or []
    print(f"[checkpoint] loaded texture_memory_bank_list size={len(bank)}")
    return list(bank)


def refresh_baseline_masks_inplace(
    cfgs,
    args,
    train_dataset,
    test_dataset,
    model1,
    model1_encoder,
    net,
    val_texture_bank_template,
    epoch,
    device,
):
    """Refresh train-split coverage maps used by PMS self-bootstrap."""
    if not cfgs.baseline_masks_dir:
        print("[coverage-refresh] skipped: --baseline_masks_dir is empty")
        return None

    saved_dump_dir = getattr(cfgs, "dump_baseline_masks_dir", "") or ""
    refresh_dataset = build_eval_dataset(cfgs, args, split="train")
    cfgs.dump_baseline_masks_dir = cfgs.baseline_masks_dir

    temp_loader = DataLoader(
        refresh_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    print(
        "[coverage-refresh] train split inference only; "
        f"n={len(refresh_dataset.paths)}; dump={cfgs.baseline_masks_dir}; "
        f"manifest={cfgs.train_manifest or 'legacy_directory'}"
    )

    seg_pq = None
    try:
        val_bank = list(val_texture_bank_template) if val_texture_bank_template is not None else []
        net.eval()
        model1.eval()
        _, _, _, _, _, _, seg_pq = validation_on_epoch(
            cfgs,
            args,
            temp_loader,
            epoch,
            model1,
            model1_encoder,
            net,
            cfgs.load,
            args.data.post.iou_threshold,
            val_bank,
            device,
        )
    finally:
        cfgs.dump_baseline_masks_dir = saved_dump_dir

    n_reloaded = train_dataset.reload_baseline_masks()
    epoch_label = "init" if int(epoch) < 0 else f"epoch {epoch}"
    if seg_pq is not None:
        print(
            f"[coverage-refresh] {epoch_label}: reloaded {n_reloaded} maps; "
            f"train PQ={seg_pq * 100:.2f}"
        )
    return seg_pq


def apply_cli_overrides(args, cfgs):
    if cfgs.pms_loss_coef >= 0:
        args.criterion.pms_loss_coef = float(cfgs.pms_loss_coef)
    if cfgs.pms_object_weight >= 0:
        args.criterion.pms_object_weight = float(cfgs.pms_object_weight)
    if cfgs.pms_residual_mask_weight >= 0:
        args.criterion.pms_residual_mask_weight = float(cfgs.pms_residual_mask_weight)
    if cfgs.pms_preserve_loss_coef >= 0:
        args.criterion.pms_preserve_loss_coef = float(cfgs.pms_preserve_loss_coef)
    if cfgs.pms_gt_match_radius >= 0:
        args.criterion.pms_gt_match_radius = int(cfgs.pms_gt_match_radius)
    if cfgs.pms_baseline_prompts or cfgs.pms_preserve_covered:
        args.criterion.pms_baseline_prompts = True
    if cfgs.pms_preserve_max_prompts >= 0:
        args.criterion.pms_preserve_max_prompts = int(cfgs.pms_preserve_max_prompts)

    if cfgs.stain_baseline_dilate_radius >= 0:
        args.criterion.stain_baseline_dilate_radius = int(cfgs.stain_baseline_dilate_radius)
    if cfgs.stain_min_distance >= 0:
        args.criterion.stain_min_distance = int(cfgs.stain_min_distance)
    if cfgs.stain_top_k >= 0:
        args.criterion.stain_top_k = int(cfgs.stain_top_k)
    if cfgs.stain_sigma >= 0:
        args.criterion.stain_sigma = float(cfgs.stain_sigma)
    if cfgs.stain_merge_aware:
        args.criterion.stain_merge_aware = True
    if cfgs.stain_merge_min_distance >= 0:
        args.criterion.stain_merge_min_distance = int(cfgs.stain_merge_min_distance)
    if cfgs.stain_merge_num_peaks >= 0:
        args.criterion.stain_merge_num_peaks = int(cfgs.stain_merge_num_peaks)

    if cfgs.test_nms_thr >= 0:
        args.test.nms_thr = int(cfgs.test_nms_thr)
    if cfgs.test_filtering in ("true", "false"):
        args.test.filtering = cfgs.test_filtering == "true"


DATASET_CLASSES = {
    "monuseg": MONUSEG,
    "tnbc": TNBC,
}


def dataset_class_for(dataset_name):
    try:
        return DATASET_CLASSES[str(dataset_name).lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported dataset: {dataset_name}") from exc


def build_eval_dataset(cfgs, args, *, split):
    if split == "train":
        manifest_path = cfgs.train_manifest
        data_split = "train"
    elif split == "eval":
        manifest_path = cfgs.eval_manifest
        data_split = "train" if manifest_path else "test"
    else:
        raise ValueError(f"Unsupported evaluation split selector: {split}")
    dataset_class = dataset_class_for(cfgs.dataset)
    return dataset_class(
        cfgs,
        args,
        cfgs.data_path,
        cfgs.load,
        mode="test",
        manifest_path=manifest_path or None,
        data_split=data_split,
        verify_manifest_hashes=cfgs.verify_manifest_hashes,
    )


def build_dataloaders(cfgs, args):
    dataset_class = dataset_class_for(cfgs.dataset)
    train_dataset = dataset_class(
        cfgs,
        args,
        cfgs.data_path,
        cfgs.load,
        mode="train",
        manifest_path=cfgs.train_manifest or None,
        data_split="train",
        verify_manifest_hashes=cfgs.verify_manifest_hashes,
    )
    smoke_steps = int(cfgs.train_only_smoke_steps or 0)
    phase2a_timing = bool(cfgs.phase2a_timing_profile)
    warmstart_train_only = bool(cfgs.warmstart_stage)
    phase2a_no_eval = bool(cfgs.phase2a_baseline and cfgs.phase2a_eval_policy == "none")
    train_only_protocol = smoke_steps > 0 or phase2a_timing or warmstart_train_only
    if train_only_protocol:
        if not cfgs.train_manifest:
            raise ValueError("train-only protocol requires --train_manifest")
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=False,
        # A smoke must not prefetch unreported samples.  Full training retains
        # the configured worker count below its separate branch.
        num_workers=0 if train_only_protocol else cfgs.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    if train_only_protocol or phase2a_no_eval:
        return train_dataset, None, train_loader, None

    if cfgs.train_manifest and not cfgs.eval_manifest:
        raise ValueError(
            "manifest-backed training requires an explicit --eval_manifest; "
            "the legacy test directory will not be selected implicitly"
        )

    test_dataset = build_eval_dataset(cfgs, args, split="eval")
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfgs.num_workers,
        pin_memory=True,
    )
    return train_dataset, test_dataset, train_loader, test_loader


def run_train_only_smoke(
    cfgs,
    args,
    train_dataset,
    train_loader,
    model1,
    model1_encoder,
    net,
    criterion,
    optimizer,
    device,
    val_texture_bank_template,
):
    if not cfgs.smoke_output:
        raise ValueError("--smoke_output is required for train-only smoke")
    runtime_stats = {}
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    net.train()
    model1.train()
    started = time.perf_counter()
    log_info = train_on_epoch(
        cfgs,
        model1,
        model1_encoder,
        net,
        train_loader,
        criterion,
        optimizer,
        0,
        [],
        device,
        runtime_stats=runtime_stats,
        max_optimizer_steps=int(cfgs.train_only_smoke_steps),
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    images_seen = int(runtime_stats.get("images_seen", 0))
    optimizer_steps = int(runtime_stats.get("optimizer_steps", 0))
    checkpoint_path = Path(cfgs.sam_ckpt).resolve()
    try:
        driver = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).splitlines()[0].strip()
    except (OSError, subprocess.CalledProcessError, IndexError):
        driver = None
    gpu_name = torch.cuda.get_device_name(device) if torch.cuda.is_available() else None
    peak_allocated = (
        int(torch.cuda.max_memory_allocated(device)) if torch.cuda.is_available() else 0
    )
    peak_reserved = (
        int(torch.cuda.max_memory_reserved(device)) if torch.cuda.is_available() else 0
    )
    requested_optimizer_steps = int(cfgs.train_only_smoke_steps)
    all_losses_finite = bool(log_info) and all(
        np.isfinite(value) for value in log_info.values()
    )
    skip_count = sum(
        int(runtime_stats.get(name, 0))
        for name in (
            "shape_skips",
            "nonfinite_loss_skips",
            "nonfinite_gradient_skips",
        )
    )
    smoke_complete = (
        optimizer_steps == requested_optimizer_steps
        and skip_count == 0
        and all_losses_finite
    )
    report = {
        "schema_version": 1,
        "phase": "0.5",
        "status": "complete" if smoke_complete else "issues_found",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "train_only_smoke_no_development_or_test_loader",
        "command": list(sys.argv),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "cuda_available": torch.cuda.is_available(),
            "gpu": gpu_name,
            "driver": driver,
        },
        "determinism": {
            "seed": int(cfgs.seed),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        },
        "repository": {
            "branch": subprocess.check_output(
                ["git", "branch", "--show-current"], text=True
            ).strip(),
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
        },
        "data": {
            "manifest_path": str(cfgs.train_manifest),
            "manifest_sha256": train_dataset.manifest.get("manifest_sha256"),
            "protocol_id": train_dataset.manifest.get("protocol_id"),
            "full_manifest_image_count": len(train_dataset),
            "smoke_image_count": images_seen,
            "smoke_requested_optimizer_steps": requested_optimizer_steps,
            "sample_ids": train_dataset.sample_names[:images_seen],
            "hashes_verified": bool(cfgs.verify_manifest_hashes),
        },
        "initialization": {
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "task_warm_start_detected": val_texture_bank_template is not None,
            "policy": "generic SAM2 allowed; task checkpoint exposed to development is forbidden",
        },
        "runtime": {
            **runtime_stats,
            "wall_seconds": elapsed,
            "wall_seconds_per_image": elapsed / images_seen if images_seen else None,
            "wall_seconds_per_optimizer_step": (
                elapsed / optimizer_steps if optimizer_steps else None
            ),
            "extrapolated_full_train_epoch_seconds": None,
            "peak_memory_allocated_bytes": peak_allocated,
            "peak_memory_reserved_bytes": peak_reserved,
            "peak_memory_allocated_mib": peak_allocated / (1024 ** 2),
            "peak_memory_reserved_mib": peak_reserved / (1024 ** 2),
        },
        "numerics": {
            "losses": {key: float(value) for key, value in log_info.items()},
            "all_losses_finite": all_losses_finite,
        },
        "preliminary_budget": {
            "basis": "per-optimizer-step measurement from an exact 1-2 update train-only smoke",
            "estimates_by_epoch_count": None,
            "formal_budget_status": (
                "not_estimated_until the owner locks total crop/update budget, "
                "epoch count, and validation cadence"
            ),
        },
        "sealed_data_attestation": {
            "eval_manifest": None,
            "development_loader_constructed": False,
            "test_loader_constructed": False,
        },
    }
    output = Path(cfgs.smoke_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(f"[train-only-smoke] wrote {output}; status={report['status']}")
    return report


def _cuda_device_index(device):
    if isinstance(device, torch.device):
        return device.index if device.index is not None else torch.cuda.current_device()
    return int(device)


def run_phase2a_timing(
    cfgs,
    args,
    train_dataset,
    train_loader,
    model1,
    model1_encoder,
    net,
    criterion,
    optimizer,
    device,
    *,
    coverage_refresh_record,
):
    """Measure exactly 10 warm-up and 100 synchronized optimizer updates."""

    warmup_updates = int(cfgs.phase2a_warmup_updates)
    timed_updates = int(cfgs.phase2a_timed_updates)
    if warmup_updates <= 0 or timed_updates <= 0:
        raise ValueError("Phase 2A warm-up and timed update counts must be positive")
    output = Path(cfgs.phase2a_timing_output).resolve()
    if output.exists():
        raise ValueError(f"Phase 2A timing output already exists: {output}")
    checkpoint_path = Path(cfgs.sam_ckpt).resolve()
    checkpoint_sha256 = sha256_file(checkpoint_path)
    if checkpoint_sha256 != str(cfgs.phase2a_generic_checkpoint_sha256).lower():
        raise ValueError(
            "Phase 2A initialization SHA256 mismatch: "
            f"{checkpoint_sha256} != {cfgs.phase2a_generic_checkpoint_sha256}"
        )

    warmup_stats = {
        "capture_gradient_audit": False,
        "collect_candidate_audit": False,
    }
    warmup_losses = train_on_epoch(
        cfgs,
        model1,
        model1_encoder,
        net,
        train_loader,
        criterion,
        optimizer,
        0,
        [],
        device,
        runtime_stats=warmup_stats,
        max_optimizer_steps=warmup_updates,
    )
    if int(warmup_stats.get("optimizer_steps", 0)) != warmup_updates:
        raise RuntimeError(f"Phase 2A warm-up did not reach {warmup_updates} updates")

    timed_stats = {
        "capture_gradient_audit": False,
        "collect_candidate_audit": False,
    }
    cuda_index = _cuda_device_index(device) if torch.cuda.is_available() else None
    if torch.cuda.is_available():
        torch.cuda.synchronize(cuda_index)
        torch.cuda.reset_peak_memory_stats(cuda_index)
    started = time.perf_counter()
    timed_losses = train_on_epoch(
        cfgs,
        model1,
        model1_encoder,
        net,
        train_loader,
        criterion,
        optimizer,
        0,
        [],
        device,
        runtime_stats=timed_stats,
        max_optimizer_steps=timed_updates,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize(cuda_index)
    elapsed = time.perf_counter() - started
    peak_allocated = int(torch.cuda.max_memory_allocated(cuda_index)) if torch.cuda.is_available() else 0
    peak_reserved = int(torch.cuda.max_memory_reserved(cuda_index)) if torch.cuda.is_available() else 0

    skip_keys = ("shape_skips", "nonfinite_loss_skips", "nonfinite_gradient_skips")
    timed_skip_count = sum(int(timed_stats.get(key, 0)) for key in skip_keys)
    numerics_finite = bool(timed_losses) and all(
        np.isfinite(float(value)) for value in timed_losses.values()
    )
    completed = (
        int(timed_stats.get("optimizer_steps", 0)) == timed_updates
        and timed_skip_count == 0
        and numerics_finite
    )
    manifest = train_dataset.manifest or {}
    report = {
        "schema_version": 1,
        "phase": "2A",
        "status": "complete" if completed else "issues_found",
        "protocol": "phase2a_train_only_synchronized_update_timing_v1",
        "profile": str(cfgs.phase2a_timing_profile),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": list(sys.argv),
        "repository": {
            "branch": subprocess.check_output(["git", "branch", "--show-current"], text=True).strip(),
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        },
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(cuda_index) if torch.cuda.is_available() else None,
        },
        "determinism": {
            "seed": int(cfgs.seed),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        },
        "initialization": {
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha256,
            "expected_generic_checkpoint_sha256": str(cfgs.phase2a_generic_checkpoint_sha256),
            "point_head_task_checkpoint_loaded": False,
        },
        "data": {
            "manifest_path": str(cfgs.train_manifest),
            "manifest_sha256": manifest.get("manifest_sha256"),
            "protocol_id": manifest.get("protocol_id"),
            "record_count": len(train_dataset),
            "hashes_verified": bool(cfgs.verify_manifest_hashes),
            "eval_manifest": None,
        },
        "objective": {
            "use_pms": bool(cfgs.use_pms),
            "pms_loss_coef": float(criterion.pms_loss_coef),
            "pms_self_bootstrap": bool(cfgs.pms_self_bootstrap),
            "pms_start_epoch": int(cfgs.pms_start_epoch),
            "coverage_refresh_interval_epochs": int(cfgs.iterative_baseline_refresh_every),
            "coverage_accumulate": bool(cfgs.coverage_accumulate),
        },
        "coverage_refresh": coverage_refresh_record,
        "warmup": {
            **warmup_stats,
            "requested_optimizer_updates": warmup_updates,
            "losses": {key: float(value) for key, value in warmup_losses.items()},
        },
        "timed": {
            **timed_stats,
            "requested_optimizer_updates": timed_updates,
            "wall_seconds": elapsed,
            "seconds_per_optimizer_update": elapsed / timed_updates,
            "peak_memory_allocated_mib": peak_allocated / (1024 ** 2),
            "peak_memory_reserved_mib": peak_reserved / (1024 ** 2),
            "losses": {key: float(value) for key, value in timed_losses.items()},
            "all_losses_finite": numerics_finite,
        },
        "sealed_data_attestation": {
            "development_loader_constructed": False,
            "test_loader_constructed": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(f"[phase2a-timing] wrote {output}; status={report['status']}")
    return report


def maybe_load_warm_start(cfgs, model1):
    if cfgs.eval:
        return None
    ckpt = torch.load(cfgs.sam_ckpt, map_location="cpu")
    has_point_head = "model1" in ckpt
    del ckpt
    if not has_point_head:
        return None
    load_ca_sam2_point_head_checkpoint(cfgs, model1)
    if cfgs.warmstart_stage:
        print(
            "[warmstart] loaded model/model1 weights only; embedded texture bank discarded"
        )
        return None
    return load_ca_sam2_texture_bank(cfgs)


def _repository_identity():
    status_lines = [
        line
        for line in subprocess.check_output(
            ["git", "status", "--short"], text=True
        ).splitlines()
        if line.strip()
    ]
    return {
        "branch": subprocess.check_output(
            ["git", "branch", "--show-current"], text=True
        ).strip(),
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "dirty": bool(status_lines),
        "dirty_files": status_lines,
    }


def _warmstart_training_configuration(cfgs, args, model1, net, optimizer, scheduler):
    decoder = net.sam_mask_decoder
    return {
        "arm": str(cfgs.warmstart_candidate_arm),
        "optimizer": {
            "type": "AdamW",
            "state_source": "fresh",
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "weight_decay": float(optimizer.param_groups[0]["weight_decay"]),
            "betas": list(optimizer.param_groups[0]["betas"]),
            "eps": float(optimizer.param_groups[0]["eps"]),
            "amsgrad": bool(optimizer.param_groups[0]["amsgrad"]),
            "maximize": bool(optimizer.param_groups[0]["maximize"]),
            "foreach": optimizer.param_groups[0].get("foreach"),
            "capturable": bool(optimizer.param_groups[0].get("capturable", False)),
            "differentiable": bool(
                optimizer.param_groups[0].get("differentiable", False)
            ),
        },
        "scheduler": {
            "type": type(scheduler).__name__,
            "state_source": "fresh",
            "milestones": list(cfgs.lr_milestones),
            "gamma": 0.3,
            "last_epoch_without_step_calls": int(scheduler.last_epoch),
            "step_calls_during_smoke_or_timing": 0,
        },
        "amp": {"enabled": False},
        "gradient_clipping": {
            "enabled": float(cfgs.clip_grad) > 0,
            "max_norm": float(cfgs.clip_grad),
            "audit_norm_position": "before_clipping",
        },
        "trainable_parameters": {
            "point_head": count_trainable_params(model1),
            "sam2": count_trainable_params(net),
            "total": count_trainable_params(model1, net),
            "image_encoder_policy": "frozen except prompt_generator",
            "C1_new_parameters": 0,
        },
        "decoder": {
            "native_mask_token_count": int(decoder.num_mask_tokens),
            "legacy_and_C0_supervised_token": 0,
            "C0_C1_common_forward": "sam_mask_decoder.predict_masks tokens 0..3",
            "multimask_flag_difference_between_C0_C1": False,
            "decoder_call_count_difference_between_C0_C1": False,
        },
        "objective": {
            "stainpms_preserved": True,
            "candidate_coverage_tau": float(cfgs.candidate_coverage_tau),
            "candidate_coverage_coefficient": float(
                cfgs.candidate_coverage_coefficient
            ),
            "candidate_quality_coefficient": float(
                cfgs.candidate_quality_coefficient
            ),
            "pms_loss_coef": float(args.criterion.pms_loss_coef),
            "pms_residual_mask_weight": float(
                args.criterion.pms_residual_mask_weight
            ),
            "pms_preserve_loss_coef": float(args.criterion.pms_preserve_loss_coef),
        },
        "data_order": {
            "shuffle": False,
            "crop_batch_size": int(cfgs.b),
            "seed": int(cfgs.seed),
        },
    }


def run_warmstart_prepare_coverage(
    cfgs,
    args,
    train_dataset,
    model1,
    model1_encoder,
    net,
    device,
    train_manifest_identity,
):
    output = Path(cfgs.warmstart_output).resolve()
    checkpoint_path = Path(cfgs.sam_ckpt).resolve()
    checkpoint_sha = sha256_file(checkpoint_path)
    started = time.perf_counter()
    refresh_baseline_masks_inplace(
        cfgs,
        args,
        train_dataset,
        None,
        model1,
        model1_encoder,
        net,
        None,
        -1,
        device,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize(_cuda_device_index(device))
    elapsed = time.perf_counter() - started
    report = build_coverage_manifest(
        cache_dir=Path(cfgs.baseline_masks_dir),
        train_manifest_identity=train_manifest_identity,
        dataset=str(cfgs.dataset),
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha,
        wall_seconds=elapsed,
        repository=_repository_identity(),
        command=list(sys.argv),
    )
    _json_write_atomic(output, report)
    print(json.dumps({"status": "complete", "coverage_manifest": str(output)}))
    return report


def _warmstart_base_report(
    cfgs,
    args,
    train_dataset,
    model1,
    net,
    optimizer,
    scheduler,
    coverage_identity,
):
    checkpoint_path = Path(cfgs.sam_ckpt).resolve()
    return {
        "schema_version": 1,
        "phase": "2A-warmstart-feasibility",
        "protocol": "native_candidate_C0_C1_train_only_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": list(sys.argv),
        "repository": _repository_identity(),
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(_cuda_device_index(torch.device("cuda")))
            if torch.cuda.is_available()
            else None,
        },
        "initialization": {
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "loaded_fields": ["model", "model1"],
            "optimizer_scheduler_rng_loaded": False,
            "embedded_texture_bank_loaded": False,
            "evidence_class": "exploratory_weight_warm_start",
        },
        "data": {
            "manifest_path": str(cfgs.train_manifest),
            "manifest_sha256": train_dataset.manifest.get("manifest_sha256"),
            "protocol_id": train_dataset.manifest.get("protocol_id"),
            "record_count": len(train_dataset),
            "hashes_verified": bool(cfgs.verify_manifest_hashes),
            "coverage": coverage_identity,
            "eval_manifest": None,
        },
        "training_configuration": _warmstart_training_configuration(
            cfgs, args, model1, net, optimizer, scheduler
        ),
        "determinism": {
            "seed": int(cfgs.seed),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        },
        "sealed_data_attestation": {
            "development_loader_constructed": False,
            "test_loader_constructed": False,
            "TNBC_p7_p11_accessed": False,
            "MoNuSeg_test14_accessed": False,
        },
    }


def run_warmstart_smoke(
    cfgs,
    args,
    train_dataset,
    train_loader,
    model1,
    model1_encoder,
    net,
    criterion,
    optimizer,
    scheduler,
    device,
    coverage_identity,
):
    requested = int(cfgs.warmstart_smoke_updates)
    runtime_stats = {}
    cuda_index = _cuda_device_index(device) if torch.cuda.is_available() else None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(cuda_index)
        torch.cuda.synchronize(cuda_index)
    started = time.perf_counter()
    losses = train_on_epoch(
        cfgs,
        model1,
        model1_encoder,
        net,
        train_loader,
        criterion,
        optimizer,
        0,
        [],
        device,
        runtime_stats=runtime_stats,
        max_optimizer_steps=requested,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize(cuda_index)
    elapsed = time.perf_counter() - started
    finite = bool(losses) and all(np.isfinite(float(value)) for value in losses.values())
    skips = sum(
        int(runtime_stats.get(key, 0))
        for key in ("shape_skips", "nonfinite_loss_skips", "nonfinite_gradient_skips")
    )
    arm = str(cfgs.warmstart_candidate_arm)
    four_candidate_ok = arm == "legacy" or (
        int(runtime_stats.get("native_mask_token_count", 0)) == 4
        and int(runtime_stats.get("native_candidate_decoder_calls", 0)) > 0
    )
    candidate_ok = arm != "c1" or "candidate_loss_audit" in runtime_stats
    complete = (
        int(runtime_stats.get("optimizer_steps", 0)) == requested
        and skips == 0
        and finite
        and four_candidate_ok
        and candidate_ok
    )
    report = _warmstart_base_report(
        cfgs,
        args,
        train_dataset,
        model1,
        net,
        optimizer,
        scheduler,
        coverage_identity,
    )
    report.update(
        {
            "status": "complete" if complete else "issues_found",
            "stage": "smoke",
            "requested_optimizer_updates": requested,
            "losses": {key: float(value) for key, value in losses.items()},
            "runtime": {
                **finalize_runtime_audits(runtime_stats),
                "wall_seconds": elapsed,
                "seconds_per_optimizer_update": elapsed / requested,
                "peak_memory_allocated_mib": (
                    torch.cuda.max_memory_allocated(cuda_index) / (1024**2)
                    if torch.cuda.is_available()
                    else 0.0
                ),
                "peak_memory_reserved_mib": (
                    torch.cuda.max_memory_reserved(cuda_index) / (1024**2)
                    if torch.cuda.is_available()
                    else 0.0
                ),
            },
            "numerical_gate": {
                "losses_finite": finite,
                "skipped_updates": skips,
                "four_candidate_forward_verified": four_candidate_ok,
                "candidate_audit_present_when_required": candidate_ok,
            },
        }
    )
    _json_write_atomic(Path(cfgs.warmstart_output).resolve(), report)
    print(json.dumps({"status": report["status"], "output": cfgs.warmstart_output}))
    return report


def run_warmstart_timing(
    cfgs,
    args,
    train_dataset,
    train_loader,
    model1,
    model1_encoder,
    net,
    criterion,
    optimizer,
    scheduler,
    device,
    coverage_identity,
):
    warmup_updates = int(cfgs.phase2a_warmup_updates)
    timed_updates = int(cfgs.phase2a_timed_updates)
    warmup_stats = new_timing_runtime_stats()
    warmup_losses = train_on_epoch(
        cfgs,
        model1,
        model1_encoder,
        net,
        train_loader,
        criterion,
        optimizer,
        0,
        [],
        device,
        runtime_stats=warmup_stats,
        max_optimizer_steps=warmup_updates,
    )
    warmup_skips = sum(
        int(warmup_stats.get(key, 0))
        for key in ("shape_skips", "nonfinite_loss_skips", "nonfinite_gradient_skips")
    )
    warmup_finite = bool(warmup_losses) and all(
        np.isfinite(float(value)) for value in warmup_losses.values()
    )
    if (
        int(warmup_stats.get("optimizer_steps", 0)) != warmup_updates
        or warmup_skips != 0
        or not warmup_finite
    ):
        raise RuntimeError(
            "warm-start timing warm-up gate failed: "
            f"updates={warmup_stats.get('optimizer_steps')} skips={warmup_skips} "
            f"finite={warmup_finite}"
        )
    timed_stats = new_timing_runtime_stats()
    cuda_index = _cuda_device_index(device) if torch.cuda.is_available() else None
    if torch.cuda.is_available():
        torch.cuda.synchronize(cuda_index)
        torch.cuda.reset_peak_memory_stats(cuda_index)
    started = time.perf_counter()
    timed_losses = train_on_epoch(
        cfgs,
        model1,
        model1_encoder,
        net,
        train_loader,
        criterion,
        optimizer,
        0,
        [],
        device,
        runtime_stats=timed_stats,
        max_optimizer_steps=timed_updates,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize(cuda_index)
    elapsed = time.perf_counter() - started
    skips = sum(
        int(timed_stats.get(key, 0))
        for key in ("shape_skips", "nonfinite_loss_skips", "nonfinite_gradient_skips")
    )
    finite = bool(timed_losses) and all(
        np.isfinite(float(value)) for value in timed_losses.values()
    )
    warmup_audit_isolation = timing_audit_isolation(warmup_stats)
    timed_audit_isolation = timing_audit_isolation(timed_stats)
    complete = (
        int(timed_stats.get("optimizer_steps", 0)) == timed_updates
        and skips == 0
        and finite
        and int(timed_stats.get("native_mask_token_count", 0)) == 4
        and warmup_audit_isolation["status"] == "pass"
        and timed_audit_isolation["status"] == "pass"
    )
    report = _warmstart_base_report(
        cfgs,
        args,
        train_dataset,
        model1,
        net,
        optimizer,
        scheduler,
        coverage_identity,
    )
    report.update(
        {
            "status": "complete" if complete else "issues_found",
            "stage": "timing",
            "timing_audit_isolation": {
                "warmup": warmup_audit_isolation,
                "timed": timed_audit_isolation,
            },
            "warmup": {
                **finalize_runtime_audits(warmup_stats),
                "requested_optimizer_updates": warmup_updates,
                "losses": {key: float(value) for key, value in warmup_losses.items()},
                "all_losses_finite": warmup_finite,
                "skipped_updates": warmup_skips,
            },
            "timed": {
                **finalize_runtime_audits(timed_stats),
                "requested_optimizer_updates": timed_updates,
                "losses": {key: float(value) for key, value in timed_losses.items()},
                "wall_seconds": elapsed,
                "seconds_per_optimizer_update": elapsed / timed_updates,
                "peak_memory_allocated_mib": (
                    torch.cuda.max_memory_allocated(cuda_index) / (1024**2)
                    if torch.cuda.is_available()
                    else 0.0
                ),
                "peak_memory_reserved_mib": (
                    torch.cuda.max_memory_reserved(cuda_index) / (1024**2)
                    if torch.cuda.is_available()
                    else 0.0
                ),
                "all_losses_finite": finite,
                "skipped_updates": skips,
            },
        }
    )
    _json_write_atomic(Path(cfgs.warmstart_output).resolve(), report)
    print(json.dumps({"status": report["status"], "output": cfgs.warmstart_output}))
    return report


def run_warmstart_formal_tnbc_5epoch(
    cfgs,
    args,
    train_dataset,
    train_loader,
    model1,
    model1_encoder,
    net,
    criterion,
    optimizer,
    scheduler,
    device,
    coverage_identity,
    train_manifest_identity,
):
    """Run the owner-approved fixed-budget exploratory TNBC C0/C1 screen.

    This intentionally constructs no development loader.  Evaluation is run
    subsequently against each immutable saved epoch checkpoint, which keeps
    p7--p8 out of the optimizer process and makes C0/C1 state isolation
    auditable.  The shared coverage cache is verified before and after the
    five epochs and is never refreshed in this stage.
    """
    output = Path(cfgs.warmstart_output).resolve()
    output_dir = output.parent
    checkpoints_dir = output_dir / "checkpoints"
    declarations_dir = output_dir / "checkpoint_declarations"

    attempted_crop_batches_per_epoch = 270
    planned_attempted_crop_batches = 1350
    if len(train_dataset) != 30:
        raise RuntimeError("formal TNBC screen requires exactly 30 p1-p6 images")
    resume_checkpoint_arg = str(getattr(cfgs, "warmstart_resume_checkpoint", "") or "")
    recovery = None
    if resume_checkpoint_arg:
        resume_path = Path(resume_checkpoint_arg).resolve()
        if resume_path.parent != checkpoints_dir:
            raise RuntimeError("formal TNBC recovery checkpoint is outside this arm's checkpoint directory")
        if output.exists():
            raise RuntimeError("formal TNBC recovery refuses an already-complete training summary")
        if not checkpoints_dir.is_dir() or not declarations_dir.is_dir():
            raise RuntimeError("formal TNBC recovery requires existing checkpoint and declaration directories")

        resume_state = torch.load(resume_path, map_location="cpu")
        required = {
            "phase": "2A-warmstart-formal-screen",
            "protocol": "tnbc_c0_c1_5epoch_exploratory_v1",
            "dataset": "tnbc",
            "arm": str(cfgs.warmstart_candidate_arm),
            "train_manifest": train_manifest_identity,
            "coverage": coverage_identity,
            "screen_config": getattr(cfgs, "warmstart_screen_config_identity", None),
            "texture_memory_bank_list": [],
            "embedded_texture_bank_loaded": False,
            "coverage_refresh_events": [],
        }
        mismatched = {
            name: (resume_state.get(name), expected)
            for name, expected in required.items()
            if resume_state.get(name) != expected
        }
        if mismatched:
            raise RuntimeError(f"formal TNBC recovery provenance mismatch: {mismatched}")
        resume_epoch = int(resume_state.get("epoch", -1))
        if resume_epoch < 1 or resume_epoch >= 5:
            raise RuntimeError("formal TNBC recovery checkpoint must be from completed epoch 1--4")
        stored_paths = sorted(checkpoints_dir.glob("*.pth"))
        if len(stored_paths) != resume_epoch or resume_path not in stored_paths:
            raise RuntimeError(
                "formal TNBC recovery requires exactly the contiguous pre-recovery epoch checkpoints"
            )

        runtime_stats = dict(resume_state.get("runtime_stats", {}))
        if runtime_stats.get("record_no_prompt_batches") is not True:
            raise RuntimeError("formal TNBC recovery checkpoint lacks no-prompt position auditing")
        if int(runtime_stats.get("crop_batches_seen", -1)) != resume_epoch * attempted_crop_batches_per_epoch:
            raise RuntimeError("formal TNBC recovery attempted-crop history is inconsistent")
        if len(runtime_stats.get("no_prompt_batches", [])) != int(
            runtime_stats.get("no_prompt_batch_count", -1)
        ):
            raise RuntimeError("formal TNBC recovery no-prompt history is inconsistent")

        epoch_records = []
        prior_states = []
        for path in stored_paths:
            state = resume_state if path == resume_path else torch.load(path, map_location="cpu")
            state_mismatched = {
                name: (state.get(name), expected)
                for name, expected in required.items()
                if state.get(name) != expected
            }
            if state_mismatched:
                raise RuntimeError(
                    f"formal TNBC recovery prior checkpoint provenance mismatch for {path}: "
                    f"{state_mismatched}"
                )
            epoch_number = int(state.get("epoch", -1))
            local_attempted = int(state.get("attempted_crop_batches", -1))
            local_updates = int(state.get("effective_optimizer_updates", -1))
            local_no_prompt = int(state.get("no_prompt_batch_count", -1))
            local_positions = list(state.get("no_prompt_batch_indices", []))
            if (
                epoch_number != len(prior_states) + 1
                or local_attempted != attempted_crop_batches_per_epoch
                or local_updates + local_no_prompt != local_attempted
                or len(local_positions) != local_no_prompt
            ):
                raise RuntimeError(f"formal TNBC recovery checkpoint contract failed: {path}")
            calculated_positions_sha = hashlib.sha256(
                json.dumps(local_positions, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            if calculated_positions_sha != state.get("no_prompt_batch_indices_sha256"):
                raise RuntimeError(f"formal TNBC recovery no-prompt hash mismatch: {path}")
            checkpoint_sha = sha256_file(path)
            declaration_path = declarations_dir / f"{path.stem}.json"
            if not declaration_path.is_file():
                raise RuntimeError(f"formal TNBC recovery declaration is missing: {declaration_path}")
            declaration = json.loads(declaration_path.read_text(encoding="utf-8"))
            if declaration.get("checkpoint_sha256") != checkpoint_sha:
                raise RuntimeError(f"formal TNBC recovery declaration hash mismatch: {declaration_path}")
            epoch_records.append(
                {
                    "epoch": epoch_number,
                    "optimizer_updates": int(state.get("optimizer_updates", -1)),
                    "attempted_crop_batches": local_attempted,
                    "effective_optimizer_updates": local_updates,
                    "no_prompt_batch_count": local_no_prompt,
                    "no_prompt_batch_indices": local_positions,
                    "no_prompt_batch_indices_sha256": calculated_positions_sha,
                    "checkpoint_path": str(path),
                    "checkpoint_sha256": checkpoint_sha,
                    "checkpoint_declaration": str(declaration_path),
                    "losses": {
                        key: float(value) for key, value in state.get("epoch_losses", {}).items()
                    },
                    "runtime": dict(state.get("epoch_runtime", {})),
                    "learning_rate_after_scheduler_step": float(
                        state.get("scheduler", {}).get("_last_lr", [optimizer.param_groups[0]["lr"]])[0]
                    ),
                    "scheduler_state_after_step": state.get("scheduler"),
                }
            )
            prior_states.append(state)
        if [record["epoch"] for record in epoch_records] != list(range(1, resume_epoch + 1)):
            raise RuntimeError("formal TNBC recovery checkpoint epochs are not contiguous")
        if int(resume_state.get("optimizer_updates", -1)) != int(
            epoch_records[-1]["optimizer_updates"]
        ):
            raise RuntimeError("formal TNBC recovery optimizer-update history is inconsistent")

        net.load_state_dict(resume_state["model"])
        model1.load_state_dict(resume_state["model1"])
        optimizer.load_state_dict(resume_state["optimizer"])
        scheduler.load_state_dict(resume_state["scheduler"])
        _restore_rng_state(resume_state["rng_state"])
        start_epoch = resume_epoch
        recovery = {
            "resumed": True,
            "resume_checkpoint_path": str(resume_path),
            "resume_checkpoint_sha256": sha256_file(resume_path),
            "resumed_after_epoch": resume_epoch,
            "restored_fields": ["model", "model1", "optimizer", "scheduler", "rng_state"],
            "preexisting_epoch_checkpoint_count": len(epoch_records),
        }
        del prior_states
    else:
        if checkpoints_dir.exists() and any(checkpoints_dir.iterdir()):
            raise FileExistsError(f"formal TNBC output already has checkpoints: {checkpoints_dir}")
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        declarations_dir.mkdir(parents=True, exist_ok=True)
        runtime_stats = new_timing_runtime_stats()
        runtime_stats["record_no_prompt_batches"] = True
        epoch_records = []
        start_epoch = 0
    coverage_before = dict(coverage_identity)
    cuda_index = _cuda_device_index(device) if torch.cuda.is_available() else None
    peak_allocated_mib = max(
        [float(record.get("runtime", {}).get("peak_memory_allocated_mib", 0.0)) for record in epoch_records]
        or [0.0]
    )
    peak_reserved_mib = max(
        [float(record.get("runtime", {}).get("peak_memory_reserved_mib", 0.0)) for record in epoch_records]
        or [0.0]
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize(cuda_index)
    training_started = time.perf_counter()

    for epoch in range(start_epoch, 5):
        crop_batches_before = int(runtime_stats.get("crop_batches_seen", 0))
        optimizer_updates_before = int(runtime_stats.get("optimizer_steps", 0))
        no_prompt_before = int(runtime_stats.get("no_prompt_batch_count", 0))
        no_prompt_positions_before = len(runtime_stats.get("no_prompt_batches", []))
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(cuda_index)
            torch.cuda.synchronize(cuda_index)
        epoch_started = time.perf_counter()
        # Empty bank is deliberate: the approved run discards the checkpoint's
        # opaque bank and never transfers any C0 state to C1.
        losses = train_on_epoch(
            cfgs,
            model1,
            model1_encoder,
            net,
            train_loader,
            criterion,
            optimizer,
            epoch,
            [],
            device,
            runtime_stats=runtime_stats,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize(cuda_index)
        epoch_seconds = time.perf_counter() - epoch_started
        if not losses or not all(np.isfinite(float(value)) for value in losses.values()):
            raise RuntimeError(f"formal TNBC epoch {epoch + 1} has empty or non-finite losses")
        skipped = sum(
            int(runtime_stats.get(key, 0))
            for key in ("shape_skips", "nonfinite_loss_skips", "nonfinite_gradient_skips")
        )
        actual_updates = int(runtime_stats.get("optimizer_steps", 0))
        attempted_crop_batches = int(runtime_stats.get("crop_batches_seen", 0)) - crop_batches_before
        effective_optimizer_updates = actual_updates - optimizer_updates_before
        no_prompt_count = int(runtime_stats.get("no_prompt_batch_count", 0)) - no_prompt_before
        no_prompt_positions = list(runtime_stats.get("no_prompt_batches", []))[no_prompt_positions_before:]
        no_prompt_positions_sha256 = hashlib.sha256(
            json.dumps(no_prompt_positions, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if (
            skipped != 0
            or attempted_crop_batches != attempted_crop_batches_per_epoch
            or effective_optimizer_updates + no_prompt_count != attempted_crop_batches
        ):
            raise RuntimeError(
                "formal TNBC crop-batch contract failed at epoch "
                f"{epoch + 1}: attempted={attempted_crop_batches}/"
                f"{attempted_crop_batches_per_epoch}, effective_updates={effective_optimizer_updates}, "
                f"no_prompt={no_prompt_count}, other_skips={skipped}"
            )
        if int(runtime_stats.get("native_mask_token_count", 0)) != 4:
            raise RuntimeError("formal TNBC screen did not use four native mask tokens")
        scheduler.step()
        epoch_peak_allocated = (
            torch.cuda.max_memory_allocated(cuda_index) / (1024**2)
            if torch.cuda.is_available()
            else 0.0
        )
        epoch_peak_reserved = (
            torch.cuda.max_memory_reserved(cuda_index) / (1024**2)
            if torch.cuda.is_available()
            else 0.0
        )
        peak_allocated_mib = max(peak_allocated_mib, epoch_peak_allocated)
        peak_reserved_mib = max(peak_reserved_mib, epoch_peak_reserved)
        checkpoint_path = checkpoints_dir / f"epoch_{epoch + 1:04d}_update_{actual_updates:06d}.pth"
        checkpoint_payload = {
            "schema_version": 1,
            "phase": "2A-warmstart-formal-screen",
            "protocol": "tnbc_c0_c1_5epoch_exploratory_v1",
            "dataset": "tnbc",
            "arm": str(cfgs.warmstart_candidate_arm),
            "model": net.state_dict(),
            "model1": model1.state_dict(),
            "epoch": int(epoch + 1),
            "optimizer_updates": actual_updates,
            "attempted_crop_batches": attempted_crop_batches,
            "effective_optimizer_updates": effective_optimizer_updates,
            "no_prompt_batch_count": no_prompt_count,
            "no_prompt_batch_indices": no_prompt_positions,
            "no_prompt_batch_indices_sha256": no_prompt_positions_sha256,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "rng_state": _capture_rng_state(),
            "texture_memory_bank_list": [],
            "embedded_texture_bank_loaded": False,
            "coverage_refresh_events": [],
            "train_manifest": train_manifest_identity,
            "coverage": coverage_identity,
            "screen_config": getattr(cfgs, "warmstart_screen_config_identity", None),
            "epoch_losses": {key: float(value) for key, value in losses.items()},
            "epoch_runtime": {
                "wall_seconds": epoch_seconds,
                "peak_memory_allocated_mib": epoch_peak_allocated,
                "peak_memory_reserved_mib": epoch_peak_reserved,
            },
            "runtime_stats": dict(runtime_stats),
            "repository": _repository_identity(),
            "command": list(sys.argv),
        }
        _torch_save_atomic(checkpoint_path, checkpoint_payload)
        checkpoint_sha = sha256_file(checkpoint_path)
        declaration_path = declarations_dir / f"{checkpoint_path.stem}.json"
        _json_write_atomic(
            declaration_path,
            {
                "schema_version": 1,
                "dataset": "tnbc",
                "classification": "historical_exploratory",
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": checkpoint_sha,
                "selection_history": (
                    "approved fixed five-epoch C0/C1 exploratory warm-start screen; "
                    "development is evaluated only on the immutable saved epoch state"
                ),
                "training_manifest": train_manifest_identity,
                "p7_p8_exposure": "none during optimizer updates; fixed development evaluation only",
                "p9_p11_exposure": "none",
                "test_metric_selection": "not applicable; epoch 5 is pre-specified primary",
                "allowed_phase1_use": (
                    "exploratory TNBC p7-p8 fixed-epoch diagnosis only; not a clean "
                    "baseline, model-selection, or final-performance checkpoint"
                ),
                "source_note": "model/model1 warm-start only; fresh optimizer, scheduler, and RNG; texture bank discarded",
            },
        )
        epoch_records.append(
            {
                "epoch": epoch + 1,
                "optimizer_updates": actual_updates,
                "attempted_crop_batches": attempted_crop_batches,
                "effective_optimizer_updates": effective_optimizer_updates,
                "no_prompt_batch_count": no_prompt_count,
                "no_prompt_batch_indices": no_prompt_positions,
                "no_prompt_batch_indices_sha256": no_prompt_positions_sha256,
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": checkpoint_sha,
                "checkpoint_declaration": str(declaration_path),
                "losses": {key: float(value) for key, value in losses.items()},
                "runtime": checkpoint_payload["epoch_runtime"],
                "learning_rate_after_scheduler_step": float(optimizer.param_groups[0]["lr"]),
                "scheduler_state_after_step": scheduler.state_dict(),
            }
        )
        print(
            f"[formal-tnbc-screen] arm={cfgs.warmstart_candidate_arm} "
            f"epoch={epoch + 1}/5 attempted={attempted_crop_batches}/"
            f"{attempted_crop_batches_per_epoch} effective_updates={effective_optimizer_updates} "
            f"cumulative_updates={actual_updates} no_prompt={no_prompt_count}"
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize(cuda_index)
    coverage_after = verify_coverage_manifest(
        Path(cfgs.warmstart_coverage_manifest),
        train_manifest_identity=train_manifest_identity,
        checkpoint_sha256=sha256_file(Path(cfgs.sam_ckpt).resolve()),
        dataset="tnbc",
    )
    report = _warmstart_base_report(
        cfgs, args, train_dataset, model1, net, optimizer, scheduler, coverage_identity
    )
    report.update(
        {
            "status": "complete",
            "stage": "formal_tnbc_5epoch",
            "protocol": "tnbc_c0_c1_5epoch_exploratory_v1",
            "planned_epochs": 5,
            "attempted_crop_batches_per_epoch": attempted_crop_batches_per_epoch,
            "planned_attempted_crop_batches": planned_attempted_crop_batches,
            "actual_attempted_crop_batches": int(runtime_stats.get("crop_batches_seen", 0)),
            "actual_optimizer_updates": int(runtime_stats.get("optimizer_steps", 0)),
            "actual_no_prompt_batch_count": int(runtime_stats.get("no_prompt_batch_count", 0)),
            "coverage_refresh_events": [],
            "coverage_integrity": {"before": coverage_before, "after": coverage_after},
            "screen_config": getattr(cfgs, "warmstart_screen_config_identity", None),
            "epochs": epoch_records,
            "recovery": recovery,
            "runtime": {
                **finalize_runtime_audits(runtime_stats),
                "wall_seconds": sum(
                    float(record.get("runtime", {}).get("wall_seconds", 0.0))
                    for record in epoch_records
                ),
                "wall_seconds_this_process": time.perf_counter() - training_started,
                "peak_memory_allocated_mib": peak_allocated_mib,
                "peak_memory_reserved_mib": peak_reserved_mib,
            },
            "sealed_data_attestation": {
                "development_loader_constructed": False,
                "test_loader_constructed": False,
                "TNBC_p7_p11_accessed": False,
                "MoNuSeg_test14_accessed": False,
            },
            "evaluation_plan": {
                "epoch_0": "shared p7-p8 diagnosis before either arm trains",
                "epochs_1_to_5": "strict p7-p8 diagnosis from each immutable epoch checkpoint",
                "primary_comparison": "fixed epoch 5 C1-C0 patient-macro delta",
            },
        }
    )
    if int(runtime_stats.get("crop_batches_seen", 0)) != planned_attempted_crop_batches:
        raise RuntimeError("formal TNBC screen ended with incorrect attempted crop-batch count")
    _json_write_atomic(output, report)
    print(json.dumps({"status": "complete", "output": str(output)}))
    return report


def _json_write_atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    os.replace(temp_path, path)


def _torch_save_atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp")
    torch.save(payload, temp_path)
    os.replace(temp_path, path)


def _capture_rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _validate_phase2a_recipe(cfgs, args, train_dataset, test_dataset):
    recipe_path = Path(cfgs.phase2a_recipe).resolve()
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    dataset_recipe = recipe["datasets"][cfgs.dataset]
    optimization = recipe["optimization"]
    pms = recipe["stainpms"]
    inference_spec = recipe["inference"]
    checks = {
        "seed": (int(cfgs.seed), int(recipe["determinism"]["seed"])),
        "epochs": (int(cfgs.epochs), int(optimization["epochs"])),
        "crop_size": (int(cfgs.crop_size), int(optimization["crop_size"])),
        "crop_batch_size": (int(cfgs.b), int(optimization["crop_batch_size"])),
        "overlap": (int(cfgs.overlap), int(dataset_recipe["overlap"])),
        "pms_start_epoch": (int(cfgs.pms_start_epoch), int(pms["start_epoch"])),
        "coverage_refresh_interval": (
            int(cfgs.iterative_baseline_refresh_every),
            int(pms["coverage_refresh_interval_epochs"]),
        ),
        "cosine_t_max_epochs": (
            int(cfgs.lr_cosine_t_max),
            int(optimization["cosine_t_max_epochs"]),
        ),
        "pms_preservation_max_prompts": (
            int(args.criterion.pms_preserve_max_prompts),
            int(pms["preservation_max_prompts"]),
        ),
        "pms_gt_match_radius": (
            int(args.criterion.pms_gt_match_radius),
            int(pms["gt_match_radius"]),
        ),
        "stain_peak_min_distance": (
            int(args.criterion.stain_min_distance),
            int(pms["stain_peak_min_distance"]),
        ),
        "stain_top_k": (
            int(args.criterion.stain_top_k),
            int(pms["stain_top_k"]),
        ),
        "point_nms_threshold": (int(args.test.nms_thr), int(inference_spec["point_nms_threshold"])),
    }
    mismatches = {name: pair for name, pair in checks.items() if pair[0] != pair[1]}
    float_checks = {
        "learning_rate": (float(cfgs.lr), float(optimization["learning_rate"])),
        "minimum_learning_rate": (float(cfgs.lr_min), float(optimization["minimum_learning_rate"])),
        "weight_decay": (float(cfgs.weight_decay), float(optimization["weight_decay"])),
        "pms_loss_coefficient": (float(args.criterion.pms_loss_coef), float(pms["pms_loss_coefficient"])),
        "pms_object_weight": (float(args.criterion.pms_object_weight), float(pms["object_weight"])),
        "pms_residual_mask_weight": (
            float(args.criterion.pms_residual_mask_weight),
            float(pms["residual_mask_weight"]),
        ),
        "pms_preservation_loss": (
            float(args.criterion.pms_preserve_loss_coef),
            float(pms["preservation_loss_coefficient"]),
        ),
    }
    mismatches.update(
        {name: pair for name, pair in float_checks.items() if not math.isclose(pair[0], pair[1])}
    )
    if mismatches:
        raise ValueError(f"Phase 2A command differs from frozen recipe: {mismatches}")
    required_bools = {
        "use_pms": cfgs.use_pms,
        "pms_self_bootstrap": cfgs.pms_self_bootstrap,
        "coverage_accumulate": cfgs.coverage_accumulate is True,
        "pms_preserve_covered": cfgs.pms_preserve_covered,
        "texture": cfgs.texture,
        "context": cfgs.context,
        "strict_evaluator": cfgs.evaluator_mode == "strict",
        "tta_disabled": not cfgs.tta,
        "point_filtering": bool(args.test.filtering) is bool(inference_spec["filtering"]),
        "load_order": str(cfgs.load) == str(inference_spec["load"]),
        "manifest_hash_verification": cfgs.verify_manifest_hashes,
    }
    missing = [name for name, enabled in required_bools.items() if not enabled]
    if missing:
        raise ValueError("Phase 2A required settings are disabled: " + ", ".join(missing))
    if train_dataset.manifest.get("protocol_id") != dataset_recipe["train_protocol_id"]:
        raise ValueError("training manifest protocol does not match frozen Phase 2A recipe")
    if cfgs.dataset == "tnbc":
        if cfgs.phase2a_eval_policy != "tnbc_patient_macro" or test_dataset is None:
            raise ValueError("TNBC Phase 2A requires its p7-p8 patient-macro development loader")
        if test_dataset.manifest.get("protocol_id") != dataset_recipe["development_protocol_id"]:
            raise ValueError("TNBC development manifest protocol does not match recipe")
    elif cfgs.dataset == "monuseg":
        if cfgs.phase2a_eval_policy != "none" or test_dataset is not None or cfgs.eval_manifest:
            raise ValueError("MoNuSeg Phase 2A must not construct an evaluation loader")
    return recipe, recipe_path, dataset_recipe


def _phase2a_checkpoint_payload(
    *,
    cfgs,
    net,
    model1,
    optimizer,
    scheduler,
    epoch,
    texture_memory_bank_list,
    runtime_stats,
    coverage_events,
    evaluation_records,
    recipe_sha256,
    train_manifest_sha256,
    eval_manifest_sha256,
    include_training_state,
):
    payload = {
        "schema_version": 1,
        "phase": "2A",
        "protocol": "protocol_clean_stainpms_baseline_v1",
        "model": net.state_dict(),
        "model1": model1.state_dict(),
        "epoch": int(epoch),
        "optimizer_updates": int(runtime_stats.get("optimizer_steps", 0)),
        "texture_memory_bank_list": texture_memory_bank_list,
        "runtime_stats": dict(runtime_stats),
        "coverage_events": list(coverage_events),
        "evaluation_records": list(evaluation_records),
        "recipe_sha256": recipe_sha256,
        "train_manifest_sha256": train_manifest_sha256,
        "eval_manifest_sha256": eval_manifest_sha256,
        "generic_initialization_sha256": str(cfgs.phase2a_generic_checkpoint_sha256),
        "coverage_cache_dir": str(cfgs.baseline_masks_dir),
        "includes_training_state": bool(include_training_state),
    }
    if include_training_state:
        payload.update(
            {
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "rng_state": _capture_rng_state(),
            }
        )
    return payload


def run_phase2a_baseline(
    cfgs,
    args,
    train_dataset,
    test_dataset,
    train_loader,
    test_loader,
    model1,
    model1_encoder,
    net,
    criterion,
    optimizer,
    scheduler,
    device,
):
    from stainpms.phase2a_selection import choose_tnbc_checkpoint, tnbc_patient_macro_score

    recipe, recipe_path, dataset_recipe = _validate_phase2a_recipe(
        cfgs, args, train_dataset, test_dataset
    )
    recipe_sha256 = sha256_file(recipe_path)
    generic_sha256 = sha256_file(Path(cfgs.sam_ckpt).resolve())
    if generic_sha256 != str(cfgs.phase2a_generic_checkpoint_sha256).lower():
        raise ValueError("Phase 2A generic SAM2 checkpoint SHA256 mismatch")
    gate = json.loads(Path(cfgs.phase2a_budget_gate_report).read_text(encoding="utf-8"))
    if (
        gate.get("dataset") != cfgs.dataset
        or gate.get("status") != "gate_pass"
        or gate.get("recipe_sha256") != recipe_sha256
    ):
        raise ValueError("Phase 2A formal training requires a matching gate_pass budget report")

    output_dir = Path(cfgs.phase2a_output_dir).resolve()
    checkpoints_dir = output_dir / "checkpoints"
    metrics_dir = output_dir / "metrics"
    if not cfgs.phase2a_resume_checkpoint and (
        (checkpoints_dir.exists() and any(checkpoints_dir.iterdir()))
        or (output_dir / "training_summary.json").exists()
    ):
        raise ValueError(
            "Phase 2A output already contains checkpoints/results; use a new output "
            "directory or an explicit recovery checkpoint"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    cfgs.metrics_output_dir = str(metrics_dir)
    planned_updates = int(dataset_recipe["planned_optimizer_updates"])
    updates_per_epoch = int(dataset_recipe["optimizer_updates_per_epoch"])
    interval_updates = int(dataset_recipe["checkpoint_interval_updates"])
    if planned_updates != int(cfgs.epochs) * updates_per_epoch:
        raise ValueError("Phase 2A planned update count does not match epochs")

    runtime_stats = {}
    coverage_events = []
    evaluation_records = []
    start_epoch = 0
    if cfgs.phase2a_resume_checkpoint:
        resume_path = Path(cfgs.phase2a_resume_checkpoint).resolve()
        state = torch.load(resume_path, map_location="cpu")
        if not state.get("includes_training_state"):
            raise ValueError("Phase 2A resume requires a recovery checkpoint with training state")
        immutable = {
            "recipe_sha256": recipe_sha256,
            "train_manifest_sha256": train_dataset.manifest["manifest_sha256"],
            "eval_manifest_sha256": (
                test_dataset.manifest["manifest_sha256"] if test_dataset is not None else None
            ),
            "generic_initialization_sha256": generic_sha256,
        }
        mismatched = {
            name: (state.get(name), expected)
            for name, expected in immutable.items()
            if state.get(name) != expected
        }
        if mismatched:
            raise ValueError(f"Phase 2A resume checkpoint provenance mismatch: {mismatched}")
        net.load_state_dict(state["model"])
        model1.load_state_dict(state["model1"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        runtime_stats.update(state["runtime_stats"])
        coverage_events.extend(state["coverage_events"])
        evaluation_records.extend(state["evaluation_records"])
        start_epoch = int(state["epoch"]) + 1
        _restore_rng_state(state["rng_state"])
        if start_epoch > int(cfgs.pms_start_epoch):
            loaded_cache_count = train_dataset.reload_baseline_masks()
            if loaded_cache_count != len(train_dataset):
                raise ValueError("Phase 2A resume coverage cache is incomplete")

    if torch.cuda.is_available():
        cuda_index = _cuda_device_index(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(cuda_index)
        torch.cuda.synchronize(cuda_index)
    else:
        cuda_index = None
    training_started = time.perf_counter()
    evaluation_wall_seconds = 0.0
    refresh_wall_seconds = sum(float(item["wall_seconds"]) for item in coverage_events)

    for epoch in range(start_epoch, int(cfgs.epochs)):
        refresh_base = int(cfgs.pms_start_epoch)
        should_refresh = epoch == refresh_base or (
            epoch > refresh_base
            and (epoch - refresh_base) % int(cfgs.iterative_baseline_refresh_every) == 0
        )
        if should_refresh:
            refresh_started = time.perf_counter()
            refresh_baseline_masks_inplace(
                cfgs,
                args,
                train_dataset,
                test_dataset,
                model1,
                model1_encoder,
                net,
                None,
                epoch,
                device,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize(cuda_index)
            refresh_elapsed = time.perf_counter() - refresh_started
            refresh_wall_seconds += refresh_elapsed
            coverage_events.append(
                {
                    "epoch": epoch,
                    "optimizer_updates_before_refresh": int(runtime_stats.get("optimizer_steps", 0)),
                    "wall_seconds": refresh_elapsed,
                    "record_count": len(train_dataset),
                }
            )

        texture_memory_bank_list = []
        net.train()
        model1.train()
        log_info = train_on_epoch(
            cfgs,
            model1,
            model1_encoder,
            net,
            train_loader,
            criterion,
            optimizer,
            epoch,
            texture_memory_bank_list,
            device,
            runtime_stats=runtime_stats,
        )
        if not log_info or not all(np.isfinite(float(value)) for value in log_info.values()):
            raise RuntimeError(f"non-finite or empty Phase 2A loss at epoch {epoch}")
        if any(
            int(runtime_stats.get(key, 0))
            for key in ("shape_skips", "nonfinite_loss_skips", "nonfinite_gradient_skips")
        ):
            raise RuntimeError("Phase 2A baseline encountered a skipped optimizer update")
        scheduler.step()
        actual_updates = int(runtime_stats.get("optimizer_steps", 0))
        expected_updates = (epoch + 1) * updates_per_epoch
        if actual_updates != expected_updates:
            raise RuntimeError(
                f"epoch {epoch}: optimizer updates {actual_updates} != planned {expected_updates}"
            )
        checkpoint_due = actual_updates % interval_updates == 0
        if not checkpoint_due:
            continue

        evaluation = None
        if cfgs.dataset == "tnbc":
            eval_started = time.perf_counter()
            validation_on_epoch(
                cfgs,
                args,
                test_loader,
                epoch,
                model1,
                model1_encoder,
                net,
                cfgs.load,
                args.data.post.iou_threshold,
                list(texture_memory_bank_list),
                device,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize(cuda_index)
            eval_elapsed = time.perf_counter() - eval_started
            evaluation_wall_seconds += eval_elapsed
            metric_payload = json.loads(
                (metrics_dir / f"epoch_{epoch:04d}" / "metrics_per_image.json").read_text(
                    encoding="utf-8"
                )
            )
            patient_by_sample = {
                str(record["sample_id"]): int(record["patient"])
                for record in test_dataset.manifest["records"]
            }
            evaluation = tnbc_patient_macro_score(metric_payload["images"], patient_by_sample)
            evaluation["wall_seconds"] = eval_elapsed
            evaluation["epoch"] = epoch
            evaluation["optimizer_updates"] = actual_updates
            evaluation_records.append(evaluation)

        final_update = actual_updates == planned_updates
        common_payload_args = dict(
            cfgs=cfgs,
            net=net,
            model1=model1,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            texture_memory_bank_list=texture_memory_bank_list,
            runtime_stats=runtime_stats,
            coverage_events=coverage_events,
            evaluation_records=evaluation_records,
            recipe_sha256=recipe_sha256,
            train_manifest_sha256=train_dataset.manifest["manifest_sha256"],
            eval_manifest_sha256=(
                test_dataset.manifest["manifest_sha256"] if test_dataset is not None else None
            ),
        )
        checkpoint_path = None
        if cfgs.dataset == "tnbc":
            checkpoint_path = checkpoints_dir / f"candidate_update_{actual_updates:07d}.pth"
            _torch_save_atomic(
                checkpoint_path,
                _phase2a_checkpoint_payload(
                    **common_payload_args, include_training_state=False
                ),
            )
            _torch_save_atomic(
                checkpoints_dir / "recovery_latest.pth",
                _phase2a_checkpoint_payload(
                    **common_payload_args, include_training_state=True
                ),
            )
        elif final_update:
            checkpoint_path = checkpoints_dir / f"final_update_{actual_updates:07d}.pth"
            _torch_save_atomic(
                checkpoint_path,
                _phase2a_checkpoint_payload(
                    **common_payload_args, include_training_state=True
                ),
            )
        else:
            _torch_save_atomic(
                checkpoints_dir / "recovery_latest.pth",
                _phase2a_checkpoint_payload(
                    **common_payload_args, include_training_state=True
                ),
            )
        checkpoint_record = {
            "epoch": epoch,
            "optimizer_updates": actual_updates,
            "path": str(checkpoint_path) if checkpoint_path is not None else None,
            "sha256": sha256_file(checkpoint_path) if checkpoint_path is not None else None,
            "role": (
                "tnbc_selection_candidate"
                if cfgs.dataset == "tnbc"
                else ("fixed_final" if final_update else "recovery_latest_overwrite")
            ),
        }
        if evaluation is not None:
            checkpoint_record.update(
                {
                    "selection_score": evaluation["selection_score"],
                    "macro_patient_aji": evaluation["macro_patient_aji"],
                    "macro_patient_pq": evaluation["macro_patient_pq"],
                }
            )
        progress = {
            "schema_version": 1,
            "phase": "2A",
            "status": "running" if not final_update else "complete",
            "dataset": cfgs.dataset,
            "optimizer_updates": actual_updates,
            "planned_optimizer_updates": planned_updates,
            "coverage_refresh_count": len(coverage_events),
            "latest_checkpoint": checkpoint_record,
        }
        _json_write_atomic(output_dir / "progress.json", progress)

    if torch.cuda.is_available():
        torch.cuda.synchronize(cuda_index)
    training_wall_seconds = time.perf_counter() - training_started
    checkpoint_files = sorted(
        checkpoints_dir.glob("candidate_update_*.pth")
        if cfgs.dataset == "tnbc"
        else checkpoints_dir.glob("final_update_*.pth")
    )
    checkpoint_records = []
    eval_by_update = {int(item["optimizer_updates"]): item for item in evaluation_records}
    for path in checkpoint_files:
        updates = int(path.stem.rsplit("_", 1)[-1])
        record = {
            "epoch": updates // updates_per_epoch - 1,
            "optimizer_updates": updates,
            "path": str(path),
            "sha256": sha256_file(path),
            "role": (
                "tnbc_selection_candidate"
                if cfgs.dataset == "tnbc" else "fixed_final"
            ),
        }
        if updates in eval_by_update:
            record.update(
                {
                    "selection_score": eval_by_update[updates]["selection_score"],
                    "macro_patient_aji": eval_by_update[updates]["macro_patient_aji"],
                    "macro_patient_pq": eval_by_update[updates]["macro_patient_pq"],
                }
            )
        checkpoint_records.append(record)
    if cfgs.dataset == "tnbc":
        selected = choose_tnbc_checkpoint(checkpoint_records, tie_tolerance=0.001)
    else:
        selected = next(
            record for record in checkpoint_records if record["optimizer_updates"] == planned_updates
        )
    final_invariants = {
        "optimizer_updates": (
            int(runtime_stats.get("optimizer_steps", 0)),
            planned_updates,
        ),
        "crops_seen": (int(runtime_stats.get("crops_seen", 0)), planned_updates),
        "coverage_refresh_count": (
            len(coverage_events),
            int(recipe["stainpms"]["expected_refresh_count"]),
        ),
        "evaluation_count": (
            len(evaluation_records),
            int(dataset_recipe["checkpoint_count"]) if cfgs.dataset == "tnbc" else 0,
        ),
    }
    invariant_failures = {
        name: values for name, values in final_invariants.items() if values[0] != values[1]
    }
    if invariant_failures:
        raise RuntimeError(f"Phase 2A final budget invariant failure: {invariant_failures}")
    peak_allocated = int(torch.cuda.max_memory_allocated(cuda_index)) if torch.cuda.is_available() else 0
    peak_reserved = int(torch.cuda.max_memory_reserved(cuda_index)) if torch.cuda.is_available() else 0
    report = {
        "schema_version": 1,
        "phase": "2A",
        "status": "complete",
        "dataset": cfgs.dataset,
        "protocol_id": recipe["protocol_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": list(sys.argv),
        "repository": {
            "branch": subprocess.check_output(["git", "branch", "--show-current"], text=True).strip(),
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        },
        "recipe": {"path": str(recipe_path), "sha256": recipe_sha256},
        "initialization": {
            "path": str(Path(cfgs.sam_ckpt).resolve()),
            "sha256": generic_sha256,
            "task_checkpoint_loaded": False,
        },
        "data": {
            "train_manifest": str(cfgs.train_manifest),
            "train_manifest_sha256": train_dataset.manifest["manifest_sha256"],
            "eval_manifest": str(cfgs.eval_manifest) if cfgs.eval_manifest else None,
            "eval_manifest_sha256": (
                test_dataset.manifest["manifest_sha256"] if test_dataset is not None else None
            ),
            "sealed_test_loader_constructed": False,
        },
        "actual_budget": {
            "epochs": int(cfgs.epochs),
            "optimizer_updates": int(runtime_stats["optimizer_steps"]),
            "crop_batches_seen": int(runtime_stats["crop_batches_seen"]),
            "crops_seen": int(runtime_stats["crops_seen"]),
            "coverage_refresh_count": len(coverage_events),
            "checkpoint_save_event_count": int(dataset_recipe["checkpoint_count"]),
            "coverage_refresh_wall_seconds": refresh_wall_seconds,
            "evaluation_wall_seconds": evaluation_wall_seconds,
            "training_wall_seconds": training_wall_seconds,
            "gpu_hours": training_wall_seconds / 3600,
            "peak_memory_allocated_mib": peak_allocated / (1024 ** 2),
            "peak_memory_reserved_mib": peak_reserved / (1024 ** 2),
        },
        "coverage_events": coverage_events,
        "evaluations": evaluation_records,
        "checkpoints": checkpoint_records,
        "selected_checkpoint": selected,
        "selection_policy": dataset_recipe["selection"],
    }
    _json_write_atomic(output_dir / "training_summary.json", report)
    _json_write_atomic(output_dir / "selection.json", selected)
    print(
        json.dumps(
            {
                "status": "complete",
                "dataset": cfgs.dataset,
                "optimizer_updates": runtime_stats["optimizer_steps"],
                "selected_checkpoint": selected["path"],
                "output": str(output_dir / "training_summary.json"),
            }
        )
    )
    return report


def freeze_sam2_image_encoder(net):
    for name, param in net.named_parameters():
        if "image_encoder" in name and "prompt_generator" not in name:
            param.requires_grad_(False)


def write_eval_metric_artifact(cfgs, eval_split, metrics):
    """Write unrounded evaluation metrics alongside optional map artifacts.

    The text printed to stdout is intentionally concise and rounded, which is
    not sufficient for Stage 0 metric reconciliation.  This sidecar is written
    only when artifact dumping was explicitly requested and does not affect
    evaluation, NMS, or instance assembly.
    """

    dump_dir = str(getattr(cfgs, "dump_eval_artifacts_dir", "") or "")
    if not dump_dir:
        return
    metric_names = ("dice1", "dice2", "aji", "aji_p", "dq", "sq", "pq")
    payload = {
        "split": str(eval_split),
        "metrics": {
            name: float(value) for name, value in zip(metric_names, metrics, strict=True)
        },
        "checkpoint_path": str(cfgs.sam_ckpt),
        "dataset": str(cfgs.dataset),
        "overlap": int(cfgs.overlap),
        "nms_threshold": int(cfgs.test_nms_thr),
        "seed": int(cfgs.seed),
        "evaluator_mode": str(cfgs.evaluator_mode),
        "train_manifest": str(cfgs.train_manifest or ""),
        "eval_manifest": str(cfgs.eval_manifest or ""),
        "command": list(sys.argv),
    }
    os.makedirs(dump_dir, exist_ok=True)
    path = os.path.join(dump_dir, "main_eval_metrics.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"Wrote unrounded evaluation metrics: {path}")


def _validate_warmstart_preflight(cfgs, args):
    """Validate the approved train-only warm-start contract before data access."""
    stage = str(cfgs.warmstart_stage or "")
    if not stage:
        return None, None

    incompatible = {
        "eval": bool(cfgs.eval),
        "eval_manifest": bool(cfgs.eval_manifest),
        "stage1_coverage_oracle": bool(cfgs.stage1_coverage_oracle),
        "stage2_selective_refine": bool(cfgs.stage2_selective_refine),
        "train_only_smoke_steps": int(cfgs.train_only_smoke_steps or 0) > 0,
        "legacy_phase2a_timing": bool(cfgs.phase2a_timing_profile),
        "scratch_phase2a_baseline": bool(cfgs.phase2a_baseline),
    }
    enabled = [name for name, value in incompatible.items() if value]
    if enabled:
        raise ValueError(
            "warm-start feasibility cannot be combined with: " + ", ".join(enabled)
        )

    required_paths = {
        "train_manifest": cfgs.train_manifest,
        "warmstart_output": cfgs.warmstart_output,
        "sam_ckpt": cfgs.sam_ckpt,
        "warmstart_checkpoint_sha256": cfgs.warmstart_checkpoint_sha256,
    }
    if stage == "formal_tnbc_5epoch":
        required_paths["warmstart_screen_config"] = cfgs.warmstart_screen_config
    missing_paths = [name for name, value in required_paths.items() if not value]
    if missing_paths:
        raise ValueError("warm-start feasibility missing: " + ", ".join(missing_paths))
    output_path = Path(cfgs.warmstart_output).resolve()
    resume_checkpoint = str(getattr(cfgs, "warmstart_resume_checkpoint", "") or "")
    if output_path.exists() and not resume_checkpoint:
        raise ValueError(f"warm-start output already exists: {output_path}")
    if resume_checkpoint:
        if stage != "formal_tnbc_5epoch":
            raise ValueError("warm-start recovery is permitted only for formal_tnbc_5epoch")
        resume_path = Path(resume_checkpoint).resolve()
        expected_checkpoint_dir = output_path.parent / "checkpoints"
        if not resume_path.is_file():
            raise ValueError(f"warm-start recovery checkpoint is missing: {resume_path}")
        if resume_path.parent != expected_checkpoint_dir:
            raise ValueError(
                "warm-start recovery checkpoint must belong to the output's checkpoints directory"
            )
        if output_path.is_file():
            raise ValueError("cannot recover a formal screen that already has training_summary.json")
    if not cfgs.verify_manifest_hashes:
        raise ValueError("warm-start feasibility requires manifest hash verification")

    train_manifest_identity = validate_train_manifest_identity(
        Path(cfgs.train_manifest), str(cfgs.dataset)
    )
    expected_protocol = {
        "tnbc": "tnbc_stainpms_prepared_continuity_v1_phase1_train",
        "monuseg": "monuseg_download37_continuity_v1_phase1_trainonly",
    }[str(cfgs.dataset)]
    if train_manifest_identity["protocol_id"] != expected_protocol:
        raise ValueError(
            "warm-start train manifest protocol mismatch: "
            f"{train_manifest_identity['protocol_id']} != {expected_protocol}"
        )

    checkpoint_path = Path(cfgs.sam_ckpt).resolve()
    if not checkpoint_path.is_file():
        raise ValueError(f"warm-start checkpoint is missing: {checkpoint_path}")
    checkpoint_sha = sha256_file(checkpoint_path)
    expected_checkpoint_sha = str(cfgs.warmstart_checkpoint_sha256).lower()
    if checkpoint_sha != expected_checkpoint_sha:
        raise ValueError(
            f"warm-start checkpoint SHA256 mismatch: {checkpoint_sha} != "
            f"{expected_checkpoint_sha}"
        )

    approved_epochs = 5 if stage == "formal_tnbc_5epoch" else 10
    exact_values = {
        "seed": (int(cfgs.seed), 3407),
        "epochs": (int(cfgs.epochs), approved_epochs),
        "crop_size": (int(cfgs.crop_size), 256),
        "out_size": (int(cfgs.out_size), 256),
        "crop_batch_size": (int(cfgs.b), 1),
        "overlap": (
            int(cfgs.overlap),
            32 if str(cfgs.dataset) == "tnbc" else 92,
        ),
        "pms_start_epoch": (int(cfgs.pms_start_epoch), 0),
        "coverage_refresh_interval": (
            int(cfgs.iterative_baseline_refresh_every),
            20,
        ),
        "pms_gt_match_radius": (int(args.criterion.pms_gt_match_radius), 8),
        "pms_preserve_max_prompts": (
            int(args.criterion.pms_preserve_max_prompts),
            20,
        ),
        "stain_min_distance": (int(args.criterion.stain_min_distance), 12),
        "stain_top_k": (int(args.criterion.stain_top_k), 20),
        "point_nms_threshold": (int(args.test.nms_thr), 12),
    }
    value_mismatches = {
        name: values for name, values in exact_values.items() if values[0] != values[1]
    }
    float_values = {
        "learning_rate": (float(cfgs.lr), 1e-5),
        "weight_decay": (float(cfgs.weight_decay), 1e-4),
        "gradient_clip": (float(cfgs.clip_grad), 0.1),
        "pms_loss_coef": (float(args.criterion.pms_loss_coef), 0.5),
        "pms_object_weight": (float(args.criterion.pms_object_weight), 1.0),
        "pms_residual_mask_weight": (
            float(args.criterion.pms_residual_mask_weight),
            0.3,
        ),
        "pms_preserve_loss_coef": (
            float(args.criterion.pms_preserve_loss_coef),
            1.0,
        ),
        "candidate_coverage_tau": (float(cfgs.candidate_coverage_tau), 0.1),
        "candidate_coverage_coefficient": (
            float(cfgs.candidate_coverage_coefficient),
            1.0,
        ),
        "candidate_quality_coefficient": (
            float(cfgs.candidate_quality_coefficient),
            1.0,
        ),
    }
    value_mismatches.update(
        {
            name: values
            for name, values in float_values.items()
            if not math.isclose(values[0], values[1], rel_tol=0.0, abs_tol=1e-12)
        }
    )
    if value_mismatches:
        raise ValueError(
            f"warm-start command differs from the frozen configuration: {value_mismatches}"
        )
    if float(cfgs.lr_min) >= 0:
        raise ValueError("warm-start feasibility requires the public MultiStepLR path")
    if list(cfgs.lr_milestones) != [80, 140, 200]:
        raise ValueError("warm-start feasibility requires milestones 80 140 200")
    if str(cfgs.sam_config) != "sam2_hiera_l" or str(cfgs.net) != "sam2":
        raise ValueError("warm-start feasibility requires the frozen SAM2 Hiera-L path")
    if str(cfgs.load) != "unclockwise":
        raise ValueError("warm-start feasibility requires load=unclockwise")
    if bool(cfgs.tta):
        raise ValueError("warm-start feasibility forbids TTA")

    required_bools = {
        "use_pms": bool(cfgs.use_pms),
        "pms_self_bootstrap": bool(cfgs.pms_self_bootstrap),
        "coverage_accumulate": cfgs.coverage_accumulate is True,
        "pms_preserve_covered": bool(cfgs.pms_preserve_covered),
        "texture": bool(cfgs.texture),
        "context": bool(cfgs.context),
        "test_filtering": bool(args.test.filtering),
        "strict_evaluator": str(cfgs.evaluator_mode) == "strict",
    }
    missing_bools = [name for name, enabled in required_bools.items() if not enabled]
    if missing_bools:
        raise ValueError(
            "warm-start required settings are disabled: " + ", ".join(missing_bools)
        )

    arm = str(cfgs.warmstart_candidate_arm or "")
    if stage == "prepare_coverage":
        if arm != "c0":
            raise ValueError("coverage preparation must declare arm c0")
        if cfgs.warmstart_coverage_manifest:
            raise ValueError("coverage preparation cannot consume a coverage manifest")
        if not cfgs.baseline_masks_dir:
            raise ValueError("coverage preparation requires --baseline_masks_dir")
        cache_dir = Path(cfgs.baseline_masks_dir).resolve()
        if cache_dir.exists() and any(cache_dir.iterdir()):
            raise ValueError(f"coverage output directory must be empty: {cache_dir}")
        cache_dir.mkdir(parents=True, exist_ok=True)
        coverage_identity = None
    else:
        allowed_arms = {"legacy", "c0", "c1"} if stage == "smoke" else {"c0", "c1"}
        if arm not in allowed_arms:
            raise ValueError(
                f"warm-start {stage} arm must be one of {sorted(allowed_arms)}"
            )
        if not cfgs.warmstart_coverage_manifest:
            raise ValueError(f"warm-start {stage} requires a coverage manifest")
        coverage_identity = verify_coverage_manifest(
            Path(cfgs.warmstart_coverage_manifest),
            train_manifest_identity=train_manifest_identity,
            checkpoint_sha256=checkpoint_sha,
            dataset=str(cfgs.dataset),
        )
        if cfgs.baseline_masks_dir:
            supplied_cache = str(Path(cfgs.baseline_masks_dir).resolve())
            if supplied_cache != coverage_identity["cache_dir"]:
                raise ValueError("CLI coverage directory differs from frozen coverage manifest")
        cfgs.baseline_masks_dir = coverage_identity["cache_dir"]

    if stage == "smoke" and int(cfgs.warmstart_smoke_updates) not in {1, 2}:
        raise ValueError("approved warm-start smoke permits exactly 1 or 2 updates")
    if stage == "timing":
        if int(cfgs.phase2a_warmup_updates) != 10:
            raise ValueError("warm-start timing requires exactly 10 warm-up updates")
        if int(cfgs.phase2a_timed_updates) != 100:
            raise ValueError("warm-start timing requires exactly 100 timed updates")
    if stage == "formal_tnbc_5epoch":
        if str(cfgs.dataset) != "tnbc":
            raise ValueError("formal_tnbc_5epoch rejects all non-TNBC datasets")
        screen_config_path = Path(cfgs.warmstart_screen_config).resolve()
        screen_config = json.loads(screen_config_path.read_text(encoding="utf-8"))
        if screen_config.get("protocol_id") != "tnbc_c0_c1_5epoch_exploratory_v1":
            raise ValueError("formal_tnbc_5epoch screen config protocol mismatch")
        if int(screen_config.get("optimization", {}).get("planned_attempted_crop_batches", -1)) != 1350:
            raise ValueError("formal_tnbc_5epoch screen config must freeze 1350 attempted crop batches")
        cfgs.warmstart_screen_config_identity = {
            "path": str(screen_config_path),
            "sha256": sha256_file(screen_config_path),
            "protocol_id": screen_config["protocol_id"],
        }
        if int(cfgs.warmstart_smoke_updates) != 0:
            raise ValueError("formal_tnbc_5epoch cannot set warmstart smoke updates")
        if int(cfgs.phase2a_warmup_updates) != 10 or int(cfgs.phase2a_timed_updates) != 100:
            raise ValueError("formal_tnbc_5epoch requires the approved 10/100 timing provenance")
    return train_manifest_identity, coverage_identity


def main():
    args = Config.fromfile("./args.py")
    cfgs = cfg.parse_args()
    apply_cli_overrides(args, cfgs)
    warmstart_manifest_identity, warmstart_coverage_identity = (
        _validate_warmstart_preflight(cfgs, args)
    )
    if cfgs.train_only_smoke_steps < 0:
        raise ValueError("--train_only_smoke_steps cannot be negative")
    if cfgs.train_only_smoke_steps > 0:
        incompatible = {
            "eval": cfgs.eval,
            "stage1_coverage_oracle": cfgs.stage1_coverage_oracle,
            "stage2_selective_refine": cfgs.stage2_selective_refine,
            "use_pms": cfgs.use_pms,
            "pms_self_bootstrap": cfgs.pms_self_bootstrap,
            "eval_manifest": bool(cfgs.eval_manifest),
        }
        enabled = [name for name, value in incompatible.items() if value]
        if enabled:
            raise ValueError(
                "train-only smoke cannot be combined with: " + ", ".join(enabled)
            )
        if not cfgs.train_manifest or not cfgs.verify_manifest_hashes:
            raise ValueError(
                "Phase 0.5 smoke requires --train_manifest and --verify_manifest_hashes"
            )
        if not cfgs.smoke_output:
            raise ValueError("Phase 0.5 smoke requires --smoke_output")
    if cfgs.phase2a_timing_profile:
        incompatible = {
            "eval": cfgs.eval,
            "stage1_coverage_oracle": cfgs.stage1_coverage_oracle,
            "stage2_selective_refine": cfgs.stage2_selective_refine,
            "train_only_smoke_steps": cfgs.train_only_smoke_steps > 0,
            "eval_manifest": bool(cfgs.eval_manifest),
        }
        enabled = [name for name, value in incompatible.items() if value]
        if enabled:
            raise ValueError(
                "Phase 2A timing cannot be combined with: " + ", ".join(enabled)
            )
        if not cfgs.train_manifest or not cfgs.verify_manifest_hashes:
            raise ValueError(
                "Phase 2A timing requires --train_manifest and --verify_manifest_hashes"
            )
        if not cfgs.phase2a_timing_output:
            raise ValueError("Phase 2A timing requires --phase2a_timing_output")
        if Path(cfgs.phase2a_timing_output).resolve().exists():
            raise ValueError(
                f"Phase 2A timing output already exists: {Path(cfgs.phase2a_timing_output).resolve()}"
            )
        if cfgs.phase2a_timing_profile == "base":
            if cfgs.use_pms or cfgs.pms_self_bootstrap:
                raise ValueError("Phase 2A base timing must not enable PMS")
        elif cfgs.phase2a_timing_profile == "pms_active":
            required = {
                "use_pms": cfgs.use_pms,
                "pms_self_bootstrap": cfgs.pms_self_bootstrap,
                "pms_start_epoch_is_zero": int(cfgs.pms_start_epoch) == 0,
            }
            missing = [name for name, value in required.items() if not value]
            if missing:
                raise ValueError(
                    "Phase 2A pms_active timing requirements not met: "
                    + ", ".join(missing)
                )
    if cfgs.phase2a_baseline:
        incompatible = {
            "eval": cfgs.eval,
            "stage1_coverage_oracle": cfgs.stage1_coverage_oracle,
            "stage2_selective_refine": cfgs.stage2_selective_refine,
            "train_only_smoke_steps": cfgs.train_only_smoke_steps > 0,
            "phase2a_timing_profile": bool(cfgs.phase2a_timing_profile),
        }
        enabled = [name for name, value in incompatible.items() if value]
        if enabled:
            raise ValueError(
                "Phase 2A baseline cannot be combined with: " + ", ".join(enabled)
            )
        required_paths = {
            "train_manifest": cfgs.train_manifest,
            "phase2a_recipe": cfgs.phase2a_recipe,
            "phase2a_output_dir": cfgs.phase2a_output_dir,
            "phase2a_budget_gate_report": cfgs.phase2a_budget_gate_report,
        }
        missing_paths = [name for name, value in required_paths.items() if not value]
        if missing_paths:
            raise ValueError("Phase 2A baseline missing: " + ", ".join(missing_paths))
        if not cfgs.verify_manifest_hashes:
            raise ValueError("Phase 2A baseline requires manifest hash verification")
        if cfgs.dataset == "tnbc":
            if cfgs.phase2a_eval_policy != "tnbc_patient_macro" or not cfgs.eval_manifest:
                raise ValueError("TNBC Phase 2A requires p7-p8 eval manifest and patient-macro policy")
        elif cfgs.dataset == "monuseg":
            if cfgs.phase2a_eval_policy != "none" or cfgs.eval_manifest:
                raise ValueError("MoNuSeg Phase 2A forbids any evaluation manifest")
    set_seed(cfgs)

    device = torch.device(
        "cuda:" + str(cfgs.gpu_device) if torch.cuda.is_available() else "cpu"
    )
    net = get_network(
        cfgs,
        cfgs.net,
        use_gpu=cfgs.gpu,
        gpu_device=device,
        distribution=cfgs.distributed,
    )
    model1, model1_encoder = build_model(args)
    model1.to(device)
    model1_encoder.to(device)

    val_texture_bank_template = maybe_load_warm_start(cfgs, model1)
    if cfgs.phase2a_timing_profile and val_texture_bank_template is not None:
        raise ValueError("Phase 2A timing forbids task-specific point-head warm starts")
    if cfgs.phase2a_baseline and val_texture_bank_template is not None:
        raise ValueError("Phase 2A baseline forbids task-specific point-head warm starts")
    freeze_sam2_image_encoder(net)

    actual_lr = args.optimizer.lr if cfgs.lr < 0 else cfgs.lr
    actual_wd = args.optimizer.weight_decay if cfgs.weight_decay < 0 else cfgs.weight_decay
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, list(model1.parameters()) + list(net.parameters())),
        lr=actual_lr,
        weight_decay=actual_wd,
    )
    if cfgs.lr_min >= 0:
        t_max = cfgs.lr_cosine_t_max if cfgs.lr_cosine_t_max > 0 else cfgs.epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=t_max,
            eta_min=cfgs.lr_min,
            last_epoch=-1,
        )
        if t_max != cfgs.epochs:
            final_lr = cfgs.lr_min + 0.5 * (actual_lr - cfgs.lr_min) * (
                1 + math.cos(cfgs.epochs * math.pi / t_max)
            )
            print(
                f"[lr] cosine schedule: peak={actual_lr}, eta_min={cfgs.lr_min}, "
                f"T_max={t_max}; lr at epoch {cfgs.epochs} ~= {final_lr:.2e}"
            )
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=list(cfgs.lr_milestones),
            gamma=0.3,
            last_epoch=-1,
        )

    criterion, _ = build_criterion(args, device)
    print(
        f"[trainable] point_head+sam2={count_trainable_params(model1, net):,} "
        f"(image encoder frozen except prompt_generator)"
    )

    cfgs.path_helper = set_log_dir("logs", cfgs.exp_name)
    logger = create_logger(cfgs.path_helper["log_path"])
    logger.info(cfgs)

    # The dataset captures its coverage-cache directory at construction time.
    # Configure self-bootstrap before building loaders so refresh/reload refer
    # to the same path.  Defaults remain unchanged when self-bootstrap is off.
    if cfgs.pms_self_bootstrap:
        if not cfgs.use_pms:
            print("[pms-self-bootstrap] disabled because --use_pms was not set.")
            cfgs.pms_self_bootstrap = False
        else:
            if cfgs.warmstart_stage:
                if not cfgs.baseline_masks_dir:
                    raise ValueError(
                        "warm-start self-bootstrap requires its validated coverage directory"
                    )
                self_cache_dir = str(Path(cfgs.baseline_masks_dir).resolve())
            elif cfgs.phase2a_baseline:
                self_cache_dir = str(
                    Path(cfgs.phase2a_output_dir).resolve() / "coverage_cache"
                )
            else:
                self_cache_dir = os.path.join(
                    cfgs.path_helper["prefix"], "PMS_SelfBootstrapCoverage"
                )
            cfgs.baseline_masks_dir = self_cache_dir
            os.makedirs(cfgs.baseline_masks_dir, exist_ok=True)
            if cfgs.iterative_baseline_refresh_every <= 0:
                cfgs.iterative_baseline_refresh_every = 10
            if cfgs.coverage_accumulate is None:
                cfgs.coverage_accumulate = True
            print(
                "[pms-self-bootstrap] enabled: "
                f"cache={cfgs.baseline_masks_dir}, "
                f"refresh={cfgs.iterative_baseline_refresh_every}, "
                f"accumulate={cfgs.coverage_accumulate}"
            )

    train_dataset, test_dataset, train_loader, test_loader = build_dataloaders(cfgs, args)

    if cfgs.warmstart_stage == "prepare_coverage":
        run_warmstart_prepare_coverage(
            cfgs,
            args,
            train_dataset,
            model1,
            model1_encoder,
            net,
            device,
            warmstart_manifest_identity,
        )
        return

    if cfgs.warmstart_stage == "smoke":
        run_warmstart_smoke(
            cfgs,
            args,
            train_dataset,
            train_loader,
            model1,
            model1_encoder,
            net,
            criterion,
            optimizer,
            scheduler,
            device,
            warmstart_coverage_identity,
        )
        return

    if cfgs.warmstart_stage == "timing":
        run_warmstart_timing(
            cfgs,
            args,
            train_dataset,
            train_loader,
            model1,
            model1_encoder,
            net,
            criterion,
            optimizer,
            scheduler,
            device,
            warmstart_coverage_identity,
        )
        return

    if cfgs.warmstart_stage == "formal_tnbc_5epoch":
        run_warmstart_formal_tnbc_5epoch(
            cfgs,
            args,
            train_dataset,
            train_loader,
            model1,
            model1_encoder,
            net,
            criterion,
            optimizer,
            scheduler,
            device,
            warmstart_coverage_identity,
            warmstart_manifest_identity,
        )
        return

    if cfgs.train_only_smoke_steps > 0:
        run_train_only_smoke(
            cfgs,
            args,
            train_dataset,
            train_loader,
            model1,
            model1_encoder,
            net,
            criterion,
            optimizer,
            device,
            val_texture_bank_template,
        )
        return

    if cfgs.stage1_coverage_oracle:
        ckpt = torch.load(cfgs.sam_ckpt, map_location="cpu")
        if "model1" in ckpt:
            model1.load_state_dict(ckpt["model1"])
        else:
            print(f"[checkpoint] no model1 state found in {cfgs.sam_ckpt}")
        texture_memory_bank_list = ckpt.get("texture_memory_bank_list", []) or []

        oracle_loader = test_loader
        if cfgs.oracle_split == "train":
            oracle_dataset = build_eval_dataset(cfgs, args, split="train")
            oracle_loader = DataLoader(
                oracle_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=cfgs.num_workers,
                pin_memory=True,
            )
            print(
                f"[stage1-oracle] train split; n={len(oracle_dataset.paths)}; "
                f"manifest={cfgs.train_manifest or 'legacy_directory'}"
            )
        else:
            print(f"[stage1-oracle] test split; n={len(test_loader.dataset)}")

        from stainpqr.coverage_oracle import run_coverage_oracle

        run_coverage_oracle(
            cfgs,
            oracle_loader,
            net,
            model1_encoder,
            texture_memory_bank_list,
            device,
        )
        return

    if cfgs.stage2_selective_refine:
        ckpt = torch.load(cfgs.sam_ckpt, map_location="cpu")
        if "model1" in ckpt:
            model1.load_state_dict(ckpt["model1"])
        else:
            print(f"[checkpoint] no model1 state found in {cfgs.sam_ckpt}")
        texture_memory_bank_list = ckpt.get("texture_memory_bank_list", []) or []

        selective_loader = test_loader
        if cfgs.selective_split == "train":
            selective_dataset = build_eval_dataset(cfgs, args, split="train")
            selective_loader = DataLoader(
                selective_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=cfgs.num_workers,
                pin_memory=True,
            )
            print(
                f"[stage2-selective] train split; n={len(selective_dataset.paths)}; "
                f"manifest={cfgs.train_manifest or 'legacy_directory'}"
            )
        else:
            print(f"[stage2-selective] test split; n={len(test_loader.dataset)}")

        from stainpqr.coverage_oracle import run_selective_coverage_refinement

        run_selective_coverage_refinement(
            cfgs,
            selective_loader,
            net,
            model1_encoder,
            texture_memory_bank_list,
            device,
        )
        return

    if cfgs.eval:
        ckpt = torch.load(cfgs.sam_ckpt, map_location="cpu")
        if "model1" in ckpt:
            model1.load_state_dict(ckpt["model1"])
        if "epoch" in ckpt:
            settings.EPOCH = ckpt["epoch"]
        texture_memory_bank_list = ckpt.get("texture_memory_bank_list", []) or []

        eval_loader = test_loader
        eval_split = "test"
        if cfgs.eval_on_train:
            eval_dataset = build_eval_dataset(cfgs, args, split="train")
            eval_loader = DataLoader(
                eval_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=cfgs.num_workers,
                pin_memory=True,
            )
            eval_split = "train"
            print(
                f"[eval] train split; n={len(eval_dataset.paths)}; "
                f"manifest={cfgs.train_manifest or 'legacy_directory'}"
            )

        metrics = validation_on_epoch(
            cfgs,
            args,
            eval_loader,
            settings.EPOCH,
            model1,
            model1_encoder,
            net,
            cfgs.load,
            args.data.post.iou_threshold,
            texture_memory_bank_list,
            device,
        )
        seg_dice1, seg_dice2, seg_aji, seg_aji_p, seg_dq, seg_sq, seg_pq = metrics
        print(
            f"split: {eval_split} epoch: {settings.EPOCH} "
            f"dice1: {seg_dice1 * 100:.2f} dice2: {seg_dice2 * 100:.2f} "
            f"aji: {seg_aji * 100:.2f} aji_p: {seg_aji_p * 100:.2f} "
            f"dq: {seg_dq * 100:.2f} sq: {seg_sq * 100:.2f} pq: {seg_pq * 100:.2f}"
        )
        write_eval_metric_artifact(cfgs, eval_split, metrics)
        return

    if cfgs.phase2a_timing_profile:
        refresh_record = {
            "count": 0,
            "wall_seconds": 0.0,
            "train_record_count": len(train_dataset),
        }
        if cfgs.phase2a_timing_profile == "pms_active":
            started = time.perf_counter()
            refresh_baseline_masks_inplace(
                cfgs,
                args,
                train_dataset,
                None,
                model1,
                model1_encoder,
                net,
                None,
                -1,
                device,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize(_cuda_device_index(device))
            refresh_record["count"] = 1
            refresh_record["wall_seconds"] = time.perf_counter() - started
        run_phase2a_timing(
            cfgs,
            args,
            train_dataset,
            train_loader,
            model1,
            model1_encoder,
            net,
            criterion,
            optimizer,
            device,
            coverage_refresh_record=refresh_record,
        )
        return

    if cfgs.phase2a_baseline:
        run_phase2a_baseline(
            cfgs,
            args,
            train_dataset,
            test_dataset,
            train_loader,
            test_loader,
            model1,
            model1_encoder,
            net,
            criterion,
            optimizer,
            scheduler,
            device,
        )
        return

    iter_refresh_every = int(cfgs.iterative_baseline_refresh_every or 0)
    pms_self_bootstrap = bool(cfgs.pms_self_bootstrap)
    pms_start_epoch = int(cfgs.pms_start_epoch or 0)
    if iter_refresh_every > 0 and not cfgs.use_pms:
        print("[coverage-refresh] disabled because --use_pms was not set.")
        iter_refresh_every = 0
    if iter_refresh_every > 0 and not cfgs.baseline_masks_dir:
        print("[coverage-refresh] disabled because --baseline_masks_dir is empty.")
        iter_refresh_every = 0

    detect_loss = []
    segment_loss = []
    all_loss = []
    dice1 = []
    dice2 = []
    aji = []
    aji_p = []
    dq = []
    sq = []
    pq = []
    best_pq = 0.0
    best_aji = 0.0

    settings.EPOCH = cfgs.epochs
    if pms_self_bootstrap and iter_refresh_every > 0 and pms_start_epoch <= 0:
        print("[pms-self-bootstrap] generating initial coverage maps before epoch 0.")
        refresh_baseline_masks_inplace(
            cfgs,
            args,
            train_dataset,
            test_dataset,
            model1,
            model1_encoder,
            net,
            val_texture_bank_template,
            -1,
            device,
        )

    for epoch in range(settings.EPOCH):
        deferred_c0 = (
            pms_self_bootstrap
            and iter_refresh_every > 0
            and pms_start_epoch > 0
            and epoch == pms_start_epoch
        )
        if deferred_c0:
            print(f"[pms-self-bootstrap] epoch {epoch}: generating initial coverage maps.")
            refresh_baseline_masks_inplace(
                cfgs,
                args,
                train_dataset,
                test_dataset,
                model1,
                model1_encoder,
                net,
                val_texture_bank_template,
                epoch,
                device,
            )
        else:
            refresh_base = max(0, pms_start_epoch)
            should_refresh = (
                iter_refresh_every > 0
                and epoch > refresh_base
                and (epoch - refresh_base) % iter_refresh_every == 0
            )
            if should_refresh:
                refresh_baseline_masks_inplace(
                    cfgs,
                    args,
                    train_dataset,
                    test_dataset,
                    model1,
                    model1_encoder,
                    net,
                    val_texture_bank_template,
                    epoch,
                    device,
                )

        texture_memory_bank_list = []
        net.train()
        start = time.time()
        log_info = train_on_epoch(
            cfgs,
            model1,
            model1_encoder,
            net,
            train_loader,
            criterion,
            optimizer,
            epoch,
            texture_memory_bank_list,
            device,
        )
        logger.info(f"Train loss: {log_info} || epoch {epoch}.")
        print("time_for_training", time.time() - start)

        detect_loss_tmp = (
            log_info.get("loss_reg", 0.0)
            + log_info.get("loss_cls", 0.0)
            + log_info.get("loss_mask", 0.0)
        )
        segment_loss_tmp = (
            log_info.get("loss_focal", 0.0)
            + log_info.get("loss_dice", 0.0)
            + log_info.get("loss_iou", 0.0)
        )
        all_loss_tmp = sum(log_info.values()) if log_info else 0.0
        detect_loss.append(detect_loss_tmp)
        segment_loss.append(segment_loss_tmp)
        all_loss.append(all_loss_tmp)

        scheduler.step()
        net.eval()
        should_validate = (
            epoch > cfgs.val_start_epoch
            and (epoch % cfgs.val_freq == 0 or epoch == settings.EPOCH - 1)
        )
        if not should_validate:
            continue

        val_texture_bank = (
            list(val_texture_bank_template)
            if val_texture_bank_template is not None
            else texture_memory_bank_list.copy()
        )
        print(f"[test-eval] test split evaluation; n={len(test_loader.dataset)}")
        metrics = validation_on_epoch(
            cfgs,
            args,
            test_loader,
            epoch,
            model1,
            model1_encoder,
            net,
            cfgs.load,
            args.data.post.iou_threshold,
            val_texture_bank,
            device,
        )
        seg_dice1, seg_dice2, seg_aji, seg_aji_p, seg_dq, seg_sq, seg_pq = metrics
        print(
            f"dice1: {seg_dice1 * 100:.2f} dice2: {seg_dice2 * 100:.2f} "
            f"aji: {seg_aji * 100:.2f} aji_p: {seg_aji_p * 100:.2f} "
            f"dq: {seg_dq * 100:.2f} sq: {seg_sq * 100:.2f} pq: {seg_pq * 100:.2f}"
        )
        dice1.append(seg_dice1)
        dice2.append(seg_dice2)
        aji.append(seg_aji)
        aji_p.append(seg_aji_p)
        dq.append(seg_dq)
        sq.append(seg_sq)
        pq.append(seg_pq)

        if seg_pq > best_pq:
            best_pq = seg_pq
            torch.save(
                {
                    "model": net.state_dict(),
                    "model1": model1.state_dict(),
                    "parameter": net._parameters,
                    "epoch": epoch,
                    "texture_memory_bank_list": texture_memory_bank_list,
                },
                os.path.join(cfgs.path_helper["ckpt_path"], "base_pq_epoch.pth"),
            )
        if seg_aji > best_aji:
            best_aji = seg_aji
            torch.save(
                {
                    "model": net.state_dict(),
                    "model1": model1.state_dict(),
                    "parameter": net._parameters,
                    "epoch": epoch,
                    "texture_memory_bank_list": texture_memory_bank_list,
                },
                os.path.join(cfgs.path_helper["ckpt_path"], "base_aji_epoch.pth"),
            )

    if detect_loss:
        epochs = np.arange(1, len(detect_loss) + 1)
        fig, ax1 = plt.subplots(figsize=(20, 12))
        ax1.plot(epochs, detect_loss, marker="o", linestyle="-", color="b", label="Detect Loss")
        ax1.plot(epochs, segment_loss, marker="o", linestyle="-", color="g", label="Segment Loss")
        ax1.plot(epochs, all_loss, marker="o", linestyle="-", color="r", label="Total Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.grid(True)
        ax1.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(cfgs.path_helper["sample_path"], "Loss.png"))
        plt.close()

    if dice1:
        epochs = np.arange(1, len(dice1) + 1)
        fig, ax2 = plt.subplots(figsize=(20, 12))
        ax2.plot(epochs, dice1, marker="o", linestyle="-", color="b", label="Dice1")
        ax2.plot(epochs, dice2, marker="o", linestyle="-", color="navy", label="Dice2")
        ax2.plot(epochs, aji, marker="o", linestyle="-", color="g", label="AJI")
        ax2.plot(epochs, aji_p, marker="o", linestyle="-", color="r", label="AJI+")
        ax2.plot(epochs, dq, marker="o", linestyle="-", color="c", label="DQ")
        ax2.plot(epochs, sq, marker="o", linestyle="-", color="m", label="SQ")
        ax2.plot(epochs, pq, marker="o", linestyle="-", color="y", label="PQ")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Metric")
        ax2.grid(True)
        ax2.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(cfgs.path_helper["sample_path"], "Metrics.png"))
        plt.close()


if __name__ == "__main__":
    main()
