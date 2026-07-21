import hashlib
import json
import math
import os
import platform
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
    if smoke_steps > 0:
        if not cfgs.train_manifest:
            raise ValueError("train-only smoke requires --train_manifest")
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=False,
        # A smoke must not prefetch unreported samples.  Full training retains
        # the configured worker count below its separate branch.
        num_workers=0 if smoke_steps > 0 else cfgs.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    if smoke_steps > 0:
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


def maybe_load_warm_start(cfgs, model1):
    if cfgs.eval:
        return None
    ckpt = torch.load(cfgs.sam_ckpt, map_location="cpu")
    has_point_head = "model1" in ckpt
    del ckpt
    if not has_point_head:
        return None
    load_ca_sam2_point_head_checkpoint(cfgs, model1)
    return load_ca_sam2_texture_bank(cfgs)


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


def main():
    args = Config.fromfile("./args.py")
    cfgs = cfg.parse_args()
    apply_cli_overrides(args, cfgs)
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

    train_dataset, test_dataset, train_loader, test_loader = build_dataloaders(cfgs, args)

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

    if cfgs.pms_self_bootstrap:
        if not cfgs.use_pms:
            print("[pms-self-bootstrap] disabled because --use_pms was not set.")
            cfgs.pms_self_bootstrap = False
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
