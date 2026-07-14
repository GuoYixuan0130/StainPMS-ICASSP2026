import copy
import csv
import glob
import hashlib
import json
import math
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from mmengine.config import Config
from torch.utils.data import DataLoader, Subset

import cfg
from conf import settings
from run.dataset.monuseg import MONUSEG
from run.run_on_epoch import train_on_epoch, validation_on_epoch
from run.utils import create_logger, get_network, set_log_dir
from sam2_train.modeling.criterion import build_criterion
from sam2_train.modeling.dpa_p2pnet import build_model
from sam2_train.modeling.utils import collate_fn, set_seed
from setpms import L2SPAnchor


def count_trainable_params(*modules):
    return sum(
        param.numel()
        for module in modules
        for param in module.parameters()
        if param.requires_grad
    )


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


def _train_split_paths(cfgs):
    if cfgs.dataset == "monuseg":
        return (
            os.path.join(cfgs.data_path, "train_12", "images"),
            os.path.join(cfgs.data_path, "train_12", "labels"),
        )
    raise ValueError(f"Unsupported dataset: {cfgs.dataset}")


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
    train_img_root, train_lbl_root = _train_split_paths(cfgs)
    refresh_dataset = copy.copy(test_dataset)
    refresh_dataset.image_root = train_img_root
    refresh_dataset.label_root = train_lbl_root
    refresh_dataset.paths = sorted(os.listdir(train_img_root))
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
        f"n={len(refresh_dataset.paths)}; dump={cfgs.baseline_masks_dir}"
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


def build_dataloaders(cfgs, args):
    if cfgs.dataset != "monuseg":
        raise ValueError(f"Unsupported dataset: {cfgs.dataset}")

    train_dataset = MONUSEG(cfgs, args, cfgs.data_path, cfgs.load, mode="train")
    eval_mode = "eval_train" if cfgs.eval_on_train else "test"
    test_dataset = MONUSEG(cfgs, args, cfgs.data_path, cfgs.load, mode=eval_mode)
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfgs.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfgs.num_workers,
        pin_memory=True,
    )
    return train_dataset, test_dataset, train_loader, test_loader


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
        "command": list(sys.argv),
    }
    os.makedirs(dump_dir, exist_ok=True)
    path = os.path.join(dump_dir, "main_eval_metrics.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"Wrote unrounded evaluation metrics: {path}")


_METRIC_NAMES = ("dice", "dice2", "aji", "aji_plus", "dq", "sq", "pq")


def _parse_epoch_set(raw_value):
    if not raw_value:
        return set()
    values = set()
    for token in str(raw_value).split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value < 0:
            raise ValueError(f"Continuation epoch must be non-negative, got {value}")
        values.add(value)
    return values


def _append_csv(path, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _save_continuation_checkpoint(
    cfgs,
    epoch,
    net,
    model1,
    optimizer,
    scheduler,
    texture_memory_bank_list,
    *,
    filename=None,
):
    """Save an archival continuation checkpoint without duplicating AdamW state.

    The required epoch nodes retain both learned model state dicts and the
    texture-memory state used by the continuation.  Optimizer and scheduler
    state are intentionally excluded from normal archival nodes: persisting
    AdamW moments at every required node would exceed the authorised data-disk
    capacity without adding an evaluation or selection capability.
    """

    filename = filename or f"continuation_epoch_{int(epoch)}.pth"
    path = os.path.join(cfgs.path_helper["ckpt_path"], filename)
    torch.save(
        {
            "model": net.state_dict(),
            "model1": model1.state_dict(),
            "epoch": int(epoch),
            "texture_memory_bank_list": texture_memory_bank_list,
            "checkpoint_kind": "continuation_model_weights",
            "optimizer_state_included": False,
        },
        path,
    )
    return path


def _prune_interrupted_continuation_rows(metrics_dir, run_label, resume_epoch):
    """Discard only stale post-resume rows from the active interrupted run."""

    for filename in ("training_curves.csv", "metrics.csv", "per_image.csv"):
        path = os.path.join(metrics_dir, filename)
        if not os.path.isfile(path):
            continue
        with open(path, "r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames
            rows = list(reader)
        if not fieldnames:
            continue
        retained = [
            row
            for row in rows
            if row.get("run_label") != str(run_label)
            or int(row.get("epoch", -1)) <= int(resume_epoch)
        ]
        if len(retained) == len(rows):
            continue
        temporary = path + ".resume_tmp"
        with open(temporary, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(retained)
        os.replace(temporary, path)


def _restore_continuation_state(path, net, model1, optimizer, scheduler, device, epochs):
    """Restore a full-state node after an external storage interruption."""

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Continuation recovery checkpoint is absent: {path}")
    checkpoint = torch.load(path, map_location=device)
    required = {"model", "model1", "epoch", "optimizer", "scheduler"}
    missing = sorted(required.difference(checkpoint))
    if missing:
        raise RuntimeError(
            "Continuation recovery requires a full optimizer/scheduler checkpoint; "
            f"{path} is missing {missing}"
        )
    completed_epoch = int(checkpoint["epoch"])
    if completed_epoch <= 0 or completed_epoch >= int(epochs):
        raise ValueError(
            f"Recovery epoch {completed_epoch} must be in [1, {int(epochs) - 1}]"
        )
    net.load_state_dict(checkpoint["model"], strict=True)
    model1.load_state_dict(checkpoint["model1"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    return completed_epoch


def _continuation_evaluate(
    cfgs,
    args,
    test_loader,
    epoch,
    model1,
    model1_encoder,
    net,
    val_texture_bank,
    device,
):
    """Evaluate a frozen continuation node and emit aggregate/per-image CSV."""

    records = []
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
        metric_records=records,
    )
    if cfgs.metrics_output_dir:
        row = {"run_label": cfgs.run_label, "epoch": int(epoch)}
        row.update({name: float(value) for name, value in zip(_METRIC_NAMES, metrics)})
        _append_csv(os.path.join(cfgs.metrics_output_dir, "metrics.csv"), row)
        for record in records:
            record = {"run_label": cfgs.run_label, "epoch": int(epoch), **record}
            _append_csv(os.path.join(cfgs.metrics_output_dir, "per_image.csv"), record)
    return metrics


def _smoke_baseline_equivalence(
    cfgs,
    args,
    test_loader,
    model1,
    model1_encoder,
    net,
    val_texture_bank_template,
    device,
):
    """Prove pixel-level inference equality with SetPMS toggled off/on.

    The comparison runs the unmodified canonical validation path on one frozen
    holdout image twice, writes no scientific evaluation conclusion, and
    compares the assembled instance-map pixels byte-for-byte.
    """

    if len(test_loader.dataset) == 0:
        raise ValueError("Cannot run baseline-equivalence smoke with an empty holdout")
    output_root = cfgs.metrics_output_dir or cfgs.path_helper["prefix"]
    off_dir = os.path.join(output_root, "baseline_equivalence_off")
    on_dir = os.path.join(output_root, "baseline_equivalence_on")
    saved_flag = bool(cfgs.setpms)
    saved_dump_dir = str(cfgs.dump_eval_artifacts_dir or "")
    saved_vis = bool(cfgs.vis)
    smoke_loader = DataLoader(
        Subset(test_loader.dataset, [0]),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    outputs = []
    try:
        for enabled, artifact_dir in ((False, off_dir), (True, on_dir)):
            cfgs.setpms = enabled
            cfgs.dump_eval_artifacts_dir = artifact_dir
            cfgs.vis = False
            validation_on_epoch(
                cfgs,
                args,
                smoke_loader,
                0,
                model1,
                model1_encoder,
                net,
                cfgs.load,
                args.data.post.iou_threshold,
                list(val_texture_bank_template or []),
                device,
            )
            paths = sorted(glob.glob(os.path.join(artifact_dir, "*_pred.npy")))
            if len(paths) != 1:
                raise RuntimeError(
                    f"Expected one smoke prediction in {artifact_dir}, found {len(paths)}"
                )
            prediction = np.load(paths[0])
            outputs.append(prediction)
    finally:
        cfgs.setpms = saved_flag
        cfgs.dump_eval_artifacts_dir = saved_dump_dir
        cfgs.vis = saved_vis

    identical = bool(np.array_equal(outputs[0], outputs[1]))
    payload = {
        "checked_images": 1,
        "pixel_identical": identical,
        "sha256_setpms_off": hashlib.sha256(outputs[0].tobytes()).hexdigest(),
        "sha256_setpms_on": hashlib.sha256(outputs[1].tobytes()).hexdigest(),
    }
    if not identical:
        raise RuntimeError("SetPMS changed canonical inference pixels during smoke")
    return payload


def main():
    args = Config.fromfile("./args.py")
    cfgs = cfg.parse_args()
    apply_cli_overrides(args, cfgs)
    if cfgs.setpms and (
        cfgs.use_pms
        or cfgs.pms_self_bootstrap
        or cfgs.iterative_baseline_refresh_every > 0
        or cfgs.dump_baseline_masks_dir
        or cfgs.baseline_masks_dir
    ):
        raise ValueError(
            "SetPMS continuation forbids PMS mining, coverage refresh, and pseudo prompts."
        )
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
    anchor_regularizer = L2SPAnchor((model1, net)) if cfgs.setpms else None

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

    cfgs.path_helper = set_log_dir("logs", cfgs.exp_name, cfgs.run_dir)
    logger = create_logger(cfgs.path_helper["log_path"])
    logger.info(cfgs)

    train_dataset, test_dataset, train_loader, test_loader = build_dataloaders(cfgs, args)

    if cfgs.stage1_coverage_oracle:
        ckpt = torch.load(cfgs.sam_ckpt, map_location="cpu")
        if "model1" in ckpt:
            model1.load_state_dict(ckpt["model1"])
        else:
            print(f"[checkpoint] no model1 state found in {cfgs.sam_ckpt}")
        texture_memory_bank_list = ckpt.get("texture_memory_bank_list", []) or []

        oracle_loader = test_loader
        if cfgs.oracle_split == "train":
            train_img_root, train_lbl_root = _train_split_paths(cfgs)
            oracle_dataset = copy.copy(test_dataset)
            oracle_dataset.image_root = train_img_root
            oracle_dataset.label_root = train_lbl_root
            oracle_dataset.paths = sorted(os.listdir(train_img_root))
            oracle_loader = DataLoader(
                oracle_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=cfgs.num_workers,
                pin_memory=True,
            )
            print(f"[stage1-oracle] train split; n={len(oracle_dataset.paths)} from {train_img_root}")
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
            train_img_root, train_lbl_root = _train_split_paths(cfgs)
            selective_dataset = copy.copy(test_dataset)
            selective_dataset.image_root = train_img_root
            selective_dataset.label_root = train_lbl_root
            selective_dataset.paths = sorted(os.listdir(train_img_root))
            selective_loader = DataLoader(
                selective_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=cfgs.num_workers,
                pin_memory=True,
            )
            print(f"[stage2-selective] train split; n={len(selective_dataset.paths)} from {train_img_root}")
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
        eval_split = "train" if cfgs.eval_on_train else "test"
        if cfgs.eval_on_train:
            print(
                f"[eval] frozen train-split holdout; n={len(eval_loader.dataset.paths)} "
                f"from {eval_loader.dataset.image_root}"
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

    smoke_batches = int(cfgs.setpms_smoke_batches or 0)
    if smoke_batches < 0:
        raise ValueError("--setpms_smoke_batches must be non-negative")
    if smoke_batches and not cfgs.setpms:
        raise ValueError("SetPMS smoke requires --setpms")

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
    continuation_save_epochs = _parse_epoch_set(cfgs.continuation_save_epochs)
    continuation_eval_epochs = _parse_epoch_set(cfgs.continuation_eval_epochs)
    controlled_continuation = bool(
        continuation_save_epochs or continuation_eval_epochs
    )
    if smoke_batches and controlled_continuation:
        raise ValueError("Mechanical SetPMS smoke cannot include continuation checkpoints/evaluation")
    if controlled_continuation and not continuation_eval_epochs:
        raise ValueError("Controlled continuation requires explicit evaluation epochs")
    if controlled_continuation and not cfgs.eval_on_train:
        raise ValueError(
            "Controlled continuation is restricted to an explicit train-split holdout manifest"
        )
    best_selection_score = float("-inf")
    resume_checkpoint = str(cfgs.continuation_resume_checkpoint or "")
    resume_epoch = 0
    if resume_checkpoint:
        if smoke_batches or not controlled_continuation:
            raise ValueError(
                "Continuation recovery requires a non-smoke controlled continuation run"
            )
        resume_checkpoint = os.path.abspath(resume_checkpoint)
        resume_epoch = _restore_continuation_state(
            resume_checkpoint,
            net,
            model1,
            optimizer,
            scheduler,
            device,
            cfgs.epochs,
        )
        if not cfgs.metrics_output_dir or not cfgs.run_label:
            raise ValueError("Continuation recovery requires metrics_output_dir and run_label")
        _prune_interrupted_continuation_rows(
            cfgs.metrics_output_dir,
            cfgs.run_label,
            resume_epoch,
        )
        print(
            f"[continuation-resume] restored completed epoch {resume_epoch} from "
            f"{resume_checkpoint}; continuing at epoch {resume_epoch}"
        )

    settings.EPOCH = 1 if smoke_batches else cfgs.epochs
    smoke_equivalence = None
    if smoke_batches:
        smoke_equivalence = _smoke_baseline_equivalence(
            cfgs,
            args,
            test_loader,
            model1,
            model1_encoder,
            net,
            val_texture_bank_template,
            device,
        )
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
    if controlled_continuation and not resume_checkpoint:
        initial_texture_bank = (
            list(val_texture_bank_template)
            if val_texture_bank_template is not None
            else []
        )
        if 0 in continuation_save_epochs:
            _save_continuation_checkpoint(
                cfgs,
                0,
                net,
                model1,
                optimizer,
                scheduler,
                initial_texture_bank,
            )
        if 0 in continuation_eval_epochs:
            print(
                f"[continuation-eval] frozen holdout epoch 0; n={len(test_loader.dataset)}"
            )
            initial_metrics = _continuation_evaluate(
                cfgs,
                args,
                test_loader,
                0,
                model1,
                model1_encoder,
                net,
                initial_texture_bank,
                device,
            )
            if cfgs.baseline_reference_aji >= 0 and (
                abs(float(initial_metrics[2]) - float(cfgs.baseline_reference_aji))
                > float(cfgs.baseline_reference_tolerance)
            ):
                raise RuntimeError(
                    "TNBC step-0 AJI disagrees with the fixed canonical reference; "
                    "baseline attribution is required before continuation."
                )
            if cfgs.baseline_reference_pq >= 0 and (
                abs(float(initial_metrics[6]) - float(cfgs.baseline_reference_pq))
                > float(cfgs.baseline_reference_tolerance)
            ):
                raise RuntimeError(
                    "TNBC step-0 PQ disagrees with the fixed canonical reference; "
                    "baseline attribution is required before continuation."
                )
            initial_score = float(initial_metrics[2] + initial_metrics[6])
            if initial_score > best_selection_score:
                best_selection_score = initial_score
                print(
                    "[continuation-best] epoch 0 is currently selected; "
                    "the required continuation_epoch_0.pth is the best checkpoint."
                )
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

    for epoch in range(resume_epoch, settings.EPOCH):
        if hasattr(train_dataset, "set_epoch"):
            train_dataset.set_epoch(epoch)
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
            anchor_regularizer=anchor_regularizer,
        )
        logger.info(f"Train loss: {log_info} || epoch {epoch}.")
        print("time_for_training", time.time() - start)

        if smoke_batches:
            observed_steps = int(log_info.get("setpms_smoke_optimizer_steps", 0))
            if observed_steps != smoke_batches:
                raise RuntimeError(
                    f"SetPMS smoke requested {smoke_batches} optimizer steps, observed {observed_steps}"
                )
            finite_losses = all(math.isfinite(float(value)) for value in log_info.values())
            smoke_payload = {
                "requested_optimizer_steps": smoke_batches,
                "observed_optimizer_steps": observed_steps,
                "finite_losses": finite_losses,
                "cuda_max_memory_bytes": (
                    int(torch.cuda.max_memory_allocated(device))
                    if torch.cuda.is_available()
                    else 0
                ),
                "baseline_equivalence": smoke_equivalence,
                "seed": int(cfgs.seed),
                "tta": bool(cfgs.tta),
                "set_loss_weight": 1.0,
                "note": "Mechanical smoke only; formal continuation retains the fixed 0->0.1 warm-up.",
            }
            if not finite_losses:
                raise RuntimeError("SetPMS smoke produced non-finite logged losses")
            output_root = cfgs.metrics_output_dir or cfgs.path_helper["prefix"]
            os.makedirs(output_root, exist_ok=True)
            with open(os.path.join(output_root, "smoke_report.json"), "w", encoding="utf-8") as handle:
                json.dump(smoke_payload, handle, indent=2)
            with open(os.path.join(output_root, "baseline_equivalence.json"), "w", encoding="utf-8") as handle:
                json.dump(smoke_equivalence, handle, indent=2)
            return

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
        completed_epoch = epoch + 1
        if cfgs.metrics_output_dir:
            curve_row = {
                "run_label": cfgs.run_label,
                "epoch": int(completed_epoch),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "detect_loss": float(detect_loss_tmp),
                "segment_loss": float(segment_loss_tmp),
                "total_loss": float(all_loss_tmp),
            }
            for key, value in sorted(log_info.items()):
                curve_row[key] = float(value)
            _append_csv(
                os.path.join(cfgs.metrics_output_dir, "training_curves.csv"), curve_row
            )

        scheduler.step()
        if controlled_continuation and completed_epoch in continuation_save_epochs:
            _save_continuation_checkpoint(
                cfgs,
                completed_epoch,
                net,
                model1,
                optimizer,
                scheduler,
                texture_memory_bank_list,
            )
        net.eval()
        if controlled_continuation:
            should_validate = completed_epoch in continuation_eval_epochs
        else:
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
        split_label = "frozen train-split holdout" if cfgs.eval_on_train else "test split"
        print(f"[test-eval] {split_label} evaluation; n={len(test_loader.dataset)}")
        if controlled_continuation:
            metrics = _continuation_evaluate(
                cfgs,
                args,
                test_loader,
                completed_epoch,
                model1,
                model1_encoder,
                net,
                val_texture_bank,
                device,
            )
        else:
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

        if controlled_continuation:
            selection_score = float(seg_aji + seg_pq)
            if selection_score > best_selection_score:
                best_selection_score = selection_score
                print(
                    f"[continuation-best] epoch {completed_epoch} is currently selected; "
                    f"the required continuation_epoch_{completed_epoch}.pth is the best checkpoint."
                )
        elif seg_pq > best_pq:
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
        if not controlled_continuation and seg_aji > best_aji:
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

    if cfgs.metrics_output_dir:
        os.makedirs(cfgs.metrics_output_dir, exist_ok=True)
        runtime_payload = {
            "run_label": cfgs.run_label,
            "cuda_max_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if torch.cuda.is_available()
                else 0
            ),
            "cuda_max_memory_reserved_bytes": (
                int(torch.cuda.max_memory_reserved(device))
                if torch.cuda.is_available()
                else 0
            ),
        }
        with open(
            os.path.join(cfgs.metrics_output_dir, "runtime_memory.json"),
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(runtime_payload, handle, indent=2)

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
