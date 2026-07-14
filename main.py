import copy
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from mmengine.config import Config
from torch.utils.data import DataLoader

import cfg
from conf import settings
from run.dataset.monuseg import MONUSEG
from run.run_on_epoch import train_on_epoch, validation_on_epoch
from run.utils import create_logger, get_network, set_log_dir
from resimixpms.experiment import EvaluationSchedule, append_csv, write_json
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


def _train_evaluation_dataset(test_dataset):
    """Create the manifest-filtered train view used by static coverage/audits."""
    dataset = copy.copy(test_dataset)
    if not hasattr(dataset, "use_train_split_for_evaluation"):
        raise TypeError("dataset lacks a manifest-safe train evaluation view")
    dataset.use_train_split_for_evaluation()
    return dataset


def _set_explicit_log_dir(prefix):
    """Create the historical log layout under one pre-registered artifact dir."""
    prefix = os.path.abspath(prefix)
    os.makedirs(prefix, exist_ok=False)
    path_dict = {"prefix": prefix}
    for key, name in (("ckpt_path", "Model"), ("log_path", "Log"), ("sample_path", "Samples")):
        path = os.path.join(prefix, name)
        os.makedirs(path, exist_ok=False)
        path_dict[key] = path
    return path_dict


def _save_full_eval_checkpoint(path, net, model1, epoch, texture_memory_bank_list):
    """Persist every pre-registered evaluation state; never replace another node."""
    torch.save(
        {
            "model": net.state_dict(),
            "model1": model1.state_dict(),
            "parameter": net._parameters,
            "epoch": int(epoch),
            "texture_memory_bank_list": list(texture_memory_bank_list),
        },
        path,
    )


def _metric_payload(metrics):
    names = ("dice1", "dice2", "aji", "aji_p", "dq", "sq", "pq")
    return {name: float(value) for name, value in zip(names, metrics, strict=True)}


_RESIMIX_EVENT_FIELDS = (
    "epoch", "sample_key", "status", "donor_id", "donor_category", "donor_source_id", "donor_patient_id", "requested_host_mode",
    "host_mode", "host_fallback", "context_distance", "nearest_gt_distance", "proposal_ranked_count",
    "coverage_overlap_pixels", "source_area", "placed_area", "area_ratio", "clip_fraction", "seam_gradient",
    "synthetic_instance_id", "instance_count_before", "instance_count_after", "synthetic_prompt_added", "reason",
)


def record_resimix_events(cfgs, train_dataset):
    """Persist dataset-local deterministic transplant telemetry after each epoch."""
    if not hasattr(train_dataset, "consume_resimix_events"):
        return []
    events = train_dataset.consume_resimix_events()
    if not events:
        return events
    target = os.path.join(cfgs.path_helper["prefix"], "synthetic_acceptance.csv")
    for event in events:
        append_csv(target, event, _RESIMIX_EVENT_FIELDS)
    return events


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
    refresh_dataset = _train_evaluation_dataset(test_dataset)
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

    if cfgs.train_only_eval and not (cfgs.eval and cfgs.eval_on_train):
        raise ValueError("--train_only_eval is valid only with --eval --eval_on_train")

    # Static-PMS Control is a formal arm in its own right.  Keep its protocol
    # gate independent of the ResiMix switch so invoking ``main.py`` directly
    # cannot silently turn the matched control into a self-bootstrapped or
    # refreshed-coverage variant.  Coverage construction itself does not set
    # --use_pms and is intentionally outside this training-arm gate.
    formal_static_pms = cfgs.data_identity in ("tnbc", "monuseg_lite") and cfgs.use_pms
    if formal_static_pms:
        if not cfgs.train_manifest or not cfgs.test_manifest:
            raise ValueError("formal Static-PMS requires explicit immutable train/test manifests")
        if int(cfgs.seed) != 3407 or int(cfgs.b) != 1:
            raise ValueError("formal Static-PMS fixes seed=3407 and batch size=1")
        if cfgs.tta or not cfgs.texture or not cfgs.context or int(cfgs.test_nms_thr) != 12:
            raise ValueError("formal Static-PMS fixes TTA off, texture/context on, and NMS=12")
        if int(cfgs.crop_size) != 256:
            raise ValueError("formal Static-PMS fixes crop_size=256")
        fixed_weights = {
            "pms_loss_coef": (cfgs.pms_loss_coef, 0.5),
            "pms_residual_mask_weight": (cfgs.pms_residual_mask_weight, 0.3),
            "pms_preserve_loss_coef": (cfgs.pms_preserve_loss_coef, 1.0),
            "pms_object_weight": (cfgs.pms_object_weight, 1.0),
        }
        for name, (actual, expected) in fixed_weights.items():
            if abs(float(actual) - expected) > 1e-12:
                raise ValueError(f"formal Static-PMS fixes {name}={expected}")
        if cfgs.pms_self_bootstrap or int(cfgs.iterative_baseline_refresh_every or 0) != 0:
            raise ValueError("formal Static-PMS requires one static coverage cache; refresh is forbidden")
        if bool(cfgs.coverage_accumulate):
            raise ValueError("formal Static-PMS forbids coverage accumulation")
        if not cfgs.baseline_masks_dir:
            raise ValueError("formal Static-PMS requires an explicit static --baseline_masks_dir")
        if not cfgs.coverage_manifest:
            raise ValueError("formal Static-PMS requires --coverage_manifest")
        if Path(cfgs.baseline_masks_dir).resolve() != Path(cfgs.coverage_manifest).resolve().parent:
            raise ValueError("baseline_masks_dir must be the exact parent of the sealed coverage manifest")

    if cfgs.resimix_enabled:
        if not cfgs.resimix_config:
            raise ValueError("--resimix_enabled requires --resimix_config")
        if not formal_static_pms:
            raise ValueError("ResiMix-PMS requires an explicit formal Static-PMS configuration")

    train_dataset = MONUSEG(cfgs, args, cfgs.data_path, cfgs.load, mode="train")
    test_dataset = MONUSEG(
        cfgs,
        args,
        cfgs.data_path,
        cfgs.load,
        mode="test",
        source_split="train" if cfgs.train_only_eval else "test",
    )
    # Dataset-side ResiMix telemetry is deliberately process-local so every
    # accepted/rejected transplant can be recorded deterministically.
    num_workers = 0 if cfgs.resimix_enabled else cfgs.num_workers
    if cfgs.resimix_enabled and cfgs.num_workers != 0:
        print("[resimix] forcing num_workers=0 for deterministic event accounting")
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
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


def run_registered_evaluation(
    cfgs,
    args,
    test_loader,
    completed_epochs,
    model1,
    model1_encoder,
    net,
    val_texture_bank_template,
    fallback_texture_bank,
    device,
):
    """Evaluate/save one pre-registered node without choosing by dev results."""
    val_texture_bank = (
        list(val_texture_bank_template)
        if val_texture_bank_template is not None
        else list(fallback_texture_bank)
    )
    print(
        f"[registered-eval] completed_epochs={completed_epochs}; "
        f"n={len(test_loader.dataset)}"
    )
    # Per-image rows, when requested, must not collide across evaluation nodes.
    original_per_image = str(getattr(cfgs, "per_image_metrics_path", "") or "")
    if original_per_image:
        root, ext = os.path.splitext(original_per_image)
        cfgs.per_image_metrics_path = f"{root}_epoch_{int(completed_epochs):02d}{ext or '.csv'}"
    try:
        metrics = validation_on_epoch(
            cfgs,
            args,
            test_loader,
            int(completed_epochs),
            model1,
            model1_encoder,
            net,
            cfgs.load,
            args.data.post.iou_threshold,
            val_texture_bank,
            device,
        )
    finally:
        cfgs.per_image_metrics_path = original_per_image

    payload = {"completed_epochs": int(completed_epochs), **_metric_payload(metrics)}
    csv_path = os.path.join(cfgs.path_helper["prefix"], "training_curves.csv")
    append_csv(csv_path, {"record_type": "evaluation", **payload}, [
        "record_type", "completed_epochs", "loss_total", "loss_detect", "loss_segment",
        "dice1", "dice2", "aji", "aji_p", "dq", "sq", "pq",
    ])
    write_json(
        os.path.join(cfgs.path_helper["prefix"], f"evaluation_epoch_{int(completed_epochs):02d}.json"),
        payload,
    )
    if cfgs.save_eval_checkpoints:
        _save_full_eval_checkpoint(
            os.path.join(cfgs.path_helper["ckpt_path"], f"epoch_{int(completed_epochs):02d}.pth"),
            net,
            model1,
            completed_epochs,
            val_texture_bank,
        )
    return metrics


def main():
    args = Config.fromfile("./args.py")
    cfgs = cfg.parse_args()
    apply_cli_overrides(args, cfgs)
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

    cfgs.path_helper = (
        _set_explicit_log_dir(cfgs.artifact_dir)
        if cfgs.artifact_dir
        else set_log_dir("logs", cfgs.exp_name)
    )
    logger = create_logger(cfgs.path_helper["log_path"])
    logger.info(cfgs)

    if cfgs.resimix_enabled:
        with open(cfgs.resimix_config, "r", encoding="utf-8") as handle:
            resimix_config_payload = json.load(handle)
        write_json(
            os.path.join(cfgs.path_helper["prefix"], "resimix_config.json"),
            resimix_config_payload,
        )

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
            oracle_dataset = _train_evaluation_dataset(test_dataset)
            oracle_loader = DataLoader(
                oracle_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=cfgs.num_workers,
                pin_memory=True,
            )
            print(f"[stage1-oracle] train split; n={len(oracle_dataset)} from manifest")
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
            selective_dataset = _train_evaluation_dataset(test_dataset)
            selective_loader = DataLoader(
                selective_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=cfgs.num_workers,
                pin_memory=True,
            )
            print(f"[stage2-selective] train split; n={len(selective_dataset)} from manifest")
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
            if cfgs.train_only_eval:
                eval_dataset, eval_loader = test_dataset, test_loader
            else:
                eval_dataset = _train_evaluation_dataset(test_dataset)
                eval_loader = DataLoader(
                    eval_dataset,
                    batch_size=1,
                    shuffle=False,
                    num_workers=cfgs.num_workers,
                    pin_memory=True,
                )
            eval_split = "train"
            print(f"[eval] train split; n={len(eval_dataset)} from manifest")

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
    evaluation_schedule = EvaluationSchedule.from_cli(cfgs.epochs, cfgs.evaluation_epochs)
    registered_evaluation = bool(evaluation_schedule.evaluation_epochs)
    if registered_evaluation and 0 in evaluation_schedule.evaluation_epochs:
        run_registered_evaluation(
            cfgs,
            args,
            test_loader,
            0,
            model1,
            model1_encoder,
            net,
            val_texture_bank_template,
            [],
            device,
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

    for epoch in range(settings.EPOCH):
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
        )
        if cfgs.resimix_enabled:
            record_resimix_events(cfgs, train_dataset)
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

        if registered_evaluation:
            append_csv(
                os.path.join(cfgs.path_helper["prefix"], "training_curves.csv"),
                {
                    "record_type": "train",
                    "completed_epochs": int(epoch + 1),
                    "loss_total": float(all_loss_tmp),
                    "loss_detect": float(detect_loss_tmp),
                    "loss_segment": float(segment_loss_tmp),
                },
                [
                    "record_type", "completed_epochs", "loss_total", "loss_detect", "loss_segment",
                    "dice1", "dice2", "aji", "aji_p", "dq", "sq", "pq",
                ],
            )

        scheduler.step()
        net.eval()
        completed_epochs = int(epoch + 1)
        if registered_evaluation:
            if evaluation_schedule.should_evaluate(completed_epochs):
                run_registered_evaluation(
                    cfgs,
                    args,
                    test_loader,
                    completed_epochs,
                    model1,
                    model1_encoder,
                    net,
                    val_texture_bank_template,
                    texture_memory_bank_list,
                    device,
                )
            continue

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
