"""Authorized PromptCredit v1 two-crop mechanism smoke test.

This runner is GPU-only and reads only the first fixed router-train image.  It
is intentionally not a full TNBC training loop or a general evaluation tool.
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from promptcredit.audit.data import AuditCrop, iter_selected_tnbc_crops
from promptcredit.audit.gradient import focal_dice_per_prompt
from promptcredit.audit.guardrails import sha256_file, validate_stage0_inputs
from promptcredit.audit.runner import _decode_standard, _load_models, _prepare_decoder_features
from promptcredit.method import (
    build_quality_targets,
    configure_promptcredit_v1_trainable,
    directional_credit,
    frozen_parameters_have_no_grad,
    gather_nearest_coordinates,
    legacy_nearest_indices,
    load_point_checkpoint_compat,
    module_state_sha256,
    optimizer_excludes_frozen,
    quality_focal_loss,
)
from promptcredit.metrics.utility import score_utility_summary


SEED = 3407
STEPS = 100
LR = 1e-4
WEIGHT_DECAY = 1e-4
ALPHA_MAX = 0.10
WARMUP_STEPS = 20


@dataclass
class SmokeExperiment:
    name: str
    bundle: Any
    criterion: Any
    optimizer: torch.optim.Optimizer
    trainable_manifest: dict[str, Any]
    sam2_checksum_before: str
    context_bank: list[Any]


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def _set_seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _alpha_for_step(step: int, enabled: bool) -> float:
    if not enabled:
        return 0.0
    return ALPHA_MAX * min(step, WARMUP_STEPS) / WARMUP_STEPS


def _hard_iou(prediction_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    hard = prediction_logits > 0
    truth = targets.bool()
    intersection = (hard & truth).sum(dim=(1, 2)).float()
    union = (hard | truth).sum(dim=(1, 2)).float()
    return torch.where(union > 0, intersection / union, torch.ones_like(union))


def _point_distance(points: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(points[:, 0] - centroids, dim=1)


def _set_train_modes(point_net: torch.nn.Module, sam2_net: torch.nn.Module) -> None:
    point_net.eval()
    sam2_net.eval()
    for name in ("conv", "deform_layer", "reg_head", "cls_head", "quality_head"):
        getattr(point_net, name).train()


def _decode_train(bundle: Any, image_embed: torch.Tensor, high_res_features: list[torch.Tensor], coordinates: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    labels = torch.ones(coordinates.size(0), 1, dtype=torch.int, device=bundle.device)
    sparse, dense = bundle.net.sam_prompt_encoder(
        points=(coordinates, labels), boxes=None, masks=None, batch_size=1
    )
    low_res_masks, iou_predictions, _, _ = bundle.net.sam_mask_decoder(
        image_embeddings=image_embed,
        image_pe=bundle.net.sam_prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=dense,
        multimask_output=False,
        repeat_image=False,
        cell_nums=torch.as_tensor([coordinates.size(0)], device=bundle.device),
        high_res_features=high_res_features,
    )
    logits = F.interpolate(low_res_masks, size=(256, 256), mode="bilinear", align_corners=False)[:, 0]
    return logits, torch.max(iou_predictions, dim=1).values


def _targets_for_crop(crop: AuditCrop, device: torch.device) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
    centroids = torch.as_tensor(crop.gt_centroids_xy, dtype=torch.float32, device=device)
    # pytorch_toolbelt DiceLoss internally uses view(), so preserve the exact
    # mask values while requiring a contiguous target layout at this boundary.
    instances = torch.as_tensor(crop.gt_masks, dtype=torch.float32, device=device).contiguous()
    union_mask = instances.any(dim=0, keepdim=True).float()
    targets = {
        "gt_masks": union_mask,
        "gt_nums": [len(centroids)],
        "gt_points": [centroids],
        "gt_labels": [torch.zeros(len(centroids), dtype=torch.long, device=device)],
    }
    return targets, instances, centroids


def _quality_spearman(logits: torch.Tensor, target_values: torch.Tensor, matched: torch.Tensor) -> float | None:
    if int(matched.sum()) < 2:
        return None
    scores = torch.sigmoid(logits)[matched].detach().cpu().numpy()
    target = target_values[matched].detach().cpu().numpy()
    if np.all(scores == scores[0]) or np.all(target == target[0]):
        return None
    result = float(spearmanr(scores, target).statistic)
    return result if np.isfinite(result) else None


def _grad_norm(parameters: list[torch.nn.Parameter]) -> float:
    values = [parameter.grad.detach().norm() for parameter in parameters if parameter.grad is not None]
    return float(torch.linalg.vector_norm(torch.stack(values)).cpu()) if values else 0.0


def _build_experiment(
    *, name: str, config_path: Path, sam_config: str, checkpoint: Path, device: torch.device
) -> SmokeExperiment:
    from mmengine.config import Config

    # `criterion` imports a legacy utility module whose module-level code calls
    # cfg.parse_args().  The smoke CLI has already parsed its own arguments, so
    # hide them only during this one legacy import rather than letting that
    # unrelated parser reject --data-root/--checkpoint/--out-dir.
    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0]]
        from sam2_train.modeling.criterion import build_criterion
    finally:
        sys.argv = original_argv

    _set_seed()
    bundle = _load_models(config_path, sam_config, checkpoint, device, enable_quality_head=True)
    manifest = configure_promptcredit_v1_trainable(bundle.point_net, bundle.net)
    if manifest["quality_head_parameter_count"] >= 100_000:
        raise ValueError("PromptCredit v1 quality head exceeds 0.1M parameters")
    config = Config.fromfile(str(config_path))
    criterion, _ = build_criterion(config, device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in bundle.point_net.parameters() if parameter.requires_grad],
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    if not optimizer_excludes_frozen(optimizer):
        raise RuntimeError("PromptCredit optimizer contains a frozen parameter")
    _set_train_modes(bundle.point_net, bundle.net)
    return SmokeExperiment(
        name=name,
        bundle=bundle,
        criterion=criterion,
        optimizer=optimizer,
        trainable_manifest=manifest,
        sam2_checksum_before=module_state_sha256(bundle.net),
        context_bank=[],
    )


def _forward_losses(
    experiment: SmokeExperiment,
    crop: AuditCrop,
    *,
    alpha: float,
    quality_loss_coef: float,
    retain_coordinate_grad: bool,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    image = crop.image.unsqueeze(0).to(experiment.bundle.device)
    targets, instance_masks, centroids = _targets_for_crop(crop, experiment.bundle.device)
    output, _, _, _ = experiment.bundle.point_net(image)
    selection = gather_nearest_coordinates(output["pred_coords"], targets["gt_points"])
    selected = selection.coordinates
    if retain_coordinate_grad:
        selected.retain_grad()
    credited = directional_credit(selected, alpha)
    image_embed, high_res_features = _prepare_decoder_features(experiment.bundle, image, experiment.context_bank, crop)
    decoded_logits, predicted_iou = _decode_train(experiment.bundle, image_embed, high_res_features, credited)
    loss_dict = experiment.criterion(output, targets, decoded_logits, predicted_iou, instance_masks, epoch=0)
    # Only focal+dice use directional credit.  Decoder-IoU is retained as a
    # logged value but intentionally detached from the point generator.
    loss_dict["loss_iou"] = loss_dict["loss_iou"].detach()
    hard_iou = _hard_iou(decoded_logits.detach(), instance_masks)
    quality_targets = build_quality_targets(output["pred_quality_logits"], selection.source_indices, hard_iou)
    quality_loss = quality_focal_loss(output["pred_quality_logits"], quality_targets)
    loss_dict["loss_prompt_credit_quality"] = quality_loss * quality_loss_coef
    diagnostics = {
        "selected": selected,
        "decoded_logits": decoded_logits,
        "hard_iou": hard_iou,
        "point_distance": _point_distance(selected.detach(), centroids),
        "quality_targets": quality_targets,
        "quality_spearman": _quality_spearman(
            output["pred_quality_logits"], quality_targets.values, quality_targets.matched_proposals
        ),
        "duplicate_source_events": quality_targets.duplicate_source_events,
    }
    return loss_dict, diagnostics


def _step(experiment: SmokeExperiment, crop: AuditCrop, *, step: int, alpha: float, quality_loss_coef: float) -> dict[str, Any]:
    torch.cuda.synchronize(experiment.bundle.device)
    started = time.perf_counter()
    experiment.optimizer.zero_grad(set_to_none=True)
    loss_dict, diagnostics = _forward_losses(
        experiment, crop, alpha=alpha, quality_loss_coef=quality_loss_coef, retain_coordinate_grad=True
    )
    optimization_loss = (
        loss_dict["loss_cls"]
        + loss_dict["loss_reg"]
        + loss_dict["loss_focal"]
        + loss_dict["loss_dice"]
        + loss_dict["loss_prompt_credit_quality"]
    )
    if not torch.isfinite(optimization_loss):
        raise FloatingPointError(f"{experiment.name} step {step}: non-finite loss")
    optimization_loss.backward()
    if not frozen_parameters_have_no_grad(experiment.bundle.net):
        raise RuntimeError(f"{experiment.name} step {step}: frozen SAM2 parameter received a gradient")
    trainable = [parameter for parameter in experiment.bundle.point_net.parameters() if parameter.requires_grad]
    if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all() for parameter in trainable):
        raise FloatingPointError(f"{experiment.name} step {step}: non-finite trainable gradient")
    coordinate_grad = diagnostics["selected"].grad
    coordinate_grad_norm = float(coordinate_grad.norm().detach().cpu()) if coordinate_grad is not None else 0.0
    trainable_grad_norm = _grad_norm(trainable)
    experiment.optimizer.step()
    torch.cuda.synchronize(experiment.bundle.device)
    return {
        "step": step,
        "alpha": alpha,
        "detection_classification_loss": float(loss_dict["loss_cls"].detach().cpu()),
        "coordinate_regression_loss": float(loss_dict["loss_reg"].detach().cpu()),
        "mask_focal_loss": float(loss_dict["loss_focal"].detach().cpu()),
        "mask_dice_loss": float(loss_dict["loss_dice"].detach().cpu()),
        "quality_loss": float(loss_dict["loss_prompt_credit_quality"].detach().cpu()),
        "selected_prompt_hard_iou": float(diagnostics["hard_iou"].mean().detach().cpu()),
        "point_to_centroid_distance": float(diagnostics["point_distance"].mean().detach().cpu()),
        "quality_target_spearman": diagnostics["quality_spearman"],
        "coordinate_gradient_norm": coordinate_grad_norm,
        "trainable_parameter_gradient_norm": trainable_grad_norm,
        "duplicate_source_events": int(diagnostics["duplicate_source_events"]),
        "step_seconds": time.perf_counter() - started,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(experiment.bundle.device)),
    }


@torch.no_grad()
def _evaluate(experiment: SmokeExperiment, crops: list[AuditCrop]) -> dict[str, Any]:
    context_bank: list[Any] = []
    mask_loss_values: list[float] = []
    quality_loss_values: list[float] = []
    hard_iou_values: list[float] = []
    point_distance_values: list[float] = []
    score_values: list[float] = []
    matchability_values: list[float] = []
    quality_scores: list[float] = []
    quality_targets: list[float] = []
    for crop in crops:
        image = crop.image.unsqueeze(0).to(experiment.bundle.device)
        targets, instance_masks, centroids = _targets_for_crop(crop, experiment.bundle.device)
        output, _, _, _ = experiment.bundle.point_net(image)
        selection = gather_nearest_coordinates(output["pred_coords"], targets["gt_points"])
        image_embed, high_res_features = _prepare_decoder_features(experiment.bundle, image, context_bank, crop)
        decoded_logits, _ = _decode_standard(experiment.bundle, image_embed, high_res_features, selection.coordinates)
        hard_iou = _hard_iou(decoded_logits, instance_masks)
        quality = build_quality_targets(output["pred_quality_logits"], selection.source_indices, hard_iou)
        quality_loss = quality_focal_loss(output["pred_quality_logits"], quality)
        mask_loss_values.extend(focal_dice_per_prompt(decoded_logits, instance_masks).cpu().tolist())
        quality_loss_values.append(float(quality_loss.cpu()))
        hard_iou_values.extend(hard_iou.cpu().tolist())
        point_distance_values.extend(_point_distance(selection.coordinates, centroids).cpu().tolist())
        matched_logits = output["pred_quality_logits"][quality.matched_proposals]
        quality_scores.extend(torch.sigmoid(matched_logits).cpu().tolist())
        quality_targets.extend(quality.values[quality.matched_proposals].cpu().tolist())
        score_values.extend(torch.softmax(output["pred_logits"][0], dim=-1)[selection.source_indices[0], 0].cpu().tolist())
        matchability_values.extend((hard_iou >= 0.5).float().cpu().tolist())
    utility = score_utility_summary(score_values, hard_iou_values)
    quality_spearman = None
    if len(quality_scores) > 1 and not (np.allclose(quality_scores, quality_scores[0]) or np.allclose(quality_targets, quality_targets[0])):
        quality_spearman = float(spearmanr(quality_scores, quality_targets).statistic)
    return {
        "mean_mask_iou": float(np.mean(hard_iou_values)),
        "mean_mask_loss": float(np.mean(mask_loss_values)),
        "mean_quality_loss": float(np.mean(quality_loss_values)),
        "mean_point_localization_error": float(np.mean(point_distance_values)),
        "quality_target_spearman": quality_spearman,
        "matchability_auroc": utility["auroc_iou_ge_0_5"],
        "matchability_brier": utility["brier_iou_ge_0_5"],
        "matchability_ece": utility["ece_10_equal_frequency"],
    }


def _baseline_equivalence(
    *, config_path: Path, sam_config: str, checkpoint: Path, crop: AuditCrop, device: torch.device
) -> dict[str, Any]:
    """Compare old point architecture with new disabled-credit configuration on one fixed crop."""
    _set_seed()
    legacy = _load_models(config_path, sam_config, checkpoint, device, enable_quality_head=False)
    _set_seed()
    promptcredit = _load_models(config_path, sam_config, checkpoint, device, enable_quality_head=True)
    image = crop.image.unsqueeze(0).to(device)
    targets, _, _ = _targets_for_crop(crop, device)
    with torch.no_grad():
        legacy_output, _, _, _ = legacy.point_net(image)
        promptcredit_output, _, _, _ = promptcredit.point_net(image)
        legacy_indices = legacy_nearest_indices(legacy_output["pred_coords"], targets["gt_points"])[0].to(device)
        new_selection = gather_nearest_coordinates(promptcredit_output["pred_coords"], targets["gt_points"])
        legacy_coords = legacy_output["pred_coords"][0].index_select(0, legacy_indices).unsqueeze(1)
        legacy_embed, legacy_high = _prepare_decoder_features(legacy, image, [], crop)
        new_embed, new_high = _prepare_decoder_features(promptcredit, image, [], crop)
        legacy_logits, legacy_iou = _decode_standard(legacy, legacy_embed, legacy_high, legacy_coords)
        new_logits, new_iou = _decode_standard(promptcredit, new_embed, new_high, new_selection.coordinates)
        legacy_scores = torch.softmax(legacy_output["pred_logits"], dim=-1)[..., 0]
        new_scores = torch.softmax(promptcredit_output["pred_logits"], dim=-1)[..., 0]
    errors = {
        "pred_coords_max_abs_error": float((legacy_output["pred_coords"] - promptcredit_output["pred_coords"]).abs().max().cpu()),
        "pred_logits_max_abs_error": float((legacy_output["pred_logits"] - promptcredit_output["pred_logits"]).abs().max().cpu()),
        "selected_prompts_max_abs_error": float((legacy_coords - new_selection.coordinates).abs().max().cpu()),
        "decoded_mask_logits_max_abs_error": float((legacy_logits - new_logits).abs().max().cpu()),
        "predicted_iou_max_abs_error": float((legacy_iou - new_iou).abs().max().cpu()),
        "point_scores_max_abs_error": float((legacy_scores - new_scores).abs().max().cpu()),
        "nearest_indices_equal": bool(torch.equal(legacy_indices, new_selection.source_indices[0])),
        "old_checkpoint_compatibility": promptcredit.point_checkpoint_compatibility,
    }
    errors["passed"] = bool(errors["nearest_indices_equal"] and all(value == 0.0 for key, value in errors.items() if key.endswith("max_abs_error")))
    del legacy, promptcredit
    torch.cuda.empty_cache()
    return errors


def _select_two_crops(data_root: Path, first_image_id: str) -> list[AuditCrop]:
    crops = list(iter_selected_tnbc_crops(data_root, [first_image_id]))
    if len(crops) < 2:
        raise RuntimeError("First fixed image has fewer than two nucleus-containing training crops")
    return crops[:2]


def run_smoke(
    *,
    data_root: Path,
    split_manifest_path: Path,
    selection_path: Path,
    checkpoint: Path,
    config_path: Path,
    sam_config: str,
    out_dir: Path,
    device_name: str,
) -> dict[str, Any]:
    if device_name != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("PromptCredit smoke is GPU-only; run the documented command on AutoDL 4090")
    split = _read_json(split_manifest_path)
    selection = _read_json(selection_path)
    image_ids = validate_stage0_inputs(data_root=data_root, split_manifest=split, selection=selection, checkpoint=checkpoint)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing smoke artifacts: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda")
    # The list and crop/GT checksums are saved before model construction or loss/utility observation.
    crops = _select_two_crops(data_root, image_ids[0])
    crop_manifest = {
        "image_id": image_ids[0],
        "selection_rule": "first two nucleus-containing crops in frozen unclockwise traversal",
        "crops": [
            {"crop_id": crop.crop_id, "crop_box_xyxy": list(crop.crop_box_xyxy), "image_crop_sha256": crop.image_crop_sha256, "gt_crop_sha256": crop.gt_crop_sha256}
            for crop in crops
        ],
    }
    _write_json(out_dir / "smoke_crop_selection.json", crop_manifest)
    baseline = _baseline_equivalence(config_path=config_path, sam_config=sam_config, checkpoint=checkpoint, crop=crops[0], device=device)
    if not baseline["passed"]:
        raise RuntimeError(f"Baseline-equivalence failed: {baseline}")
    torch.cuda.reset_peak_memory_stats(device)
    control = _build_experiment(name="control", config_path=config_path, sam_config=sam_config, checkpoint=checkpoint, device=device)
    promptcredit = _build_experiment(name="promptcredit", config_path=config_path, sam_config=sam_config, checkpoint=checkpoint, device=device)
    if control.trainable_manifest != promptcredit.trainable_manifest:
        raise RuntimeError("Control and PromptCredit have different trainable parameter manifests")
    control_start = _evaluate(control, crops)
    promptcredit_start = _evaluate(promptcredit, crops)
    control_rows = [{"step": 0, **control_start, "alpha": 0.0, "step_seconds": 0.0, "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device))}]
    promptcredit_rows = [{"step": 0, **promptcredit_start, "alpha": 0.0, "step_seconds": 0.0, "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device))}]
    for step in range(1, STEPS + 1):
        crop = crops[(step - 1) % len(crops)]
        control_record = _step(control, crop, step=step, alpha=0.0, quality_loss_coef=0.0)
        promptcredit_record = _step(
            promptcredit, crop, step=step, alpha=_alpha_for_step(step, enabled=True), quality_loss_coef=1.0
        )
        if step % 10 == 0:
            control_rows.append(control_record)
            promptcredit_rows.append(promptcredit_record)
    control_end = _evaluate(control, crops)
    promptcredit_end = _evaluate(promptcredit, crops)
    control_rows.append({"step": STEPS, **control_end, "alpha": 0.0, "summary": True, "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device))})
    promptcredit_rows.append({"step": STEPS, **promptcredit_end, "alpha": ALPHA_MAX, "summary": True, "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device))})
    sam2_checks = {
        "control_before": control.sam2_checksum_before,
        "control_after": module_state_sha256(control.bundle.net),
        "promptcredit_before": promptcredit.sam2_checksum_before,
        "promptcredit_after": module_state_sha256(promptcredit.bundle.net),
    }
    frozen_ok = sam2_checks["control_before"] == sam2_checks["control_after"] and sam2_checks["promptcredit_before"] == sam2_checks["promptcredit_after"]
    runtime_ratio = float(np.mean([row["step_seconds"] for row in promptcredit_rows if "step_seconds" in row and row["step_seconds"] > 0]) / max(np.mean([row["step_seconds"] for row in control_rows if "step_seconds" in row and row["step_seconds"] > 0]), np.finfo(float).eps))
    promptcredit_step_rows = [row for row in promptcredit_rows if "coordinate_gradient_norm" in row]
    coordinate_gradients_stable = bool(
        promptcredit_step_rows
        and all(np.isfinite(row["coordinate_gradient_norm"]) and row["coordinate_gradient_norm"] > 0 for row in promptcredit_step_rows)
    )
    quality_loss_decreased = bool(promptcredit_end["mean_quality_loss"] < promptcredit_start["mean_quality_loss"])
    quality_spearman_improved = bool(
        promptcredit_start["quality_target_spearman"] is not None
        and promptcredit_end["quality_target_spearman"] is not None
        and promptcredit_end["quality_target_spearman"] > promptcredit_start["quality_target_spearman"]
    )
    localization_preserved = bool(
        promptcredit_end["mean_point_localization_error"] <= 1.10 * promptcredit_start["mean_point_localization_error"]
    )
    smoke_checks = {
        "baseline_equivalence": baseline["passed"],
        "frozen_sam2_unchanged": frozen_ok,
        "finite_losses_and_gradients": True,
        "coordinate_gradients_stable": coordinate_gradients_stable,
        "quality_loss_decreased": quality_loss_decreased,
        "quality_spearman_improved": quality_spearman_improved,
        "promptcredit_mask_loss_below_step_0": promptcredit_end["mean_mask_loss"] < promptcredit_start["mean_mask_loss"],
        "point_localization_preserved_within_10_percent": localization_preserved,
        "step_time_within_1_30x_control": runtime_ratio <= 1.30,
        "peak_memory_accepted_no_oom": True,
    }
    passed = bool(
        baseline["passed"]
        and all(value for key, value in smoke_checks.items() if key != "baseline_equivalence")
    )
    report = {
        "title": "REPORT FOR PROJECT LEAD — PROMPTCREDIT STAGE 1 SMOKE",
        "recommendation": "PASS" if passed else "FAIL",
        "git_sha": _git_sha(),
        "environment": {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__, "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(device)},
        "corrected_verdict_commit": "4fe29104878248e3af0263b5a120bb4ddeee3283",
        "trainable_frozen_parameter_manifest": control.trainable_manifest,
        "baseline_equivalence": baseline,
        "control_start": control_start,
        "control_step_100": control_end,
        "promptcredit_start": promptcredit_start,
        "promptcredit_step_100": promptcredit_end,
        "sam2_parameter_checksums": sam2_checks,
        "frozen_parameters_unchanged": frozen_ok,
        "smoke_checks": smoke_checks,
        "runtime_memory": {"mean_step_time_ratio_promptcredit_over_control": runtime_ratio, "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device))},
        "smoke_crop_selection": crop_manifest,
        "artifact_paths": {"control_curve": "control_metrics.csv", "promptcredit_curve": "promptcredit_metrics.csv", "baseline_equivalence": "baseline_equivalence.json", "crop_selection": "smoke_crop_selection.json"},
        "scope": "Two fixed router-train crops, 100 steps each; not a generalization or final-method result.",
    }
    _write_csv(out_dir / "control_metrics.csv", control_rows)
    _write_csv(out_dir / "promptcredit_metrics.csv", promptcredit_rows)
    _write_json(out_dir / "baseline_equivalence.json", baseline)
    _write_json(out_dir / "report.json", report)
    _write_json(out_dir / "run_manifest.json", {
        "git_sha": _git_sha(), "command": sys.argv, "seed": SEED, "steps": STEPS, "optimizer": "AdamW", "lr": LR,
        "weight_decay": WEIGHT_DECAY, "alpha_max": ALPHA_MAX, "alpha_warmup_steps": WARMUP_STEPS,
        "quality_loss_coef": 1.0, "score_mode": "objectness_x_quality",
        "quality_head_initialization": "MLP default initialization under torch.manual_seed(3407), with the ambient CPU RNG state restored afterward",
        "checkpoint_sha256": sha256_file(checkpoint),
        "selection_manifest_sha256": sha256_file(selection_path), "split_manifest_sha256": sha256_file(split_manifest_path),
    })
    return report
