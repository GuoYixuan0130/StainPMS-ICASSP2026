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
    quality_focal_loss_with_audit,
    prompt_ranking_scores,
)
from promptcredit.metrics.utility import score_utility_summary
from promptcredit.smoke.evaluation import (
    capture_rng_snapshot,
    evaluation_snapshot,
    model_state_snapshot,
    restore_rng_snapshot,
)


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
    nms_radius: float
    semantic_filtering: bool


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


def _state_dict_sha256(module: torch.nn.Module, *, excluded_prefixes: tuple[str, ...] = ()) -> str:
    """Hash an exact parameter/buffer subset without changing it."""
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        if name.startswith(excluded_prefixes):
            continue
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        # ``num_batches_tracked`` is a zero-dimensional Long buffer.  NumPy
        # exports its contiguous scalar bytes directly, unlike Tensor.view()
        # which rejects a 0-D dtype reinterpretation.
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _loss_gradient_norm(loss: torch.Tensor, parameters: list[torch.nn.Parameter]) -> float:
    gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    values = [gradient.detach().norm() for gradient in gradients if gradient is not None]
    return float(torch.linalg.vector_norm(torch.stack(values)).cpu()) if values else 0.0


def _ranking_order(output: dict[str, torch.Tensor], mode: str) -> torch.Tensor:
    foreground_probability = torch.softmax(output["pred_logits"][0], dim=-1)[:, 0]
    quality_logits = output.get("pred_quality_logits")
    quality_for_crop = quality_logits[0] if quality_logits is not None else None
    scores = prompt_ranking_scores(foreground_probability, quality_for_crop, mode)
    return torch.argsort(scores, descending=True, stable=True)


def _prompt_action_source_ids(
    output: dict[str, torch.Tensor], *, mode: str, nms_radius: float, semantic_filtering: bool
) -> torch.Tensor:
    """Replicate crop-local current point-NMS actions while retaining source IDs."""
    points = output["pred_coords"][0].detach().cpu().numpy().copy()
    logits = output["pred_logits"][0].detach()
    probabilities = torch.softmax(logits, dim=-1)
    classes = torch.argmax(probabilities, dim=-1).cpu().numpy()
    foreground_probability = probabilities[:, 0]
    quality_logits = output.get("pred_quality_logits")
    quality_for_crop = quality_logits[0] if quality_logits is not None else None
    ranking = prompt_ranking_scores(foreground_probability, quality_for_crop, mode).detach().cpu().numpy()
    height, width = output["pred_masks"].shape[-2:]
    np.clip(points[:, 0], a_min=0, a_max=width - 1, out=points[:, 0])
    np.clip(points[:, 1], a_min=0, a_max=height - 1, out=points[:, 1])
    valid = classes < (probabilities.shape[-1] - 1)
    source_ids = np.flatnonzero(valid)
    points = points[valid]
    ranking = ranking[valid]
    if semantic_filtering and len(points):
        semantic_mask = output["pred_masks"][0, 0].detach().cpu().numpy() > 0
        keep_semantic = semantic_mask[points.astype(np.int64)[:, 1], points.astype(np.int64)[:, 0]]
        source_ids = source_ids[keep_semantic]
        points = points[keep_semantic]
        ranking = ranking[keep_semantic]
    if len(points) == 0:
        return torch.empty(0, dtype=torch.long, device=output["pred_coords"].device)
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    np.fill_diagonal(distances, np.inf)
    reserved = np.ones(len(points), dtype=bool)
    for local_index in np.argsort(-ranking):
        if reserved[local_index]:
            reserved[distances[local_index] <= nms_radius] = False
    return torch.as_tensor(source_ids[reserved], dtype=torch.long, device=output["pred_coords"].device)


def _selected_mask(source_indices: torch.Tensor, action_source_ids: torch.Tensor) -> torch.Tensor:
    if action_source_ids.numel() == 0:
        return torch.zeros_like(source_indices, dtype=torch.bool)
    return (source_indices[:, None] == action_source_ids[None, :]).any(dim=1)


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
        nms_radius=float(config.test.nms_thr),
        semantic_filtering=bool(config.test.filtering),
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
    quality_loss, quality_loss_audit = quality_focal_loss_with_audit(
        output["pred_quality_logits"], quality_targets
    )
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
        "quality_loss_audit": quality_loss_audit,
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
    shared_parameters = list(experiment.bundle.point_net.conv.parameters())
    shared_gradient_norms = {
        "detection_classification_to_shared_fusion_conv": _loss_gradient_norm(loss_dict["loss_cls"], shared_parameters),
        "coordinate_regression_to_shared_fusion_conv": _loss_gradient_norm(loss_dict["loss_reg"], shared_parameters),
        "quality_to_shared_fusion_conv": _loss_gradient_norm(
            loss_dict["loss_prompt_credit_quality"], shared_parameters
        ),
    }
    if not all(np.isfinite(value) for value in shared_gradient_norms.values()):
        raise FloatingPointError(f"{experiment.name} step {step}: non-finite shared-fusion gradient")
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
        "shared_fusion_conv_gradient_norms": shared_gradient_norms,
        "duplicate_source_events": int(diagnostics["duplicate_source_events"]),
        "step_seconds": time.perf_counter() - started,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(experiment.bundle.device)),
    }


def _evaluate(experiment: SmokeExperiment, crops: list[AuditCrop], *, score_mode: str) -> dict[str, Any]:
    """One deterministic eval snapshot, including crop-local point-NMS ranking."""
    context_bank: list[Any] = []
    mask_loss_values: list[float] = []
    quality_loss_values: list[float] = []
    hard_iou_values: list[float] = []
    point_distance_values: list[float] = []
    score_values: list[float] = []
    matchability_values: list[float] = []
    quality_scores: list[float] = []
    quality_targets: list[float] = []
    selected_source_ids_by_crop: list[dict[str, Any]] = []
    decoded_mask_calls = 0
    image_encoder_calls = 0
    with evaluation_snapshot(experiment.bundle.point_net, experiment.bundle.net):
        for crop in crops:
            image = crop.image.unsqueeze(0).to(experiment.bundle.device)
            targets, instance_masks, centroids = _targets_for_crop(crop, experiment.bundle.device)
            output, _, _, _ = experiment.bundle.point_net(image)
            selection = gather_nearest_coordinates(output["pred_coords"], targets["gt_points"])
            action_source_ids = _prompt_action_source_ids(
                output,
                mode=score_mode,
                nms_radius=experiment.nms_radius,
                semantic_filtering=experiment.semantic_filtering,
            )
            image_embed, high_res_features = _prepare_decoder_features(experiment.bundle, image, context_bank, crop)
            image_encoder_calls += 1
            decoded_logits, _ = _decode_standard(experiment.bundle, image_embed, high_res_features, selection.coordinates)
            decoded_mask_calls += 1
            hard_iou = _hard_iou(decoded_logits, instance_masks)
            quality = build_quality_targets(output["pred_quality_logits"], selection.source_indices, hard_iou)
            quality_loss = quality_focal_loss(output["pred_quality_logits"], quality)
            matched_logits = output["pred_quality_logits"][quality.matched_proposals]
            quality_scores.extend(torch.sigmoid(matched_logits).cpu().tolist())
            quality_targets.extend(quality.values[quality.matched_proposals].cpu().tolist())
            source_indices = selection.source_indices[0]
            selected = _selected_mask(source_indices, action_source_ids)
            selected_source_ids_by_crop.append(
                {
                    "crop_id": int(crop.crop_id),
                    "matched_source_ids": [int(value) for value in source_indices.detach().cpu().tolist()],
                    "point_nms_kept_source_ids": [int(value) for value in action_source_ids.detach().cpu().tolist()],
                    "matched_prompt_kept": [bool(value) for value in selected.detach().cpu().tolist()],
                }
            )
            if not bool(selected.any()):
                continue
            matched_ranking = prompt_ranking_scores(
                torch.softmax(output["pred_logits"][0], dim=-1)[source_indices, 0],
                output["pred_quality_logits"][0, source_indices],
                score_mode,
            )
            selected_indices = torch.nonzero(selected, as_tuple=False).squeeze(1)
            mask_loss_values.extend(
                focal_dice_per_prompt(decoded_logits[selected_indices], instance_masks[selected_indices]).cpu().tolist()
            )
            quality_loss_values.append(float(quality_loss.cpu()))
            hard_iou_values.extend(hard_iou[selected_indices].cpu().tolist())
            point_distance_values.extend(
                _point_distance(selection.coordinates[selected_indices], centroids[selected_indices]).cpu().tolist()
            )
            score_values.extend(matched_ranking[selected_indices].cpu().tolist())
            matchability_values.extend((hard_iou[selected_indices] >= 0.5).float().cpu().tolist())
    if not hard_iou_values:
        raise RuntimeError(f"{experiment.name} evaluation retained no matched prompts after current point NMS")
    utility = score_utility_summary(score_values, hard_iou_values)
    quality_spearman = None
    if len(quality_scores) > 1 and not (np.allclose(quality_scores, quality_scores[0]) or np.allclose(quality_targets, quality_targets[0])):
        quality_spearman = float(spearmanr(quality_scores, quality_targets).statistic)
    return {
        "score_mode": score_mode,
        "n_matched_prompts_after_point_nms": int(len(hard_iou_values)),
        "selected_point_source_ids_by_crop": selected_source_ids_by_crop,
        "decoded_mask_calls": decoded_mask_calls,
        "image_encoder_calls": image_encoder_calls,
        "mean_mask_iou": float(np.mean(hard_iou_values)),
        "mean_mask_loss": float(np.mean(mask_loss_values)),
        "mean_quality_loss": float(np.mean(quality_loss_values)),
        "mean_point_localization_error": float(np.mean(point_distance_values)),
        "quality_target_spearman": quality_spearman,
        "matchability_auroc": utility["auroc_iou_ge_0_5"],
        "matchability_auprc": utility["auprc_iou_ge_0_5"],
        "matchability_brier": utility["brier_iou_ge_0_5"],
        "matchability_ece": utility["ece_10_equal_frequency"],
        "score_hard_iou_spearman": utility["spearman_point_score_vs_hard_iou"],
        "quality_prediction_std": float(np.std(quality_scores)),
    }


def _quality_loss_scale_audit(experiment: SmokeExperiment, crop: AuditCrop) -> dict[str, Any]:
    """Audit the frozen, pre-registered Quality Focal Loss before any update."""
    shared_parameters = list(experiment.bundle.point_net.conv.parameters())
    original_context_bank = experiment.context_bank
    experiment.context_bank = []
    try:
        with model_state_snapshot(experiment.bundle.point_net, experiment.bundle.net):
            experiment.optimizer.zero_grad(set_to_none=True)
            loss_dict, diagnostics = _forward_losses(
                experiment, crop, alpha=0.0, quality_loss_coef=1.0, retain_coordinate_grad=False
            )
            accounting = diagnostics["quality_loss_audit"]
            if accounting.proposal_total != accounting.matched_positive_count + accounting.unmatched_negative_count:
                raise RuntimeError("Quality-loss accounting does not partition proposals")
            if accounting.normalization_denominator != max(accounting.matched_positive_count, 1):
                raise RuntimeError("Quality-loss denominator is not matched-positive normalized")
            gradient_norms = {
                "detection_classification_to_shared_fusion_conv": _loss_gradient_norm(loss_dict["loss_cls"], shared_parameters),
                "coordinate_regression_to_shared_fusion_conv": _loss_gradient_norm(loss_dict["loss_reg"], shared_parameters),
                "quality_to_shared_fusion_conv": _loss_gradient_norm(loss_dict["loss_prompt_credit_quality"], shared_parameters),
            }
            result = {
                "proposal_total": accounting.proposal_total,
                "matched_positive_count": accounting.matched_positive_count,
                "unmatched_negative_count": accounting.unmatched_negative_count,
                "positive_to_negative_ratio": (
                    float(accounting.matched_positive_count / accounting.unmatched_negative_count)
                    if accounting.unmatched_negative_count
                    else None
                ),
                "positive_loss_sum": accounting.positive_loss_sum,
                "negative_loss_sum": accounting.negative_loss_sum,
                "negative_loss_scaled_sum": accounting.negative_loss_scaled_sum,
                "normalization_denominator": accounting.normalization_denominator,
                "quality_focal_loss": float(loss_dict["loss_prompt_credit_quality"].detach().cpu()),
                "shared_fusion_conv_gradient_norms": gradient_norms,
                "gamma": 2.0,
                "quality_loss_coefficient": 1.0,
                "utility_target": "hard_iou * sigmoid((hard_iou - 0.5) / 0.1)",
                "negative_balancing": "negative_sum * (matched_positive_count / max(unmatched_negative_count, 1))",
            }
    finally:
        experiment.context_bank = original_context_bank
    experiment.optimizer.zero_grad(set_to_none=True)
    if not all(np.isfinite(value) for value in result["shared_fusion_conv_gradient_norms"].values()):
        raise FloatingPointError("Quality-loss scale audit produced a non-finite gradient norm")
    return result


def _strict_step0_equivalence(
    control: SmokeExperiment, promptcredit: SmokeExperiment, crops: list[AuditCrop]
) -> dict[str, Any]:
    """Fail closed unless both independent initializations are exactly paired."""
    common_control = _state_dict_sha256(control.bundle.point_net, excluded_prefixes=("quality_head.",))
    common_promptcredit = _state_dict_sha256(promptcredit.bundle.point_net, excluded_prefixes=("quality_head.",))
    quality_control = module_state_sha256(control.bundle.point_net.quality_head)
    quality_promptcredit = module_state_sha256(promptcredit.bundle.point_net.quality_head)
    errors = {
        "pred_coords_max_abs_error": 0.0,
        "pred_logits_max_abs_error": 0.0,
        "selected_prompts_max_abs_error": 0.0,
        "decoded_mask_logits_max_abs_error": 0.0,
        "common_score_mean_iou_abs_error": 0.0,
        "common_score_mask_loss_abs_error": 0.0,
    }
    rankings_equal = True
    combined_rankings_equal = True
    point_nms_actions_equal = True
    common_score_metrics_equal = True
    quality_prediction_constant = True
    quality_prediction_is_low_prior = True
    per_crop: list[dict[str, Any]] = []
    control_context: list[Any] = []
    promptcredit_context: list[Any] = []
    with evaluation_snapshot(
        control.bundle.point_net,
        control.bundle.net,
        promptcredit.bundle.point_net,
        promptcredit.bundle.net,
    ):
        for crop in crops:
            image = crop.image.unsqueeze(0).to(control.bundle.device)
            targets, instance_masks, _ = _targets_for_crop(crop, control.bundle.device)
            control_output, _, _, _ = control.bundle.point_net(image)
            promptcredit_output, _, _, _ = promptcredit.bundle.point_net(image)
            control_selection = gather_nearest_coordinates(control_output["pred_coords"], targets["gt_points"])
            promptcredit_selection = gather_nearest_coordinates(promptcredit_output["pred_coords"], targets["gt_points"])
            control_embed, control_high = _prepare_decoder_features(control.bundle, image, control_context, crop)
            promptcredit_embed, promptcredit_high = _prepare_decoder_features(
                promptcredit.bundle, image, promptcredit_context, crop
            )
            control_logits, _ = _decode_standard(
                control.bundle, control_embed, control_high, control_selection.coordinates
            )
            promptcredit_logits, _ = _decode_standard(
                promptcredit.bundle, promptcredit_embed, promptcredit_high, promptcredit_selection.coordinates
            )
            errors["pred_coords_max_abs_error"] = max(
                errors["pred_coords_max_abs_error"],
                float((control_output["pred_coords"] - promptcredit_output["pred_coords"]).abs().max().cpu()),
            )
            errors["pred_logits_max_abs_error"] = max(
                errors["pred_logits_max_abs_error"],
                float((control_output["pred_logits"] - promptcredit_output["pred_logits"]).abs().max().cpu()),
            )
            errors["selected_prompts_max_abs_error"] = max(
                errors["selected_prompts_max_abs_error"],
                float((control_selection.coordinates - promptcredit_selection.coordinates).abs().max().cpu()),
            )
            errors["decoded_mask_logits_max_abs_error"] = max(
                errors["decoded_mask_logits_max_abs_error"],
                float((control_logits - promptcredit_logits).abs().max().cpu()),
            )
            control_order = _ranking_order(control_output, "objectness")
            promptcredit_order = _ranking_order(promptcredit_output, "objectness")
            control_combined_order = _ranking_order(control_output, "objectness_x_quality")
            promptcredit_combined_order = _ranking_order(promptcredit_output, "objectness_x_quality")
            rankings_equal &= bool(torch.equal(control_order, promptcredit_order))
            combined_rankings_equal &= bool(
                torch.equal(control_combined_order, promptcredit_combined_order)
                and torch.equal(control_order, control_combined_order)
                and torch.equal(promptcredit_order, promptcredit_combined_order)
            )
            control_actions = _prompt_action_source_ids(
                control_output,
                mode="objectness",
                nms_radius=control.nms_radius,
                semantic_filtering=control.semantic_filtering,
            )
            control_combined_actions = _prompt_action_source_ids(
                control_output,
                mode="objectness_x_quality",
                nms_radius=control.nms_radius,
                semantic_filtering=control.semantic_filtering,
            )
            promptcredit_actions = _prompt_action_source_ids(
                promptcredit_output,
                mode="objectness",
                nms_radius=promptcredit.nms_radius,
                semantic_filtering=promptcredit.semantic_filtering,
            )
            promptcredit_combined_actions = _prompt_action_source_ids(
                promptcredit_output,
                mode="objectness_x_quality",
                nms_radius=promptcredit.nms_radius,
                semantic_filtering=promptcredit.semantic_filtering,
            )
            point_nms_actions_equal &= bool(
                torch.equal(control_actions, promptcredit_actions)
                and torch.equal(control_actions, control_combined_actions)
                and torch.equal(promptcredit_actions, promptcredit_combined_actions)
            )
            control_keep = _selected_mask(control_selection.source_indices[0], control_actions)
            promptcredit_keep = _selected_mask(promptcredit_selection.source_indices[0], promptcredit_actions)
            if not bool(control_keep.any()) or not bool(promptcredit_keep.any()):
                common_score_metrics_equal = False
                control_mean_iou = None
                promptcredit_mean_iou = None
                control_mean_loss = None
                promptcredit_mean_loss = None
            else:
                control_hard_iou = _hard_iou(control_logits, instance_masks)
                promptcredit_hard_iou = _hard_iou(promptcredit_logits, instance_masks)
                control_mean_iou = float(control_hard_iou[control_keep].mean().cpu())
                promptcredit_mean_iou = float(promptcredit_hard_iou[promptcredit_keep].mean().cpu())
                control_mean_loss = float(
                    focal_dice_per_prompt(control_logits[control_keep], instance_masks[control_keep]).mean().cpu()
                )
                promptcredit_mean_loss = float(
                    focal_dice_per_prompt(
                        promptcredit_logits[promptcredit_keep], instance_masks[promptcredit_keep]
                    ).mean().cpu()
                )
                errors["common_score_mean_iou_abs_error"] = max(
                    errors["common_score_mean_iou_abs_error"], abs(control_mean_iou - promptcredit_mean_iou)
                )
                errors["common_score_mask_loss_abs_error"] = max(
                    errors["common_score_mask_loss_abs_error"], abs(control_mean_loss - promptcredit_mean_loss)
                )
            quality_probability = torch.sigmoid(control_output["pred_quality_logits"])
            quality_prediction_constant &= bool(torch.all(quality_probability == quality_probability.flatten()[0]))
            quality_prediction_is_low_prior &= bool(
                torch.allclose(quality_probability, torch.full_like(quality_probability, 0.01), atol=1e-7, rtol=0.0)
            )
            per_crop.append(
                {
                    "crop_id": int(crop.crop_id),
                    "control_point_nms_kept_source_ids": [int(value) for value in control_actions.cpu().tolist()],
                    "promptcredit_point_nms_kept_source_ids": [int(value) for value in promptcredit_actions.cpu().tolist()],
                    "common_score_control_mean_iou": control_mean_iou,
                    "common_score_promptcredit_mean_iou": promptcredit_mean_iou,
                    "common_score_control_mean_mask_loss": control_mean_loss,
                    "common_score_promptcredit_mean_mask_loss": promptcredit_mean_loss,
                }
            )
    passed = bool(
        common_control == common_promptcredit
        and quality_control == quality_promptcredit
        and rankings_equal
        and combined_rankings_equal
        and point_nms_actions_equal
        and common_score_metrics_equal
        and quality_prediction_constant
        and quality_prediction_is_low_prior
        and all(value == 0.0 for value in errors.values())
    )
    return {
        "old_checkpoint_common_parameter_checksum": {
            "control": common_control,
            "promptcredit": common_promptcredit,
            "equal": common_control == common_promptcredit,
        },
        "quality_head_parameter_checksum": {
            "control": quality_control,
            "promptcredit": quality_promptcredit,
            "equal": quality_control == quality_promptcredit,
        },
        "max_abs_errors": errors,
        "objectness_ranking_equal": rankings_equal,
        "objectness_x_quality_ranking_equal": combined_rankings_equal,
        "point_nms_action_source_ids_equal": point_nms_actions_equal,
        "quality_prediction_constant_at_step_0": quality_prediction_constant,
        "quality_prediction_is_fixed_0_01_prior": quality_prediction_is_low_prior,
        "per_crop": per_crop,
        "passed": passed,
    }


def _evaluation_scalars(summary: dict[str, Any], *, prefix: str) -> dict[str, Any]:
    return {
        f"{prefix}{key}": value
        for key, value in summary.items()
        if not isinstance(value, (dict, list)) and key != "score_mode"
    }


def _curve_row(
    *, step: int, alpha: float, train_record: dict[str, Any] | None, common: dict[str, Any], deployment: dict[str, Any] | None = None
) -> dict[str, Any]:
    row: dict[str, Any] = {"step": step, "alpha": alpha}
    if train_record is not None:
        row.update(
            {
                key: value
                for key, value in train_record.items()
                if key not in {"step", "alpha", "selected_prompt_hard_iou", "point_to_centroid_distance", "quality_target_spearman"}
            }
        )
    row.update(_evaluation_scalars(common, prefix="common_score_"))
    if deployment is not None:
        row.update(_evaluation_scalars(deployment, prefix="deployment_score_"))
    return row


def _mask_metric_relation(candidate: dict[str, Any], reference: dict[str, Any]) -> dict[str, bool]:
    loss_not_higher = bool(candidate["mean_mask_loss"] <= reference["mean_mask_loss"])
    iou_not_lower = bool(candidate["mean_mask_iou"] >= reference["mean_mask_iou"])
    return {
        "mask_loss_not_higher": loss_not_higher,
        "mean_iou_not_lower": iou_not_lower,
        "metrics_conflict": loss_not_higher != iou_not_lower,
        "strict_relative_gain": bool(
            candidate["mean_mask_loss"] < reference["mean_mask_loss"]
            or candidate["mean_mask_iou"] > reference["mean_mask_iou"]
        ),
        "clearly_harmful": bool(
            candidate["mean_mask_loss"] > reference["mean_mask_loss"]
            and candidate["mean_mask_iou"] < reference["mean_mask_iou"]
        ),
    }


def _quality_score_relation_better(deployment: dict[str, Any], common: dict[str, Any]) -> bool:
    """Fixed pre-run criterion: deployment score must strictly improve IoU rank correlation."""
    deployment_value = deployment["score_hard_iou_spearman"]
    common_value = common["score_hard_iou_spearman"]
    return bool(
        deployment_value is not None
        and common_value is not None
        and deployment_value > common_value
    )


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
    with evaluation_snapshot(legacy.point_net, legacy.net, promptcredit.point_net, promptcredit.net):
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
    quality_loss_scale_audit = _quality_loss_scale_audit(promptcredit, crops[0])
    if control.context_bank or promptcredit.context_bank:
        raise RuntimeError("Pre-run audit must not seed either paired experiment's context bank")
    step0_equivalence = _strict_step0_equivalence(control, promptcredit, crops)
    if not step0_equivalence["passed"]:
        _write_json(out_dir / "baseline_equivalence.json", baseline)
        _write_json(out_dir / "step0_strict_equivalence.json", step0_equivalence)
        _write_json(out_dir / "quality_loss_scale_audit.json", quality_loss_scale_audit)
        raise RuntimeError(f"Strict paired step-0 equivalence failed: {step0_equivalence}")
    control_start_common = _evaluate(control, crops, score_mode="objectness")
    promptcredit_start_common = _evaluate(promptcredit, crops, score_mode="objectness")
    promptcredit_start_deployment = _evaluate(promptcredit, crops, score_mode="objectness_x_quality")
    control_rows = [_curve_row(step=0, alpha=0.0, train_record=None, common=control_start_common)]
    promptcredit_rows = [
        _curve_row(
            step=0,
            alpha=0.0,
            train_record=None,
            common=promptcredit_start_common,
            deployment=promptcredit_start_deployment,
        )
    ]
    # Keep independently built runs on identical stochastic sequences.  Eval
    # snapshots restore their state and therefore cannot perturb either stream.
    control_rng = capture_rng_snapshot()
    promptcredit_rng = control_rng
    control_step_records: list[dict[str, Any]] = []
    promptcredit_step_records: list[dict[str, Any]] = []
    control_end_common = control_start_common
    promptcredit_end_common = promptcredit_start_common
    promptcredit_end_deployment = promptcredit_start_deployment
    for step in range(1, STEPS + 1):
        crop = crops[(step - 1) % len(crops)]
        restore_rng_snapshot(control_rng)
        control_record = _step(control, crop, step=step, alpha=0.0, quality_loss_coef=0.0)
        control_rng = capture_rng_snapshot()
        restore_rng_snapshot(promptcredit_rng)
        promptcredit_record = _step(
            promptcredit, crop, step=step, alpha=_alpha_for_step(step, enabled=True), quality_loss_coef=1.0
        )
        promptcredit_rng = capture_rng_snapshot()
        control_step_records.append(control_record)
        promptcredit_step_records.append(promptcredit_record)
        if step % 10 == 0:
            control_end_common = _evaluate(control, crops, score_mode="objectness")
            promptcredit_end_common = _evaluate(promptcredit, crops, score_mode="objectness")
            promptcredit_end_deployment = _evaluate(promptcredit, crops, score_mode="objectness_x_quality")
            control_rows.append(
                _curve_row(step=step, alpha=0.0, train_record=control_record, common=control_end_common)
            )
            promptcredit_rows.append(
                _curve_row(
                    step=step,
                    alpha=_alpha_for_step(step, enabled=True),
                    train_record=promptcredit_record,
                    common=promptcredit_end_common,
                    deployment=promptcredit_end_deployment,
                )
            )
    restore_rng_snapshot(promptcredit_rng)
    sam2_checks = {
        "control_before": control.sam2_checksum_before,
        "control_after": module_state_sha256(control.bundle.net),
        "promptcredit_before": promptcredit.sam2_checksum_before,
        "promptcredit_after": module_state_sha256(promptcredit.bundle.net),
    }
    frozen_ok = sam2_checks["control_before"] == sam2_checks["control_after"] and sam2_checks["promptcredit_before"] == sam2_checks["promptcredit_after"]
    runtime_ratio = float(
        np.mean([row["step_seconds"] for row in promptcredit_step_records])
        / max(np.mean([row["step_seconds"] for row in control_step_records]), np.finfo(float).eps)
    )
    promptcredit_step_rows = promptcredit_step_records
    coordinate_gradients_stable = bool(
        promptcredit_step_rows
        and all(np.isfinite(row["coordinate_gradient_norm"]) and row["coordinate_gradient_norm"] > 0 for row in promptcredit_step_rows)
    )
    finite_losses_and_gradients = bool(
        all(
            np.isfinite(value)
            for record in control_step_records + promptcredit_step_records
            for key, value in record.items()
            if isinstance(value, (float, int)) and key != "step"
        )
    )
    quality_loss_decreased = bool(
        promptcredit_end_common["mean_quality_loss"] < promptcredit_start_common["mean_quality_loss"]
    )
    quality_prediction_nontrivial = bool(promptcredit_end_common["quality_prediction_std"] > 1e-6)
    quality_score_relation_better = _quality_score_relation_better(
        promptcredit_end_deployment, promptcredit_end_common
    )
    localization_preserved = bool(
        promptcredit_end_common["mean_point_localization_error"]
        <= 1.10 * promptcredit_start_common["mean_point_localization_error"]
    )
    common_score_comparison = _mask_metric_relation(promptcredit_end_common, control_end_common)
    deployment_score_comparison = _mask_metric_relation(promptcredit_end_deployment, promptcredit_end_common)
    decoder_call_budget_ok = bool(
        control_end_common["decoded_mask_calls"] == len(crops)
        and promptcredit_end_common["decoded_mask_calls"] == len(crops)
        and promptcredit_end_deployment["decoded_mask_calls"] == len(crops)
        and control_end_common["image_encoder_calls"] == len(crops)
        and promptcredit_end_common["image_encoder_calls"] == len(crops)
        and promptcredit_end_deployment["image_encoder_calls"] == len(crops)
    )
    smoke_checks = {
        "baseline_equivalence": baseline["passed"],
        "strict_step_0_paired_equivalence": step0_equivalence["passed"],
        "frozen_sam2_unchanged": frozen_ok,
        "finite_losses_and_gradients": finite_losses_and_gradients,
        "coordinate_gradients_stable": coordinate_gradients_stable,
        "quality_loss_decreased": quality_loss_decreased,
        "quality_prediction_nontrivial_at_step_100": quality_prediction_nontrivial,
        "quality_score_relation_better_than_raw_objectness": quality_score_relation_better,
        "common_score_promptcredit_vs_control": common_score_comparison,
        "deployment_score_promptcredit_vs_own_common_score": deployment_score_comparison,
        "promptcredit_mask_loss_below_own_step_0": promptcredit_end_common["mean_mask_loss"] < promptcredit_start_common["mean_mask_loss"],
        "point_localization_preserved_within_10_percent": localization_preserved,
        "step_time_within_1_30x_control": runtime_ratio <= 1.30,
        "peak_memory_accepted_no_oom": True,
        "no_extra_encoder_or_decoder_calls_per_evaluation_view": decoder_call_budget_ok,
    }
    hard_fail = bool(
        not step0_equivalence["passed"]
        or not frozen_ok
        or not finite_losses_and_gradients
        or not coordinate_gradients_stable
        or not quality_prediction_nontrivial
        or common_score_comparison["clearly_harmful"]
        or deployment_score_comparison["clearly_harmful"]
        or not localization_preserved
        or runtime_ratio > 1.30
        or not decoder_call_budget_ok
    )
    full_pass = bool(
        not hard_fail
        and baseline["passed"]
        and quality_loss_decreased
        and quality_score_relation_better
        and common_score_comparison["mask_loss_not_higher"]
        and common_score_comparison["mean_iou_not_lower"]
        and not common_score_comparison["metrics_conflict"]
        and common_score_comparison["strict_relative_gain"]
        and deployment_score_comparison["mask_loss_not_higher"]
        and deployment_score_comparison["mean_iou_not_lower"]
        and not deployment_score_comparison["metrics_conflict"]
    )
    recommendation = "FAIL" if hard_fail else "PASS" if full_pass else "CONDITIONAL"
    report = {
        "title": "REPORT FOR PROJECT LEAD \u2014 PROMPTCREDIT CORRECTED PAIRED SMOKE",
        "recommendation": recommendation,
        "git_sha": _git_sha(),
        "environment": {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__, "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(device)},
        "corrected_verdict_commit": "4fe29104878248e3af0263b5a120bb4ddeee3283",
        "trainable_frozen_parameter_manifest": control.trainable_manifest,
        "baseline_equivalence": baseline,
        "step0_strict_equivalence": step0_equivalence,
        "quality_loss_scale_audit": quality_loss_scale_audit,
        "control_common_score_step_0": control_start_common,
        "control_common_score_step_100": control_end_common,
        "promptcredit_common_score_step_0": promptcredit_start_common,
        "promptcredit_common_score_step_100": promptcredit_end_common,
        "promptcredit_deployment_score_step_0": promptcredit_start_deployment,
        "promptcredit_deployment_score_step_100": promptcredit_end_deployment,
        "sam2_parameter_checksums": sam2_checks,
        "frozen_parameters_unchanged": frozen_ok,
        "smoke_checks": smoke_checks,
        "runtime_memory": {"mean_step_time_ratio_promptcredit_over_control": runtime_ratio, "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device))},
        "smoke_crop_selection": crop_manifest,
        "artifact_paths": {"control_curve": "control_metrics.csv", "promptcredit_curve": "promptcredit_metrics.csv", "baseline_equivalence": "baseline_equivalence.json", "step0_strict_equivalence": "step0_strict_equivalence.json", "quality_loss_scale_audit": "quality_loss_scale_audit.json", "crop_selection": "smoke_crop_selection.json"},
        "scope": "Two fixed router-train crops, 100 steps each; a mechanism smoke only, not a generalization or final-method result.",
    }
    _write_csv(out_dir / "control_metrics.csv", control_rows)
    _write_csv(out_dir / "promptcredit_metrics.csv", promptcredit_rows)
    _write_json(out_dir / "baseline_equivalence.json", baseline)
    _write_json(out_dir / "step0_strict_equivalence.json", step0_equivalence)
    _write_json(out_dir / "quality_loss_scale_audit.json", quality_loss_scale_audit)
    _write_json(out_dir / "report.json", report)
    _write_json(out_dir / "run_manifest.json", {
        "git_sha": _git_sha(), "command": sys.argv, "seed": SEED, "steps": STEPS, "optimizer": "AdamW", "lr": LR,
        "weight_decay": WEIGHT_DECAY, "alpha_max": ALPHA_MAX, "alpha_warmup_steps": WARMUP_STEPS,
        "quality_loss_coef": 1.0, "score_mode": "objectness_x_quality",
        "quality_head_initialization": "deterministic MLP under seed 3407; final weight=0 and final bias=logit(0.01), with CPU and CUDA RNG streams restored",
        "evaluation_protocol": "evaluation_snapshot: complete point/SAM2 model eval, torch.no_grad, exact module-mode restoration, Python/NumPy/torch CPU/CUDA RNG restoration",
        "paired_training_rng": "control and PromptCredit use independent restored RNG snapshots from the same initial state",
        "checkpoint_sha256": sha256_file(checkpoint),
        "selection_manifest_sha256": sha256_file(selection_path), "split_manifest_sha256": sha256_file(split_manifest_path),
    })
    return report
