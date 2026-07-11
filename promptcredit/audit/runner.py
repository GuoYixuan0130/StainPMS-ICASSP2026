"""Executable, read-only PromptCredit PC-Stage 0 mechanism audit.

This module is deliberately not wired into ``main.py``.  It opens only the
six precommitted TNBC router-train images after all scope checks have passed.
"""

from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from promptcredit.audit.data import AuditCrop, iter_selected_tnbc_crops
from promptcredit.audit.gradient import coordinate_gradient_probe, freeze_parameters_and_clear_gradients
from promptcredit.audit.guardrails import sha256_file, validate_stage0_inputs
from promptcredit.audit.verdict import stage0_verdict
from promptcredit.matching import collision_groups, hungarian_assignment, nearest_assignment, point_inside_mask
from promptcredit.metrics import binary_iou, score_utility_summary, soft_iou
from promptcredit.utils.selection import build_selection_payload


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_selection_if_absent(split_manifest_path: Path, selection_path: Path) -> dict[str, Any]:
    """Create the fixed list before data/model/GT access; never replace a different file."""
    payload = build_selection_payload(_read_json(split_manifest_path))
    if selection_path.exists():
        observed = _read_json(selection_path)
        if observed != payload:
            raise FileExistsError(f"Refusing to overwrite differing selection manifest: {selection_path}")
        return payload
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    selection_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def _cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _quantile_labels(values: list[float], n_groups: int = 3) -> list[str]:
    if not values:
        return []
    order = np.argsort(np.asarray(values), kind="mergesort")
    labels = [""] * len(values)
    for index, group in enumerate(np.array_split(order, n_groups)):
        for item in group:
            labels[int(item)] = f"tertile_{index + 1}"
    return labels


def _grouped_utility(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = [float(row[key]) for row in rows]
    labels = _quantile_labels(values)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row, label in zip(rows, labels, strict=True):
        grouped[label].append(row)
    return {
        label: score_utility_summary(
            [float(row["point_foreground_probability"]) for row in group],
            [float(row["decoded_hard_mask_iou"]) for row in group],
        )
        for label, group in grouped.items()
    }


def _assignment_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_crop: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_crop[(str(row["image_id"]), int(row["crop_id"]))].append(row)
    collision_rows = 0
    collision_excess = 0
    collision_crops = 0
    collision_density: list[int] = []
    non_collision_density: list[int] = []
    contradictory_groups = 0
    for crop_rows in by_crop.values():
        groups = collision_groups([int(row["nearest_source_proposal_index"]) for row in crop_rows])
        colliding_gt = {gt_index for gt_indices in groups.values() for gt_index in gt_indices}
        if groups:
            collision_crops += 1
        collision_rows += len(colliding_gt)
        collision_excess += sum(len(gt_indices) - 1 for gt_indices in groups.values())
        contradictory_groups += len(groups)  # instance IDs are mutually exclusive masks by construction.
        for gt_index, row in enumerate(crop_rows):
            destination = collision_density if gt_index in colliding_gt else non_collision_density
            destination.append(int(row["local_nucleus_density_64px"]))
    total = len(rows)
    disagreement = [row for row in rows if int(row["hungarian_source_proposal_index"]) >= 0]
    return {
        "n_gt_crop_records": total,
        "nearest_collision_gt_rate": collision_rows / total if total else None,
        "nearest_collision_excess_rate": collision_excess / total if total else None,
        "crops_with_at_least_one_collision_fraction": collision_crops / len(by_crop) if by_crop else None,
        "nearest_hungarian_source_disagreement_rate": (
            sum(int(row["nearest_source_proposal_index"]) != int(row["hungarian_source_proposal_index"]) for row in disagreement) / len(disagreement)
            if disagreement else None
        ),
        "nearest_prompt_outside_corresponding_gt_fraction": (
            sum(not bool(row["nearest_point_inside_gt_mask"]) for row in rows) / total if total else None
        ),
        "collision_groups_same_point_different_gt_masks": contradictory_groups,
        "median_local_density_collision_gt": float(np.median(collision_density)) if collision_density else None,
        "median_local_density_noncollision_gt": float(np.median(non_collision_density)) if non_collision_density else None,
    }


def _gradient_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n_prompts": 0, "finite_nonzero_coordinate_gradient_fraction": 0.0}
    delta_loss = np.asarray([float(row["delta_mask_loss"]) for row in rows], dtype=np.float64)
    delta_iou = np.asarray([float(row["delta_iou"]) for row in rows], dtype=np.float64)
    norm = np.asarray([float(row["gradient_norm"]) for row in rows], dtype=np.float64)
    finite_nonzero = [bool(row["finite_coordinate_gradient"]) and bool(row["nonzero_coordinate_gradient"]) for row in rows]
    return {
        "n_prompts": len(rows),
        "frozen_prompt_encoder_and_mask_decoder_parameter_grads_none": all(bool(row["frozen_parameter_grads_none"]) for row in rows),
        "finite_nonzero_coordinate_gradient_fraction": sum(finite_nonzero) / len(rows),
        "gradient_norm": {
            "mean": float(norm.mean()),
            "median": float(np.median(norm)),
            "min": float(norm.min()),
            "max": float(norm.max()),
        },
        "mean_delta_mask_loss": float(delta_loss.mean()),
        "median_delta_mask_loss": float(np.median(delta_loss)),
        "mean_delta_iou": float(delta_iou.mean()),
        "gradient_step_prompt_loss_improvement_fraction": float((delta_loss < 0).mean()),
        "nan_or_infinite_observed": not np.isfinite(delta_loss).all() or not np.isfinite(delta_iou).all() or not np.isfinite(norm).all(),
    }


def _verdict(assignment: dict[str, Any], utility: dict[str, Any], gradient: dict[str, Any]) -> tuple[str, dict[str, bool]]:
    assignment_gap = bool(
        (assignment.get("nearest_collision_excess_rate") or 0.0) >= 0.01
        or (assignment.get("nearest_hungarian_source_disagreement_rate") or 0.0) >= 0.10
    )
    quality_gap = bool(
        (utility.get("spearman_point_score_vs_hard_iou") is not None and utility["spearman_point_score_vs_hard_iou"] <= 0.60)
        or (utility.get("ece_10_equal_frequency") or 0.0) >= 0.08
    )
    actionable = bool(
        gradient.get("finite_nonzero_coordinate_gradient_fraction", 0.0) >= 0.95
        and gradient.get("mean_delta_mask_loss", 0.0) < 0.0
        and gradient.get("gradient_step_prompt_loss_improvement_fraction", 0.0) >= 0.60
        and not gradient.get("nan_or_infinite_observed", True)
    )
    verdict = stage0_verdict(
        assignment_gap=assignment_gap,
        quality_gap=quality_gap,
        actionable_gradient=actionable,
        acceptable_cost=True,
    )
    return verdict, {"assignment_gap": assignment_gap, "quality_gap": quality_gap, "actionable_gradient": actionable}


def _draw_reliability_diagram(path: Path, reliability_bins: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    if not reliability_bins:
        return
    confidence = [float(item["mean_score"]) for item in reliability_bins]
    accuracy = [float(item["empirical_matchability"]) for item in reliability_bins]
    plt.figure(figsize=(4, 4))
    plt.plot([0, 1], [0, 1], "--", color="gray", label="ideal")
    plt.plot(confidence, accuracy, marker="o", label="PromptCredit audit")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("Point foreground probability")
    plt.ylabel("P(decoded hard-mask IoU >= 0.5)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


class _ModelBundle:
    def __init__(self, *, point_net: Any, point_encoder: Any, net: Any, texture_memory_bank: list[Any], device: torch.device):
        self.point_net = point_net
        self.point_encoder = point_encoder
        self.net = net
        self.texture_memory_bank = texture_memory_bank
        self.device = device


def _load_models(config_path: Path, sam_config: str, checkpoint: Path, device: torch.device) -> _ModelBundle:
    from mmengine.config import Config
    from sam2_train.build_sam import build_sam2
    from sam2_train.modeling.dpa_p2pnet import build_model

    config = Config.fromfile(str(config_path))
    point_net, point_encoder = build_model(config)
    checkpoint_payload = torch.load(checkpoint, map_location="cpu")
    if "model1" not in checkpoint_payload or "model" not in checkpoint_payload:
        raise ValueError("Frozen StainPMS checkpoint must contain both model1 and model states")
    point_net.load_state_dict(checkpoint_payload["model1"])
    net = build_sam2(sam_config, str(checkpoint), device=device, mode="eval")
    point_net.to(device).eval()
    point_encoder.to(device).eval()
    net.eval()
    # A local copy preserves checkpoint artifacts while retaining the frozen v1 texture context.
    texture_memory_bank = list(checkpoint_payload.get("texture_memory_bank_list", []) or [])
    return _ModelBundle(point_net=point_net, point_encoder=point_encoder, net=net, texture_memory_bank=texture_memory_bank, device=device)


def _apply_context_memory(context_bank: list[Any], feats: list[torch.Tensor], positions: list[torch.Tensor], net: Any, x: int, y: int) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Copy the existing context-memory operation for one fixed selected image."""
    batch_size = feats[-1].size(1)
    if not context_bank:
        zero = torch.zeros(1, batch_size, net.hidden_dim, device=feats[-1].device)
        feats[-1] = feats[-1] + zero
        positions[-1] = positions[-1] + zero
        return feats, positions
    memory = context_bank[-1][0].to(feats[-1].device, non_blocking=True)
    memory_pos = context_bank[-1][1].to(feats[-1].device, non_blocking=True)
    feats[-1], positions[-1] = net.memory_attention(
        state="context", curr=feats[-1], curr_pos=positions[-1], memory=memory,
        memory_pos=memory_pos, num_obj_ptr_tokens=0,
    )
    return feats, positions


def _apply_frozen_texture_memory(texture_bank: list[Any], feats: list[torch.Tensor], positions: list[torch.Tensor], net: Any) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Use the checkpoint's frozen texture bank without mutating it during an audit."""
    batch_size = feats[-1].size(1)
    device = feats[-1].device
    if not texture_bank:
        zero = torch.zeros(1, batch_size, net.hidden_dim, device=device)
        feats[-1] = feats[-1] + zero
        positions[-1] = positions[-1] + zero
        return feats, positions
    memory_entries = [entry[0].to(device, non_blocking=True).flatten(2).permute(2, 0, 1) for entry in texture_bank]
    position_entries = [entry[1].to(device, non_blocking=True).flatten(2).permute(2, 0, 1) for entry in texture_bank]
    embedding_entries = [entry[3].to(device, non_blocking=True) for entry in texture_bank]
    memory_stack = torch.stack(memory_entries, dim=0)
    position_stack = torch.stack(position_entries, dim=0)
    embedding_stack = F.normalize(torch.stack(embedding_entries, dim=0), p=2, dim=1)
    query = feats[-1].permute(1, 0, 2).reshape(batch_size, -1)
    similarity = F.softmax(torch.mm(embedding_stack, F.normalize(query, p=2, dim=1).t()).t(), dim=1)
    sampled = torch.topk(similarity, batch_size, dim=1).indices.squeeze(1)
    selected_memory = memory_stack[sampled].squeeze(3).permute(1, 2, 0, 3)
    selected_position = position_stack[sampled].squeeze(3).permute(1, 2, 0, 3)
    memory = selected_memory.reshape(-1, selected_memory.size(2), selected_memory.size(3))
    memory_pos = selected_position.reshape(-1, selected_position.size(2), selected_position.size(3))
    feats[-1], positions[-1] = net.memory_attention(
        state="texture", curr=[feats[-1]], curr_pos=[positions[-1]], memory=memory,
        memory_pos=memory_pos, num_obj_ptr_tokens=0,
    )
    return feats, positions


def _prepare_decoder_features(bundle: _ModelBundle, image: torch.Tensor, context_bank: list[Any], crop: AuditCrop) -> tuple[torch.Tensor, list[torch.Tensor]]:
    with torch.no_grad():
        backbone_features, _ = bundle.point_encoder(image)
        backbone_out, _ = bundle.net.forward_image(image, backbone_features)
        _, vision_feats, vision_positions, _ = bundle.net._prepare_backbone_features(backbone_out)
        raw_feats, raw_positions = vision_feats, vision_positions
        vision_feats, vision_positions = _apply_context_memory(context_bank, vision_feats, vision_positions, bundle.net, crop.crop_box_xyxy[0], crop.crop_box_xyxy[1])
        vision_feats, vision_positions = _apply_frozen_texture_memory(bundle.texture_memory_bank, vision_feats, vision_positions, bundle.net)
        batch_size = vision_feats[-1].size(1)
        feature_sizes = [(64, 64), (32, 32), (16, 16)]
        decoded_feats = [
            feature.permute(1, 2, 0).view(batch_size, -1, *feature_size)
            for feature, feature_size in zip(vision_feats[::-1], feature_sizes[::-1])
        ][::-1]
        if len(context_bank) < 100:
            context_bank.append([raw_feats[-1].detach(), raw_positions[-1].detach()])
    return decoded_feats[-1], decoded_feats[:-1]


def _decode_standard(bundle: _ModelBundle, image_embed: torch.Tensor, high_res_features: list[torch.Tensor], coordinates: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        labels = torch.ones(coordinates.size(0), 1, dtype=torch.int, device=bundle.device)
        sparse, dense = bundle.net.sam_prompt_encoder(points=(coordinates, labels), boxes=None, masks=None, batch_size=1)
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
        predicted_iou = torch.max(iou_predictions, dim=1).values
    return logits, predicted_iou


def _gradient_rows_for_crop(
    bundle: _ModelBundle,
    crop: AuditCrop,
    image_embed: torch.Tensor,
    high_res_features: list[torch.Tensor],
    coordinates: torch.Tensor,
    targets: torch.Tensor,
    metadata: list[dict[str, Any]],
    frozen_parameters: list[torch.nn.Parameter],
) -> list[dict[str, Any]]:
    def decode_logits(coords: torch.Tensor) -> torch.Tensor:
        labels = torch.ones(coords.size(0), 1, dtype=torch.int, device=bundle.device)
        sparse, dense = bundle.net.sam_prompt_encoder(points=(coords, labels), boxes=None, masks=None, batch_size=1)
        low_res, _, _, _ = bundle.net.sam_mask_decoder(
            image_embeddings=image_embed,
            image_pe=bundle.net.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
            repeat_image=False,
            cell_nums=torch.as_tensor([coords.size(0)], device=bundle.device),
            high_res_features=high_res_features,
        )
        return F.interpolate(low_res, size=(256, 256), mode="bilinear", align_corners=False)[:, 0]

    probe = coordinate_gradient_probe(
        decode_logits, coordinates, targets, width=256, height=256, frozen_parameters=frozen_parameters
    )
    rows: list[dict[str, Any]] = []
    for index, base in enumerate(metadata):
        row = dict(base)
        row.update(
            {
                "gradient_norm": float(probe["gradient_norm"][index].cpu()),
                "finite_coordinate_gradient": bool(probe["finite"][index].cpu()),
                "nonzero_coordinate_gradient": bool(probe["nonzero"][index].cpu()),
                "frozen_parameter_grads_none": bool(probe["frozen_parameter_grads_none"]),
                "original_mask_loss": float(probe["original_loss"][index].cpu()),
                "moved_mask_loss": float(probe["moved_loss"][index].cpu()),
                "delta_mask_loss": float((probe["moved_loss"][index] - probe["original_loss"][index]).cpu()),
                "original_hard_mask_iou": float(probe["original_iou"][index].cpu()),
                "moved_hard_mask_iou": float(probe["moved_iou"][index].cpu()),
                "delta_iou": float((probe["moved_iou"][index] - probe["original_iou"][index]).cpu()),
                "eta_pixels": 1.0,
            }
        )
        rows.append(row)
    return rows


def run_stage0(
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
    """Run the authorized six-image GPU audit and write only local audit artifacts."""
    if device_name != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("PC-Stage 0 model audit is GPU-only; run the documented command on AutoDL 4090")
    split_manifest = _read_json(split_manifest_path)
    selection = _read_json(selection_path)
    image_ids = validate_stage0_inputs(
        data_root=data_root, split_manifest=split_manifest, selection=selection, checkpoint=checkpoint
    )
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing audit artifacts: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda")
    torch.manual_seed(3407)
    np.random.seed(3407)
    torch.cuda.reset_peak_memory_stats(device)
    bundle = _load_models(config_path, sam_config, checkpoint, device)
    frozen_parameters = freeze_parameters_and_clear_gradients([bundle.net.sam_prompt_encoder, bundle.net.sam_mask_decoder])
    assignment_rows: list[dict[str, Any]] = []
    utility_rows: list[dict[str, Any]] = []
    gradient_rows: list[dict[str, Any]] = []
    runtime = {"standard_forward_seconds": 0.0, "assignment_audit_seconds": 0.0, "gradient_audit_seconds": 0.0, "standard_forward_crops": 0}
    first_image_id = image_ids[0]
    gradient_remaining = 20
    context_bank: list[Any] = []
    active_image_id: str | None = None

    for crop in iter_selected_tnbc_crops(data_root, image_ids):
        if crop.image_id != active_image_id:
            active_image_id = crop.image_id
            context_bank = []
        image = crop.image.unsqueeze(0).to(device)
        _cuda_sync(device)
        forward_started = time.perf_counter()
        with torch.no_grad():
            point_output, _, _, _ = bundle.point_net(image)
        image_embed, high_res_features = _prepare_decoder_features(bundle, image, context_bank, crop)
        probabilities = point_output["pred_logits"][0].softmax(dim=-1)[:, 0].detach().cpu().numpy()
        proposals = point_output["pred_coords"][0].detach().cpu().numpy()
        assignment_started = time.perf_counter()
        nearest = nearest_assignment(proposals, crop.gt_centroids_xy)
        hungarian = hungarian_assignment(proposals, crop.gt_centroids_xy, probabilities)
        runtime["assignment_audit_seconds"] += time.perf_counter() - assignment_started
        groups = collision_groups(nearest.source_for_gt)
        source_collision_count = {source: len(gt_indices) for source, gt_indices in groups.items()}
        for gt_index, (instance_id, mask, centroid, area, density) in enumerate(zip(crop.gt_instance_ids, crop.gt_masks, crop.gt_centroids_xy, crop.gt_areas, crop.local_density, strict=True)):
            nearest_source = int(nearest.source_for_gt[gt_index])
            hungarian_source = int(hungarian.source_for_gt[gt_index])
            point = proposals[nearest_source] if nearest_source >= 0 else np.asarray([np.nan, np.nan])
            assignment_rows.append(
                {
                    "image_id": crop.image_id,
                    "crop_id": crop.crop_id,
                    "crop_box_xyxy": json.dumps(crop.crop_box_xyxy),
                    "gt_instance_index": gt_index,
                    "gt_instance_id": int(instance_id),
                    "nearest_source_proposal_index": nearest_source,
                    "hungarian_source_proposal_index": hungarian_source,
                    "nearest_predicted_x": float(point[0]),
                    "nearest_predicted_y": float(point[1]),
                    "gt_centroid_x": float(centroid[0]),
                    "gt_centroid_y": float(centroid[1]),
                    "point_foreground_probability": float(probabilities[nearest_source]) if nearest_source >= 0 else None,
                    "nearest_coordinate_distance": float(nearest.distance_for_gt[gt_index]),
                    "nearest_point_inside_gt_mask": point_inside_mask(mask, point) if nearest_source >= 0 else False,
                    "nearest_source_gt_reuse_count": int(source_collision_count.get(nearest_source, 1)),
                    "local_nucleus_density_64px": int(density),
                    "nucleus_area_pixels": int(area),
                }
            )
        valid_gt = np.flatnonzero(nearest.source_for_gt >= 0)
        if len(valid_gt):
            coordinates = torch.as_tensor(proposals[nearest.source_for_gt[valid_gt]], dtype=torch.float32, device=device).unsqueeze(1)
            target_masks = torch.as_tensor(crop.gt_masks[valid_gt], dtype=torch.float32, device=device)
            logits, predicted_iou = _decode_standard(bundle, image_embed, high_res_features, coordinates)
            _cuda_sync(device)
            runtime["standard_forward_seconds"] += time.perf_counter() - forward_started
            runtime["standard_forward_crops"] += 1
            hard = (logits > 0).detach().cpu().numpy()
            soft = torch.sigmoid(logits).detach().cpu().numpy()
            target_np = target_masks.detach().cpu().numpy().astype(bool)
            hard_iou = binary_iou(hard, target_np)
            soft_values = soft_iou(soft, target_np)
            for local_index, gt_index in enumerate(valid_gt):
                assignment = assignment_rows[-len(crop.gt_masks) + int(gt_index)]
                utility_rows.append(
                    {
                        "image_id": crop.image_id,
                        "crop_id": crop.crop_id,
                        "gt_instance_index": int(gt_index),
                        "nearest_source_proposal_index": int(nearest.source_for_gt[gt_index]),
                        "point_foreground_probability": float(probabilities[nearest.source_for_gt[gt_index]]),
                        "point_to_centroid_distance": float(nearest.distance_for_gt[gt_index]),
                        "point_inside_gt_mask": bool(assignment["nearest_point_inside_gt_mask"]),
                        "decoded_hard_mask_iou": float(hard_iou[local_index]),
                        "decoded_soft_mask_iou": float(soft_values[local_index]),
                        "sam_predicted_iou": float(predicted_iou[local_index].cpu()),
                        "decoded_iou_ge_0_5": bool(hard_iou[local_index] >= 0.5),
                        "nucleus_area_pixels": int(crop.gt_areas[gt_index]),
                        "local_nucleus_density_64px": int(crop.local_density[gt_index]),
                    }
                )
        else:
            _cuda_sync(device)
            runtime["standard_forward_seconds"] += time.perf_counter() - forward_started
            runtime["standard_forward_crops"] += 1

        if crop.image_id == first_image_id and gradient_remaining > 0:
            legal = [index for index, source in enumerate(hungarian.source_for_gt) if source >= 0 and np.isfinite(proposals[source]).all() and 0 <= proposals[source, 0] < 256 and 0 <= proposals[source, 1] < 256]
            legal = legal[:gradient_remaining]
            if legal:
                gradient_coordinates = torch.as_tensor(proposals[hungarian.source_for_gt[legal]], dtype=torch.float32, device=device).unsqueeze(1)
                gradient_targets = torch.as_tensor(crop.gt_masks[legal], dtype=torch.float32, device=device)
                metadata = [
                    {
                        "image_id": crop.image_id,
                        "crop_id": crop.crop_id,
                        "gt_instance_index": int(index),
                        "hungarian_source_proposal_index": int(hungarian.source_for_gt[index]),
                    }
                    for index in legal
                ]
                _cuda_sync(device)
                gradient_started = time.perf_counter()
                gradient_rows.extend(_gradient_rows_for_crop(bundle, crop, image_embed, high_res_features, gradient_coordinates, gradient_targets, metadata, frozen_parameters))
                _cuda_sync(device)
                runtime["gradient_audit_seconds"] += time.perf_counter() - gradient_started
                gradient_remaining -= len(legal)

    assignment = _assignment_summary(assignment_rows)
    utility = score_utility_summary(
        [float(row["point_foreground_probability"]) for row in utility_rows],
        [float(row["decoded_hard_mask_iou"]) for row in utility_rows],
    )
    utility["grouped_by_local_density_64px"] = _grouped_utility(utility_rows, "local_nucleus_density_64px")
    utility["grouped_by_nucleus_area"] = _grouped_utility(utility_rows, "nucleus_area_pixels")
    utility["grouped_by_prompt_distance"] = _grouped_utility(utility_rows, "point_to_centroid_distance")
    gradient = _gradient_summary(gradient_rows)
    runtime["peak_gpu_memory_bytes"] = int(torch.cuda.max_memory_allocated(device))
    runtime["mean_standard_forward_seconds_per_crop"] = runtime["standard_forward_seconds"] / max(1, runtime["standard_forward_crops"])
    runtime["future_training_overhead_estimate"] = {
        "assignment_relative_to_standard_forward": runtime["assignment_audit_seconds"] / max(runtime["standard_forward_seconds"], np.finfo(float).eps),
        "gradient_diagnostic_relative_to_standard_forward": runtime["gradient_audit_seconds"] / max(runtime["standard_forward_seconds"], np.finfo(float).eps),
        "interpretation": "The diagnostic includes a prohibited-for-final-method second decoder decode. A future method must reuse its ordinary decoder call; Stage 0 cannot claim a final numeric training overhead before that method is authorized.",
    }
    verdict, criteria = _verdict(assignment, utility, gradient)
    report = {
        "title": "REPORT FOR PROJECT LEAD — PROMPTCREDIT STAGE 0",
        "verdict": verdict,
        "pre_registered_criteria": criteria,
        "git_sha": _git_sha(),
        "environment": {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__, "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(device)},
        "fixed_six_image_list": image_ids,
        "fixed_six_image_list_checksum": selection["image_ids_sha256"],
        "assignment_collision_statistics": assignment,
        "point_score_mask_utility_calibration": utility,
        "decoder_coordinate_gradient_results": gradient,
        "runtime_and_memory": runtime,
        "negative_findings_and_anomalies": [],
        "artifact_paths": {
            "assignment_records": "assignment_records.csv",
            "utility_records": "utility_records.csv",
            "gradient_records": "gradient_records.csv",
            "reliability_diagram": "reliability_diagram.png",
        },
        "recommendation": "Await project-lead decision; do not implement a quality head or alter the formal training loop.",
    }
    _write_csv(out_dir / "assignment_records.csv", assignment_rows)
    _write_csv(out_dir / "utility_records.csv", utility_rows)
    _write_csv(out_dir / "gradient_records.csv", gradient_rows)
    _draw_reliability_diagram(out_dir / "reliability_diagram.png", utility["reliability_diagram"])
    _json_dump(out_dir / "report.json", report)
    _json_dump(out_dir / "run_manifest.json", {
        "git_sha": _git_sha(), "command": sys.argv, "selection_sha256": sha256_file(selection_path),
        "split_manifest_sha256": sha256_file(split_manifest_path), "checkpoint_sha256": sha256_file(checkpoint),
        "preprocessing": {"crop_size": 256, "overlap": 32, "load_order": "unclockwise", "normalize": "Albumentations Normalize", "density_radius_pixels": 64},
        "memory_policy": "Frozen checkpoint texture bank; local per-image context bank; no checkpoint/artifact mutation.",
    })
    return report
