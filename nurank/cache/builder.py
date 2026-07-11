"""One-pass frozen automatic-prompt cache construction for NuRank Stage 1."""

from __future__ import annotations

import json
import platform
import random
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from nuset.audit.data import BASELINE_V1_TNBC_SHA256, sha256_file
from nuset.audit.metrics import hard_iou, soft_iou
from nuset.audit.models import FrozenNuSetBundle, module_state_sha256
from nuset.audit.runner import (
    CallCounts,
    NMS_RADIUS,
    OVERLAP,
    _decode_all_tokens,
    _update_baseline_memory,
    crop_with_overlap,
    point_nms,
    predict,
)
from nurank.cache.data import Role, resolve_nurank_images
from nurank.cache.io import TOKEN_COUNT, sha256_file as cache_sha256_file
from nurank.features.morphology import morphology_features


SEED = 3407
TIME_LIMIT_SECONDS = 6 * 60 * 60


@dataclass(frozen=True)
class CacheResult:
    cache_dir: Path
    manifest_path: Path
    elapsed_seconds: float
    estimated_total_seconds: float | None


def _set_seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _max_iou_against_all(logits: torch.Tensor, gt_masks: torch.Tensor, *, soft: bool) -> torch.Tensor:
    """Max per-token IoU with crop GT, used only as the unmatched analysis target."""
    if not len(gt_masks):
        return logits.new_zeros(logits.shape[:2])
    values: list[torch.Tensor] = []
    # Keeping chunks bounded avoids materializing [prompts, tokens, all GT, H, W].
    for start in range(0, logits.size(0), 16):
        current = logits[start : start + 16]
        if soft:
            probability = torch.sigmoid(current).flatten(2)
            truth = gt_masks.float().flatten(1)
            intersection = torch.einsum("ntp,gp->ntg", probability, truth)
            union = probability.sum(dim=-1)[..., None] + truth.sum(dim=-1)[None, None, :] - intersection
        else:
            prediction = (current > 0).float().flatten(2)
            truth = gt_masks.float().flatten(1)
            intersection = torch.einsum("ntp,gp->ntg", prediction, truth)
            union = prediction.sum(dim=-1)[..., None] + truth.sum(dim=-1)[None, None, :] - intersection
        values.append(torch.where(union > 0, intersection / union, torch.ones_like(union)).amax(dim=-1))
    return torch.cat(values, dim=0)


def _targets_for_prompts(
    instance_map: np.ndarray,
    crop_box: tuple[int, int, int, int],
    local_points: torch.Tensor,
    logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    """Associate existing automatic points; unmatched targets are max IoU over existing GT."""
    x1, y1, x2, y2 = crop_box
    local_map = instance_map[y1:y2, x1:x2]
    global_points = local_points[:, 0].detach().cpu().numpy() + np.asarray((x1, y1), dtype=np.float32)
    instance_ids = np.zeros(len(global_points), dtype=np.int64)
    for index, point in enumerate(global_points):
        px = int(np.clip(np.trunc(point[0]), 0, instance_map.shape[1] - 1))
        py = int(np.clip(np.trunc(point[1]), 0, instance_map.shape[0] - 1))
        instance_ids[index] = int(instance_map[py, px])
    gt_ids = np.unique(local_map)
    gt_ids = gt_ids[gt_ids != 0]
    gt_masks = torch.as_tensor(np.stack([local_map == instance_id for instance_id in gt_ids]), dtype=torch.float32, device=logits.device) if len(gt_ids) else logits.new_empty((0, *logits.shape[-2:]))
    matched = instance_ids != 0
    hard = _max_iou_against_all(logits, gt_masks, soft=False)
    soft = _max_iou_against_all(logits, gt_masks, soft=True)
    if matched.any():
        target = torch.as_tensor(np.stack([local_map == instance_id for instance_id in instance_ids[matched]]), dtype=torch.float32, device=logits.device)
        indices = torch.as_tensor(np.flatnonzero(matched), dtype=torch.long, device=logits.device)
        hard[indices] = hard_iou(logits.index_select(0, indices), target)
        soft[indices] = soft_iou(logits.index_select(0, indices), target)
    return hard, soft, matched, instance_ids


def _cache_group(
    *, out_path: Path, item, crop_id: int, crop_box: tuple[int, int, int, int], local_points: torch.Tensor,
    classes: np.ndarray, inds: np.ndarray, objectness: np.ndarray, all_tokens,
) -> dict[str, Any]:
    hard, soft, matched, instance_ids = _targets_for_prompts(item.instance_map, crop_box, local_points, all_tokens.upsampled_logits)
    morphology = morphology_features(all_tokens.upsampled_logits, local_points)
    arrays = {
        # Logits stay float32 so token-0 replay can prove exact baseline identity; only embeddings are fp16 by protocol.
        "mask_logits": all_tokens.upsampled_logits.detach().cpu().float().numpy(),
        "mask_tokens": all_tokens.mask_tokens.detach().cpu().to(torch.float16).numpy(),
        "token_index": np.tile(np.arange(TOKEN_COUNT, dtype=np.int64), (len(local_points), 1)),
        "original_predicted_iou": all_tokens.predicted_iou.detach().cpu().float().numpy(),
        "morphology": morphology.detach().cpu().float().numpy(),
        "coordinates_local_xy": local_points[:, 0].detach().cpu().float().numpy(),
        "classes": np.asarray(classes, dtype=np.int64),
        "prompt_ids": np.asarray(inds, dtype=np.int64),
        "objectness": np.asarray(objectness, dtype=np.float32),
        "true_hard_iou": hard.detach().cpu().float().numpy(),
        "true_soft_iou": soft.detach().cpu().float().numpy(),
        "matched": np.asarray(matched, dtype=np.bool_),
        "target_instance_id": np.asarray(instance_ids, dtype=np.int64),
        "crop_box_xyxy": np.asarray(crop_box, dtype=np.int64),
        "ori_shape_hw": np.asarray(item.instance_map.shape, dtype=np.int64),
    }
    np.savez_compressed(out_path, **arrays)
    # The explicit precision record makes float16 storage auditable rather than pretending it is lossless.
    return {
        "path": out_path.name,
        "sha256": cache_sha256_file(out_path),
        "image_id": item.image_id,
        "crop_id": int(crop_id),
        "prompt_count": int(len(local_points)),
        "matched_prompt_count": int(matched.sum()),
        "unmatched_prompt_count": int((~matched).sum()),
        "cached_mask_logits_max_abs_error": float(np.max(np.abs(arrays["mask_logits"].astype(np.float32) - all_tokens.upsampled_logits.detach().cpu().float().numpy()))),
        "cached_mask_tokens_max_abs_error": float(np.max(np.abs(arrays["mask_tokens"].astype(np.float32) - all_tokens.mask_tokens.detach().cpu().float().numpy()))),
        "cached_predicted_iou_max_abs_error": float(np.max(np.abs(arrays["original_predicted_iou"] - all_tokens.predicted_iou.detach().cpu().float().numpy()))),
        "cached_morphology_max_abs_error": float(np.max(np.abs(arrays["morphology"] - morphology.detach().cpu().float().numpy()))),
    }


def build_automatic_prompt_cache(
    *, bundle: FrozenNuSetBundle, data_root: Path, split_manifest_path: Path, role: Role, cache_dir: Path,
    progress: Callable[[int, int], None] | None = None, prior_stage_seconds: float = 0.0,
) -> CacheResult:
    """Build an immutable cache from one standard automatic inference traversal."""
    if cache_dir.exists():
        raise FileExistsError(f"NuRank cache destination must be new and immutable: {cache_dir}")
    cache_dir.mkdir(parents=True)
    _set_seed()
    images = resolve_nurank_images(data_root, split_manifest_path, role)
    before = {"point_net": module_state_sha256(bundle.point_net), "point_encoder": module_state_sha256(bundle.point_encoder), "sam2": module_state_sha256(bundle.net)}
    counts, groups = CallCounts(), []
    memory_bank = list(bundle.texture_memory_bank)
    started, processed, estimated_total = time.perf_counter(), 0, None
    total_crops = sum(len(crop_with_overlap(item.image, 256, 256, OVERLAP, "unclockwise")) for item in images)
    for item in images:
        image = item.image.unsqueeze(0).to(bundle.device)
        ori_shape = item.instance_map.shape
        all_points: list[np.ndarray] = []
        all_scores: list[np.ndarray] = []
        all_classes: list[np.ndarray] = []
        processed_boxes: list[list[int]] = []
        point_id_map: dict[tuple[float, float], int] = {}
        next_id, context_bank = 0, []
        boxes = crop_with_overlap(image[0], 256, 256, OVERLAP, "unclockwise").tolist()
        for crop_id, raw_box in enumerate(boxes):
            x1, y1, x2, y2 = (int(value) for value in raw_box)
            crop_box = (x1, y1, x2, y2)
            crop_image = image[..., y1:y2, x1:x2]
            with torch.no_grad():
                points, scores, classes, _, _, _, _ = predict(bundle.point_net, crop_image, ori_shape=np.asarray((y2 - y1, x2 - x1)), filtering=True, nms_thr=NMS_RADIUS, prompt_score_mode="objectness")
                counts.point_head_forward += 1
            if len(points):
                points[:, 0] += x1
                points[:, 1] += y1
                keep_new = np.ones(len(points), dtype=bool)
                for px1, py1, px2, py2 in processed_boxes:
                    keep_new &= ~((points[:, 0] >= px1 + 1) & (points[:, 0] <= px2 - 1) & (points[:, 1] >= py1 + 1) & (points[:, 1] <= py2 - 1))
                points, scores, classes = points[keep_new], scores[keep_new], classes[keep_new]
                if len(points):
                    all_points.append(points)
                    all_scores.append(scores)
                    all_classes.append(classes)
                    current_points, current_scores, current_classes = point_nms(np.vstack(all_points), np.concatenate(all_scores), np.concatenate(all_classes), NMS_RADIUS)
                    current_ids: list[int] = []
                    for point in current_points:
                        key = tuple(point.tolist())
                        if key not in point_id_map:
                            point_id_map[key] = next_id
                            next_id += 1
                        current_ids.append(point_id_map[key])
                    local = torch.from_numpy(current_points).unsqueeze(1)
                    keep = ((local[..., 0] >= x1) & (local[..., 0] < x2) & (local[..., 1] >= y1) & (local[..., 1] < y2)).squeeze(1)
                    if int(keep.sum()):
                        local_points = (local[keep] - torch.as_tensor([x1, y1])).to(bundle.device).float()
                        all_tokens, vision_feats, image_embed = _decode_all_tokens(bundle=bundle, image=crop_image, memory_bank=memory_bank, context_bank=context_bank, x=x1, y=y1, coordinates=local_points, counts=counts)
                        kept = keep.cpu().numpy()
                        group_path = cache_dir / f"{item.image_id}__crop_{crop_id:03d}__group_{len(groups):05d}.npz"
                        groups.append(_cache_group(out_path=group_path, item=item, crop_id=crop_id, crop_box=crop_box, local_points=local_points, classes=current_classes[kept], inds=np.asarray(current_ids, dtype=np.int64)[kept], objectness=current_scores[kept], all_tokens=all_tokens))
                        _update_baseline_memory(bundle=bundle, memory_bank=memory_bank, all_tokens=all_tokens, vision_feats=vision_feats, image_embed=image_embed, crop_box=crop_box, ori_shape=ori_shape, local_points=local_points)
            processed_boxes.append(raw_box)
            processed += 1
            elapsed = time.perf_counter() - started
            if processed >= max(1, int(np.ceil(total_crops * 0.10))):
                estimated_total = elapsed / processed * total_crops
                if prior_stage_seconds + estimated_total > TIME_LIMIT_SECONDS:
                    raise RuntimeError(f"NuRank stage estimate {(prior_stage_seconds + estimated_total) / 3600:.2f} GPU hours exceeds fixed 6 hour limit")
            if progress:
                progress(processed, total_crops)
    after = {"point_net": module_state_sha256(bundle.point_net), "point_encoder": module_state_sha256(bundle.point_encoder), "sam2": module_state_sha256(bundle.net)}
    if before != after:
        raise RuntimeError("Frozen baseline checksum changed during NuRank cache construction")
    if not (counts.sam_image_encoder == counts.sam_prompt_encoder == counts.sam_mask_decoder == len(groups)):
        raise RuntimeError("NuRank cache requires exactly one image/prompt/decoder call per cached group")
    elapsed = time.perf_counter() - started
    manifest = {
        "schema": "nurank_automatic_prompt_cache_v1", "role": role, "seed": SEED, "token_count": TOKEN_COUNT,
        "git_sha": _git_sha(), "checkpoint_sha256": BASELINE_V1_TNBC_SHA256,
        "image_ids": [item.image_id for item in images],
        "image_manifest": [{"image_id": item.image_id, "image_sha256": item.image_sha256, "label_sha256": item.label_sha256} for item in images],
        "groups": groups, "group_count": len(groups), "prompt_count": int(sum(group["prompt_count"] for group in groups)),
        "matched_prompt_count": int(sum(group["matched_prompt_count"] for group in groups)), "unmatched_prompt_count": int(sum(group["unmatched_prompt_count"] for group in groups)),
        "call_counts": counts.as_dict(), "elapsed_seconds": elapsed, "prior_stage_seconds": prior_stage_seconds, "estimated_total_seconds_at_10_percent": estimated_total,
        "frozen_checksums": {"before": before, "after": after},
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": str(bundle.device), "nms": NMS_RADIUS, "tta": False, "texture": True, "context": True, "overlap": OVERLAP},
    }
    _write_json(cache_dir / "manifest.json", manifest)
    return CacheResult(cache_dir=cache_dir, manifest_path=cache_dir / "manifest.json", elapsed_seconds=elapsed, estimated_total_seconds=estimated_total)
