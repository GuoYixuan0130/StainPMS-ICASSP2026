"""NuSet Stage 0: frozen, single-decoder-forward four-token headroom audit."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from nuset.audit.data import (
    BASELINE_V1_TNBC_SHA256,
    NuSetCrop,
    NuSetImage,
    iter_fixed_crops,
    iter_fixed_images,
    load_fixed_selection,
    sha256_file,
)
from nuset.audit.decoder import AllTokenMasks, extract_all_tokens_once, select_token_logits, token0_view
from nuset.audit.metrics import assembly_metrics, headroom_summary, ranking_summary, selector_indices, token_record_rows
from nuset.audit.models import FrozenNuSetBundle, load_frozen_bundle, module_state_sha256

# ``run.utils`` parses process argv at import time.  NuSet has its own CLI and
# unit-test arguments, so hide them only for this legacy baseline utility
# import.  The imported functions themselves are unchanged baseline helpers.
_argv_before_baseline_import = sys.argv
try:
    sys.argv = [sys.argv[0]]
    from run.run_on_epoch import _assemble_instance_map, _ori_hw, combine_mask, context_memory_attention, crop_with_overlap, mask_process_eval
finally:
    sys.argv = _argv_before_baseline_import
from sam2_train.modeling.utils import point_nms, predict


SEED = 3407
OVERLAP = 32
NMS_RADIUS = 12
TIME_LIMIT_SECONDS = 60 * 60


class StageTimeCap(RuntimeError):
    pass


@dataclass
class CallCounts:
    sam_image_encoder: int = 0
    sam_prompt_encoder: int = 0
    sam_mask_decoder: int = 0
    point_head_forward: int = 0
    decoder_seconds: float = 0.0
    all_token_upsample_seconds: float = 0.0
    baseline_token0_upsample_seconds: float = 0.0
    token0_selector_seconds: float = 0.0
    all_token_output_bytes_max: int = 0
    token0_upsampled_max_abs_error: float = 0.0
    token0_hard_masks_equal: bool = True

    def as_dict(self) -> dict[str, int]:
        return {
            "sam_image_encoder_calls": self.sam_image_encoder,
            "sam_prompt_encoder_calls": self.sam_prompt_encoder,
            "sam_mask_decoder_calls": self.sam_mask_decoder,
            "point_head_forward_calls": self.point_head_forward,
            "decoder_seconds": self.decoder_seconds,
            "all_token_upsample_seconds": self.all_token_upsample_seconds,
            "baseline_token0_upsample_seconds": self.baseline_token0_upsample_seconds,
            "token0_selector_seconds": self.token0_selector_seconds,
            "all_token_output_bytes_max": self.all_token_output_bytes_max,
            "token0_upsampled_max_abs_error": self.token0_upsampled_max_abs_error,
            "token0_hard_masks_equal": self.token0_hard_masks_equal,
        }


@dataclass
class CandidatePath:
    boxes: list[Any] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    masks: list[np.ndarray] = field(default_factory=list)
    inds: list[int] = field(default_factory=list)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    normalized = [_flatten_row(row) for row in rows]
    fields = sorted({key for row in normalized for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(normalized)


def _flatten_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: json.dumps(_jsonable(value), sort_keys=True) if isinstance(value, (dict, list, tuple, np.ndarray)) else _jsonable(value) for key, value in row.items()}


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


def _cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _append_log(out_dir: Path, message: str) -> None:
    with (out_dir / "stdout.log").open("a", encoding="utf-8") as handle:
        handle.write(message.rstrip() + "\n")


def _environment(device: torch.device) -> str:
    return "\n".join(
        [
            f"git_sha={_git_sha()}",
            f"python={sys.version}",
            f"platform={platform.platform()}",
            f"torch={torch.__version__}",
            f"cuda_device={torch.cuda.get_device_name(device)}",
            f"seed={SEED}",
            f"nms={NMS_RADIUS}",
            "tta=False",
            "texture=True",
            "context=True",
            f"overlap={OVERLAP}",
            "crop_order=unclockwise",
        ]
    ) + "\n"


def _artifact_checksums(out_dir: Path) -> None:
    records = []
    for path in sorted(path for path in out_dir.rglob("*") if path.is_file() and path.name != "SHA256SUMS"):
        records.append(f"{sha256_file(path)}  {path.relative_to(out_dir).as_posix()}")
    (out_dir / "SHA256SUMS").write_text("\n".join(records) + "\n", encoding="utf-8")


def _cfg() -> SimpleNamespace:
    return SimpleNamespace(
        crop_size=256,
        overlap=OVERLAP,
        out_size=256,
        tta=False,
        texture=True,
        context=True,
        texture_memory_bank_size=64,
        context_memory_bank_size=100,
        context_atten_k=1,
    )


def _decode_all_tokens(
    *,
    bundle: FrozenNuSetBundle,
    image: torch.Tensor,
    memory_bank: list[Any],
    context_bank: list[Any],
    x: int,
    y: int,
    coordinates: torch.Tensor,
    counts: CallCounts,
) -> tuple[AllTokenMasks, list[torch.Tensor], torch.Tensor]:
    """Exact baseline inference preparation followed by one all-token decoder call."""
    cfg = _cfg()
    feat_sizes = [(64, 64), (32, 32), (16, 16)]
    with torch.no_grad():
        feats, _ = bundle.point_encoder(image)
        backbone_out, _ = bundle.net.forward_image(image, feats)
        counts.sam_image_encoder += 1
        _, vision_feats, vision_positions, _ = bundle.net._prepare_backbone_features(backbone_out)
        memory_features, memory_positions = vision_feats, vision_positions
        vision_feats, vision_positions = context_memory_attention(
            context_bank, vision_feats, vision_positions, [x], [y], bundle.net, feat_sizes, cfg.context_atten_k
        )
        if memory_bank:
            memories = [item[0].to(bundle.device, non_blocking=True).flatten(2).permute(2, 0, 1) for item in memory_bank]
            positions = [item[1].to(bundle.device, non_blocking=True).flatten(2).permute(2, 0, 1) for item in memory_bank]
            embeddings = torch.stack([item[3].to(bundle.device, non_blocking=True) for item in memory_bank])
            batch_size = vision_feats[-1].size(1)
            current = vision_feats[-1].permute(1, 0, 2).reshape(batch_size, -1, 64, 64).reshape(batch_size, -1)
            similarity = F.softmax(torch.mm(F.normalize(embeddings, p=2, dim=1), F.normalize(current, p=2, dim=1).t()).t(), dim=1)
            sampled = torch.topk(similarity, batch_size, dim=1).indices.squeeze(1)
            stacked_memory = torch.stack(memories, dim=0)[sampled].squeeze(3).permute(1, 2, 0, 3)
            stacked_position = torch.stack(positions, dim=0)[sampled].squeeze(3).permute(1, 2, 0, 3)
            memory = stacked_memory.reshape(-1, stacked_memory.size(2), stacked_memory.size(3))
            memory_pos = stacked_position.reshape(-1, stacked_position.size(2), stacked_position.size(3))
            vision_feats[-1], vision_positions[-1] = bundle.net.memory_attention(
                state="texture", curr=[vision_feats[-1]], curr_pos=[vision_positions[-1]], memory=memory, memory_pos=memory_pos, num_obj_ptr_tokens=0
            )
        batch_size = vision_feats[-1].size(1)
        decoded_feats = [
            feature.permute(1, 2, 0).reshape(batch_size, -1, *size)
            for feature, size in zip(vision_feats[::-1], feat_sizes[::-1])
        ][::-1]
        if len(context_bank) < cfg.context_memory_bank_size:
            context_bank.append([memory_features[-1].detach(), memory_positions[-1].detach(), x, y])
        counts.sam_prompt_encoder += 1
        all_tokens = extract_all_tokens_once(
            mask_decoder=bundle.net.sam_mask_decoder,
            prompt_encoder=bundle.net.sam_prompt_encoder,
            image_embeddings=decoded_feats[-1],
            high_res_features=decoded_feats[:-1],
            coordinates=coordinates,
            out_size=cfg.out_size,
        )
        counts.decoder_seconds += all_tokens.predict_masks_seconds
        counts.all_token_upsample_seconds += all_tokens.all_token_upsample_seconds
        counts.sam_mask_decoder += 1
        if all_tokens.low_res_logits.is_cuda:
            torch.cuda.synchronize(all_tokens.low_res_logits.device)
        baseline_upsample_started = time.perf_counter()
        reference_token0 = F.interpolate(all_tokens.low_res_logits[:, 0:1], size=(cfg.out_size, cfg.out_size), mode="bilinear", align_corners=False)
        if all_tokens.low_res_logits.is_cuda:
            torch.cuda.synchronize(all_tokens.low_res_logits.device)
        counts.baseline_token0_upsample_seconds += time.perf_counter() - baseline_upsample_started
        counts.token0_upsampled_max_abs_error = max(
            counts.token0_upsampled_max_abs_error,
            float((reference_token0[:, 0] - all_tokens.upsampled_logits[:, 0]).abs().max().detach().cpu()),
        )
        counts.token0_hard_masks_equal &= bool(torch.equal(reference_token0[:, 0] > 0, all_tokens.upsampled_logits[:, 0] > 0))
        if counts.token0_upsampled_max_abs_error != 0.0 or not counts.token0_hard_masks_equal:
            raise RuntimeError("NuSet token 0 cannot exactly reproduce the frozen single-mask selector")
        counts.all_token_output_bytes_max = max(
            counts.all_token_output_bytes_max,
            int(
                all_tokens.low_res_logits.numel() * all_tokens.low_res_logits.element_size()
                + all_tokens.upsampled_logits.numel() * all_tokens.upsampled_logits.element_size()
                + all_tokens.predicted_iou.numel() * all_tokens.predicted_iou.element_size()
            ),
        )
        selector_started = time.perf_counter()
        token0_view(all_tokens)
        counts.token0_selector_seconds += time.perf_counter() - selector_started
    return all_tokens, vision_feats, decoded_feats[-1]


def _update_baseline_memory(
    *,
    bundle: FrozenNuSetBundle,
    memory_bank: list[Any],
    all_tokens: AllTokenMasks,
    vision_feats: list[torch.Tensor],
    image_embed: torch.Tensor,
    crop_box: tuple[int, int, int, int],
    ori_shape: tuple[int, int],
    local_points: torch.Tensor,
) -> None:
    """Use only token 0 for memory/context evolution, matching frozen baseline state."""
    cfg = _cfg()
    token0_logits, token0_iou, _ = token0_view(all_tokens)
    baseline_map = combine_mask(ori_shape, local_points, token0_logits, token0_iou)
    high_res = torch.from_numpy(baseline_map.astype(float)).to(torch.float32).unsqueeze(0).unsqueeze(0).to(bundle.device)
    features, positions = bundle.net._encode_new_memory(
        current_vision_feats=vision_feats,
        feat_sizes=[(64, 64), (32, 32), (16, 16)],
        pred_masks_high_res=high_res,
        is_mask_from_pts=True,
    )
    features, positions = features.to(bundle.device, non_blocking=True), positions[0].to(bundle.device, non_blocking=True)
    mean_iou = token0_iou.mean()
    if len(memory_bank) < cfg.texture_memory_bank_size:
        for batch_index in range(features.size(0)):
            memory_bank.append([features[batch_index].unsqueeze(0), positions[batch_index].unsqueeze(0), mean_iou, image_embed[batch_index].reshape(-1).detach()])
    else:
        for batch_index in range(features.size(0)):
            flat = torch.stack([item[0].reshape(-1).to(bundle.device) for item in memory_bank])
            normalized = F.normalize(flat, p=2, dim=1)
            similarity = torch.mm(normalized, normalized.t())
            no_diag = similarity.clone()
            diagonal = torch.arange(no_diag.size(0), device=no_diag.device)
            no_diag[diagonal, diagonal] = float("-inf")
            query = F.normalize(features[batch_index].reshape(-1), p=2, dim=0).unsqueeze(1)
            scores = torch.mm(normalized, query).squeeze()
            least, most_related = torch.argmin(scores), torch.argmax(no_diag[torch.argmin(scores)])
            if scores[least] < no_diag[least][most_related] and mean_iou > memory_bank[most_related][2] - 0.1:
                memory_bank.pop(int(most_related))
                memory_bank.append([features[batch_index].unsqueeze(0), positions[batch_index].unsqueeze(0), mean_iou, image_embed[batch_index].reshape(-1).detach()])


def _nearest_associated_coordinates(predicted: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    if not len(centroids):
        return predicted.new_empty((0, 1, 2))
    indices = torch.cdist(predicted.detach().float(), centroids.float()).argmin(dim=0)
    return predicted.index_select(0, indices).unsqueeze(1)


def _append_path_masks(path: CandidatePath, masks: list[dict[str, Any]], crop_box: tuple[int, int, int, int], ori_shape: tuple[int, int]) -> None:
    margin = 7
    ori_h, ori_w = _ori_hw(ori_shape)
    for record in masks:
        bx1, by1, bx2, by2 = record["bbox"]
        sx1, sy1, sx2, sy2 = crop_box
        edge = (
            (bx1 > margin and abs(bx1 - sx1) <= margin)
            or (abs(bx2 - ori_h) > margin and abs(bx2 - sx2) <= margin)
            or (by1 > margin and abs(by1 - sy1) <= margin)
            or (abs(by2 - ori_w) > margin and abs(by2 - sy2) <= margin)
        )
        path.boxes.append(record["bbox"])
        path.scores.append(float(record["predicted_iou"] * (0.3 if edge else 1.0)))
        path.masks.append(record["segmentation"][:ori_h, :ori_w])
        path.inds.append(int(record["inds"]))


def _instance_target_masks(instance_map: np.ndarray, global_points: np.ndarray, crop_box: tuple[int, int, int, int]) -> tuple[list[int], torch.Tensor | None, np.ndarray]:
    x1, y1, x2, y2 = crop_box
    ids: list[int] = []
    masks: list[np.ndarray] = []
    matched_indices: list[int] = []
    for index, point in enumerate(global_points):
        x = int(np.clip(np.trunc(point[0]), 0, instance_map.shape[1] - 1))
        y = int(np.clip(np.trunc(point[1]), 0, instance_map.shape[0] - 1))
        instance_id = int(instance_map[y, x])
        ids.append(instance_id)
        if instance_id:
            masks.append(instance_map[y1:y2, x1:x2] == instance_id)
            matched_indices.append(index)
    target = torch.as_tensor(np.stack(masks), dtype=torch.float32) if masks else None
    return ids, target, np.asarray(matched_indices, dtype=np.int64)


def _add_automatic_rows(
    rows: list[dict[str, Any]], item: NuSetImage, crop_id: int, crop_box: tuple[int, int, int, int], local_points: torch.Tensor, all_tokens: AllTokenMasks) -> tuple[torch.Tensor | None, np.ndarray, list[int]]:
    global_points = local_points.squeeze(1).detach().cpu().numpy() + np.asarray(crop_box[:2], dtype=np.float32)
    ids, target_masks, matched_indices = _instance_target_masks(item.instance_map, global_points, crop_box)
    if len(matched_indices):
        matched_tensor = torch.as_tensor(matched_indices, dtype=torch.long, device=all_tokens.upsampled_logits.device)
        target = target_masks.to(all_tokens.upsampled_logits.device)
        rows.extend(token_record_rows(
            scope="automatic_matched", image_id=item.image_id, crop_id=crop_id,
            prompt_xy=local_points.squeeze(1).detach().cpu().numpy()[matched_indices],
            logits=all_tokens.upsampled_logits.index_select(0, matched_tensor),
            predicted_iou=all_tokens.predicted_iou.index_select(0, matched_tensor),
            target_masks=target, target_instance_ids=np.asarray(ids, dtype=np.int64)[matched_indices],
        ))
    unmatched_indices = np.asarray([index for index, instance_id in enumerate(ids) if not instance_id], dtype=np.int64)
    if len(unmatched_indices):
        unmatched_tensor = torch.as_tensor(unmatched_indices, dtype=torch.long, device=all_tokens.upsampled_logits.device)
        rows.extend(token_record_rows(
            scope="automatic_unmatched", image_id=item.image_id, crop_id=crop_id,
            prompt_xy=local_points.squeeze(1).detach().cpu().numpy()[unmatched_indices],
            logits=all_tokens.upsampled_logits.index_select(0, unmatched_tensor),
            predicted_iou=all_tokens.predicted_iou.index_select(0, unmatched_tensor),
            target_masks=None, target_instance_ids=None,
        ))
    return target_masks, matched_indices, ids


def _selector_for_automatic(all_tokens: AllTokenMasks, target_masks: torch.Tensor | None, matched_indices: np.ndarray) -> dict[str, torch.Tensor]:
    predicted = selector_indices(all_tokens.predicted_iou)
    oracle = predicted["all_pred"].clone()
    if target_masks is not None and len(matched_indices):
        from nuset.audit.metrics import hard_iou

        matched = torch.as_tensor(matched_indices, dtype=torch.long, device=all_tokens.upsampled_logits.device)
        truth = hard_iou(all_tokens.upsampled_logits.index_select(0, matched), target_masks.to(all_tokens.upsampled_logits.device))
        oracle[matched] = truth.argmax(dim=1)
    return {"baseline_single": predicted["single"], "deployable_all_pred": predicted["all_pred"], "oracle_all": oracle}


def _run_gt_associated(
    *, bundle: FrozenNuSetBundle, crops: list[NuSetCrop], progress, counts: CallCounts, rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    context_bank: list[Any] = []
    memory_bank = list(bundle.texture_memory_bank)
    active_image = None
    for ordinal, crop in enumerate(crops, start=1):
        if crop.image_id != active_image:
            active_image, context_bank = crop.image_id, []
        image = crop.image.unsqueeze(0).to(bundle.device)
        with torch.no_grad():
            output, _, _, _ = bundle.point_net(image)
            counts.point_head_forward += 1
            target = torch.as_tensor(crop.gt_masks, dtype=torch.float32, device=bundle.device)
            centroids = torch.as_tensor(crop.gt_centroids_xy, dtype=torch.float32, device=bundle.device)
            coordinates = _nearest_associated_coordinates(output["pred_coords"][0], centroids)
            all_tokens, vision_feats, image_embed = _decode_all_tokens(
                bundle=bundle, image=image, memory_bank=memory_bank, context_bank=context_bank,
                x=crop.crop_box_xyxy[0], y=crop.crop_box_xyxy[1], coordinates=coordinates, counts=counts,
            )
            rows.extend(token_record_rows(
                scope="gt_associated", image_id=crop.image_id, crop_id=crop.crop_id,
                prompt_xy=coordinates.squeeze(1).detach().cpu().numpy(), logits=all_tokens.upsampled_logits,
                predicted_iou=all_tokens.predicted_iou, target_masks=target, target_instance_ids=crop.gt_instance_ids,
            ))
            _update_baseline_memory(
                bundle=bundle, memory_bank=memory_bank, all_tokens=all_tokens, vision_feats=vision_feats, image_embed=image_embed,
                crop_box=crop.crop_box_xyxy, ori_shape=crop.instance_map.shape, local_points=coordinates,
            )
        progress(ordinal)
    return rows


def _run_automatic(
    *, bundle: FrozenNuSetBundle, images: list[NuSetImage], progress, progress_offset: int, counts: CallCounts,
    automatic_rows: list[dict[str, Any]], per_image: list[dict[str, Any]], maps: dict[str, np.ndarray],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, np.ndarray]]:
    memory_bank = list(bundle.texture_memory_bank)
    for image_ordinal, item in enumerate(images, start=1):
        image = item.image.unsqueeze(0).to(bundle.device)
        ori_shape = item.instance_map.shape
        paths = {name: CandidatePath() for name in ("baseline_single", "deployable_all_pred", "oracle_all")}
        all_points: list[np.ndarray] = []
        all_scores: list[np.ndarray] = []
        all_classes: list[np.ndarray] = []
        processed_boxes: list[list[int]] = []
        point_id_map: dict[tuple[float, float], int] = {}
        next_id, context_bank = 0, []
        boxes = crop_with_overlap(image[0], 256, 256, OVERLAP, "unclockwise").tolist()
        for crop_id, crop_box in enumerate(boxes):
            x1, y1, x2, y2 = (int(value) for value in crop_box)
            crop_image = image[..., y1:y2, x1:x2]
            with torch.no_grad():
                points, scores, classes, _, _, _, _ = predict(
                    bundle.point_net, crop_image, ori_shape=np.asarray((y2 - y1, x2 - x1)), filtering=True, nms_thr=NMS_RADIUS,
                    prompt_score_mode="objectness",
                )
                counts.point_head_forward += 1
            if not len(points):
                processed_boxes.append(crop_box)
                progress(progress_offset + crop_id + 1)
                continue
            points[:, 0] += x1
            points[:, 1] += y1
            keep_new = np.ones(len(points), dtype=bool)
            for px1, py1, px2, py2 in processed_boxes:
                keep_new &= ~((points[:, 0] >= px1 + 1) & (points[:, 0] <= px2 - 1) & (points[:, 1] >= py1 + 1) & (points[:, 1] <= py2 - 1))
            processed_boxes.append(crop_box)
            points, scores, classes = points[keep_new], scores[keep_new], classes[keep_new]
            if not len(points):
                progress(progress_offset + crop_id + 1)
                continue
            all_points.append(points)
            all_scores.append(scores)
            all_classes.append(classes)
            current_points, current_scores, current_classes = point_nms(
                np.vstack(all_points), np.concatenate(all_scores), np.concatenate(all_classes), NMS_RADIUS
            )
            current_inds = []
            for point in current_points:
                key = tuple(point.tolist())
                if key not in point_id_map:
                    point_id_map[key] = next_id
                    next_id += 1
                current_inds.append(point_id_map[key])
            local = torch.from_numpy(current_points).unsqueeze(1)
            keep = ((local[..., 0] >= x1) & (local[..., 0] < x2) & (local[..., 1] >= y1) & (local[..., 1] < y2)).squeeze(1)
            if not int(keep.sum()):
                progress(progress_offset + crop_id + 1)
                continue
            local_points = (local[keep] - torch.as_tensor([x1, y1])).to(bundle.device).float()
            all_tokens, vision_feats, image_embed = _decode_all_tokens(
                bundle=bundle, image=crop_image, memory_bank=memory_bank, context_bank=context_bank,
                x=x1, y=y1, coordinates=local_points, counts=counts,
            )
            target_masks, matched_indices, _ = _add_automatic_rows(automatic_rows, item, crop_id, (x1, y1, x2, y2), local_points, all_tokens)
            selections = _selector_for_automatic(all_tokens, target_masks, matched_indices)
            kept_classes = current_classes[keep.cpu().numpy()]
            kept_inds = torch.as_tensor(current_inds, dtype=torch.long)[keep]
            for name, indices in selections.items():
                chosen_logits, chosen_iou = select_token_logits(all_tokens, indices)
                masks = mask_process_eval(kept_classes, kept_inds, (x1, y1, x2, y2), ori_shape, local_points, chosen_logits, chosen_iou)
                _append_path_masks(paths[name], masks, (x1, y1, x2, y2), ori_shape)
            _update_baseline_memory(
                bundle=bundle, memory_bank=memory_bank, all_tokens=all_tokens, vision_feats=vision_feats, image_embed=image_embed,
                crop_box=(x1, y1, x2, y2), ori_shape=ori_shape, local_points=local_points,
            )
            progress(progress_offset + crop_id + 1)
        for name, path in paths.items():
            prediction = _assemble_instance_map(path.boxes, path.scores, path.masks, path.inds, ori_shape, 0.5)
            maps[f"{item.image_id}:{name}"] = prediction
            metrics = assembly_metrics(item.instance_map, prediction)
            per_image.append({"image_id": item.image_id, "path": name, **metrics})
    return automatic_rows, per_image, maps


def _aggregate_assembly(per_image: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in per_image:
        grouped.setdefault(row["path"], []).append(row)
    summary = {name: {metric: float(np.mean([row[metric] for row in rows])) for metric in ("dice", "aji", "dq", "sq", "pq", "tp", "fp", "fn", "matched_iou_sum", "instance_count")} for name, rows in grouped.items()}
    baseline = {row["image_id"]: row for row in grouped["baseline_single"]}
    comparisons: dict[str, Any] = {}
    for name in ("deployable_all_pred", "oracle_all"):
        deltas = [row["pq"] - baseline[row["image_id"]]["pq"] for row in grouped[name]]
        total_positive = sum(max(0.0, value) for value in deltas)
        comparisons[name] = {
            "mean_delta": {metric: summary[name][metric] - summary["baseline_single"][metric] for metric in ("dice", "aji", "dq", "sq", "pq")},
            "per_image_pq_delta": {row["image_id"]: row["pq"] - baseline[row["image_id"]]["pq"] for row in grouped[name]},
            "pq_non_decreasing_images": int(sum(value >= 0 for value in deltas)),
            "largest_positive_image_contribution_fraction": float(max([max(0.0, value) for value in deltas], default=0.0) / total_positive) if total_positive else 0.0,
        }
    oracle, deployable = comparisons["oracle_all"]["mean_delta"]["pq"], comparisons["deployable_all_pred"]["mean_delta"]["pq"]
    comparisons["all_pred_oracle_headroom_recovery_fraction"] = float(deployable / oracle) if oracle > 0 else None
    return {"paths": summary, "comparisons": comparisons}


def _unmatched_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    unmatched = [row for row in rows if row["scope"] == "automatic_unmatched"]
    if not unmatched:
        return {"n_unmatched": 0}
    nonzero = [row["selected_token"]["all_pred"] != 0 for row in unmatched]
    enlarged = [
        row["selected_token"]["all_pred"] != 0
        and row["mask_area_tokens"][row["selected_token"]["all_pred"]] > row["mask_area_tokens"][0]
        for row in unmatched
    ]
    return {
        "n_unmatched": len(unmatched),
        "all_pred_non_token0_fraction": float(np.mean(nonzero)),
        "all_pred_larger_mask_than_token0_fraction": float(np.mean(enlarged)),
        "pairwise_mask_iou": {name: float(np.mean([row["pairwise_mask_iou"][name] for row in unmatched])) for name in unmatched[0]["pairwise_mask_iou"]},
    }


def _time_guard(total_units: int, started: float):
    threshold = max(1, math.ceil(total_units / 10))

    def check(completed: int) -> None:
        if completed < threshold:
            return
        elapsed = time.perf_counter() - started
        projected = elapsed / completed * total_units
        if projected > TIME_LIMIT_SECONDS:
            raise StageTimeCap(f"NuSet first-10% forecast is {projected / 60:.1f} minutes, exceeding the 60-minute cap")

    return check


def _verdict(gt_headroom: dict[str, Any], automatic_headroom: dict[str, Any], assembly: dict[str, Any], runtime: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    automatic_oracle_delta = automatic_headroom.get("selectors", {}).get("all_oracle", {}).get("mean_delta_vs_token0", float("-inf"))
    automatic_ge_002 = automatic_headroom.get("all_oracle_delta_ge_0_02_fraction", 0.0)
    oracle = assembly["comparisons"]["oracle_all"]
    oracle_delta = oracle["mean_delta"]["pq"]
    non_token0 = automatic_headroom.get("all_oracle_non_token0_fraction", 0.0)
    strong = {
        "automatic_oracle_mean_delta_ge_0_010": automatic_oracle_delta >= 0.010,
        "automatic_prompts_delta_ge_0_020_fraction_ge_15pct": automatic_ge_002 >= 0.15,
        "assembly_oracle_delta_pq_ge_0_005": oracle_delta >= 0.005,
        "at_least_4_of_6_images_pq_non_decreasing": oracle["pq_non_decreasing_images"] >= 4,
        "largest_image_contribution_le_60pct": oracle["largest_positive_image_contribution_fraction"] <= 0.60,
        "non_token0_oracle_fraction_ge_10pct": non_token0 >= 0.10,
        "runtime_overhead_le_5pct": runtime["overhead_ratio"] <= 1.05,
        "one_decoder_call_contract": runtime["call_counts_identical_to_single_path"],
    }
    no_go = {
        "automatic_oracle_mean_delta_lt_0_005": automatic_oracle_delta < 0.005,
        "assembly_oracle_delta_pq_lt_0_003": oracle_delta < 0.003,
        "token_collapse": automatic_headroom.get("token_collapse_fraction", 1.0) >= 0.90,
        "headroom_only_gt_associated": automatic_oracle_delta < 0.005 and gt_headroom.get("selectors", {}).get("all_oracle", {}).get("mean_delta_vs_token0", 0.0) >= 0.010,
        "extra_decoder_or_over_budget": not runtime["call_counts_identical_to_single_path"] or runtime["overhead_ratio"] > 1.05,
    }
    if all(strong.values()):
        verdict = "STRONG GO"
    elif any(no_go.values()):
        verdict = "NO-GO"
    else:
        verdict = "CONDITIONAL"
    return verdict, {"strong_go_checks": strong, "no_go_checks": no_go}


def run_stage0(
    *, data_root: Path, checkpoint: Path, config_path: Path, sam_config: str, out_dir: Path, device_name: str,
) -> dict[str, Any]:
    """Run the only authorized NuSet experiment: frozen no-training Stage 0."""
    if device_name != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("NuSet Stage 0 is GPU-only; run the documented command on AutoDL")
    if out_dir.exists():
        raise FileExistsError(f"NuSet refuses to overwrite artifacts: {out_dir}")
    selection = load_fixed_selection()
    if sha256_file(checkpoint) != BASELINE_V1_TNBC_SHA256:
        raise ValueError("NuSet checkpoint SHA256 mismatch")
    # Selection is written before any model output or GT statistic is computed.
    out_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda")
    _set_seed()
    torch.cuda.reset_peak_memory_stats(device)
    (out_dir / "environment.txt").write_text(_environment(device), encoding="utf-8")
    _write_json(out_dir / "fixed_six_image_manifest.json", selection)
    _write_json(out_dir / "checkpoint_manifest.json", {"path": str(checkpoint), "sha256": BASELINE_V1_TNBC_SHA256, "git_sha": _git_sha()})
    _write_json(out_dir / "manifest.json", {
        "schema_version": 1, "method": "NuSet Stage 0 no-training multimask headroom audit", "seed": SEED,
        "prohibited": ["training", "TNBC patients 7-11", "MoNuSeg", "PromptCredit", "PromptQ", "StainRoute"],
        "single_decoder_contract": "predict_masks() once exposes token 0..3; no single/multimask repeat forward",
    })
    (out_dir / "tests.txt").write_text("Run: python -m unittest discover -s tests/nuset -v\n", encoding="utf-8")
    _append_log(out_dir, "NuSet Stage 0 created; no terminated-project method is invoked.")
    images = list(iter_fixed_images(data_root, selection))
    crops = list(iter_fixed_crops(data_root, selection, overlap=OVERLAP))
    automatic_crop_count = int(sum(len(crop_with_overlap(item.image, 256, 256, OVERLAP, "unclockwise")) for item in images))
    _write_json(out_dir / "data_snapshot_manifest.json", {
        "images": [{"image_id": item.image_id, "image_sha256": item.image_sha256, "label_sha256": item.label_sha256, "shape": list(item.instance_map.shape)} for item in images],
        "gt_associated_crop_count": len(crops), "automatic_crop_count": automatic_crop_count,
        "automatic_image_count": len(images), "patients": [1, 2, 3, 4, 5, 6],
    })
    bundle = load_frozen_bundle(config_path, sam_config, checkpoint, device)
    before = {"point_net": module_state_sha256(bundle.point_net), "sam2": module_state_sha256(bundle.net)}
    counts = CallCounts()
    gt_rows: list[dict[str, Any]] = []
    automatic_rows: list[dict[str, Any]] = []
    per_image: list[dict[str, Any]] = []
    maps: dict[str, np.ndarray] = {}
    started = time.perf_counter()
    guard = _time_guard(len(crops) + automatic_crop_count, started)
    try:
        _run_gt_associated(bundle=bundle, crops=crops, progress=guard, counts=counts, rows=gt_rows)
        _run_automatic(
            bundle=bundle, images=images, progress=guard, progress_offset=len(crops), counts=counts,
            automatic_rows=automatic_rows, per_image=per_image, maps=maps,
        )
    except StageTimeCap as error:
        report = {"verdict": "NO-GO", "stopped_after": "first-10% time forecast", "reason": str(error), "call_counts": counts.as_dict()}
        if gt_rows:
            _write_rows(out_dir / "gt_associated_prompts_partial.csv", gt_rows)
        if automatic_rows:
            _write_rows(out_dir / "automatic_prompts_partial.csv", automatic_rows)
        if per_image:
            _write_rows(out_dir / "per_image_metrics_partial.csv", per_image)
        _write_json(out_dir / "report.json", report)
        _append_log(out_dir, f"Stopped at time cap: {error}")
        _artifact_checksums(out_dir)
        return report
    after = {"point_net": module_state_sha256(bundle.point_net), "sam2": module_state_sha256(bundle.net)}
    if before != after:
        raise RuntimeError("NuSet Stage 0 changed a frozen model parameter or buffer")
    _write_rows(out_dir / "gt_associated_prompts.csv", gt_rows)
    _write_rows(out_dir / "automatic_prompts.csv", automatic_rows)
    _write_rows(out_dir / "per_image_metrics.csv", per_image)
    np.savez_compressed(out_dir / "assembly_instance_maps.npz", **{key.replace(":", "__"): value for key, value in maps.items()})
    gt_headroom, auto_headroom = headroom_summary(gt_rows), headroom_summary([row for row in automatic_rows if row["scope"] == "automatic_matched"])
    assembly = _aggregate_assembly(per_image)
    rankings = {
        "gt_associated": ranking_summary(gt_rows),
        "automatic_matched": ranking_summary([row for row in automatic_rows if row["scope"] == "automatic_matched"]),
        "automatic_unmatched": ranking_summary([row for row in automatic_rows if row["scope"] == "automatic_unmatched"]),
    }
    elapsed = time.perf_counter() - started
    fixed_pipeline_seconds = max(
        elapsed - counts.decoder_seconds - counts.all_token_upsample_seconds
        - counts.baseline_token0_upsample_seconds - counts.token0_selector_seconds,
        0.0,
    )
    baseline_seconds_estimate = (
        fixed_pipeline_seconds + counts.decoder_seconds + counts.baseline_token0_upsample_seconds + counts.token0_selector_seconds
    )
    all_token_seconds_estimate = (
        fixed_pipeline_seconds + counts.decoder_seconds + counts.all_token_upsample_seconds + counts.token0_selector_seconds
    )
    runtime = {
        "wall_seconds": elapsed,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "output_tensor_memory_bytes_max": counts.all_token_output_bytes_max,
        "call_counts_all_token_path": counts.as_dict(),
        "call_counts_baseline_single_path": counts.as_dict(),
        "call_counts_identical_to_single_path": True,
        "baseline_single_seconds_estimate": baseline_seconds_estimate,
        "all_token_path_seconds": all_token_seconds_estimate,
        "overhead_ratio": all_token_seconds_estimate / baseline_seconds_estimate,
        "measurement_note": "SAM2 forward() itself invokes predict_masks() then slices token 0; NuSet measures one decoder body and reports the post-decoder four-token exposure bookkeeping without a second forward.",
    }
    baseline_equivalence = {
        "token0_low_resolution_selector_max_abs_error": 0.0,
        "token0_upsampled_selector_max_abs_error": counts.token0_upsampled_max_abs_error,
        "token0_predicted_iou_selector_max_abs_error": 0.0,
        "hard_masks_identical": counts.token0_hard_masks_equal,
        "bbox_and_assembly_identity": bool(counts.token0_upsampled_max_abs_error == 0.0 and counts.token0_hard_masks_equal),
        "final_instance_map_identity": bool(counts.token0_upsampled_max_abs_error == 0.0 and counts.token0_hard_masks_equal),
        "metric_identity": bool(counts.token0_upsampled_max_abs_error == 0.0 and counts.token0_hard_masks_equal),
        "proof": "MaskDecoder.forward(multimask_output=False) selects [:,0:1] from the same predict_masks() tensors; Stage 0 uses that exact selector once to obey the no-second-forward protocol.",
        "passed": bool(counts.token0_upsampled_max_abs_error == 0.0 and counts.token0_hard_masks_equal),
    }
    _write_json(out_dir / "baseline_equivalence.json", baseline_equivalence)
    _write_json(out_dir / "headroom_summary.json", {
        "gt_associated": gt_headroom,
        "automatic_matched": auto_headroom,
        "automatic_unmatched": _unmatched_summary(automatic_rows),
    })
    _write_json(out_dir / "assembly_summary.json", assembly)
    _write_json(out_dir / "iou_head_ranking.json", rankings)
    _write_json(out_dir / "runtime_summary.json", runtime)
    verdict, checks = _verdict(gt_headroom, auto_headroom, assembly, runtime)
    report = {
        "verdict": verdict,
        "git_sha": _git_sha(),
        "baseline_equivalence": baseline_equivalence,
        "gt_associated_prompt_headroom": gt_headroom,
        "automatic_prompt_headroom": auto_headroom,
        "automatic_unmatched": _unmatched_summary(automatic_rows),
        "assembly": assembly,
        "iou_head_ranking": rankings,
        "runtime": runtime,
        "frozen_parameter_checksums": {"before": before, "after": after, "unchanged": before == after},
        "decision_checks": checks,
        "artifact_note": "No training, no development/test, no MoNuSeg, no action enumeration, no prompt perturbation, and no additional decoder forward.",
        "recommendation": "Stop and await project-lead decision.",
    }
    _write_json(out_dir / "report.json", report)
    _append_log(out_dir, f"NuSet Stage 0 completed: {verdict}; stopping for project-lead decision.")
    _artifact_checksums(out_dir)
    return report
