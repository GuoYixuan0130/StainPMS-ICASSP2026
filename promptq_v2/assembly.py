"""Canonical StainPMS crop/context/mask assembly primitives without CLI imports."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from torchvision.ops.boxes import batched_nms

from sam2_train.utils.amg import (
    MaskData,
    batched_mask_to_box,
    calculate_stability_score,
    mask_to_rle_pytorch,
    rle_to_mask,
    uncrop_boxes_xyxy,
    uncrop_masks,
    uncrop_points,
)


def crop_with_overlap(image: torch.Tensor, split_width: int = 256, split_height: int = 256, overlap: int = 32, load: str = "unclockwise") -> np.ndarray:
    """Exact canonical crop traversal, copied to avoid importing cfg.parse_args."""
    def start_points(size: int, split_size: int, overlap_pixels: int) -> list[int]:
        points, counter, stride = [0], 1, 256 - overlap_pixels
        while True:
            point = stride * counter
            if point + split_size >= size:
                if split_size != size:
                    points.append(size - split_size)
                break
            points.append(point)
            counter += 1
        return points
    _, image_h, image_w = image.shape
    x_points, y_points = start_points(image_w, split_width, overlap), start_points(image_h, split_height, overlap)
    boxes: list[list[int]] = []
    if load == "sequence":
        for x in x_points:
            for y in y_points:
                boxes.append([x, y, min(x + split_width, image_w), min(y + split_height, image_h)])
    elif load == "unsequence":
        forward = True
        for x in x_points:
            for y in (y_points if forward else np.flip(y_points)):
                boxes.append([x, y, min(x + split_width, image_w), min(y + split_height, image_h)])
            forward = not forward
    elif load in ("clockwise", "unclockwise"):
        top, bottom, left, right = 0, len(y_points) - 1, 0, len(x_points) - 1
        while top <= bottom or left <= right:
            if top <= bottom:
                for y in range(left, right + 1):
                    boxes.append([x_points[top], y_points[y], min(x_points[top] + split_width, image_w), min(y_points[y] + split_height, image_h)])
                top += 1
            if left <= right:
                for x in range(top, bottom + 1):
                    boxes.append([x_points[x], y_points[right], min(x_points[x] + split_width, image_w), min(y_points[right] + split_height, image_h)])
                right -= 1
            if top <= bottom:
                for y in np.flip(range(left, right + 1)):
                    boxes.append([x_points[bottom], y_points[y], min(x_points[bottom] + split_width, image_w), min(y_points[y] + split_height, image_h)])
                bottom -= 1
            if left <= right:
                for x in np.flip(range(top, bottom + 1)):
                    boxes.append([x_points[x], y_points[left], min(x_points[x] + split_width, image_w), min(y_points[left] + split_height, image_h)])
                left += 1
        if load == "unclockwise":
            boxes = boxes[::-1]
    else:
        raise ValueError(f"unsupported crop traversal: {load}")
    return np.asarray(boxes)


def context_memory_attention(context_bank: list[Any], feats: list[torch.Tensor], feats_pos: list[torch.Tensor], xs: list[int], ys: list[int], net: Any, k: int = 1) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    batch_size, device = feats[-1].size(1), feats[-1].device
    if not context_bank:
        zero = torch.zeros(1, batch_size, net.hidden_dim, device=device)
        feats[-1], feats_pos[-1] = feats[-1] + zero, feats_pos[-1] + zero
        return feats, feats_pos
    choices = [[] for _ in range(batch_size)]
    for feature, position, x, y in context_bank:
        for index in range(batch_size):
            choices[index].append([feature.to(device, non_blocking=True), position.to(device, non_blocking=True), math.sqrt((x - xs[index]) ** 2 + (y - ys[index]) ** 2)])
    for choice in choices:
        choice.sort(key=lambda item: item[2])
    for index in range(min(k, len(choices[0]))):
        memory = torch.stack([choice[index][0] for choice in choices]).transpose(0, 1).squeeze(2)
        memory_pos = torch.stack([choice[index][1] for choice in choices]).transpose(0, 1).squeeze(2)
        feats[-1], feats_pos[-1] = net.memory_attention(state="context", curr=feats[-1], curr_pos=feats_pos[-1], memory=memory, memory_pos=memory_pos, num_obj_ptr_tokens=0)
    return feats, feats_pos


def _ori_hw(shape: Any) -> tuple[int, int]:
    array = shape.detach().cpu().numpy() if torch.is_tensor(shape) else np.asarray(shape)
    array = array.reshape(-1)
    return int(array[0]), int(array[1])


def combine_mask(ori_shape: Any, points: torch.Tensor, pred: torch.Tensor, iou_predictions: torch.Tensor, mask_threshold: float = 0.0, box_nms_thresh: float = 1.0) -> np.ndarray:
    """Canonical post-decoder crop mask used only for baseline texture memory."""
    if pred.shape[0] == 0:
        return np.zeros(pred.shape[-2:], dtype=float)
    data = MaskData(masks=pred, iou_preds=iou_predictions, points=points, categories=np.ones(points.shape[0], dtype=np.int64), inds=torch.arange(points.shape[0], dtype=torch.int64, device=points.device))
    data["masks"] = data["masks"] > mask_threshold
    data["boxes"] = batched_mask_to_box(data["masks"])
    data["rles"] = mask_to_rle_pytorch(data["masks"])
    del data["masks"]
    data.filter(batched_nms(data["boxes"].float(), data["iou_preds"], torch.zeros_like(data["boxes"][:, 0]), iou_threshold=box_nms_thresh))
    masks = [rle_to_mask(rle) for rle in data["rles"]]
    scores, inds = torch.as_tensor(data["iou_preds"]), np.asarray(data["inds"])
    keep = np.ones(len(inds), dtype=bool)
    for value in np.unique(inds):
        duplicate = np.flatnonzero(inds == value)
        if len(duplicate) > 1:
            keep[np.delete(duplicate, np.argmax(scores[duplicate]))] = False
    crop_h, crop_w = _ori_hw(ori_shape)
    output = np.zeros((pred.shape[1], pred.shape[2]), dtype=float)
    for index in np.flatnonzero(keep):
        mask = masks[index][:crop_h, :crop_w]
        if output[mask].all() == 0:
            output[mask] = index + 1
    return output


def mask_process_eval(cell_types: np.ndarray, sub_inds: torch.Tensor, crop_box: list[int], ori_shape: Any, points: torch.Tensor, pred: torch.Tensor, iou_predictions: torch.Tensor, mask_threshold: float = 0.0, stability_score_offset: float = 1.0, box_nms_thresh: float = 1.0) -> list[dict]:
    if pred.shape[0] == 0:
        return []
    orig_h, orig_w = _ori_hw(ori_shape)
    data = MaskData(masks=pred, iou_preds=iou_predictions, points=points, categories=cell_types, inds=sub_inds)
    data["stability_score"] = calculate_stability_score(data["masks"], mask_threshold, stability_score_offset)
    data["masks"] = data["masks"] > mask_threshold
    data["boxes"] = batched_mask_to_box(data["masks"])
    data["masks"] = uncrop_masks(data["masks"], crop_box, orig_h, orig_w)
    data["rles"] = mask_to_rle_pytorch(data["masks"])
    del data["masks"]
    data.filter(batched_nms(data["boxes"].float(), data["iou_preds"], torch.zeros_like(data["boxes"][:, 0]), iou_threshold=box_nms_thresh))
    data["boxes"] = uncrop_boxes_xyxy(data["boxes"], crop_box)
    data["points"] = uncrop_points(data["points"], crop_box)
    data["segmentations"] = [rle_to_mask(rle) for rle in data["rles"]]
    return [{"segmentation": data["segmentations"][index], "bbox": data["boxes"][index].tolist(), "predicted_iou": data["iou_preds"][index].item(), "stability_score": data["stability_score"][index].item(), "point": data["points"][index].tolist(), "categories": data["categories"][index].tolist(), "inds": data["inds"][index].tolist()} for index in range(len(data["segmentations"]))]


def assemble_instance_map(all_boxes: list, all_scores: list, all_masks: list, all_inds: list, inst_shape: tuple[int, int], iou_threshold: float, all_records: list[dict] | None = None, return_records: bool = False):
    if not all_masks:
        return (np.zeros(inst_shape, dtype=int), []) if return_records else np.zeros(inst_shape, dtype=int)
    boxes, scores, inds = torch.as_tensor(all_boxes), torch.as_tensor(all_scores), np.asarray(all_inds)
    keep_prior = np.ones(len(inds), dtype=bool)
    for value in np.unique(inds):
        duplicate = np.flatnonzero(inds == value)
        if len(duplicate) > 1:
            keep_prior[np.delete(duplicate, np.argmax(scores[duplicate]))] = False
    original = np.flatnonzero(keep_prior)
    boxes, scores, masks = boxes[torch.from_numpy(keep_prior)], scores[torch.from_numpy(keep_prior)], [all_masks[index] for index in original]
    categories = torch.zeros_like(boxes if boxes.ndim == 1 else boxes[:, 0])
    keep_by_nms = batched_nms(boxes.float(), scores, categories, iou_threshold=iou_threshold).cpu().numpy()
    output, selected = np.zeros(inst_shape, dtype=int), []
    for instance_id, index in enumerate(keep_by_nms[::-1]):
        if output[masks[index]].all() == 0:
            output[masks[index]] = instance_id + 1
            if return_records and all_records is not None:
                record = dict(all_records[int(original[int(index)])])
                record.update({"source_candidate_index": int(original[int(index)]), "final_id": int(instance_id + 1), "final_area": int(np.asarray(masks[index]).sum())})
                selected.append(record)
    return (output, selected) if return_records else output
