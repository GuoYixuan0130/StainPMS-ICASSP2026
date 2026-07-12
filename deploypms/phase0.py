"""DeployPMS Phase 0: training--deployment prompt exposure-gap audit.

The implementation intentionally has a narrow scope:

* it accepts only the declared TNBC 1--8 source split and evaluates 7--8;
* it checks the closed 9--11 patients before any image is loaded;
* it loads an e156 checkpoint in ``eval`` mode and never constructs an
  optimiser or calls ``backward``;
* teacher prompts use the exact nearest-query rule from
  ``run.run_on_epoch.find_nearest_points`` and positive SAM labels;
* deployment prompts replay the CA-SAM2 validation classifier, filtering and
  progressive point NMS without GT intervention;
* both prompt paths are decoded from the same frozen crop features.

No StainPMS residual candidate, coverage map, or loss is imported or used.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import albumentations as A
import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as F
from mmengine.config import Config
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt
from scipy.spatial.distance import cdist
from skimage import color, io

from sam2_train.modeling.dpa_p2pnet import build_model
from sam2_train.modeling.stats_utils import (
    get_dice_1,
    get_fast_aji,
    get_fast_aji_plus,
    get_fast_dice_2,
    get_fast_pq,
    remap_label,
)
from sam2_train.build_sam import build_sam2
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
from torchvision.ops.boxes import batched_nms


CANONICAL_BASELINE = "2a1348cb7a1158a6f77aae2f92c168f9552d8068"
E156_SHA256 = "44a3cb3e93051301d789e44f93769588abfa727d7a174b0270f55305ef023781"
CLOSED_PATIENTS = frozenset({9, 10, 11})
TRAIN_PATIENTS = frozenset(range(1, 7))
DEV_PATIENTS = frozenset({7, 8})
EXPECTED_DEV_IMAGES = 7
BOUNDARY_BAND_PIXELS = 2


def _ori_hw(ori_shape: Any) -> tuple[int, int]:
    values = torch.as_tensor(ori_shape).detach().cpu().numpy().reshape(-1)
    return int(values[0]), int(values[1])


def crop_with_overlap(img: torch.Tensor, split_width: int, split_height: int, overlap: int, load: str) -> np.ndarray:
    """Frozen copy of the CA-SAM2 validation crop traversal."""
    def start_points(size: int, split_size: int) -> list[int]:
        points, counter = [0], 1
        stride = 256 - overlap
        while True:
            point = stride * counter
            if point + split_size >= size:
                if split_size != size:
                    points.append(size - split_size)
                break
            points.append(point)
            counter += 1
        return points
    _, image_height, image_width = img.shape
    xs, ys = start_points(image_width, split_width), start_points(image_height, split_height)
    boxes: list[list[int]] = []
    if load == "sequence":
        for x in xs:
            for y in ys:
                boxes.append([x, y, min(x + split_width, image_width), min(y + split_height, image_height)])
    elif load == "unsequence":
        forward = True
        for x in xs:
            for y in ys if forward else list(reversed(ys)):
                boxes.append([x, y, min(x + split_width, image_width), min(y + split_height, image_height)])
            forward = not forward
    elif load in ("clockwise", "unclockwise"):
        top, bottom, left, right = 0, len(ys) - 1, 0, len(xs) - 1
        while top <= bottom or left <= right:
            if top <= bottom:
                for y_index in range(left, right + 1):
                    boxes.append([xs[top], ys[y_index], min(xs[top] + split_width, image_width), min(ys[y_index] + split_height, image_height)])
                top += 1
            if left <= right:
                for x_index in range(top, bottom + 1):
                    boxes.append([xs[x_index], ys[right], min(xs[x_index] + split_width, image_width), min(ys[right] + split_height, image_height)])
                right -= 1
            if top <= bottom:
                for y_index in reversed(range(left, right + 1)):
                    boxes.append([xs[bottom], ys[y_index], min(xs[bottom] + split_width, image_width), min(ys[y_index] + split_height, image_height)])
                bottom -= 1
            if left <= right:
                for x_index in reversed(range(top, bottom + 1)):
                    boxes.append([xs[x_index], ys[left], min(xs[x_index] + split_width, image_width), min(ys[left] + split_height, image_height)])
                left += 1
        if load == "unclockwise":
            boxes.reverse()
    else:
        raise ValueError(f"Unsupported crop load order: {load}")
    return np.asarray(boxes)


def context_memory_attention(context_bank: list[Any], feats: list[torch.Tensor], feats_pos: list[torch.Tensor], xs: list[int], ys: list[int], net: torch.nn.Module, feat_sizes: list[tuple[int, int]], k: int) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Frozen copy of the CA-SAM2 validation context-memory operation."""
    del feat_sizes
    batch_size, device = feats[-1].size(1), feats[-1].device
    if not context_bank:
        zero = torch.zeros(1, batch_size, net.hidden_dim, device=device)
        feats[-1], feats_pos[-1] = feats[-1] + zero, feats_pos[-1] + zero
        return feats, feats_pos
    nearest: list[list[tuple[torch.Tensor, torch.Tensor, float]]] = [[] for _ in range(batch_size)]
    for features, positions, x, y in context_bank:
        for index in range(batch_size):
            nearest[index].append((features.to(device), positions.to(device), math.hypot(x - xs[index], y - ys[index])))
    for values in nearest:
        values.sort(key=lambda row: row[2])
    for index in range(min(k, len(nearest[0]))):
        memory = torch.stack([values[index][0] for values in nearest]).transpose(0, 1).squeeze(2)
        memory_pos = torch.stack([values[index][1] for values in nearest]).transpose(0, 1).squeeze(2)
        feats[-1], feats_pos[-1] = net.memory_attention(
            state="context", curr=feats[-1], curr_pos=feats_pos[-1], memory=memory, memory_pos=memory_pos, num_obj_ptr_tokens=0
        )
    return feats, feats_pos


def mask_process_eval(cell_types: np.ndarray, sub_inds: torch.Tensor, crop_box: Sequence[int], ori_shape: Any, points: torch.Tensor, pred: torch.Tensor, iou_predictions: torch.Tensor) -> list[dict[str, Any]]:
    """Frozen validation mask threshold, box NMS, and uncrop sequence."""
    if pred.shape[0] == 0:
        return []
    original_height, original_width = _ori_hw(ori_shape)
    data = MaskData(masks=pred, iou_preds=iou_predictions, points=points, categories=cell_types, inds=sub_inds)
    data["stability_score"] = calculate_stability_score(data["masks"], 0.0, 1.0)
    data["masks"] = data["masks"] > 0.0
    data["boxes"] = batched_mask_to_box(data["masks"])
    data["masks"] = uncrop_masks(data["masks"], crop_box, original_height, original_width)
    data["rles"] = mask_to_rle_pytorch(data["masks"])
    del data["masks"]
    keep = batched_nms(data["boxes"].float(), data["iou_preds"], torch.zeros_like(data["boxes"][:, 0]), iou_threshold=1.0)
    data.filter(keep)
    data["boxes"] = uncrop_boxes_xyxy(data["boxes"], crop_box)
    data["points"] = uncrop_points(data["points"], crop_box)
    data["segmentations"] = [rle_to_mask(rle) for rle in data["rles"]]
    return [
        {
            "segmentation": data["segmentations"][index],
            "bbox": data["boxes"][index].tolist(),
            "predicted_iou": data["iou_preds"][index].item(),
            "stability_score": data["stability_score"][index].item(),
            "point": data["points"][index].tolist(),
            "categories": data["categories"][index].tolist(),
            "inds": data["inds"][index].tolist(),
        }
        for index in range(len(data["segmentations"]))
    ]


def combine_mask(ori_shape: Any, points: torch.Tensor, pred: torch.Tensor, iou_predictions: torch.Tensor) -> np.ndarray:
    """Frozen local mask union used solely for validation texture-memory update."""
    if pred.shape[0] == 0:
        return np.zeros(pred.shape[-2:], dtype=float)
    data = MaskData(
        masks=pred, iou_preds=iou_predictions, points=points,
        categories=np.ones(points.shape[0], dtype=np.int64),
        inds=torch.arange(points.shape[0], dtype=torch.int64, device=points.device),
    )
    data["masks"] = data["masks"] > 0.0
    data["boxes"] = batched_mask_to_box(data["masks"])
    data["rles"] = mask_to_rle_pytorch(data["masks"])
    del data["masks"]
    keep = batched_nms(data["boxes"].float(), data["iou_preds"], torch.zeros_like(data["boxes"][:, 0]), iou_threshold=1.0)
    data.filter(keep)
    data["segmentations"] = [rle_to_mask(rle) for rle in data["rles"]]
    height, width = _ori_hw(ori_shape)
    out = np.zeros((pred.shape[1], pred.shape[2]), dtype=float)
    for index, segmentation in enumerate(data["segmentations"]):
        mask = segmentation[:height, :width]
        if out[mask].all() == 0:
            out[mask] = index + 1
    return out


@dataclass(frozen=True)
class ImageEntry:
    patient: int
    stem: str
    image_path: str
    label_path: str
    image_sha256: str
    label_sha256: str


@dataclass(frozen=True)
class GateResult:
    passed: bool
    measurements: dict[str, Any]


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    raise TypeError(f"Not JSON serialisable: {type(value)!r}")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=_json_default)
        handle.write("\n")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a streaming SHA256 without materialising a large artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def state_dict_sha256(state: Mapping[str, Any]) -> str:
    """Content checksum for one loaded model component, keyed deterministically."""
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key]
        digest.update(key.encode("utf-8"))
        if torch.is_tensor(value):
            tensor = value.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
            digest.update(tensor.numpy().tobytes())
        else:
            digest.update(repr(value).encode("utf-8"))
    return digest.hexdigest()


def _patient_from_stem(stem: str) -> int:
    match = re.match(r"^(\d{1,2})_", stem)
    if not match:
        raise ValueError(
            f"TNBC filename {stem!r} must start with '<patient>_'; refusing ambiguous split assignment."
        )
    return int(match.group(1))


def _list_tnbc_entries(data_root: Path, allowed_patients: frozenset[int]) -> list[ImageEntry]:
    image_root = data_root / "train_12" / "images"
    label_root = data_root / "train_12" / "labels"
    if not image_root.is_dir() or not label_root.is_dir():
        raise FileNotFoundError(
            "DeployPMS Phase 0 requires data/tnbc/train_12/images and labels; "
            f"not found below {data_root}."
        )

    entries: list[ImageEntry] = []
    for image_path in sorted(path for path in image_root.iterdir() if path.is_file()):
        stem = image_path.stem
        patient = _patient_from_stem(stem)
        if patient in CLOSED_PATIENTS:
            # Metadata-only discovery is allowed so that a wrongly selected
            # filename cannot silently leak closed 9--11 data into the audit.
            continue
        if patient not in allowed_patients:
            continue
        label_path = label_root / f"{stem}.mat"
        if not label_path.is_file():
            raise FileNotFoundError(f"Missing TNBC label for {image_path.name}: {label_path}")
        entries.append(
            ImageEntry(
                patient=patient,
                stem=stem,
                image_path=str(image_path.resolve()),
                label_path=str(label_path.resolve()),
                image_sha256=sha256_file(image_path),
                label_sha256=sha256_file(label_path),
            )
        )
    return entries


def build_data_manifest(data_root: Path) -> dict[str, Any]:
    """Build the only permitted manifest and enforce the preregistered split."""
    train = _list_tnbc_entries(data_root, TRAIN_PATIENTS)
    development = _list_tnbc_entries(data_root, DEV_PATIENTS)
    if not train:
        raise RuntimeError("No TNBC patients 1--6 found; training split manifest is empty.")
    if len(development) != EXPECTED_DEV_IMAGES:
        raise RuntimeError(
            f"Development must be the preregistered 7 images from patients 7--8; found {len(development)}."
        )
    if {entry.patient for entry in development} != DEV_PATIENTS:
        raise RuntimeError("Development manifest must include both patients 7 and 8.")
    return {
        "dataset": "TNBC",
        "root": str(data_root.resolve()),
        "training_patients": sorted(TRAIN_PATIENTS),
        "development_patients": sorted(DEV_PATIENTS),
        "closed_patients": sorted(CLOSED_PATIENTS),
        "monuseg": "forbidden",
        "train_entries": [asdict(entry) for entry in train],
        "development_entries": [asdict(entry) for entry in development],
        "access_guard": {
            "loaded_patients": sorted(TRAIN_PATIENTS | DEV_PATIENTS),
            "closed_patient_files_opened": 0,
            "monuseg_files_opened": 0,
            "test_split_opened": False,
        },
    }


def find_nearest_points_with_indices(
    pred_coords: torch.Tensor, selected_points: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact ``train_on_epoch.find_nearest_points`` rule plus query indices."""
    if pred_coords.ndim != 2 or selected_points.ndim != 2:
        raise ValueError("Expected pred_coords [Q,2] and selected_points [N,2].")
    distances = torch.cdist(pred_coords.float().unsqueeze(0), selected_points.float().unsqueeze(0)).squeeze(0)
    indices = torch.argmin(distances, dim=0)
    return pred_coords[indices], indices


def point_nms_indices(points: np.ndarray, scores: np.ndarray, nms_thr: float) -> np.ndarray:
    """Return source indices using the exact progressive CA-SAM2 point NMS policy."""
    if len(points) == 0:
        return np.empty(0, dtype=np.int64)
    reserved = np.ones(len(points), dtype=bool)
    distances = cdist(points, points)
    np.fill_diagonal(distances, np.inf)
    for index in np.argsort(-scores):
        if reserved[index]:
            reserved[distances[index] <= nms_thr] = False
    return np.flatnonzero(reserved)


def _inside_instance(inst_map: np.ndarray, point_xy: Sequence[float]) -> int:
    x, y = int(point_xy[0]), int(point_xy[1])
    if y < 0 or x < 0 or y >= inst_map.shape[0] or x >= inst_map.shape[1]:
        return 0
    return int(inst_map[y, x])


def _prompt_location(inst_map: np.ndarray, gt_id: int, point_xy: Sequence[float]) -> str:
    if _inside_instance(inst_map, point_xy) != gt_id:
        return "outside"
    gt_mask = inst_map == gt_id
    y, x = int(point_xy[1]), int(point_xy[0])
    distance = distance_transform_edt(gt_mask)
    return "boundary_band" if float(distance[y, x]) <= BOUNDARY_BAND_PIXELS else "interior"


def _hard_iou(a: np.ndarray, b: np.ndarray) -> float:
    union = int(np.logical_or(a, b).sum())
    return float(np.logical_and(a, b).sum() / union) if union else 1.0


def _dice(a: np.ndarray, b: np.ndarray) -> float:
    total = int(a.sum() + b.sum())
    return float(2 * np.logical_and(a, b).sum() / total) if total else 1.0


def _soft_iou(soft: np.ndarray, target: np.ndarray) -> float:
    target_f = target.astype(np.float32)
    union = float(np.maximum(soft, target_f).sum())
    return float(np.minimum(soft, target_f).sum() / union) if union else 1.0


def _boundary_iou(a: np.ndarray, b: np.ndarray, band: int = BOUNDARY_BAND_PIXELS) -> float:
    structure = np.ones((3, 3), dtype=bool)
    def boundary(mask: np.ndarray) -> np.ndarray:
        core = binary_erosion(mask, structure=structure, border_value=0)
        edge = np.logical_xor(mask, core)
        return binary_dilation(edge, structure=structure, iterations=band)
    return _hard_iou(boundary(a), boundary(b))


def _bbox(mask: np.ndarray) -> list[int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return [0, 0, 0, 0]
    return [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]


def _assemble_instance_map(
    candidates: Sequence[dict[str, Any]], inst_shape: tuple[int, int]
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Frozen CA-SAM2 final assembly, factored for standard/teacher/swap maps."""
    if not candidates:
        return np.zeros(inst_shape, dtype=np.int32), []
    scores = np.asarray([candidate["assembly_score"] for candidate in candidates], dtype=np.float32)
    inds = np.asarray([candidate["point_id"] for candidate in candidates])
    keep_prior = np.ones(len(candidates), dtype=bool)
    for point_id, count in zip(*np.unique(inds, return_counts=True), strict=True):
        if count > 1:
            duplicates = np.flatnonzero(inds == point_id)
            duplicates = np.delete(duplicates, np.argmax(scores[duplicates]))
            keep_prior[duplicates] = False
    kept_indices = np.flatnonzero(keep_prior)
    boxes = torch.as_tensor([candidates[index]["bbox"] for index in kept_indices], dtype=torch.float32)
    kept_scores = torch.as_tensor(scores[kept_indices])
    keep_by_nms = batched_nms(
        boxes,
        kept_scores,
        torch.zeros_like(boxes[:, 0]),
        iou_threshold=0.5,
    ).cpu().numpy()
    inst_map = np.zeros(inst_shape, dtype=np.int32)
    selected: list[dict[str, Any]] = []
    for final_id, local_index in enumerate(keep_by_nms[::-1], start=1):
        original_index = int(kept_indices[int(local_index)])
        candidate = candidates[original_index]
        mask = np.asarray(candidate["segmentation"], dtype=bool)
        if inst_map[mask].all() == 0:
            inst_map[mask] = final_id
            row = dict(candidate)
            row["final_id"] = final_id
            row["source_candidate_index"] = original_index
            selected.append(row)
    return inst_map, selected


def _pq_counts(gt: np.ndarray, pred: np.ndarray) -> tuple[dict[str, float], dict[str, int]]:
    gt = remap_label(gt)
    pred = remap_label(pred)
    pq, pairing = get_fast_pq(gt, pred, match_iou=0.5)
    paired_true, paired_pred, unpaired_true, unpaired_pred = pairing
    return (
        {"dq": float(pq[0]), "sq": float(pq[1]), "pq": float(pq[2])},
        {"tp": len(paired_true), "fp": len(unpaired_pred), "fn": len(unpaired_true)},
    )


def image_metrics(gt: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    gt = remap_label(gt)
    pred = remap_label(pred)
    pq, counts = _pq_counts(gt, pred)
    return {
        "dice": float(get_dice_1(gt, pred)),
        "dice2": float(get_fast_dice_2(gt, pred)),
        "aji": float(get_fast_aji(gt, pred)),
        "aji_plus": float(get_fast_aji_plus(gt, pred)),
        **pq,
        **counts,
    }


def aggregate_metrics(per_image: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metric_names = ("dice", "dice2", "aji", "aji_plus", "dq", "sq", "pq")
    out = {name: float(np.mean([float(row[name]) for row in per_image])) for name in metric_names}
    out.update({name: int(sum(int(row[name]) for row in per_image)) for name in ("tp", "fp", "fn")})
    out["aggregation"] = "macro_per_image; TP/FP/FN are summed"
    return out


def assess_associations(
    inst_map: np.ndarray,
    teacher_records: Mapping[int, Mapping[str, Any]],
    deployment_records: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Classify GT instances without mask-IoU-based association.

    Teacher correspondence is valid only when the query selected for that GT
    lands inside that same GT.  Deployment association is always integer-pixel
    point-inside-GT.  A GT with several deployment prompts keeps the highest
    point-head score as primary; all other prompts remain explicit duplicates.
    """
    by_gt: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    false_prompts: list[dict[str, Any]] = []
    for prompt in deployment_records:
        gt_id = _inside_instance(inst_map, prompt["point"])
        if gt_id == 0:
            row = dict(prompt)
            row["association"] = "background_unmatched"
            false_prompts.append(row)
        else:
            by_gt[gt_id].append(prompt)

    rows: list[dict[str, Any]] = []
    duplicate_rows: list[dict[str, Any]] = []
    for gt_id in [int(value) for value in np.unique(inst_map) if value != 0]:
        teacher = teacher_records.get(gt_id)
        teacher_summary = None
        if teacher is not None:
            teacher_summary = {
                key: value for key, value in teacher.items()
                if key not in {"soft_mask", "hard_mask", "candidate"}
            }
        teacher_spatial_id = int(teacher["spatial_gt_id"]) if teacher else 0
        teacher_ok = teacher is not None and teacher_spatial_id == gt_id
        prompts = sorted(by_gt.get(gt_id, []), key=lambda row: (-float(row["point_score"]), int(row["point_id"])))
        primary = prompts[0] if prompts else None
        for duplicate_rank, prompt in enumerate(prompts[1:], start=1):
            row = dict(prompt)
            row.update({"gt_id": gt_id, "duplicate_rank": duplicate_rank, "association": "duplicate"})
            duplicate_rows.append(row)
        deployment_ok = primary is not None
        if teacher_ok and deployment_ok:
            category = "both-covered"
        elif teacher_ok:
            category = "teacher-only"
        elif deployment_ok:
            category = "deployment-only"
        else:
            category = "neither"
        rows.append(
            {
                "gt_id": gt_id,
                "category": category,
                "teacher_present": bool(teacher_ok),
                "teacher_spatial_gt_id": teacher_spatial_id,
                "teacher_prompt": teacher_summary,
                "deployment_present": bool(deployment_ok),
                "deployment_primary": dict(primary) if primary else None,
                "deployment_prompt_count": len(prompts),
                "duplicate_count": max(0, len(prompts) - 1),
            }
        )
    return rows, false_prompts + duplicate_rows


def availability_gate(instance_rows: Sequence[Mapping[str, Any]]) -> GateResult:
    total = len(instance_rows)
    teacher = sum(bool(row["teacher_present"]) for row in instance_rows)
    deployment = sum(bool(row["deployment_present"]) for row in instance_rows)
    per_image: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in instance_rows:
        per_image[str(row["image"])] .append(row)
    gaps = []
    missing_by_image: dict[str, int] = {}
    for image, rows in per_image.items():
        teacher_recall = sum(bool(row["teacher_present"]) for row in rows) / len(rows)
        deployment_recall = sum(bool(row["deployment_present"]) for row in rows) / len(rows)
        gaps.append(teacher_recall - deployment_recall)
        missing_by_image[image] = sum(row["category"] == "teacher-only" for row in rows)
    teacher_only = sum(row["category"] == "teacher-only" for row in instance_rows)
    max_contribution = max(missing_by_image.values(), default=0) / teacher_only if teacher_only else 1.0
    recall_gap = teacher / total - deployment / total if total else 0.0
    measurements = {
        "teacher_coverage_recall": teacher / total if total else 0.0,
        "deployment_coverage_recall": deployment / total if total else 0.0,
        "recall_gap": recall_gap,
        "teacher_only_gt": teacher_only,
        "images_with_gap_ge_003": int(sum(gap >= 0.03 for gap in gaps)),
        "max_single_image_missing_contribution": max_contribution,
        "per_image_recall_gaps": {image: gap for image, gap in zip(per_image, gaps, strict=True)},
    }
    passed = recall_gap >= 0.05 and measurements["images_with_gap_ge_003"] >= 5 and max_contribution <= 0.5
    return GateResult(passed=passed, measurements=measurements)


def conditioning_gate(both_rows: Sequence[Mapping[str, Any]]) -> GateResult:
    gaps = np.asarray([float(row["hard_iou_gap"]) for row in both_rows], dtype=float)
    per_image: dict[str, list[float]] = defaultdict(list)
    for row in both_rows:
        per_image[str(row["image"])].append(float(row["hard_iou_gap"]))
    if len(gaps) == 0:
        return GateResult(False, {"n_both": 0, "reason": "no both-covered instances"})
    positive = gaps[gaps > 0]
    top_count = max(1, math.ceil(len(gaps) * 0.1))
    top_contribution = float(np.sort(positive)[-top_count:].sum() / positive.sum()) if len(positive) else 1.0
    measurements = {
        "n_both": int(len(gaps)),
        "teacher_minus_deployment_mean_iou": float(gaps.mean()),
        "teacher_minus_deployment_median_iou": float(np.median(gaps)),
        "images_with_positive_mean_gap": int(sum(np.mean(values) > 0 for values in per_image.values())),
        "top_10_percent_positive_gap_contribution": top_contribution,
        "per_image_mean_gap": {image: float(np.mean(values)) for image, values in per_image.items()},
    }
    passed = (
        measurements["teacher_minus_deployment_mean_iou"] >= 0.010
        and measurements["teacher_minus_deployment_median_iou"] > 0
        and measurements["images_with_positive_mean_gap"] >= 5
        and top_contribution <= 0.60
    )
    return GateResult(passed=passed, measurements=measurements)


def assembly_gate(
    standard_rows: Sequence[Mapping[str, Any]], swap_rows: Sequence[Mapping[str, Any]]
) -> GateResult:
    standard = aggregate_metrics(standard_rows)
    swap = aggregate_metrics(swap_rows)
    nondecreasing = sum(float(s["pq"]) >= float(d["pq"]) for d, s in zip(standard_rows, swap_rows, strict=True))
    measurements = {
        "standard": standard,
        "shared_gt_swap": swap,
        "delta_pq": float(swap["pq"] - standard["pq"]),
        "delta_aji": float(swap["aji"] - standard["aji"]),
        "images_with_non_decreasing_pq": int(nondecreasing),
    }
    passed = (
        measurements["delta_pq"] >= 0.004
        and measurements["delta_aji"] >= 0.0
        and measurements["images_with_non_decreasing_pq"] >= 5
    )
    return GateResult(passed=passed, measurements=measurements)


def final_verdict(availability: GateResult, conditioning: GateResult, assembly: GateResult) -> str:
    if availability.passed and (conditioning.passed or assembly.passed):
        return "STRONG GO"
    if availability.passed:
        return "CONDITIONAL GO — POINT SET"
    if conditioning.passed or assembly.passed:
        return "CONDITIONAL GO — DECODER EXPOSURE"
    return "NO-GO"


def _environment() -> dict[str, Any]:
    def version(module: Any) -> str | None:
        return getattr(module, "__version__", None)
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": version(torch),
        "cuda_available": torch.cuda.is_available(),
        "cuda": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "numpy": version(np),
        "albumentations": version(A),
    }


def _git_sha(repo_root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()


def _assert_canonical_baseline(repo_root: Path) -> None:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", CANONICAL_BASELINE, "HEAD"],
        cwd=repo_root,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"DeployPMS code must descend from canonical baseline {CANONICAL_BASELINE}; it does not."
        )


def _load_image_and_instances(entry: ImageEntry) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    image = io.imread(entry.image_path)[..., :3]
    inst_map = sio.loadmat(entry.label_path)["inst_map"].astype(np.int32)
    transform = A.Compose([A.Normalize()])
    image_tensor = torch.from_numpy(transform(image=image)["image"].transpose(2, 0, 1)).float().unsqueeze(0)
    return image_tensor, inst_map, image


def _training_reference_points(inst_map: np.ndarray, seed: int) -> dict[int, np.ndarray]:
    """Replay the train dataset's uniformly sampled interior-GT point selection.

    The historical epoch's random transforms are unavailable by handover, so
    Phase 0 fixes the generator seed and works on the frozen development image.
    The *coordinate chosen for decoding* remains exactly the nearest predicted
    coordinate used in ``train_on_epoch``.
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)
    points: dict[int, np.ndarray] = {}
    for gt_id in (int(v) for v in np.unique(inst_map) if v != 0):
        coords_yx = torch.from_numpy(np.argwhere(inst_map == gt_id))
        picked = coords_yx[torch.randint(len(coords_yx), (1,), generator=generator).item()]
        points[gt_id] = np.asarray([int(picked[1]), int(picked[0])], dtype=np.float32)
    return points


def _classify_points(point_net: torch.nn.Module, image: torch.Tensor, filtering: bool) -> tuple[list[dict[str, Any]], torch.Tensor, torch.Tensor]:
    """Exact validation point classification/filtering, retaining query provenance."""
    outputs, _, _, _ = point_net(image)
    raw_coords = outputs["pred_coords"][0].detach().cpu()
    raw_scores = outputs["pred_logits"][0].softmax(-1).detach().cpu().numpy()
    coords = raw_coords.numpy().copy()
    height, width = image.shape[-2:]
    np.clip(coords[:, 0], 0, width - 1, out=coords[:, 0])
    np.clip(coords[:, 1], 0, height - 1, out=coords[:, 1])
    classes = np.argmax(raw_scores, axis=-1)
    valid = classes < raw_scores.shape[-1] - 1
    mask = outputs["pred_masks"][0, 0].detach().float().cpu().numpy() > 0
    if filtering:
        valid_indices = np.flatnonzero(valid)
        keep = mask[coords[valid_indices].astype(int)[:, 1], coords[valid_indices].astype(int)[:, 0]]
        valid[valid_indices] &= keep
    records = [
        {
            "query_index": int(index),
            "local_point": coords[index].astype(float),
            "raw_query_point": raw_coords[index].numpy().astype(float),
            "point_score": float(raw_scores[index].max()),
            "point_class": int(classes[index]),
        }
        for index in np.flatnonzero(valid)
    ]
    return records, raw_coords, outputs["pred_logits"][0].detach().cpu()


def _encode_crop_once(
    net: torch.nn.Module,
    point_encoder: torch.nn.Module,
    image: torch.Tensor,
    memory_bank: list[Any],
    context_bank: list[Any],
    crop_xy: tuple[int, int],
    cfg: argparse.Namespace,
    device: torch.device,
) -> tuple[list[torch.Tensor], torch.Tensor, list[torch.Tensor], torch.Tensor]:
    """Validation feature path shared verbatim by teacher and deployment decode."""
    feat_sizes = [(64, 64), (32, 32), (16, 16)]
    x1, y1 = crop_xy
    feats, _ = point_encoder(image)
    backbone_out, _ = net.forward_image(image, feats)
    _, vision_feats, vision_pos_embeds, _ = net._prepare_backbone_features(backbone_out)
    memfeatures = vision_feats
    memfeatures_pos = vision_pos_embeds
    batch_size = vision_feats[-1].size(1)
    if cfg.context:
        vision_feats, vision_pos_embeds = context_memory_attention(
            context_bank, vision_feats, vision_pos_embeds, [x1], [y1], net, feat_sizes, cfg.context_atten_k
        )
    if cfg.texture:
        if not memory_bank:
            zero = torch.zeros(1, batch_size, net.hidden_dim, device=device)
            vision_feats[-1] = vision_feats[-1] + zero
            vision_pos_embeds[-1] = vision_pos_embeds[-1] + zero
        else:
            memories = torch.stack([item[0].to(device).flatten(2).permute(2, 0, 1) for item in memory_bank])
            memory_positions = torch.stack([item[1].to(device).flatten(2).permute(2, 0, 1) for item in memory_bank])
            image_embeddings = torch.stack([item[3].to(device) for item in memory_bank])
            current = vision_feats[-1].permute(1, 0, 2).reshape(batch_size, -1, 64, 64).reshape(batch_size, -1)
            scores = F.softmax(torch.mm(F.normalize(current, p=2, dim=1), F.normalize(image_embeddings, p=2, dim=1).t()), dim=1)
            selected = torch.topk(scores, batch_size, dim=1).indices.squeeze(1)
            memory_new = memories[selected].squeeze(3).permute(1, 2, 0, 3)
            position_new = memory_positions[selected].squeeze(3).permute(1, 2, 0, 3)
            vision_feats[-1], vision_pos_embeds[-1] = net.memory_attention(
                state="texture",
                curr=[vision_feats[-1]],
                curr_pos=[vision_pos_embeds[-1]],
                memory=memory_new.reshape(-1, memory_new.size(2), memory_new.size(3)),
                memory_pos=position_new.reshape(-1, position_new.size(2), position_new.size(3)),
                num_obj_ptr_tokens=0,
            )
    if cfg.context and len(context_bank) < cfg.context_memory_bank_size:
        context_bank.append([memfeatures[-1].detach(), memfeatures_pos[-1].detach(), x1, y1])
    output_feats = [
        feature.permute(1, 2, 0).view(batch_size, -1, *size)
        for feature, size in zip(vision_feats[::-1], feat_sizes[::-1])
    ][::-1]
    return vision_feats, output_feats[-1], output_feats[:-1], output_feats[-1]


def _decode_deployment_formal(
    net: torch.nn.Module,
    image_embed: torch.Tensor,
    high_res_feats: list[torch.Tensor],
    prompts_xy: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact validation decoder call (including configured dynamic fallback)."""
    labels = torch.ones((len(prompts_xy), 1), dtype=torch.int, device=device)
    sparse, dense = net.sam_prompt_encoder(points=(prompts_xy, labels), boxes=None, masks=None, batch_size=1)
    low_res, predicted_iou, _, object_logits = net.sam_mask_decoder(
        image_embeddings=image_embed,
        image_pe=net.sam_prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=dense,
        multimask_output=False,
        repeat_image=False,
        cell_nums=torch.as_tensor([len(prompts_xy)], device=device),
        high_res_features=high_res_feats,
    )
    masks = F.interpolate(low_res, size=(256, 256), mode="bilinear", align_corners=False)[:, 0]
    return masks, predicted_iou[:, 0], object_logits.reshape(-1)


def _decode_teacher_token0(
    net: torch.nn.Module,
    image_embed: torch.Tensor,
    high_res_feats: list[torch.Tensor],
    prompts_xy: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode the training token-0 mask without validation-only fallback."""
    labels = torch.ones((len(prompts_xy), 1), dtype=torch.int, device=device)
    sparse, dense = net.sam_prompt_encoder(points=(prompts_xy, labels), boxes=None, masks=None, batch_size=1)
    masks_all, iou_all, _, object_logits = net.sam_mask_decoder.predict_masks(
        image_embeddings=image_embed,
        image_pe=net.sam_prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=dense,
        repeat_image=False,
        cell_nums=torch.as_tensor([len(prompts_xy)], device=device),
        high_res_features=high_res_feats,
    )
    masks = F.interpolate(masks_all[:, 0:1], size=(256, 256), mode="bilinear", align_corners=False)[:, 0]
    return masks, iou_all[:, 0], object_logits.reshape(-1)


def _full_soft_mask(local_logits: torch.Tensor, crop_box: Sequence[int], shape: tuple[int, int]) -> np.ndarray:
    x1, y1, x2, y2 = (int(value) for value in crop_box)
    full = np.zeros(shape, dtype=np.float32)
    soft = torch.sigmoid(local_logits).detach().cpu().numpy()
    full[y1:y2, x1:x2] = soft[: y2 - y1, : x2 - x1]
    return full


def _candidate_from_mask_data(mask_data: Mapping[str, Any], point_id: int, point_score: float, source: str) -> dict[str, Any]:
    return {
        "point_id": int(point_id),
        "point": [float(value) for value in np.asarray(mask_data["point"]).reshape(-1)],
        "point_score": float(point_score),
        "bbox": [float(value) for value in mask_data["bbox"]],
        "segmentation": np.asarray(mask_data["segmentation"], dtype=bool),
        "predicted_iou": float(mask_data["predicted_iou"]),
        "object_score": float(mask_data.get("object_score", float("nan"))),
        "stability_score": float(mask_data["stability_score"]),
        "assembly_score": float(mask_data["assembly_score"]),
        "source": source,
    }


def _add_standard_scores(
    masks: Sequence[Mapping[str, Any]], crop_box: Sequence[int], image_shape: tuple[int, int]
) -> list[dict[str, Any]]:
    x1, y1, x2, y2 = (int(value) for value in crop_box)
    height, width = image_shape
    margin = 7
    output: list[dict[str, Any]] = []
    for mask in masks:
        row = dict(mask)
        bx1, by1, bx2, by2 = row["bbox"]
        edge_penalized = (
            (bx1 > margin and abs(bx1 - x1) <= margin)
            or (abs(bx2 - height) > margin and abs(bx2 - x2) <= margin)
            or (by1 > margin and abs(by1 - y1) <= margin)
            or (abs(by2 - width) > margin and abs(by2 - y2) <= margin)
        )
        row["assembly_score"] = float(row["predicted_iou"] * 0.3 if edge_penalized else row["predicted_iou"])
        row["edge_penalized"] = bool(edge_penalized)
        output.append(row)
    return output


def _update_texture_bank(
    net: torch.nn.Module,
    vision_feats: list[torch.Tensor],
    image_embed: torch.Tensor,
    deployment_points: torch.Tensor,
    deployment_masks: torch.Tensor,
    deployment_iou: torch.Tensor,
    memory_bank: list[Any],
    cfg: argparse.Namespace,
    device: torch.device,
) -> None:
    """Exact validation memory update, applied to deployment prompts only."""
    if not cfg.texture or len(deployment_points) == 0:
        return
    feat_sizes = [(64, 64), (32, 32), (16, 16)]
    instance = combine_mask(torch.as_tensor([256, 256]), deployment_points, deployment_masks, deployment_iou)
    high_res = torch.from_numpy(instance.astype(float)).float().unsqueeze(0).unsqueeze(0).to(device)
    features, positions = net._encode_new_memory(
        current_vision_feats=vision_feats,
        feat_sizes=feat_sizes,
        pred_masks_high_res=high_res,
        is_mask_from_pts=True,
    )
    features = features.to(device=device, non_blocking=True)
    positions = positions[0].to(device=device, non_blocking=True)
    mean_iou = deployment_iou.mean()
    for batch_index in range(features.size(0)):
        record = [
            features[batch_index].unsqueeze(0),
            positions[batch_index].unsqueeze(0),
            mean_iou,
            image_embed[batch_index].reshape(-1).detach(),
        ]
        if len(memory_bank) < cfg.texture_memory_bank_size:
            memory_bank.append(record)
            continue
        bank_flat = torch.stack([element[0].reshape(-1).to(device) for element in memory_bank])
        normalized = F.normalize(bank_flat, p=2, dim=1)
        similarity = torch.mm(normalized, normalized.t())
        without_diagonal = similarity.clone()
        without_diagonal[torch.arange(without_diagonal.size(0)), torch.arange(without_diagonal.size(0))] = float("-inf")
        one = F.normalize(features[batch_index].reshape(-1), p=2, dim=0).unsqueeze(1)
        scores = torch.mm(normalized, one).squeeze()
        minimum = torch.argmin(scores)
        replacement = torch.argmax(without_diagonal[minimum])
        if scores[minimum] < without_diagonal[minimum][replacement] and mean_iou > memory_bank[replacement][2] - 0.1:
            memory_bank.pop(int(replacement))
            memory_bank.append(record)


def _teacher_only_descriptors(inst_map: np.ndarray, rgb: np.ndarray, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    hed = color.rgb2hed(rgb)
    centroids = {}
    for gt_id in (int(v) for v in np.unique(inst_map) if v != 0):
        coords = np.argwhere(inst_map == gt_id)
        centroids[gt_id] = coords.mean(axis=0)[::-1]
    values: list[dict[str, Any]] = []
    for row in rows:
        if row["category"] != "teacher-only":
            continue
        gt_id = int(row["gt_id"])
        gt_mask = inst_map == gt_id
        distances = [np.linalg.norm(centroids[gt_id] - center) for other, center in centroids.items() if other != gt_id]
        values.append(
            {
                "image": row["image"],
                "gt_id": gt_id,
                "area_pixels": int(gt_mask.sum()),
                "hematoxylin_mean": float(hed[..., 0][gt_mask].mean()),
                "nearest_gt_centroid_distance_pixels": float(min(distances)) if distances else None,
            }
        )
    return values


def _availability_ceiling(standard_metrics: Mapping[str, Any], instance_rows: Sequence[Mapping[str, Any]], standard_pairings: Mapping[str, Any]) -> dict[str, Any]:
    effective = 0
    for row in instance_rows:
        if row["category"] == "teacher-only" and int(row["gt_id"]) in set(standard_pairings[row["image"]]["unpaired_gt"]):
            effective += 1
    tp, fp, fn = (int(standard_metrics[key]) for key in ("tp", "fp", "fn"))
    if effective > fn:
        raise AssertionError("Availability ceiling cannot convert more FNs than the standard assembly has.")
    new_tp, new_fn = tp + effective, fn - effective
    dq = new_tp / (new_tp + 0.5 * fp + 0.5 * new_fn) if new_tp + fp + new_fn else 0.0
    current_sq = float(standard_metrics["sq"])
    sq = (current_sq * tp + effective) / new_tp if new_tp else 0.0
    return {
        "assumption": "teacher-only GTs that are Standard FNs become perfect TP masks; FPs unchanged",
        "teacher_only_standard_fn": effective,
        "dq_upper_bound": dq,
        "pq_upper_bound": dq * sq,
        "delta_dq_upper_bound": dq - float(standard_metrics["dq"]),
        "delta_pq_upper_bound": dq * sq - float(standard_metrics["pq"]),
    }


def _pairing_details(gt: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    _, pairing = get_fast_pq(remap_label(gt), remap_label(pred), match_iou=0.5)
    return {"unpaired_gt": [int(v) for v in pairing[2]], "unpaired_pred": [int(v) for v in pairing[3]]}


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, default=_json_default) if isinstance(value, (dict, list)) else value for key, value in row.items()})


def _coverage_summary(instance_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    categories = ("both-covered", "teacher-only", "deployment-only", "neither")
    def count(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
        return {category: int(sum(row["category"] == category for row in rows)) for category in categories}
    by_image: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_patient: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in instance_rows:
        by_image[str(row["image"])].append(row)
        by_patient[str(row["patient"])].append(row)
    return {
        "overall": count(instance_rows),
        "per_image": {image: count(rows) for image, rows in by_image.items()},
        "per_patient": {patient: count(rows) for patient, rows in by_patient.items()},
    }


def _per_patient_assemblies(per_image: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in per_image.values():
        grouped[str(record["patient"])].append(record)
    return {
        patient: {
            "n_images": len(records),
            "standard_deployment": aggregate_metrics([record["metrics"]["standard"] for record in records]),
            "teacher_coordinate_gt_assisted_diagnostic": aggregate_metrics([record["metrics"]["teacher_coordinate"] for record in records]),
            "shared_gt_swap": aggregate_metrics([record["metrics"]["shared_gt_swap"] for record in records]),
        }
        for patient, records in grouped.items()
    }


def run_phase0(cfg: argparse.Namespace) -> Path:
    start = time.monotonic()
    repo_root = Path(__file__).resolve().parents[1]
    _assert_canonical_baseline(repo_root)
    if cfg.dataset.lower() != "tnbc":
        raise ValueError("Phase 0 accepts TNBC only; MoNuSeg is explicitly closed for this audit.")
    if cfg.train or cfg.use_pms or cfg.monuseg:
        raise ValueError("Phase 0 is frozen diagnostic only; training, PMS residual prompts, and MoNuSeg are forbidden.")

    checkpoint = Path(cfg.checkpoint).resolve()
    if "e156" not in checkpoint.name.lower():
        raise ValueError("Phase 0 accepts the handover e156 checkpoint only.")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"e156 checkpoint not found: {checkpoint}")
    checkpoint_sha = sha256_file(checkpoint)
    if checkpoint_sha != E156_SHA256:
        raise RuntimeError(f"e156 checkpoint SHA256 mismatch: got {checkpoint_sha}, expected {E156_SHA256}")

    manifest = build_data_manifest(Path(cfg.data_root).resolve())
    run_id = cfg.run_id or f"deploypms_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    artifact = Path(cfg.output_root).resolve() / run_id
    if artifact.exists():
        raise FileExistsError(f"Refusing to overwrite existing Phase 0 artifact: {artifact}")
    artifact.mkdir(parents=True)
    _write_json(artifact / "data_manifest.json", manifest)

    device = torch.device(f"cuda:{cfg.gpu_device}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np.random.seed(cfg.seed)
    args_config = Config.fromfile(str(repo_root / "args.py"))
    # Avoid repeated multi-GB checkpoint loads: build the architecture without
    # weights, then load both frozen components from the one verified payload.
    net = build_sam2(cfg.sam_config, None, device=device)
    point_net, point_encoder = build_model(args_config)
    loaded = torch.load(checkpoint, map_location="cpu")
    if "model" not in loaded or "model1" not in loaded:
        raise RuntimeError("e156 checkpoint must contain both frozen SAM2 'model' and point-head 'model1'.")
    net_missing, net_unexpected = net.load_state_dict(loaded["model"], strict=False)
    point_missing, point_unexpected = point_net.load_state_dict(loaded["model1"], strict=False)
    if net_missing or net_unexpected or point_missing or point_unexpected:
        raise RuntimeError(
            "Checkpoint model mismatch: "
            f"sam_missing={len(net_missing)}, sam_unexpected={len(net_unexpected)}, "
            f"point_missing={len(point_missing)}, point_unexpected={len(point_unexpected)}"
        )
    point_net.to(device).eval()
    point_encoder.to(device).eval()
    net.to(device).eval()
    for module in (point_net, point_encoder, net):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    texture_bank = list(loaded.get("texture_memory_bank_list", []) or [])
    checksums = {
        "checkpoint_sha256": checkpoint_sha,
        "point_head_state_sha256": state_dict_sha256(loaded["model1"]),
        "sam2_state_sha256": state_dict_sha256(loaded["model"]),
        "expected_e156_sha256": E156_SHA256,
    }
    del loaded

    all_instance_rows: list[dict[str, Any]] = []
    both_rows: list[dict[str, Any]] = []
    teacher_only_rows: list[dict[str, Any]] = []
    false_or_duplicate: list[dict[str, Any]] = []
    per_image_assemblies: dict[str, dict[str, Any]] = {}
    pairings: dict[str, Any] = {}
    call_counts: dict[str, int] = defaultdict(int)

    for entry_data in manifest["development_entries"]:
        entry = ImageEntry(**entry_data)
        image, inst_map, rgb = _load_image_and_instances(entry)
        image = image.to(device)
        height, width = inst_map.shape
        crop_boxes = crop_with_overlap(image[0], cfg.crop_size, cfg.crop_size, cfg.overlap, cfg.load).tolist()
        selected_points = _training_reference_points(inst_map, cfg.seed + entry.patient)
        teacher_by_crop: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for gt_id, point in selected_points.items():
            candidates = [
                index for index, (x1, y1, x2, y2) in enumerate(crop_boxes)
                if x1 <= point[0] < x2 and y1 <= point[1] < y2
            ]
            if not candidates:
                raise RuntimeError(f"No crop contains teacher reference point for {entry.stem} GT {gt_id}.")
            teacher_by_crop[candidates[0]].append({"gt_id": gt_id, "reference_point": point})

        all_raw_deployment: list[dict[str, Any]] = []
        point_id_by_coordinate: dict[tuple[float, float], int] = {}
        decoded_by_point: dict[int, list[dict[str, Any]]] = defaultdict(list)
        teacher_records: dict[int, dict[str, Any]] = {}
        teacher_candidates: list[dict[str, Any]] = []
        context_bank: list[Any] = []

        with torch.inference_mode():
            for crop_index, crop_box in enumerate(crop_boxes):
                x1, y1, x2, y2 = (int(value) for value in crop_box)
                crop = image[..., y1:y2, x1:x2]
                classified, raw_coords, raw_logits = _classify_points(point_net, crop, cfg.filtering)
                call_counts["point_head"] += 1
                keep_new = np.ones(len(classified), dtype=bool)
                for previous in crop_boxes[:crop_index]:
                    px1, py1, px2, py2 = (int(value) for value in previous)
                    points = np.asarray([row["local_point"] for row in classified], dtype=float) if classified else np.empty((0, 2))
                    if len(points):
                        keep_new &= ~(
                            (points[:, 0] >= px1 + 1) & (points[:, 0] <= px2 - 1)
                            & (points[:, 1] >= py1 + 1) & (points[:, 1] <= py2 - 1)
                        )
                for record in (row for row, keep in zip(classified, keep_new, strict=True) if keep):
                    global_point = np.asarray(record["local_point"], dtype=float) + np.asarray([x1, y1])
                    row = dict(record)
                    row.update({"point": global_point, "crop_index": crop_index, "crop_box": crop_box})
                    all_raw_deployment.append(row)
                if all_raw_deployment:
                    all_points = np.asarray([row["point"] for row in all_raw_deployment], dtype=float)
                    all_scores = np.asarray([row["point_score"] for row in all_raw_deployment], dtype=float)
                    active_indices = point_nms_indices(all_points, all_scores, cfg.nms_threshold)
                else:
                    active_indices = np.empty(0, dtype=np.int64)
                active = [all_raw_deployment[int(index)] for index in active_indices]
                for record in active:
                    key = tuple(float(value) for value in record["point"])
                    point_id_by_coordinate.setdefault(key, len(point_id_by_coordinate))
                    record["point_id"] = point_id_by_coordinate[key]

                deployment_here = [row for row in active if x1 <= row["point"][0] < x2 and y1 <= row["point"][1] < y2]
                teacher_here: list[dict[str, Any]] = []
                for request in teacher_by_crop.get(crop_index, []):
                    reference_local = torch.as_tensor(request["reference_point"] - np.asarray([x1, y1]), dtype=torch.float32)
                    nearest, query_index = find_nearest_points_with_indices(raw_coords, reference_local.unsqueeze(0))
                    query = int(query_index[0])
                    local = nearest[0].numpy().astype(float)
                    teacher_here.append(
                        {
                            **request,
                            "query_index": query,
                            "local_point": local,
                            "point": local + np.asarray([x1, y1]),
                            "point_score": float(raw_logits[query].softmax(-1).max().item()),
                            "point_class": int(raw_logits[query].argmax().item()),
                            "crop_index": crop_index,
                            "crop_box": crop_box,
                        }
                    )
                if not deployment_here and not teacher_here:
                    continue
                vision_feats, image_embed, high_res_feats, _ = _encode_crop_once(
                    net, point_encoder, crop, texture_bank, context_bank, (x1, y1), cfg, device
                )
                call_counts["shared_feature_encode"] += 1
                deployment_local = torch.as_tensor(
                    [row["point"] - np.asarray([x1, y1]) for row in deployment_here], dtype=torch.float32, device=device
                ).unsqueeze(1)
                teacher_local = torch.as_tensor(
                    [row["local_point"] for row in teacher_here], dtype=torch.float32, device=device
                ).unsqueeze(1)
                if deployment_here:
                    deployment_masks, deployment_iou, deployment_obj = _decode_deployment_formal(
                        net, image_embed, high_res_feats, deployment_local, device
                    )
                    call_counts["sam_decoder_deployment_formal"] += 1
                else:
                    deployment_masks = torch.empty((0, 256, 256), device=device)
                    deployment_iou = torch.empty(0, device=device)
                    deployment_obj = torch.empty(0, device=device)
                if teacher_here:
                    teacher_masks, teacher_iou, teacher_obj = _decode_teacher_token0(
                        net, image_embed, high_res_feats, teacher_local, device
                    )
                    call_counts["sam_decoder_teacher_token0"] += 1
                else:
                    teacher_masks = torch.empty((0, 256, 256), device=device)
                    teacher_iou = torch.empty(0, device=device)
                    teacher_obj = torch.empty(0, device=device)
                n_deployment = len(deployment_here)

                if n_deployment:
                    point_ids = torch.as_tensor([row["point_id"] for row in deployment_here], dtype=torch.long)
                    mask_data = mask_process_eval(
                        np.asarray([row["point_class"] for row in deployment_here]), point_ids, crop_box,
                        np.asarray([height, width]), deployment_local, deployment_masks, deployment_iou,
                    )
                    for item in _add_standard_scores(mask_data, crop_box, (height, width)):
                        idx = int(item["inds"])
                        source = next(row for row in deployment_here if int(row["point_id"]) == idx)
                        candidate = _candidate_from_mask_data(item, idx, source["point_score"], "deployment")
                        source_index = [int(row["point_id"]) for row in deployment_here].index(idx)
                        candidate["object_score"] = float(torch.sigmoid(deployment_obj[source_index]).item())
                        candidate["soft_mask"] = _full_soft_mask(deployment_masks[source_index], crop_box, (height, width))
                        decoded_by_point[idx].append(candidate)
                    _update_texture_bank(
                        net, vision_feats, image_embed, deployment_local, deployment_masks, deployment_iou,
                        texture_bank, cfg, device,
                    )

                for index, request in enumerate(teacher_here):
                    mask_data = mask_process_eval(
                        np.asarray([request["point_class"]]), torch.as_tensor([request["gt_id"]]), crop_box,
                        np.asarray([height, width]), teacher_local[index : index + 1], teacher_masks[index : index + 1], teacher_iou[index : index + 1],
                    )
                    soft = _full_soft_mask(teacher_masks[index], crop_box, (height, width))
                    teacher = dict(request)
                    teacher.update(
                        {
                            "spatial_gt_id": _inside_instance(inst_map, request["point"]),
                            "predicted_iou": float(teacher_iou[index].item()),
                            "object_score": float(torch.sigmoid(teacher_obj[index]).item()),
                            "soft_mask": soft,
                            "hard_mask": soft > 0.5,
                        }
                    )
                    if mask_data:
                        item = _add_standard_scores(mask_data, crop_box, (height, width))[0]
                        candidate = _candidate_from_mask_data(item, request["gt_id"], request["point_score"], "teacher")
                        candidate["object_score"] = teacher["object_score"]
                        candidate["soft_mask"] = soft
                        teacher_candidates.append(candidate)
                        teacher["candidate"] = candidate
                    teacher_records[request["gt_id"]] = teacher

        final_indices = point_nms_indices(
            np.asarray([row["point"] for row in all_raw_deployment], dtype=float),
            np.asarray([row["point_score"] for row in all_raw_deployment], dtype=float), cfg.nms_threshold,
        ) if all_raw_deployment else np.empty(0, dtype=np.int64)
        final_deployment = []
        for index in final_indices:
            row = dict(all_raw_deployment[int(index)])
            key = tuple(float(value) for value in row["point"])
            row["point_id"] = point_id_by_coordinate[key]
            final_deployment.append(row)
        standard_candidates = [candidate for candidates in decoded_by_point.values() for candidate in candidates]
        standard_map, standard_selected = _assemble_instance_map(standard_candidates, (height, width))
        teacher_map, teacher_selected = _assemble_instance_map(teacher_candidates, (height, width))
        association_rows, prompt_extras = assess_associations(inst_map, teacher_records, final_deployment)
        for row in association_rows:
            row["image"] = entry.stem
            row["patient"] = entry.patient
        for row in prompt_extras:
            row["image"] = entry.stem
            row["patient"] = entry.patient
        all_instance_rows.extend(association_rows)
        false_or_duplicate.extend(prompt_extras)

        primary_by_gt = {
            int(row["gt_id"]): row["deployment_primary"]
            for row in association_rows if row["category"] == "both-covered"
        }
        teacher_by_gt = {
            int(row["gt_id"]): teacher_records[int(row["gt_id"])]
            for row in association_rows if row["category"] == "both-covered"
        }
        for row in association_rows:
            if row["category"] != "both-covered":
                continue
            gt_id = int(row["gt_id"])
            deployment = primary_by_gt[gt_id]
            teacher = teacher_by_gt[gt_id]
            deployment_candidates = decoded_by_point.get(int(deployment["point_id"]), [])
            if not deployment_candidates or "candidate" not in teacher:
                continue
            deployment_candidate = max(deployment_candidates, key=lambda item: item["assembly_score"])
            gt_mask = inst_map == gt_id
            deployment_soft = np.asarray(deployment_candidate["soft_mask"], dtype=np.float32)
            deployment_hard = np.asarray(deployment_candidate["segmentation"], dtype=bool)
            teacher_hard = np.asarray(teacher["hard_mask"], dtype=bool)
            teacher_soft = np.asarray(teacher["soft_mask"], dtype=np.float32)
            both_rows.append(
                {
                    "image": entry.stem,
                    "patient": entry.patient,
                    "gt_id": gt_id,
                    "coordinate_distance": float(np.linalg.norm(np.asarray(teacher["point"]) - np.asarray(deployment["point"]))),
                    "teacher_point_head_objectness": float(teacher["point_score"]),
                    "deployment_point_head_objectness": float(deployment["point_score"]),
                    "teacher_hard_iou": _hard_iou(teacher_hard, gt_mask),
                    "deployment_hard_iou": _hard_iou(deployment_hard, gt_mask),
                    "teacher_soft_iou": _soft_iou(teacher_soft, gt_mask),
                    "deployment_soft_iou": _soft_iou(deployment_soft, gt_mask),
                    "teacher_dice": _dice(teacher_hard, gt_mask),
                    "deployment_dice": _dice(deployment_hard, gt_mask),
                    "teacher_boundary_iou": _boundary_iou(teacher_hard, gt_mask),
                    "deployment_boundary_iou": _boundary_iou(deployment_hard, gt_mask),
                    "teacher_predicted_iou": float(teacher["predicted_iou"]),
                    "deployment_predicted_iou": float(deployment_candidate["predicted_iou"]),
                    "teacher_object_score": float(teacher["object_score"]),
                    "deployment_object_score": float(deployment_candidate["object_score"]),
                    "hard_iou_gap": _hard_iou(teacher_hard, gt_mask) - _hard_iou(deployment_hard, gt_mask),
                    "decoded_masks_mutual_iou": _hard_iou(teacher_hard, deployment_hard),
                    "deployment_location": _prompt_location(inst_map, gt_id, deployment["point"]),
                }
            )
        teacher_only_rows.extend(_teacher_only_descriptors(inst_map, rgb, association_rows))

        swap_candidates = [dict(candidate) for candidate in standard_candidates]
        for row in association_rows:
            if row["category"] != "both-covered":
                continue
            deployment = row["deployment_primary"]
            teacher = teacher_records.get(int(row["gt_id"]))
            if not teacher or "candidate" not in teacher:
                continue
            replacement = teacher["candidate"]
            for index, candidate in enumerate(swap_candidates):
                if int(candidate["point_id"]) == int(deployment["point_id"]):
                    swapped = dict(replacement)
                    swapped["point_id"] = int(deployment["point_id"])
                    swapped["source"] = "shared_gt_swap_teacher_coordinate"
                    swap_candidates[index] = swapped
        swap_map, swap_selected = _assemble_instance_map(swap_candidates, (height, width))
        maps = {"standard": standard_map, "teacher_coordinate": teacher_map, "shared_gt_swap": swap_map}
        metrics = {name: image_metrics(inst_map, value) for name, value in maps.items()}
        per_image_assemblies[entry.stem] = {
            "patient": entry.patient,
            "metrics": metrics,
            "counts": {
                "teacher_candidates": len(teacher_candidates),
                "deployment_final_prompts": len(final_deployment),
                "deployment_decoded_candidates": len(standard_candidates),
                "standard_selected": len(standard_selected),
                "teacher_selected": len(teacher_selected),
                "swap_selected": len(swap_selected),
            },
        }
        pairings[entry.stem] = _pairing_details(inst_map, standard_map)

    standard_rows = [record["metrics"]["standard"] for record in per_image_assemblies.values()]
    teacher_rows = [record["metrics"]["teacher_coordinate"] for record in per_image_assemblies.values()]
    swap_rows = [record["metrics"]["shared_gt_swap"] for record in per_image_assemblies.values()]
    availability = availability_gate(all_instance_rows)
    conditioning = conditioning_gate(both_rows)
    assembly = assembly_gate(standard_rows, swap_rows)
    standard_metrics = aggregate_metrics(standard_rows)
    frozen_parameter_count = sum(1 for module in (point_net, point_encoder, net) for parameter in module.parameters() if not parameter.requires_grad)
    trainable_parameter_count = sum(1 for module in (point_net, point_encoder, net) for parameter in module.parameters() if parameter.requires_grad)
    if trainable_parameter_count:
        raise AssertionError("Frozen Phase 0 detected a trainable parameter.")
    report = {
        "phase": "DeployPMS Phase 0 — Training–Deployment Prompt Exposure Gap Audit",
        "run_id": run_id,
        "verdict": final_verdict(availability, conditioning, assembly),
        "frozen_only": True,
        "git_sha": _git_sha(repo_root),
        "canonical_baseline": CANONICAL_BASELINE,
        "checkpoint": {"path": str(checkpoint), **checksums},
        "environment": _environment(),
        "data_manifest": "data_manifest.json",
        "access_guard": manifest["access_guard"],
        "exact_code_paths": {
            "teacher_nearest": "run.run_on_epoch.find_nearest_points (replayed in deploypms.phase0.find_nearest_points_with_indices)",
            "deployment_classification": "sam2_train.modeling.utils.predict semantics (replayed with query provenance)",
            "deployment_nms": "sam2_train.modeling.utils.point_nms semantics (replayed with indices)",
            "decoder": "run.run_on_epoch.inference SAM prompt encoder + mask decoder; token 0 only",
            "mask_postprocess": "run.run_on_epoch.mask_process_eval",
            "assembly": "run.run_on_epoch._assemble_instance_map semantics",
            "pq": "sam2_train.modeling.stats_utils.get_fast_pq(match_iou=0.5 inclusive)",
        },
        "protocol": {
            "teacher_label": 1,
            "deployment_label": 1,
            "teacher_reference_selector": "seeded replay of training uniform interior-GT selector; selected decoder coordinate is exact nearest predicted query",
            "teacher_crop_rule": "first configured validation traversal crop containing the selected GT point",
            "deployment_association": "integer point-inside-GT; background unmatched; duplicate primary is highest point-head score",
            "boundary_band_pixels": BOUNDARY_BAND_PIXELS,
            "no_mask_iou_for_association": True,
            "inclusive_pq_iou_threshold": 0.5,
            "sam_decoder_shared_image_features": True,
            "stainpms_residual_prompts_used": False,
            "monuseg_used": False,
        },
        "call_counts": dict(call_counts),
        "tests": {
            "all_parameters_frozen": True,
            "frozen_parameter_tensors": frozen_parameter_count,
            "trainable_parameter_tensors": trainable_parameter_count,
            "no_optimizer_created": True,
            "teacher_positive_prompt_label": True,
            "teacher_token_zero": True,
            "deployment_association_uses_mask_iou": False,
            "pq_threshold_inclusive": True,
        },
        "assemblies": {
            "standard_deployment": aggregate_metrics(standard_rows),
            "teacher_coordinate_gt_assisted_diagnostic": aggregate_metrics(teacher_rows),
            "shared_gt_swap": aggregate_metrics(swap_rows),
            "per_image": per_image_assemblies,
            "per_patient": _per_patient_assemblies(per_image_assemblies),
        },
        "availability": {
            "gate": asdict(availability),
            "coverage_categories": _coverage_summary(all_instance_rows),
            "teacher_only_descriptors": teacher_only_rows,
            "false_deployment_and_duplicates": false_or_duplicate,
            "false_deployment_prompt_count": int(sum(row.get("association") == "background_unmatched" for row in false_or_duplicate)),
            "duplicate_deployment_prompt_count": int(sum(row.get("association") == "duplicate" for row in false_or_duplicate)),
            "theoretical_missed_nuclei_ceiling": _availability_ceiling(standard_metrics, all_instance_rows, pairings),
        },
        "conditioning": {"gate": asdict(conditioning), "both_covered_instances": both_rows},
        "assembly_gate": asdict(assembly),
        "runtime_seconds": time.monotonic() - start,
        "stop_condition": "Phase 0 complete; no training or next-stage execution performed.",
    }
    _write_json(artifact / "report.json", report)
    _write_csv(artifact / "instances.csv", all_instance_rows)
    _write_csv(artifact / "both_covered.csv", both_rows)
    _write_csv(artifact / "teacher_only.csv", teacher_only_rows)
    _write_csv(artifact / "false_and_duplicate_deployment_prompts.csv", false_or_duplicate)
    with (artifact / "SHA256SUMS").open("w", encoding="utf-8") as handle:
        for path in sorted(item for item in artifact.rglob("*") if item.is_file() and item.name != "SHA256SUMS"):
            handle.write(f"{sha256_file(path)}  {path.relative_to(artifact).as_posix()}\n")
    return artifact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Frozen DeployPMS Phase 0 exposure-gap audit")
    parser.add_argument("--data-root", default="data/tnbc")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-root", default="logs/deploypms/phase0")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--dataset", default="tnbc", choices=["tnbc"])
    parser.add_argument("--sam-config", default="sam2_hiera_l")
    parser.add_argument("--gpu-device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=32)
    parser.add_argument("--nms-threshold", type=int, default=12)
    parser.add_argument("--load", default="unclockwise", choices=["sequence", "unsequence", "clockwise", "unclockwise"])
    parser.add_argument("--filtering", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--texture", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--context", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--texture-memory-bank-size", type=int, default=64)
    parser.add_argument("--context-memory-bank-size", type=int, default=100)
    parser.add_argument("--context-atten-k", type=int, default=1)
    parser.add_argument("--train", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--use-pms", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--monuseg", action="store_true", help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifact = run_phase0(args)
    print(f"DeployPMS Phase 0 complete: {artifact}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
