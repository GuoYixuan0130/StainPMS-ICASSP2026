"""TNBC router-train-only crop reader for PromptCredit Stage 0."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import albumentations as A
import numpy as np
import scipy.io as sio
import torch
from skimage import io

from promptcredit.audit.guardrails import selected_tnbc_paths


@dataclass(frozen=True)
class AuditCrop:
    image_id: str
    crop_id: int
    crop_box_xyxy: tuple[int, int, int, int]
    image: torch.Tensor
    gt_instance_ids: np.ndarray
    gt_masks: np.ndarray
    gt_centroids_xy: np.ndarray
    gt_areas: np.ndarray
    local_density: np.ndarray


def _starts(size: int, crop_size: int, overlap: int) -> list[int]:
    if crop_size != 256:
        raise ValueError("PC-Stage 0 preprocessing is frozen to 256-pixel crops")
    stride = crop_size - overlap
    if stride <= 0:
        raise ValueError("overlap must be smaller than crop_size")
    points = [0]
    counter = 1
    while True:
        point = stride * counter
        if point + crop_size >= size:
            if crop_size != size:
                points.append(size - crop_size)
            break
        points.append(point)
        counter += 1
    return points


def crop_boxes_unclockwise(height: int, width: int, *, crop_size: int = 256, overlap: int = 32) -> list[tuple[int, int, int, int]]:
    """Reproduce the 256-pixel ``unclockwise`` baseline crop traversal."""
    xs, ys = _starts(width, crop_size, overlap), _starts(height, crop_size, overlap)
    boxes: list[tuple[int, int, int, int]] = []
    top, bottom, left, right = 0, len(xs) - 1, 0, len(ys) - 1
    while top <= bottom or left <= right:
        if top <= bottom:
            for y_index in range(left, right + 1):
                boxes.append((xs[top], ys[y_index], min(xs[top] + crop_size, width), min(ys[y_index] + crop_size, height)))
            top += 1
        if left <= right:
            for x_index in range(top, bottom + 1):
                boxes.append((xs[x_index], ys[right], min(xs[x_index] + crop_size, width), min(ys[right] + crop_size, height)))
            right -= 1
        if top <= bottom:
            for y_index in reversed(range(left, right + 1)):
                boxes.append((xs[bottom], ys[y_index], min(xs[bottom] + crop_size, width), min(ys[y_index] + crop_size, height)))
            bottom -= 1
        if left <= right:
            for x_index in reversed(range(top, bottom + 1)):
                boxes.append((xs[x_index], ys[left], min(xs[x_index] + crop_size, width), min(ys[left] + crop_size, height)))
            left += 1
    return list(reversed(boxes))


def _centroid_inside(mask: np.ndarray) -> np.ndarray:
    coordinates_yx = np.argwhere(mask)
    center_yx = np.rint(coordinates_yx.mean(axis=0)).astype(np.int64)
    if not mask[tuple(center_yx)]:
        nearest = np.sum((coordinates_yx - center_yx[None, :]) ** 2, axis=1).argmin()
        center_yx = coordinates_yx[nearest]
    return center_yx[[1, 0]].astype(np.float32)


def _normalize(image: np.ndarray) -> torch.Tensor:
    normalized = A.Normalize()(image=image)["image"]
    return torch.from_numpy(np.ascontiguousarray(normalized.transpose(2, 0, 1))).float()


def iter_selected_tnbc_crops(
    data_root: Path,
    image_ids: list[str],
    *,
    overlap: int = 32,
    density_radius_pixels: float = 64.0,
) -> Iterator[AuditCrop]:
    """Read only selected image/GT pairs after Stage 0 scope validation."""
    if overlap != 32:
        raise ValueError("PC-Stage 0 overlap is frozen to TNBC baseline v1 value 32")
    if density_radius_pixels != 64.0:
        raise ValueError("PC-Stage 0 local-density radius is frozen to 64 pixels")
    for image_id, image_path, label_path in selected_tnbc_paths(data_root, image_ids):
        image = io.imread(image_path)[..., :3]
        instance_map = np.asarray(sio.loadmat(label_path)["inst_map"])
        if image.shape[:2] != instance_map.shape:
            raise ValueError(f"Image/GT shape mismatch for {image_id}: {image.shape[:2]} vs {instance_map.shape}")
        for crop_id, (x1, y1, x2, y2) in enumerate(crop_boxes_unclockwise(*image.shape[:2], overlap=overlap)):
            crop_image = image[y1:y2, x1:x2]
            crop_instance = instance_map[y1:y2, x1:x2]
            instance_ids = np.unique(crop_instance)
            instance_ids = instance_ids[instance_ids != 0]
            if len(instance_ids) == 0:
                continue
            masks = np.stack([crop_instance == instance_id for instance_id in instance_ids]).astype(bool)
            centroids = np.stack([_centroid_inside(mask) for mask in masks]).astype(np.float32)
            areas = masks.sum(axis=(1, 2)).astype(np.int64)
            distances = np.linalg.norm(centroids[:, None, :] - centroids[None, :, :], axis=-1)
            density = ((distances <= density_radius_pixels) & (distances > 0)).sum(axis=1).astype(np.int64)
            yield AuditCrop(
                image_id=image_id,
                crop_id=crop_id,
                crop_box_xyxy=(x1, y1, x2, y2),
                image=_normalize(crop_image),
                gt_instance_ids=instance_ids.astype(np.int64),
                gt_masks=masks,
                gt_centroids_xy=centroids,
                gt_areas=areas,
                local_density=density,
            )

