"""Explicit TNBC 1-6 train / 7-8 development access for PromptQ only."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterator, Literal

import albumentations as A
import numpy as np
import scipy.io as sio
import torch
from skimage import io
from torch.utils.data import Dataset

from promptcredit.audit.data import crop_boxes_unclockwise


Role = Literal["train", "development"]
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
TRAIN_PATIENTS = frozenset(range(1, 7))
DEVELOPMENT_PATIENTS = frozenset((7, 8))


@dataclass(frozen=True)
class PromptQImage:
    image_id: str
    image_path: Path
    label_path: Path


@dataclass(frozen=True)
class PromptQCrop:
    image_id: str
    crop_id: int
    crop_box_xyxy: tuple[int, int, int, int]
    image: torch.Tensor
    gt_instance_ids: np.ndarray
    gt_masks: np.ndarray
    gt_centroids_xy: np.ndarray
    gt_areas: np.ndarray
    local_density: np.ndarray
    image_crop_sha256: str
    gt_crop_sha256: str


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _normalize(image: np.ndarray) -> torch.Tensor:
    normalized = A.Normalize()(image=image)["image"]
    return torch.from_numpy(np.ascontiguousarray(normalized.transpose(2, 0, 1))).float()


def _patient(image_id: str) -> int:
    try:
        return int(image_id.split("_", 1)[0])
    except (ValueError, IndexError) as error:
        raise ValueError(f"Invalid TNBC image id: {image_id}") from error


def _exact_image_path(root: Path, image_id: str) -> Path:
    matches = [root / f"{image_id}{suffix}" for suffix in IMAGE_SUFFIXES if (root / f"{image_id}{suffix}").is_file()]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one explicit PromptQ image for {image_id}, found {matches}")
    return matches[0]


def resolve_promptq_images(data_root: Path, split_manifest_path: Path, role: Role) -> list[PromptQImage]:
    """Resolve only manifest-listed IDs; never enumerate or open patients 9-11."""
    payload = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    if payload.get("dataset") != "TNBC" or payload.get("split_method") != "fixed_patient_level":
        raise ValueError("PromptQ requires the frozen TNBC patient-level split manifest")
    if role == "train":
        image_ids = list(payload["router_train"])
        allowed_patients = TRAIN_PATIENTS
        root = data_root / payload["train_image_root_relative"]
        label_root = data_root / "train_12" / "labels"
    elif role == "development":
        image_ids = list(payload["calibration"])
        allowed_patients = DEVELOPMENT_PATIENTS
        # The frozen TNBC conversion places patients 1--8 together under
        # train_12; only patients 9--11 live under the closed test directory.
        # Resolve exact 7--8 IDs here rather than touching the test root.
        root = data_root / payload["train_image_root_relative"]
        label_root = data_root / "train_12" / "labels"
    else:
        raise ValueError(f"Unsupported PromptQ role: {role}")
    if not root.is_dir() or not label_root.is_dir():
        raise FileNotFoundError(f"Missing explicit PromptQ {role} TNBC roots")
    if any(_patient(image_id) not in allowed_patients for image_id in image_ids):
        raise ValueError(f"PromptQ {role} manifest contains an unauthorized patient")
    if role == "development" and set(_patient(image_id) for image_id in image_ids) != DEVELOPMENT_PATIENTS:
        raise ValueError("PromptQ development must include exactly patients 7 and 8")
    resolved: list[PromptQImage] = []
    for image_id in image_ids:
        label_path = label_root / f"{image_id}.mat"
        if not label_path.is_file():
            raise FileNotFoundError(f"Missing PromptQ label: {label_path}")
        resolved.append(PromptQImage(image_id, _exact_image_path(root, image_id), label_path))
    return resolved


def _centroid_inside(mask: np.ndarray) -> np.ndarray:
    coordinates_yx = np.argwhere(mask)
    center_yx = np.rint(coordinates_yx.mean(axis=0)).astype(np.int64)
    if not mask[tuple(center_yx)]:
        nearest = np.sum((coordinates_yx - center_yx[None, :]) ** 2, axis=1).argmin()
        center_yx = coordinates_yx[nearest]
    return center_yx[[1, 0]].astype(np.float32)


def iter_promptq_crops(data_root: Path, split_manifest_path: Path, role: Role) -> Iterator[PromptQCrop]:
    """Deterministic 256/32/unclockwise crops with no augmentation."""
    for item in resolve_promptq_images(data_root, split_manifest_path, role):
        image = io.imread(item.image_path)[..., :3]
        instance_map = np.asarray(sio.loadmat(item.label_path)["inst_map"])
        if image.shape[:2] != instance_map.shape:
            raise ValueError(f"Image/GT shape mismatch for {item.image_id}")
        for crop_id, (x1, y1, x2, y2) in enumerate(crop_boxes_unclockwise(*image.shape[:2], overlap=32)):
            crop_image = image[y1:y2, x1:x2]
            crop_instance = instance_map[y1:y2, x1:x2]
            instance_ids = np.unique(crop_instance)
            instance_ids = instance_ids[instance_ids != 0]
            masks = (
                np.stack([crop_instance == instance_id for instance_id in instance_ids]).astype(bool)
                if len(instance_ids)
                else np.empty((0, y2 - y1, x2 - x1), dtype=bool)
            )
            centroids = (
                np.stack([_centroid_inside(mask) for mask in masks]).astype(np.float32)
                if len(masks)
                else np.empty((0, 2), dtype=np.float32)
            )
            areas = masks.sum(axis=(1, 2)).astype(np.int64)
            if len(centroids):
                distances = np.linalg.norm(centroids[:, None, :] - centroids[None, :, :], axis=-1)
                density = ((distances <= 64.0) & (distances > 0)).sum(axis=1).astype(np.int64)
            else:
                density = np.empty(0, dtype=np.int64)
            yield PromptQCrop(
                image_id=item.image_id,
                crop_id=crop_id,
                crop_box_xyxy=(x1, y1, x2, y2),
                image=_normalize(crop_image),
                gt_instance_ids=instance_ids.astype(np.int64),
                gt_masks=masks,
                gt_centroids_xy=centroids,
                gt_areas=areas,
                local_density=density,
                image_crop_sha256=_array_sha256(crop_image),
                gt_crop_sha256=_array_sha256(crop_instance),
            )


class PromptQDevelopmentDataset(Dataset):
    """Validation-on-epoch compatible direct-ID dataset for TNBC patients 7-8."""

    def __init__(self, data_root: Path, split_manifest_path: Path) -> None:
        self.items = resolve_promptq_images(data_root, split_manifest_path, "development")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        item = self.items[index]
        image = io.imread(item.image_path)[..., :3]
        instance_map = np.asarray(sio.loadmat(item.label_path)["inst_map"])
        image_tensor = _normalize(image)
        inst_tensor = torch.as_tensor(instance_map, dtype=torch.long)
        type_tensor = (inst_tensor > 0).to(torch.float32)
        binary = (inst_tensor > 0).to(torch.uint8)
        return (
            image_tensor,
            inst_tensor,
            type_tensor,
            torch.empty(0, 2, dtype=torch.float32),
            torch.empty(0, dtype=torch.long),
            binary,
            torch.as_tensor(instance_map.shape, dtype=torch.long),
            index,
            item.image_id,
        )
