"""Strict fixed-six-image TNBC reader for NuSet Stage 0 only."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterator

import albumentations as A
import numpy as np
import scipy.io as sio
import torch
from skimage import io


BASELINE_V1_TNBC_SHA256 = "44a3cb3e93051301d789e44f93769588abfa727d7a174b0270f55305ef023781"
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
EXPECTED_SELECTION_PATH = Path("configs/promptcredit/pc_stage0_tnbc_router_train_six.json")


@dataclass(frozen=True)
class NuSetImage:
    image_id: str
    image_path: Path
    label_path: Path
    image: torch.Tensor
    instance_map: np.ndarray
    image_sha256: str
    label_sha256: str


@dataclass(frozen=True)
class NuSetCrop:
    image_id: str
    crop_id: int
    crop_box_xyxy: tuple[int, int, int, int]
    image: torch.Tensor
    instance_map: np.ndarray
    gt_instance_ids: np.ndarray
    gt_masks: np.ndarray
    gt_centroids_xy: np.ndarray
    image_crop_sha256: str
    gt_crop_sha256: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _normalize(image: np.ndarray) -> torch.Tensor:
    normalized = A.Normalize()(image=image)["image"]
    return torch.from_numpy(np.ascontiguousarray(normalized.transpose(2, 0, 1))).float()


def _centroid_inside(mask: np.ndarray) -> np.ndarray:
    coordinates_yx = np.argwhere(mask)
    center_yx = np.rint(coordinates_yx.mean(axis=0)).astype(np.int64)
    if not mask[tuple(center_yx)]:
        center_yx = coordinates_yx[np.sum((coordinates_yx - center_yx[None, :]) ** 2, axis=1).argmin()]
    return center_yx[[1, 0]].astype(np.float32)


def _image_path(image_root: Path, image_id: str) -> Path:
    matches = [image_root / f"{image_id}{suffix}" for suffix in IMAGE_SUFFIXES if (image_root / f"{image_id}{suffix}").is_file()]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one fixed NuSet image for {image_id}, got {matches}")
    return matches[0]


def load_fixed_selection(selection_path: Path = EXPECTED_SELECTION_PATH) -> dict:
    """Read, never regenerate, the PromptCredit Stage-0 fixed-six manifest."""
    payload = json.loads(selection_path.read_text(encoding="utf-8"))
    ids = list(payload.get("image_ids", []))
    if len(ids) != 6 or len(set(ids)) != 6:
        raise ValueError("NuSet requires exactly the immutable six-image Stage-0 selection")
    if any(int(image_id.split("_", 1)[0]) not in range(1, 7) for image_id in ids):
        raise ValueError("NuSet selection contains a closed TNBC patient")
    if payload.get("selection_seed") != 3407:
        raise ValueError("NuSet must recover the fixed seed-3407 selection")
    return payload


def resolve_fixed_images(data_root: Path, selection: dict) -> list[tuple[str, Path, Path]]:
    """Resolve direct permitted files only; never enumerate test/calibration roots."""
    if data_root.name.lower() != "tnbc":
        raise ValueError("NuSet Stage 0 authorizes TNBC only; MoNuSeg is prohibited")
    image_root, label_root = data_root / "train_12" / "images", data_root / "train_12" / "labels"
    if not image_root.is_dir() or not label_root.is_dir():
        raise FileNotFoundError("Expected TNBC train_12 image and label roots")
    result: list[tuple[str, Path, Path]] = []
    for image_id in selection["image_ids"]:
        label_path = label_root / f"{image_id}.mat"
        if not label_path.is_file():
            raise FileNotFoundError(f"Missing fixed NuSet label: {label_path}")
        result.append((image_id, _image_path(image_root, image_id), label_path))
    return result


def iter_fixed_images(data_root: Path, selection: dict) -> Iterator[NuSetImage]:
    for image_id, image_path, label_path in resolve_fixed_images(data_root, selection):
        image = io.imread(image_path)[..., :3]
        instance_map = np.asarray(sio.loadmat(label_path)["inst_map"])
        if image.shape[:2] != instance_map.shape:
            raise ValueError(f"NuSet image/GT shape mismatch for {image_id}")
        yield NuSetImage(
            image_id=image_id,
            image_path=image_path,
            label_path=label_path,
            image=_normalize(image),
            instance_map=instance_map,
            image_sha256=sha256_file(image_path),
            label_sha256=sha256_file(label_path),
        )


def iter_fixed_crops(data_root: Path, selection: dict, *, overlap: int = 32) -> Iterator[NuSetCrop]:
    """Yield only nonempty 256px unclockwise training-style crops."""
    from run.run_on_epoch import crop_with_overlap

    if overlap != 32:
        raise ValueError("NuSet Stage 0 fixes baseline overlap to 32")
    for item in iter_fixed_images(data_root, selection):
        boxes = crop_with_overlap(item.image, 256, 256, overlap, "unclockwise").tolist()
        for crop_id, (x1, y1, x2, y2) in enumerate(boxes):
            crop_map = item.instance_map[y1:y2, x1:x2]
            ids = np.unique(crop_map)
            ids = ids[ids != 0]
            if not len(ids):
                continue
            masks = np.stack([crop_map == instance_id for instance_id in ids]).astype(bool)
            yield NuSetCrop(
                image_id=item.image_id,
                crop_id=crop_id,
                crop_box_xyxy=(int(x1), int(y1), int(x2), int(y2)),
                image=item.image[:, y1:y2, x1:x2],
                instance_map=crop_map,
                gt_instance_ids=ids.astype(np.int64),
                gt_masks=masks,
                gt_centroids_xy=np.stack([_centroid_inside(mask) for mask in masks]).astype(np.float32),
                image_crop_sha256=_array_sha256(item.image[:, y1:y2, x1:x2].numpy()),
                gt_crop_sha256=_array_sha256(crop_map),
            )
