"""Explicit TNBC 1--6 / 7--8 access with no test-root enumeration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal

import albumentations as A
import numpy as np
import scipy.io as sio
import torch
from skimage import io

from .protocol import json_dump, sha256_file


Role = Literal["train", "development"]
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


@dataclass(frozen=True)
class AuthorizedImage:
    image_id: str
    patient: int
    image_path: Path
    label_path: Path


@dataclass(frozen=True)
class ImageCrop:
    image_id: str
    patient: int
    crop_id: int
    crop_box_xyxy: tuple[int, int, int, int]
    image: torch.Tensor


def _patient(image_id: str) -> int:
    return int(image_id.split("_", 1)[0])


def read_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    required = {"dataset", "train_image_root_relative", "label_root_relative", "train_image_ids", "development_image_ids"}
    if manifest.get("dataset") != "TNBC" or not required.issubset(manifest):
        raise ValueError("PromptQ-v2 needs the committed explicit TNBC manifest")
    if any("9_" in item or "10_" in item or "11_" in item for item in manifest["train_image_ids"] + manifest["development_image_ids"]):
        raise ValueError("PromptQ-v2 manifest must not name closed patients")
    if len(manifest["development_image_ids"]) != 7 or {_patient(item) for item in manifest["development_image_ids"]} != {7, 8}:
        raise ValueError("development audit must contain exactly seven patients-7--8 images")
    return manifest


def _image_path(root: Path, image_id: str) -> Path:
    matches = [root / f"{image_id}{suffix}" for suffix in IMAGE_SUFFIXES if (root / f"{image_id}{suffix}").is_file()]
    if len(matches) != 1:
        raise FileNotFoundError(f"expected exactly one image for authorized id {image_id}: {matches}")
    return matches[0]


def authorized_images(data_root: Path, manifest_path: Path, role: Role) -> list[AuthorizedImage]:
    manifest = read_manifest(manifest_path)
    ids = list(manifest["train_image_ids"] if role == "train" else manifest["development_image_ids"])
    allowed = set(manifest["train_patients"] if role == "train" else manifest["development_patients"])
    if any(_patient(image_id) not in allowed for image_id in ids):
        raise ValueError(f"unauthorized {role} patient in manifest")
    image_root = data_root / manifest["train_image_root_relative"]
    label_root = data_root / manifest["label_root_relative"]
    if not image_root.is_dir() or not label_root.is_dir():
        raise FileNotFoundError("authorized TNBC train_12 roots are unavailable")
    # These exact paths are formed from committed IDs; test/ is never listed or opened.
    items = [AuthorizedImage(image_id, _patient(image_id), _image_path(image_root, image_id), label_root / f"{image_id}.mat") for image_id in ids]
    if any(not item.label_path.is_file() for item in items):
        raise FileNotFoundError("an authorized TNBC label file is missing")
    return items


def normalize_image(image: np.ndarray) -> torch.Tensor:
    normalized = A.Normalize()(image=image)["image"]
    return torch.from_numpy(np.ascontiguousarray(normalized.transpose(2, 0, 1))).to(torch.float32)


def crop_boxes(image_tensor: torch.Tensor, *, overlap: int = 32) -> list[tuple[int, int, int, int]]:
    from .assembly import crop_with_overlap

    boxes = crop_with_overlap(image_tensor, 256, 256, overlap, "unclockwise").tolist()
    return [tuple(int(value) for value in box) for box in boxes]


def iter_crops(data_root: Path, manifest_path: Path, role: Role) -> Iterator[ImageCrop]:
    for item in authorized_images(data_root, manifest_path, role):
        image = io.imread(item.image_path)[..., :3]
        image_tensor = normalize_image(image)
        for crop_id, box in enumerate(crop_boxes(image_tensor)):
            x1, y1, x2, y2 = box
            yield ImageCrop(item.image_id, item.patient, crop_id, box, image_tensor[:, y1:y2, x1:x2])


def materialize_labels(data_root: Path, manifest_path: Path, role: Role, out_dir: Path) -> dict:
    """Write GT in a label-only store, deliberately outside deployment cache."""
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite label store: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=False)
    records = []
    for item in authorized_images(data_root, manifest_path, role):
        instance_map = np.asarray(sio.loadmat(item.label_path)["inst_map"], dtype=np.int32)
        if instance_map.ndim != 2:
            raise ValueError(f"invalid instance map for {item.image_id}")
        path = out_dir / f"{item.image_id}.npz"
        np.savez_compressed(path, instance_map=instance_map)
        records.append({
            "image_id": item.image_id,
            "patient": item.patient,
            "label_file": path.name,
            "label_sha256": sha256_file(path),
            "source_image_sha256": sha256_file(item.image_path),
            "source_label_sha256": sha256_file(item.label_path),
        })
    payload = {"schema_version": 1, "role": role, "contains_gt": True, "records": records}
    json_dump(out_dir / "manifest.json", payload)
    return payload


def load_label(label_dir: Path, image_id: str) -> np.ndarray:
    with np.load(label_dir / f"{image_id}.npz", allow_pickle=False) as payload:
        return np.asarray(payload["instance_map"], dtype=np.int32)
