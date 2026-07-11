"""Direct, closed-split TNBC data access for NuRank Stage 1."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from nuset.audit.data import NuSetImage, _image_path, _normalize, sha256_file

import numpy as np
import scipy.io as sio
from skimage import io


Role = Literal["train", "development"]
TRAIN_PATIENTS = frozenset(range(1, 7))
DEVELOPMENT_PATIENTS = frozenset((7, 8))


def _patient(image_id: str) -> int:
    try:
        return int(image_id.split("_", 1)[0])
    except (IndexError, ValueError) as error:
        raise ValueError(f"Invalid TNBC image ID: {image_id}") from error


def resolve_nurank_images(data_root: Path, split_manifest_path: Path, role: Role) -> list[NuSetImage]:
    """Resolve only named 1-6 or 7-8 files from train_12; never list test."""
    if data_root.name.lower() != "tnbc":
        raise ValueError("NuRank authorizes TNBC only; MoNuSeg is prohibited")
    payload = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    if payload.get("dataset") != "TNBC" or payload.get("split_method") != "fixed_patient_level":
        raise ValueError("NuRank requires the fixed TNBC patient-level split manifest")
    if role == "train":
        image_ids, allowed = list(payload["router_train"]), TRAIN_PATIENTS
    elif role == "development":
        image_ids, allowed = list(payload["calibration"]), DEVELOPMENT_PATIENTS
    else:
        raise ValueError(f"Unsupported NuRank role {role}")
    if any(_patient(image_id) not in allowed for image_id in image_ids):
        raise ValueError(f"NuRank {role} manifest contains an unauthorized patient")
    if role == "development" and set(_patient(image_id) for image_id in image_ids) != DEVELOPMENT_PATIENTS:
        raise ValueError("NuRank development must contain exactly patients 7 and 8")
    image_root, label_root = data_root / "train_12" / "images", data_root / "train_12" / "labels"
    if not image_root.is_dir() or not label_root.is_dir():
        raise FileNotFoundError("NuRank requires the frozen TNBC train_12 snapshot")
    items: list[NuSetImage] = []
    for image_id in image_ids:
        image_path, label_path = _image_path(image_root, image_id), label_root / f"{image_id}.mat"
        if not label_path.is_file():
            raise FileNotFoundError(f"Missing NuRank fixed label {label_path}")
        image = io.imread(image_path)[..., :3]
        instance_map = np.asarray(sio.loadmat(label_path)["inst_map"])
        if image.shape[:2] != instance_map.shape:
            raise ValueError(f"NuRank image/GT shape mismatch for {image_id}")
        items.append(NuSetImage(
            image_id=image_id, image_path=image_path, label_path=label_path,
            image=_normalize(image), instance_map=instance_map,
            image_sha256=sha256_file(image_path), label_sha256=sha256_file(label_path),
        ))
    return items
