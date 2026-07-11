"""Hard scope guards for the authorized PromptCredit Stage 0 audit."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from promptcredit.utils.selection import validate_selection_payload


IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
BASELINE_V1_TNBC_SHA256 = "44a3cb3e93051301d789e44f93769588abfa727d7a174b0270f55305ef023781"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_stage0_inputs(
    *, data_root: Path, split_manifest: dict[str, Any], selection: dict[str, Any], checkpoint: Path
) -> list[str]:
    """Validate scope before opening an image, GT label, or checkpoint tensor."""
    if data_root.name.lower() != "tnbc":
        raise ValueError("PC-Stage 0 only accepts a TNBC data root; MoNuSeg is prohibited")
    image_ids = validate_selection_payload(selection, split_manifest)
    if any(int(image_id.split("_", 1)[0]) not in range(1, 7) for image_id in image_ids):
        raise ValueError("PC-Stage 0 selection contains a non-router-train patient")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Frozen TNBC baseline checkpoint is missing: {checkpoint}")
    observed = sha256_file(checkpoint)
    if observed != BASELINE_V1_TNBC_SHA256:
        raise ValueError("Checkpoint SHA256 does not match frozen TNBC StainPMS baseline v1")
    return image_ids


def selected_tnbc_paths(data_root: Path, image_ids: list[str]) -> list[tuple[str, Path, Path]]:
    """Resolve only exact allowed router-train image/label pairs; never list test paths."""
    image_root = data_root / "train_12" / "images"
    label_root = data_root / "train_12" / "labels"
    if not image_root.is_dir() or not label_root.is_dir():
        raise FileNotFoundError("Expected TNBC router-train train_12/images and train_12/labels directories")
    resolved: list[tuple[str, Path, Path]] = []
    for image_id in image_ids:
        matches = [image_root / f"{image_id}{suffix}" for suffix in IMAGE_SUFFIXES]
        existing = [path for path in matches if path.is_file()]
        if len(existing) != 1:
            raise FileNotFoundError(f"Expected exactly one authorized image for {image_id}, found {existing}")
        label_path = label_root / f"{image_id}.mat"
        if not label_path.is_file():
            raise FileNotFoundError(f"Missing authorized GT label for {image_id}: {label_path}")
        resolved.append((image_id, existing[0], label_path))
    return resolved

