"""Immutable on-disk cache format for NuRank automatic prompt groups."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np


TOKEN_COUNT = 4


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(cache_dir: Path) -> dict[str, Any]:
    path = cache_dir / "manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "nurank_automatic_prompt_cache_v1":
        raise ValueError(f"Unsupported NuRank cache schema in {path}")
    if payload.get("token_count") != TOKEN_COUNT:
        raise ValueError("NuRank requires exactly four cached mask tokens")
    return payload


def iter_groups(cache_dir: Path) -> Iterator[dict[str, Any]]:
    """Yield CPU arrays, preserving one four-token group per decoder call."""
    manifest = load_manifest(cache_dir)
    for entry in manifest["groups"]:
        path = cache_dir / entry["path"]
        if sha256_file(path) != entry["sha256"]:
            raise RuntimeError(f"NuRank cache checksum mismatch: {path}")
        with np.load(path, allow_pickle=False) as payload:
            group = {name: payload[name] for name in payload.files}
        count = int(group["mask_tokens"].shape[0])
        if group["mask_tokens"].shape[1:] != (TOKEN_COUNT, 256):
            raise RuntimeError(f"Invalid cached token layout: {path}")
        for key in ("original_predicted_iou", "morphology", "true_hard_iou", "true_soft_iou"):
            if group[key].shape[:2] != (count, TOKEN_COUNT):
                raise RuntimeError(f"Invalid four-token field {key}: {path}")
        if group["mask_logits"].shape[:2] != (count, TOKEN_COUNT):
            raise RuntimeError(f"Invalid cached mask layout: {path}")
        if "low_res_logits" in group and group["low_res_logits"].shape[:2] != (count, TOKEN_COUNT):
            raise RuntimeError(f"Invalid cached low-resolution mask layout: {path}")
        if not np.array_equal(group.get("token_index"), np.tile(np.arange(TOKEN_COUNT, dtype=np.int64), (count, 1))):
            raise RuntimeError(f"Invalid cached token index record: {path}")
        group["_entry"] = entry
        yield group


def group_feature_matrix(group: dict[str, Any]) -> np.ndarray:
    """Return token feature vectors [prompts, 4, 264], before train normalization."""
    tokens = np.asarray(group["mask_tokens"], dtype=np.float32)
    scalar = np.concatenate(
        (
            np.asarray(group["original_predicted_iou"], dtype=np.float32)[..., None],
            np.asarray(group["morphology"], dtype=np.float32),
        ),
        axis=-1,
    )
    if tokens.shape[:2] != scalar.shape[:2] or scalar.shape[-1] != 8:
        raise ValueError("NuRank cache feature shape is invalid")
    return np.concatenate((tokens, scalar), axis=-1)


def cache_patient_ids(manifest: dict[str, Any]) -> set[int]:
    return {int(image_id.split("_", 1)[0]) for image_id in manifest["image_ids"]}
