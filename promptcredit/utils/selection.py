"""Deterministic, GT-free selection of the PromptCredit Stage 0 images."""

from __future__ import annotations

import hashlib
import json
from typing import Any


IMAGE_SELECTION_SEED = 3407
SELECTION_COUNT = 6


def canonical_json_sha256(payload: dict[str, Any]) -> str:
    """Hash JSON canonically, excluding no fields supplied by the caller."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def image_list_sha256(image_ids: list[str]) -> str:
    return hashlib.sha256(("\n".join(image_ids) + "\n").encode("utf-8")).hexdigest()


def _validate_router_train_manifest(manifest: dict[str, Any]) -> list[str]:
    if manifest.get("dataset") != "TNBC":
        raise ValueError("PromptCredit Stage 0 is restricted to the TNBC manifest")
    image_ids = manifest.get("router_train")
    if not isinstance(image_ids, list) or not image_ids:
        raise ValueError("TNBC split manifest lacks a non-empty router_train list")
    if any(not isinstance(image_id, str) for image_id in image_ids):
        raise ValueError("router_train image IDs must be strings")
    patients = {int(image_id.split("_", 1)[0]) for image_id in image_ids}
    if patients != set(range(1, 7)):
        raise ValueError("PromptCredit Stage 0 permits TNBC patients 1--6 only")
    if set(image_ids) & set(manifest.get("calibration", [])):
        raise ValueError("router_train and calibration must be disjoint")
    if set(image_ids) & set(manifest.get("test", [])):
        raise ValueError("router_train and test must be disjoint")
    return sorted(image_ids)


def derive_selected_image_ids(
    manifest: dict[str, Any], *, seed: int = IMAGE_SELECTION_SEED, count: int = SELECTION_COUNT
) -> list[str]:
    """Return the pre-registered SHA256 ordering without opening any data file."""
    if seed != IMAGE_SELECTION_SEED or count != SELECTION_COUNT:
        raise ValueError("Stage 0 image selection seed and count are frozen")
    candidates = _validate_router_train_manifest(manifest)
    if len(candidates) < count:
        raise ValueError(f"Need at least {count} router-train images, found {len(candidates)}")
    return sorted(
        candidates,
        key=lambda image_id: (
            hashlib.sha256(f"{seed}:{image_id}".encode("utf-8")).hexdigest(),
            image_id,
        ),
    )[:count]


def build_selection_payload(manifest: dict[str, Any]) -> dict[str, Any]:
    """Build the committed selection manifest from a committed split manifest."""
    image_ids = derive_selected_image_ids(manifest)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "project": "PromptCredit",
        "stage": "PC-Stage 0",
        "dataset": "TNBC",
        "authorized_split": "router_train",
        "allowed_patients": [1, 2, 3, 4, 5, 6],
        "selection_algorithm": "ascending SHA256('3407:' + image_id), then image_id",
        "selection_seed": IMAGE_SELECTION_SEED,
        "selection_count": SELECTION_COUNT,
        "source_split_content_sha256": manifest.get("content_sha256"),
        "image_ids": image_ids,
        "image_ids_sha256": image_list_sha256(image_ids),
        "prohibited": ["calibration", "test", "MoNuSeg"],
    }
    payload["content_sha256"] = canonical_json_sha256(payload)
    return payload


def validate_selection_payload(selection: dict[str, Any], manifest: dict[str, Any]) -> list[str]:
    """Reject changed image membership, split leakage, or stale source metadata."""
    expected = build_selection_payload(manifest)
    expected_checksum = expected.pop("content_sha256")
    observed_checksum = selection.get("content_sha256")
    observed = dict(selection)
    observed.pop("content_sha256", None)
    if observed_checksum != canonical_json_sha256(observed):
        raise ValueError("PromptCredit selection manifest checksum mismatch")
    if observed_checksum != expected_checksum or observed != expected:
        raise ValueError("PromptCredit selection manifest does not match the frozen router-train selection")
    return list(expected["image_ids"])

