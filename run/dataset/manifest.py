"""Strict ordered dataset-manifest loading for StainPMS experiments."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class ManifestError(RuntimeError):
    """Raised before dataset samples are opened when a manifest is invalid."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(value: str, manifest_path: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def load_dataset_manifest(
    manifest_value: str | Path,
    *,
    expected_dataset: str,
    require_labels: bool = True,
    verify_hashes: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = Path(manifest_value).resolve()
    if not manifest_path.is_file():
        raise ManifestError(f"manifest not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ManifestError("manifest must be a JSON object")
    if str(payload.get("dataset", "")).lower() != expected_dataset.lower():
        raise ManifestError(
            f"manifest dataset {payload.get('dataset')!r} != {expected_dataset!r}"
        )
    raw_records = payload.get("records")
    if not isinstance(raw_records, list) or not raw_records:
        raise ManifestError("manifest must contain a non-empty records list")
    if payload.get("record_count") not in (None, len(raw_records)):
        raise ManifestError(
            f"record_count={payload.get('record_count')} != records={len(raw_records)}"
        )

    records: list[dict[str, Any]] = []
    sample_ids: set[str] = set()
    image_paths: set[str] = set()
    for index, raw in enumerate(raw_records):
        if not isinstance(raw, dict):
            raise ManifestError(f"record {index} is not an object")
        sample_id = str(raw.get("sample_id") or "")
        if not sample_id:
            raise ManifestError(f"record {index} has no sample_id")
        if sample_id in sample_ids:
            raise ManifestError(f"duplicate sample_id in manifest: {sample_id}")
        sample_ids.add(sample_id)
        image_value = raw.get("image_path")
        if not image_value:
            raise ManifestError(
                f"record {sample_id} has no loader-runnable image_path; "
                "identity-only manifests cannot construct a dataset"
            )
        image_path = _resolve(str(image_value), manifest_path)
        if not image_path.is_file():
            raise ManifestError(f"image not found for {sample_id}: {image_path}")
        image_key = str(image_path).casefold()
        if image_key in image_paths:
            raise ManifestError(f"duplicate image path in manifest: {image_path}")
        image_paths.add(image_key)
        label_path = None
        if raw.get("label_path"):
            label_path = _resolve(str(raw["label_path"]), manifest_path)
        if require_labels and (label_path is None or not label_path.is_file()):
            raise ManifestError(f"label not found for {sample_id}: {label_path}")
        if verify_hashes:
            expected_image_sha = raw.get("image_sha256")
            if not expected_image_sha:
                raise ManifestError(f"image_sha256 missing for {sample_id}")
            actual_image_sha = sha256_file(image_path)
            if actual_image_sha.lower() != str(expected_image_sha).lower():
                raise ManifestError(f"image SHA256 mismatch for {sample_id}")
            if require_labels:
                expected_label_sha = raw.get("label_sha256")
                if not expected_label_sha:
                    raise ManifestError(f"label_sha256 missing for {sample_id}")
                assert label_path is not None
                actual_label_sha = sha256_file(label_path)
                if actual_label_sha.lower() != str(expected_label_sha).lower():
                    raise ManifestError(f"label SHA256 mismatch for {sample_id}")
        record = dict(raw)
        record["sample_id"] = sample_id
        record["image_path"] = str(image_path)
        record["label_path"] = str(label_path) if label_path is not None else None
        record["manifest_index"] = index
        records.append(record)

    payload = dict(payload)
    payload["manifest_path"] = str(manifest_path)
    payload["manifest_sha256"] = sha256_file(manifest_path)
    return payload, records
