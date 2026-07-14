"""Fail-closed preparation of the frozen ResiMix-PMS input protocol.

The supplied SetPMS artifact contains the authoritative MoNuSeg-Lite split
and coordinates, but this route must not inherit SetPMS code or silently
reinterpret its records.  This module copies the checked raw bundle and
derives run manifests only through explicit JSON pointers registered in the
stage specification.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from resimixpms.experiment import sha256_file, write_json
from resimixpms.manifests import (
    FROZEN_CROP_INDEX_SCHEDULE_FORMAT,
    REQUIRED_FROZEN_FILES,
    ManifestPreflightError,
    copy_validated_frozen_bundle,
    load_allowed_image_names,
    load_crop_plan,
    load_crop_records,
    validate_frozen_bundle,
)


class ProtocolError(RuntimeError):
    """A pre-registered ResiMix input protocol is malformed or incomplete."""


def _read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"cannot read frozen JSON input {path}") from exc


def _resolve_pointer(payload: Any, pointer: str) -> Any:
    if pointer in ("", "/"):
        return payload
    if not pointer.startswith("/"):
        raise ProtocolError(f"JSON pointer must begin with '/': {pointer!r}")
    value = payload
    for raw_piece in pointer[1:].split("/"):
        piece = raw_piece.replace("~1", "/").replace("~0", "~")
        if isinstance(value, Mapping):
            if piece not in value:
                raise ProtocolError(f"frozen JSON pointer is absent: {pointer!r}")
            value = value[piece]
        elif isinstance(value, list):
            try:
                index = int(piece)
            except ValueError as exc:
                raise ProtocolError(f"list JSON pointer token is not an index: {piece!r}") from exc
            if index < 0 or index >= len(value):
                raise ProtocolError(f"frozen JSON pointer index is outside its list: {pointer!r}")
            value = value[index]
        else:
            raise ProtocolError(f"frozen JSON pointer descends through a scalar: {pointer!r}")
    return value


def _source_and_pointer(reference: str) -> tuple[str, str]:
    source, marker, pointer = str(reference).partition("#")
    if not source:
        raise ProtocolError("frozen protocol reference lacks a source filename")
    if Path(source).name != source or source not in REQUIRED_FROZEN_FILES[:-1]:
        raise ProtocolError(f"frozen protocol source is not an approved raw manifest: {source!r}")
    return source, pointer if marker else ""


def _records(value: Any, label: str) -> list[Any]:
    if isinstance(value, list):
        rows = value
    elif isinstance(value, Mapping):
        rows = None
        for key in ("records", "items", "images", "patches", "crops", "data"):
            if isinstance(value.get(key), list):
                rows = value[key]
                break
        if rows is None and any(key in value for key in ("image_name", "image", "name")):
            rows = [dict(value)]
    else:
        rows = None
    if not isinstance(rows, list) or not rows:
        raise ProtocolError(f"frozen protocol selection {label!r} is not a non-empty record list")
    return rows


def _selected_records(bundle_dir: Path, reference: str, label: str) -> list[Any]:
    source_name, pointer = _source_and_pointer(reference)
    return _records(_resolve_pointer(_read_json(bundle_dir / source_name), pointer), label)


def _write_records(path: Path, rows: list[Any]) -> None:
    write_json(path, {"records": rows})


def _is_epoch_crop_index_mapping(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and bool(value)
        and all(str(key).isdigit() and isinstance(child, Mapping) for key, child in value.items())
    )


def _write_crop_index_schedule(
    path: Path,
    *,
    parent_manifest: Any,
    selected_indices: Mapping[str, Any],
    source_name: str,
    pointer: str,
    source_sha256: str,
    crop_size: int,
    overlap: int,
    load: str,
) -> None:
    if not isinstance(parent_manifest, Mapping):
        raise ProtocolError("frozen crop-index source must be an object")
    try:
        max_crops = int(parent_manifest["max_train_crops_per_image"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ProtocolError("frozen crop-index source lacks max_train_crops_per_image") from exc
    if max_crops <= 0:
        raise ProtocolError("frozen crop-index source has invalid max_train_crops_per_image")
    write_json(path, {
        "format": FROZEN_CROP_INDEX_SCHEDULE_FORMAT,
        "source": {"file": source_name, "pointer": pointer, "sha256": source_sha256},
        "crop_size": int(crop_size),
        "overlap": int(overlap),
        "load": str(load),
        "epochs": len(selected_indices),
        "max_crops_per_image": max_crops,
        "epoch_crop_indices": dict(selected_indices),
    })


def derive_monuseg_lite_protocol(
    source_bundle: str | Path,
    selection: Mapping[str, str],
    destination: str | Path,
    *,
    crop_size: int = 256,
    overlap: int = 92,
    load: str = "unclockwise",
    epochs: int = 10,
) -> dict[str, Any]:
    """Copy and derive the exact frozen MoNuSeg-Lite inputs for this run.

    ``selection`` must have four explicit source#JSON-pointer entries.  There
    is deliberately no heuristic fallback: absent schema information stops the
    stage before any image, label, or official-test path is touched.
    """
    required = {"train_images", "development_images", "train_crops", "evaluation_patches"}
    if set(selection) != required or not all(isinstance(selection[key], str) for key in required):
        raise ProtocolError(f"frozen_protocol must contain exactly {sorted(required)}")
    root = Path(destination)
    if root.exists():
        raise FileExistsError(f"refusing to replace frozen protocol destination: {root}")
    copied = root / "raw"
    copy_validated_frozen_bundle(source_bundle, copied)
    copied_bundle = validate_frozen_bundle(copied)
    derived = root / "derived"
    derived.mkdir()

    train_rows = _selected_records(copied, selection["train_images"], "train_images")
    dev_rows = _selected_records(copied, selection["development_images"], "development_images")
    crop_source_name, crop_pointer = _source_and_pointer(selection["train_crops"])
    crop_parent = _read_json(copied / crop_source_name)
    crop_value = _resolve_pointer(crop_parent, crop_pointer)
    patch_rows = _selected_records(copied, selection["evaluation_patches"], "evaluation_patches")
    paths = {
        "train_manifest": derived / "train_manifest.json",
        "test_manifest": derived / "development_manifest.json",
        "train_crop_manifest": derived / "train_crops.json",
        "eval_crop_manifest": derived / "evaluation_patches.json",
    }
    _write_records(paths["train_manifest"], train_rows)
    _write_records(paths["test_manifest"], dev_rows)
    if _is_epoch_crop_index_mapping(crop_value):
        _write_crop_index_schedule(
            paths["train_crop_manifest"],
            parent_manifest=crop_parent,
            selected_indices=crop_value,
            source_name=crop_source_name,
            pointer=crop_pointer,
            source_sha256=copied_bundle.file_sha256[crop_source_name],
            crop_size=crop_size,
            overlap=overlap,
            load=load,
        )
    else:
        _write_records(paths["train_crop_manifest"], _records(crop_value, "train_crops"))
    _write_records(paths["eval_crop_manifest"], patch_rows)

    train_names = load_allowed_image_names(paths["train_manifest"])
    dev_names = load_allowed_image_names(paths["test_manifest"])
    if len(dev_names) != 6:
        raise ProtocolError(f"frozen MoNuSeg-Lite development set must contain exactly 6 images, got {len(dev_names)}")
    if set(train_names) & set(dev_names):
        raise ProtocolError("frozen MoNuSeg-Lite train and development image manifests overlap")
    train_crops, crop_schedule = load_crop_plan(
        paths["train_crop_manifest"],
        allowed_image_names=train_names,
        expected_crop_size=crop_size,
        expected_overlap=overlap,
        expected_load=load,
        expected_epochs=epochs,
    )
    patches = load_crop_records(paths["eval_crop_manifest"])
    if len(patches) != 12:
        raise ProtocolError(f"frozen MoNuSeg-Lite evaluation manifest must contain exactly 12 patches, got {len(patches)}")
    if crop_schedule is None and not {str(row["image_name"]) for row in train_crops} <= set(train_names):
        raise ProtocolError("a frozen training crop references an image outside the frozen training split")
    if not {str(row["image_name"]) for row in patches} <= set(dev_names):
        raise ProtocolError("a frozen evaluation patch references an image outside the frozen development split")

    provenance = {
        "raw_bundle": copied_bundle.as_dict(),
        "selection": dict(selection),
        "derived": {
            key: {"path": str(path), "sha256": sha256_file(path)} for key, path in paths.items()
        },
        "counts": {
            "train_images": len(train_names),
            "development_images": len(dev_names),
            "train_crops": len(train_crops),
            "train_crop_schedule_epochs": len(crop_schedule.epochs) if crop_schedule else 0,
            "train_crop_schedule_assignments": crop_schedule.total_assignments if crop_schedule else 0,
            "evaluation_patches": len(patches),
        },
    }
    write_json(root / "protocol_provenance.json", provenance)
    return {key: str(path) for key, path in paths.items()} | {"provenance": provenance}
