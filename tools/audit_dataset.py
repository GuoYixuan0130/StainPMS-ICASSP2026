"""Manifest-gated Phase 0 dataset audit for F3C-StainPMS.

The audit never walks a test directory.  Every sample must either come from an
explicit manifest or from an explicitly enabled *training-pool discovery* block
whose root is checked before enumeration.  TNBC patients 9--11 and MoNuSeg
official-test paths are rejected before any image or label is opened.

The JSON output is the machine-readable source of truth.  ``--summary-output``
adds a short human-readable companion without changing the audit result.
"""

from __future__ import annotations

import argparse
import importlib.util
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import types
from collections import Counter, defaultdict
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy import ndimage as ndi
from scipy.io import loadmat
from skimage import io
from skimage.feature import peak_local_max
from skimage.segmentation import relabel_sequential, watershed


IMAGE_EXTENSIONS = {".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"}
TNBC_FORBIDDEN_PATIENTS = {9, 10, 11}
MONUSEG_FORBIDDEN_SPLITS = {"test", "official_test", "official-test"}


class ProtocolViolation(RuntimeError):
    """Raised before I/O when a manifest violates a closed-data rule."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_git_value(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _environment_record(argv: list[str]) -> dict[str, Any]:
    versions: dict[str, str] = {}
    for name in ("numpy", "scipy", "skimage"):
        try:
            module = __import__(name)
            versions[name] = str(getattr(module, "__version__", "unknown"))
        except Exception as exc:  # pragma: no cover - defensive reporting
            versions[name] = f"unavailable:{type(exc).__name__}"
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": argv,
        "python": sys.version,
        "platform": platform.platform(),
        "packages": versions,
        "git_branch": _safe_git_value("branch", "--show-current"),
        "git_commit": _safe_git_value("rev-parse", "HEAD"),
    }


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _extract_manifest_entries(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        raise ValueError("manifest must be a JSON list or object")
    for key in (
        "samples",
        "entries",
        "items",
        "paths",
        "files",
        "images",
        "records",
        "crops",
        "patches",
        "allowed_images",
    ):
        value = payload.get(key)
        if isinstance(value, list):
            return value
    data = payload.get("data")
    if isinstance(data, list):
        return data
    # Support an id -> metadata mapping, but do not silently reinterpret a
    # configuration object as a sample map.
    ignored = {
        "schema_version",
        "dataset",
        "split",
        "metadata",
        "protocol",
        "roots",
    }
    if payload and not (set(payload) & ignored):
        return [
            ({"sample_id": key, **value} if isinstance(value, dict) else {"sample_id": key, "image": value})
            for key, value in payload.items()
        ]
    raise ValueError("manifest has no supported sample-list key")


def _path_segments(value: str | os.PathLike[str]) -> set[str]:
    normalized = str(value).replace("\\", "/").lower()
    return {part for part in normalized.split("/") if part}


def _resolve_path(value: str | None, root: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute() or not root:
        return path
    return Path(root) / path


def _infer_tnbc_patient(stem: str) -> int | None:
    match = re.match(r"^(?:patient[_-]?)?(\d{1,2})(?:[_-]|$)", stem, re.IGNORECASE)
    return int(match.group(1)) if match else None


def _infer_tcga_case(stem: str) -> str | None:
    match = re.match(r"^(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})", stem, re.IGNORECASE)
    return match.group(1).upper() if match else None


def _first_present(mapping: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value not in (None, ""):
            return value
    return None


def _normalize_entry(
    raw: Any,
    *,
    dataset: str,
    split_name: str,
    roots: dict[str, Any],
    metadata_map: dict[str, Any],
) -> dict[str, Any]:
    if isinstance(raw, str):
        entry: dict[str, Any] = {"image": raw}
    elif isinstance(raw, dict):
        entry = dict(raw)
    else:
        raise ValueError(f"unsupported manifest entry type: {type(raw).__name__}")

    image_value = _first_present(
        entry,
        (
            "image",
            "image_name",
            "image_path",
            "img",
            "path",
            "file",
            "filename",
            "name",
        ),
    )
    if image_value is None:
        raise ValueError(f"sample has no image path: {entry}")
    image_value_text = str(image_value)
    image_extension = str(roots.get("image_extension") or "")
    if not Path(image_value_text).suffix and image_extension:
        if not image_extension.startswith("."):
            image_extension = "." + image_extension
        image_value_text += image_extension
    image_path = _resolve_path(image_value_text, roots.get("image_root"))
    assert image_path is not None
    stem = image_path.stem

    metadata = metadata_map.get(stem, {})
    if not isinstance(metadata, dict):
        metadata = {}
    combined = {**metadata, **entry}
    label_value = _first_present(combined, ("label", "label_path", "gt", "mask"))
    if label_value is None:
        label_path = _resolve_path(stem + ".mat", roots.get("label_root"))
    else:
        label_path = _resolve_path(str(label_value), roots.get("label_root"))
    raw_label_value = _first_present(
        combined, ("raw_label", "raw_label_path", "official_binary_gt")
    )
    raw_label_path = _resolve_path(
        str(raw_label_value) if raw_label_value is not None else None,
        roots.get("raw_label_root"),
    )

    patient = _first_present(
        combined, ("patient", "patient_id", "patient_number", "subject")
    )
    if dataset == "tnbc" and patient is None:
        patient = _infer_tnbc_patient(stem)
    case = _first_present(combined, ("case", "case_id", "slide", "wsi"))
    if case is None and dataset == "monuseg":
        case = _infer_tcga_case(stem)
    if dataset == "monuseg" and patient is None:
        patient = case

    return {
        "sample_id": str(combined.get("sample_id") or stem),
        "split": split_name,
        "image_path": str(image_path),
        "label_path": str(label_path) if label_path is not None else None,
        "raw_label_path": str(raw_label_path) if raw_label_path is not None else None,
        "patient": patient,
        "case": case,
        "organ": _first_present(combined, ("organ", "tissue", "site")),
        "source": combined.get("source"),
    }


def _validate_protocol(
    dataset: str,
    split_name: str,
    entries: list[dict[str, Any]],
    split_cfg: dict[str, Any],
) -> None:
    split_lower = split_name.lower()
    if dataset == "monuseg" and split_lower in MONUSEG_FORBIDDEN_SPLITS:
        raise ProtocolViolation(f"MoNuSeg split {split_name!r} is forbidden in Phase 0 audit")

    for entry in entries:
        if dataset == "tnbc":
            patient = entry.get("patient")
            if patient is None:
                raise ProtocolViolation(
                    f"TNBC patient cannot be derived before I/O: {entry['sample_id']}"
                )
            try:
                patient_id = int(patient)
            except (TypeError, ValueError) as exc:
                raise ProtocolViolation(
                    f"invalid TNBC patient id for {entry['sample_id']}: {patient!r}"
                ) from exc
            if patient_id in TNBC_FORBIDDEN_PATIENTS:
                raise ProtocolViolation(
                    f"TNBC patient {patient_id} is closed and will not be accessed"
                )
            allowed = split_cfg.get("allowed_patient_ids")
            if allowed is not None and patient_id not in {int(value) for value in allowed}:
                raise ProtocolViolation(
                    f"TNBC patient {patient_id} is not allowed in split {split_name!r}"
                )
        if dataset == "monuseg":
            for key in ("image_path", "label_path", "raw_label_path"):
                value = entry.get(key)
                if value and (_path_segments(value) & MONUSEG_FORBIDDEN_SPLITS):
                    raise ProtocolViolation(
                        f"MoNuSeg official-test path rejected before I/O: {value}"
                    )


def _discover_training_pool(split_cfg: dict[str, Any], roots: dict[str, Any]) -> list[str]:
    if not split_cfg.get("audit_discovery_from_training_root", False):
        return []
    image_root_value = str(roots.get("image_root") or "")
    image_root = Path(image_root_value)
    if "test" in _path_segments(image_root_value):
        raise ProtocolViolation(f"refusing discovery under test path: {image_root_value}")
    if not image_root.is_dir():
        raise FileNotFoundError(f"training image root not found: {image_root_value}")
    return [
        path.name
        for path in sorted(image_root.iterdir(), key=lambda item: item.name)
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def _load_label(path: Path) -> tuple[np.ndarray, str]:
    if path.suffix.lower() == ".mat":
        payload = loadmat(path)
        if "inst_map" not in payload:
            raise KeyError(f"MAT label has no 'inst_map': {path}")
        array = np.asarray(payload["inst_map"])
        source = "mat:inst_map"
    else:
        array = np.asarray(io.imread(path))
        source = "raster"
    array = np.squeeze(array)
    if array.ndim != 2:
        raise ValueError(f"label must be 2-D after squeeze, got {array.shape}: {path}")
    return array, source


def _distribution(values: list[int] | np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"count": 0, "min": None, "q25": None, "median": None, "q75": None, "max": None, "mean": None}
    return {
        "count": int(arr.size),
        "min": float(arr.min()),
        "q25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "q75": float(np.percentile(arr, 75)),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
    }


def _analyze_label(array: np.ndarray) -> tuple[dict[str, Any], np.ndarray | None]:
    finite = np.isfinite(array) if np.issubdtype(array.dtype, np.number) else None
    nonfinite_count = int((~finite).sum()) if finite is not None else int(array.size)
    integer_like = bool(
        finite is not None
        and finite.all()
        and np.allclose(array, np.rint(array), rtol=0.0, atol=0.0)
    )
    negative_count = int((array < 0).sum()) if finite is not None and finite.all() else None
    valid = integer_like and negative_count == 0
    if not valid:
        return (
            {
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "valid_nonnegative_integer_label": False,
                "nonfinite_pixels": nonfinite_count,
                "negative_pixels": negative_count,
                "empty": None,
            },
            None,
        )

    labels = np.rint(array).astype(np.int64, copy=False)
    ids = np.unique(labels)
    positive_ids = ids[ids > 0]
    contiguous = np.array_equal(
        positive_ids, np.arange(1, len(positive_ids) + 1, dtype=positive_ids.dtype)
    )
    foreground = labels > 0
    components, component_count = ndi.label(foreground, structure=np.ones((3, 3), dtype=np.uint8))
    del components
    areas = [int((labels == instance_id).sum()) for instance_id in positive_ids]
    disconnected: dict[str, int] = {}
    for instance_id in positive_ids:
        _, count = ndi.label(
            labels == instance_id, structure=np.ones((3, 3), dtype=np.uint8)
        )
        if count > 1:
            disconnected[str(int(instance_id))] = int(count)
    boundary_ids = np.unique(
        np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1]))
    )
    boundary_ids = boundary_ids[boundary_ids > 0]
    unique_values = ids.tolist() if ids.size <= 64 else ids[:64].tolist()
    if len(positive_ids) == 0:
        label_kind = "empty"
    elif len(ids) <= 2:
        label_kind = "binary_or_single_instance"
    else:
        label_kind = "instance"
    return (
        {
            "shape": list(labels.shape),
            "dtype": str(array.dtype),
            "valid_nonnegative_integer_label": True,
            "nonfinite_pixels": 0,
            "negative_pixels": 0,
            "label_kind": label_kind,
            "unique_value_count": int(ids.size),
            "unique_values_head": unique_values,
            "unique_values_truncated": bool(ids.size > 64),
            "empty": bool(len(positive_ids) == 0),
            "instance_count": int(len(positive_ids)),
            "foreground_connected_components_8": int(component_count),
            "contiguous_positive_ids": bool(contiguous),
            "disconnected_instance_ids": disconnected,
            "boundary_instance_count": int(len(boundary_ids)),
            "boundary_instance_ids": [int(value) for value in boundary_ids],
            "instance_areas": areas,
            "instance_area_distribution": _distribution(areas),
        },
        labels,
    )


@lru_cache(maxsize=1)
def _load_project_evaluator():
    """Load the tracked evaluator without importing SAM2/Hydra package setup."""
    path = Path(__file__).resolve().parents[1] / "sam2_train" / "modeling" / "stats_utils.py"
    spec = importlib.util.spec_from_file_location("f3c_phase0_stats_utils", path)
    if spec is None or spec.loader is None:  # pragma: no cover - corrupt checkout
        raise ImportError(f"cannot load evaluator from {path}")
    module = importlib.util.module_from_spec(spec)
    # stats_utils has a legacy, unused top-level cv2 import.  Do not require
    # OpenCV merely to call its NumPy/SciPy AJI/PQ routines in a CPU audit.
    inserted_cv2_stub = False
    try:
        import cv2  # noqa: F401
    except ModuleNotFoundError:
        sys.modules["cv2"] = types.ModuleType("cv2")
        inserted_cv2_stub = True
    try:
        spec.loader.exec_module(module)
    finally:
        if inserted_cv2_stub:
            sys.modules.pop("cv2", None)
    return module


def _label_agreement(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    evaluator = _load_project_evaluator()

    ref = evaluator.remap_label(reference.astype(np.int32))
    pred = evaluator.remap_label(candidate.astype(np.int32))
    if ref.max() == 0 and pred.max() == 0:
        return {"aji": 1.0, "dq": 1.0, "sq": 1.0, "pq": 1.0}
    if ref.max() == 0 or pred.max() == 0:
        return {"aji": 0.0, "dq": 0.0, "sq": 0.0, "pq": 0.0}
    aji = float(evaluator.get_fast_aji(ref, pred))
    (dq, sq, pq), _ = evaluator.get_fast_pq(ref, pred, match_iou=0.5)
    return {"aji": aji, "dq": float(dq), "sq": float(sq), "pq": float(pq)}


def _current_prep_watershed(
    raw_binary: np.ndarray,
    *,
    min_distance: int = 10,
    sigma: float = 1.0,
) -> np.ndarray:
    binary = np.asarray(raw_binary, dtype=bool)
    if not binary.any():
        return np.zeros(binary.shape, dtype=np.int32)
    distance = ndi.distance_transform_edt(binary)
    if sigma > 0:
        distance = ndi.gaussian_filter(distance, sigma)
    coords = peak_local_max(distance, min_distance=min_distance, labels=binary)
    if coords.shape[0] == 0:
        labels, _ = ndi.label(binary)
    else:
        markers = np.zeros(binary.shape, dtype=np.int32)
        markers[tuple(coords.T)] = np.arange(1, len(coords) + 1)
        labels = watershed(-distance, markers, mask=binary)
    labels, _, _ = relabel_sequential(labels.astype(np.int32))
    return labels.astype(np.int32)


def _split_merge_counts(reference: np.ndarray, candidate: np.ndarray) -> dict[str, int]:
    split_reference = 0
    for reference_id in np.unique(reference):
        if reference_id <= 0:
            continue
        overlap_ids = np.unique(candidate[reference == reference_id])
        if int((overlap_ids > 0).sum()) > 1:
            split_reference += 1
    merged_candidate = 0
    for candidate_id in np.unique(candidate):
        if candidate_id <= 0:
            continue
        overlap_ids = np.unique(reference[candidate == candidate_id])
        if int((overlap_ids > 0).sum()) > 1:
            merged_candidate += 1
    return {
        "reference_instances_split_by_candidate": int(split_reference),
        "candidate_instances_merging_reference": int(merged_candidate),
    }


def _label_area_summary(label: np.ndarray) -> dict[str, Any]:
    areas = [int((label == value).sum()) for value in np.unique(label) if value > 0]
    return {
        "distribution": _distribution(areas),
        "area_le_4_count": sum(value <= 4 for value in areas),
        "area_le_8_count": sum(value <= 8 for value in areas),
        "area_le_16_count": sum(value <= 16 for value in areas),
    }


def _compare_raw_binary(
    raw: np.ndarray,
    prepared: np.ndarray,
    *,
    watershed_min_distance: int = 10,
    watershed_sigma: float = 1.0,
) -> dict[str, Any]:
    raw_binary = np.asarray(raw) > 0
    raw_components_4, raw_count_4 = ndi.label(raw_binary)
    raw_components, raw_count = ndi.label(
        raw_binary, structure=np.ones((3, 3), dtype=np.uint8)
    )
    current_watershed = _current_prep_watershed(
        raw_binary,
        min_distance=watershed_min_distance,
        sigma=watershed_sigma,
    )
    watershed_count = int((np.unique(current_watershed) > 0).sum())
    prepared_binary = prepared > 0
    cc_vs_prepared = _split_merge_counts(raw_components, prepared)
    watershed_vs_prepared = _split_merge_counts(current_watershed, prepared)
    prepared_count = int((np.unique(prepared) > 0).sum())
    return {
        "raw_binary_connected_components_8": int(raw_count),
        "raw_binary_connected_components_4": int(raw_count_4),
        "current_prep_watershed": {
            "min_distance": int(watershed_min_distance),
            "sigma": float(watershed_sigma),
            "instance_count": watershed_count,
            "instance_count_delta_vs_connected_components": int(
                watershed_count - raw_count
            ),
            "foreground_equal_to_raw": bool(
                np.array_equal(current_watershed > 0, raw_binary)
            ),
            "area": _label_area_summary(current_watershed),
        },
        "prepared_instance_count": prepared_count,
        "instance_count_delta": int(prepared_count - raw_count),
        "raw_components_split_by_preparation": cc_vs_prepared[
            "reference_instances_split_by_candidate"
        ],
        "prepared_instances_merging_raw_components": cc_vs_prepared[
            "candidate_instances_merging_reference"
        ],
        "foreground_xor_pixels": int(np.logical_xor(raw_binary, prepared_binary).sum()),
        "foreground_equal": bool(np.array_equal(raw_binary, prepared_binary)),
        "prepared_area": _label_area_summary(prepared),
        "cc_vs_prepared_evaluator_agreement": _label_agreement(raw_components, prepared),
        "cc4_vs_prepared_evaluator_agreement": _label_agreement(
            raw_components_4, prepared
        ),
        "watershed_vs_prepared": {
            **watershed_vs_prepared,
            "instance_count_delta": int(prepared_count - watershed_count),
            "foreground_xor_pixels": int(
                np.logical_xor(current_watershed > 0, prepared_binary).sum()
            ),
            "exact_instance_map_equal": bool(
                np.array_equal(current_watershed, prepared)
            ),
            "evaluator_agreement": _label_agreement(current_watershed, prepared),
        },
    }


def _audit_sample(entry: dict[str, Any], dataset: str) -> dict[str, Any]:
    image_path = Path(entry["image_path"])
    label_path = Path(str(entry["label_path"]))
    result = dict(entry)
    result["errors"] = []
    if not image_path.is_file():
        result["errors"].append(f"missing image: {image_path}")
    if not label_path.is_file():
        result["errors"].append(f"missing label: {label_path}")
    if result["errors"]:
        return result

    try:
        image = np.asarray(io.imread(image_path))
        label, label_source = _load_label(label_path)
        label_audit, prepared = _analyze_label(label)
        result.update(
            {
                "image_shape": list(image.shape),
                "image_dtype": str(image.dtype),
                "image_sha256": _sha256_file(image_path),
                "label_sha256": _sha256_file(label_path),
                "label_source": label_source,
                "label": label_audit,
                "image_gt_spatial_shape_match": bool(image.shape[:2] == label.shape),
            }
        )
        if image.shape[:2] != label.shape:
            result["errors"].append(
                f"image/GT shape mismatch: {image.shape[:2]} vs {label.shape}"
            )
        raw_label_value = entry.get("raw_label_path")
        if dataset == "tnbc" and raw_label_value:
            raw_path = Path(str(raw_label_value))
            if not raw_path.is_file():
                result["errors"].append(f"missing raw TNBC label: {raw_path}")
            elif prepared is not None:
                raw_label, raw_source = _load_label(raw_path)
                result["raw_label_sha256"] = _sha256_file(raw_path)
                result["raw_label_source"] = raw_source
                if raw_label.shape != prepared.shape:
                    result["errors"].append(
                        f"raw/prepared GT shape mismatch: {raw_label.shape} vs {prepared.shape}"
                    )
                else:
                    result["tnbc_raw_vs_prepared"] = _compare_raw_binary(
                        raw_label, prepared
                    )
    except Exception as exc:  # report per-file failures without hiding other samples
        result["errors"].append(f"{type(exc).__name__}: {exc}")
    return result


def _duplicate_groups(records: list[dict[str, Any]], key: str) -> list[list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for record in records:
        value = record.get(key)
        if value:
            groups[str(value)].append(str(record["sample_id"]))
    return [sorted(group) for group in groups.values() if len(group) > 1]


def _count_group(records: list[dict[str, Any]], key: str) -> dict[str, int]:
    counter = Counter(
        "<unknown>" if record.get(key) in (None, "") else str(record[key])
        for record in records
    )
    return dict(sorted(counter.items()))


def _audit_split(
    config: dict[str, Any], split_name: str, split_cfg: dict[str, Any]
) -> dict[str, Any]:
    dataset = str(config["dataset"]).lower()
    roots = dict(config.get("roots") or {})
    roots.update(split_cfg.get("roots") or {})
    metadata_map = dict(config.get("metadata_map") or {})
    blockers: list[str] = []
    source_hash = None
    raw_entries: list[Any] = list(split_cfg.get("samples") or [])
    source_manifest = split_cfg.get("source_manifest")
    if source_manifest:
        source_path = Path(str(source_manifest))
        if source_path.is_file():
            payload = _load_json(source_path)
            raw_entries.extend(_extract_manifest_entries(payload))
            source_hash = _sha256_file(source_path)
        else:
            blockers.append(f"source manifest not found: {source_manifest}")
    if not raw_entries and not blockers:
        try:
            raw_entries = _discover_training_pool(split_cfg, roots)
        except (FileNotFoundError, ProtocolViolation) as exc:
            blockers.append(str(exc))

    try:
        entries = [
            _normalize_entry(
                raw,
                dataset=dataset,
                split_name=split_name,
                roots=roots,
                metadata_map=metadata_map,
            )
            for raw in raw_entries
        ]
        _validate_protocol(dataset, split_name, entries, split_cfg)
    except (ValueError, ProtocolViolation) as exc:
        return {
            "status": "protocol_violation" if isinstance(exc, ProtocolViolation) else "blocked",
            "split": split_name,
            "source_manifest": source_manifest,
            "source_manifest_sha256": source_hash,
            "blockers": blockers + [str(exc)],
            "manifest_count": len(raw_entries),
            "samples": [],
        }

    records = [_audit_sample(entry, dataset) for entry in entries]
    errors = [
        {"sample_id": record["sample_id"], "errors": record["errors"]}
        for record in records
        if record.get("errors")
    ]
    expected = split_cfg.get("expected_count")
    if expected is not None and int(expected) != len(entries):
        blockers.append(f"expected {int(expected)} samples, manifest resolved {len(entries)}")
    all_areas = [
        area
        for record in records
        for area in ((record.get("label") or {}).get("instance_areas") or [])
    ]
    total_instances = sum(
        int((record.get("label") or {}).get("instance_count") or 0)
        for record in records
    )
    raw_comparisons = [
        record["tnbc_raw_vs_prepared"]
        for record in records
        if "tnbc_raw_vs_prepared" in record
    ]
    if not entries or blockers:
        status = "blocked"
    elif errors:
        status = "issues_found"
    else:
        status = "complete"
    return {
        "status": status,
        "split": split_name,
        "role": split_cfg.get("role"),
        "expected_count": expected,
        "source_manifest": source_manifest,
        "source_manifest_sha256": source_hash,
        "audit_discovery_from_training_root": bool(
            split_cfg.get("audit_discovery_from_training_root", False)
        ),
        "roots": roots,
        "manifest_count": len(entries),
        "audited_without_error_count": sum(not record.get("errors") for record in records),
        "blockers": blockers,
        "errors": errors,
        "counts_by_patient": _count_group(records, "patient"),
        "counts_by_case": _count_group(records, "case"),
        "counts_by_organ": _count_group(records, "organ"),
        "total_instances": int(total_instances),
        "instance_area_distribution": _distribution(all_areas),
        "empty_label_count": sum(
            bool((record.get("label") or {}).get("empty")) for record in records
        ),
        "invalid_label_count": sum(
            (record.get("label") or {}).get("valid_nonnegative_integer_label") is False
            for record in records
        ),
        "noncontiguous_id_count": sum(
            (record.get("label") or {}).get("contiguous_positive_ids") is False
            for record in records
            if (record.get("label") or {}).get("contiguous_positive_ids") is not None
        ),
        "disconnected_id_image_count": sum(
            bool((record.get("label") or {}).get("disconnected_instance_ids"))
            for record in records
        ),
        "duplicate_image_sha256_groups": _duplicate_groups(records, "image_sha256"),
        "duplicate_label_sha256_groups": _duplicate_groups(records, "label_sha256"),
        "duplicate_sample_ids": [
            sample_id
            for sample_id, count in Counter(record["sample_id"] for record in records).items()
            if count > 1
        ],
        "tnbc_raw_vs_prepared_summary": {
            "status": "complete" if len(raw_comparisons) == len(records) and records else "not_run_for_all_samples",
            "compared_images": len(raw_comparisons),
            "instance_count_delta_total": int(
                sum(item["instance_count_delta"] for item in raw_comparisons)
            ),
            "raw_components_split_total": int(
                sum(item["raw_components_split_by_preparation"] for item in raw_comparisons)
            ),
            "foreground_xor_pixels_total": int(
                sum(item["foreground_xor_pixels"] for item in raw_comparisons)
            ),
        }
        if dataset == "tnbc"
        else None,
        "samples": records,
    }


def _shared_values(
    left: list[dict[str, Any]], right: list[dict[str, Any]], key: str
) -> list[str]:
    left_values = {
        str(record[key])
        for record in left
        if record.get(key) not in (None, "")
    }
    right_values = {
        str(record[key])
        for record in right
        if record.get(key) not in (None, "")
    }
    return sorted(left_values & right_values)


def _audit_split_isolation(
    dataset: str, split_reports: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Check group and content isolation across every declared split pair."""

    names = list(split_reports)
    pairs: list[dict[str, Any]] = []
    violation_count = 0
    not_checkable_count = 0
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            left_records = split_reports[left_name].get("samples") or []
            right_records = split_reports[right_name].get("samples") or []
            if not left_records or not right_records:
                not_checkable_count += 1
                pairs.append(
                    {
                        "left_split": left_name,
                        "right_split": right_name,
                        "status": "not_checkable",
                        "reason": "one or both splits have no audited sample records",
                        "overlap": {},
                        "violations": {},
                    }
                )
                continue
            overlap = {
                "sample_ids": _shared_values(left_records, right_records, "sample_id"),
                "patients": _shared_values(left_records, right_records, "patient"),
                "cases": _shared_values(left_records, right_records, "case"),
                "image_sha256": _shared_values(
                    left_records, right_records, "image_sha256"
                ),
                "label_sha256": _shared_values(
                    left_records, right_records, "label_sha256"
                ),
            }
            violations = {
                key: values
                for key, values in overlap.items()
                if values
                and (
                    key in {"sample_ids", "image_sha256", "label_sha256"}
                    or key == "patients"
                    or (key == "cases" and dataset == "monuseg")
                )
            }
            violation_count += sum(len(values) for values in violations.values())
            pairs.append(
                {
                    "left_split": left_name,
                    "right_split": right_name,
                    "status": "protocol_violation" if violations else "isolated",
                    "overlap": overlap,
                    "violations": violations,
                }
            )
    if violation_count:
        status = "protocol_violation"
    elif not_checkable_count:
        status = "not_checkable"
    elif not pairs:
        status = "not_applicable"
    else:
        status = "isolated"
    return {
        "status": status,
        "checked_pair_count": len(pairs),
        "not_checkable_pair_count": not_checkable_count,
        "violation_value_count": violation_count,
        "pairs": pairs,
    }


def audit_configs(config_paths: list[Path], argv: list[str]) -> dict[str, Any]:
    reports = []
    for config_path in config_paths:
        config = _load_json(config_path)
        dataset = str(config.get("dataset") or "").lower()
        if dataset not in {"tnbc", "monuseg"}:
            reports.append(
                {
                    "config_path": str(config_path),
                    "status": "blocked",
                    "blockers": [f"unsupported or missing dataset: {dataset!r}"],
                    "splits": {},
                }
            )
            continue
        splits = config.get("splits")
        if not isinstance(splits, dict) or not splits:
            reports.append(
                {
                    "config_path": str(config_path),
                    "dataset": dataset,
                    "status": "blocked",
                    "blockers": ["config has no splits"],
                    "splits": {},
                }
            )
            continue
        split_reports = {
            name: _audit_split(config, name, dict(split_cfg or {}))
            for name, split_cfg in splits.items()
        }
        split_isolation = _audit_split_isolation(dataset, split_reports)
        statuses = {report["status"] for report in split_reports.values()}
        if (
            "protocol_violation" in statuses
            or split_isolation["status"] == "protocol_violation"
        ):
            status = "protocol_violation"
        elif statuses == {"complete"}:
            status = "complete"
        elif "issues_found" in statuses:
            status = "issues_found"
        else:
            status = "blocked"
        reports.append(
            {
                "config_path": str(config_path),
                "config_sha256": _sha256_file(config_path),
                "dataset": dataset,
                "protocol_id": config.get("protocol_id"),
                "status": status,
                "protocol": config.get("protocol"),
                "grouped_development": config.get("grouped_development"),
                "split_isolation": split_isolation,
                "splits": split_reports,
            }
        )
    statuses = {report["status"] for report in reports}
    if "protocol_violation" in statuses:
        status = "protocol_violation"
    elif statuses == {"complete"}:
        status = "complete"
    elif "issues_found" in statuses:
        status = "issues_found"
    else:
        status = "blocked"
    return {
        "schema_version": 2,
        "phase": "phase0_dataset_audit",
        "status": status,
        "environment": _environment_record(argv),
        "configs": reports,
    }


def _write_summary(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Phase 0 dataset audit summary",
        "",
        f"- Overall status: `{report['status']}`",
        f"- Generated: `{report['environment']['created_at_utc']}`",
        f"- Git: `{report['environment'].get('git_branch')}` @ `{report['environment'].get('git_commit')}`",
        "",
    ]
    for dataset_report in report["configs"]:
        lines.extend(
            [
                f"## {dataset_report.get('dataset', 'unknown')}",
                "",
                f"- Status: `{dataset_report['status']}`",
                f"- Config: `{dataset_report['config_path']}`",
            ]
        )
        for split_name, split in dataset_report.get("splits", {}).items():
            lines.extend(
                [
                    f"- `{split_name}`: status=`{split['status']}`, manifest={split.get('manifest_count', 0)}, "
                    f"audited={split.get('audited_without_error_count', 0)}, instances={split.get('total_instances', 0)}",
                ]
            )
            for blocker in split.get("blockers", []):
                lines.append(f"  - Blocker: {blocker}")
        isolation = dataset_report.get("split_isolation") or {}
        lines.append(
            f"- Split isolation: status=`{isolation.get('status', 'not_checked')}`, "
            f"pairs={isolation.get('checked_pair_count', 0)}, "
            f"overlap violations={isolation.get('violation_value_count', 0)}"
        )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary-output", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    command = [sys.executable, *sys.argv]
    report = audit_configs(args.config, command)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.summary_output:
        _write_summary(report, args.summary_output)
    print(json.dumps({"status": report["status"], "output": str(args.output)}))
    return 2 if report["status"] == "protocol_violation" else 0


if __name__ == "__main__":
    raise SystemExit(main())
