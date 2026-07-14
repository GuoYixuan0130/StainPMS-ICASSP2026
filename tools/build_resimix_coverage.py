"""Build the one immutable step-0 coverage cache for a ResiMix dataset.

This is a remote/AutoDL entry point.  It first runs the canonical StainPMS
inference exactly once on the manifest-admitted training view, then seals the
result through :mod:`resimixpms.coverage`.  It never refreshes, accumulates,
or replaces a cache.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import scipy.io as sio
from skimage import io

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from resimixpms.coverage import begin_static_coverage_generation  # noqa: E402
from resimixpms.experiment import require_sha256, sha256_file, write_json  # noqa: E402
from resimixpms.manifests import (  # noqa: E402
    ManifestPreflightError,
    load_allowed_image_names,
    load_crop_records,
    validate_manifest_patient_isolation,
)


def _read_spec(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get("datasets"), dict):
        raise ValueError("stage spec must be a JSON object with a datasets mapping")
    return payload


def _dataset_spec(spec: dict[str, Any], name: str) -> dict[str, Any]:
    try:
        dataset = dict(spec["datasets"][name])
    except KeyError as exc:
        raise ValueError(f"dataset {name!r} is absent from stage spec") from exc
    required = (
        "data_path", "checkpoint_path", "checkpoint_sha256", "train_manifest", "overlap",
        "train_image_root", "train_label_root",
    )
    missing = [key for key in required if not dataset.get(key)]
    if missing:
        raise ValueError(f"dataset {name!r} lacks required fields: {missing}")
    return dataset


def _safe_image_path(image_root: Path, manifest_name: str) -> Path:
    relative = Path(str(manifest_name).replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts:
        raise ManifestPreflightError(f"unsafe manifest image name: {manifest_name}")
    candidates = [relative]
    if not relative.suffix:
        candidates.extend(relative.with_suffix(ext) for ext in (".png", ".tif", ".tiff", ".jpg", ".jpeg"))
    found = []
    root = image_root.resolve()
    for candidate in candidates:
        resolved = (root / candidate).resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ManifestPreflightError(f"manifest image escapes root: {manifest_name}") from exc
        if resolved.is_file():
            found.append(resolved)
    if len(found) != 1:
        raise ManifestPreflightError(f"image {manifest_name!r} resolves to {len(found)} files")
    return found[0]


def _train_images(dataset: dict[str, Any], dataset_name: str):
    manifest = Path(dataset["train_manifest"])
    if dataset_name == "tnbc":
        allowed = {int(value) for value in dataset.get("train_allowed_patient_ids", [1, 2, 3, 4, 5, 6])}
        validate_manifest_patient_isolation(manifest, allowed, {9, 10, 11})
    names = load_allowed_image_names(manifest)
    root = Path(dataset["train_image_root"])
    label_root = Path(dataset["train_label_root"])
    records = []
    for name in names:
        image_path = _safe_image_path(root, name)
        relative = image_path.relative_to(root)
        label_path = label_root / relative.with_suffix(".mat")
        if not label_path.is_file():
            raise ManifestPreflightError(f"missing label for {relative}: {label_path}")
        # Only manifest-admitted image/label files are opened.
        image_shape = tuple(io.imread(image_path)[..., :3].shape[:2])
        label_shape = tuple(np.asarray(sio.loadmat(label_path)["inst_map"]).shape)
        if image_shape != label_shape:
            raise ValueError(f"image/label shape mismatch for {relative}: {image_shape} != {label_shape}")
        records.append({"name": str(relative), "stem": relative.stem, "shape": image_shape})
    return records


def _crop_map(path: str | None):
    mapping: dict[str, list[dict[str, Any]]] = {}
    if not path:
        return mapping
    for record in load_crop_records(path):
        mapping.setdefault(Path(str(record["image_name"])).stem, []).append(record)
    return mapping


def _main_command(spec: dict[str, Any], dataset_name: str, dataset: dict[str, Any], raw_dir: Path, run_dir: Path):
    command = [
        sys.executable, "main.py", "--eval", "--eval_on_train", "--train_only_eval",
        "--dataset", "monuseg", "--data_path", str(dataset["data_path"]),
        "--sam_ckpt", str(dataset["checkpoint_path"]),
        "--sam_config", str(dataset.get("sam_config", "sam2_hiera_l")),
        "--seed", "3407", "--b", "1", "--num_workers", "0",
        "--crop_size", "256", "--load", "unclockwise", "--texture", "--context", "--test_nms_thr", "12",
        "--overlap", str(int(dataset["overlap"])),
        "--train_manifest", str(dataset["train_manifest"]),
        "--train_image_root", str(dataset["train_image_root"]),
        "--train_label_root", str(dataset["train_label_root"]),
        "--train_crop_manifest", str(dataset.get("train_crop_manifest", "")),
        "--data_identity", dataset_name,
        "--dump_baseline_masks_dir", str(raw_dir),
        "--artifact_dir", str(run_dir),
    ]
    if dataset_name == "tnbc":
        command.extend([
            "--train_allowed_patient_ids", ",".join(str(value) for value in dataset.get("train_allowed_patient_ids", [1, 2, 3, 4, 5, 6])),
            "--test_allowed_patient_ids", ",".join(str(value) for value in dataset.get("test_allowed_patient_ids", [7, 8])),
            "--forbidden_patient_ids", "9,10,11",
        ])
    return command


def build_coverage(spec_path: Path, dataset_name: str, artifact_dir: Path) -> Path:
    spec = _read_spec(spec_path)
    dataset = _dataset_spec(spec, dataset_name)
    artifact_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_hash = require_sha256(
        dataset["checkpoint_path"], dataset["checkpoint_sha256"], f"{dataset_name} frozen StainPMS checkpoint"
    )
    images = _train_images(dataset, dataset_name)
    raw_dir = artifact_dir / "raw_step0_predictions"
    run_dir = artifact_dir / "inference_run"
    command = _main_command(spec, dataset_name, dataset, raw_dir, run_dir)
    write_json(artifact_dir / "coverage_inference_command.json", {"command": command})
    subprocess.run(command, cwd=ROOT, check=True)

    cache_dir = artifact_dir / "static_coverage"
    shapes = {record["stem"]: record["shape"] for record in images}
    crop_records = _crop_map(dataset.get("train_crop_manifest"))
    provenance = {
        "canonical_sha": str(spec.get("canonical_sha", "")),
        "dataset": dataset_name,
        "checkpoint_path": str(dataset["checkpoint_path"]),
        "checkpoint_sha256": checkpoint_hash,
        "train_manifest": str(dataset["train_manifest"]),
        "train_manifest_sha256": sha256_file(dataset["train_manifest"]),
        "train_crop_manifest": str(dataset.get("train_crop_manifest", "")),
        "train_crop_manifest_sha256": sha256_file(dataset["train_crop_manifest"]) if dataset.get("train_crop_manifest") else "",
        "seed": 3407,
        "tta": False,
        "batch_size": 1,
        "nms": 12,
        "texture": True,
        "context": True,
    }
    writer = begin_static_coverage_generation(cache_dir, shapes, provenance=provenance)
    for record in images:
        raw_prediction = raw_dir / f"{record['stem']}.npy"
        if not raw_prediction.is_file():
            raise FileNotFoundError(f"frozen inference did not emit coverage for {record['stem']}")
        prediction = np.load(raw_prediction, allow_pickle=False)
        if tuple(prediction.shape) != tuple(record["shape"]):
            raise ValueError(f"raw coverage shape mismatch for {record['stem']}")
        fixed = crop_records.get(record["stem"], [])
        if fixed:
            for crop in fixed:
                x, y = int(crop["x"]), int(crop["y"])
                width, height = int(crop["width"]), int(crop["height"])
                writer.write_crop(record["stem"], (x, y, x + width, y + height), prediction[y:y + height, x:x + width])
        else:
            if dataset_name == "monuseg_lite":
                raise ManifestPreflightError(
                    f"MoNuSeg-Lite training image has no frozen allowed crop: {record['stem']}"
                )
            writer.write_full(record["stem"], prediction)
    sealed = writer.seal()
    write_json(artifact_dir / "coverage_manifest.json", {
        "cache_dir": str(cache_dir), "manifest": str(sealed.manifest.path), "provenance": dict(sealed.manifest.provenance),
    })
    return sealed.manifest.path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--dataset", required=True, choices=("tnbc", "monuseg_lite"))
    parser.add_argument("--artifact-dir", required=True, type=Path)
    options = parser.parse_args()
    manifest = build_coverage(options.spec, options.dataset, options.artifact_dir)
    print(manifest)


if __name__ == "__main__":
    main()
