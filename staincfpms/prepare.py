"""Freeze Phase 0 samples and deterministic stain counterfactuals before inference."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import scipy.io as sio
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from .protocol import (
    TNBC_AUDIT_PATIENTS,
    TNBC_CALIBRATION_PATIENTS,
    ProtocolError,
    assert_open_path,
    assert_tnbc_records,
    sha256_file,
    write_json,
)
from .transforms import counterfactual_views, decompose, rgb_to_od, stain_record


DEFAULT_HE = np.asarray([[0.650, 0.704, 0.286], [0.072, 0.990, 0.105]], dtype=np.float64)
VIEW_ORDER = ("V0", "V1", "V2", "V3", "V4", "V5")


def _load_records(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    records = payload["samples"] if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ProtocolError(f"manifest is not a list: {path}")
    return [dict(record) for record in records]


def _read_rgb(path: str | Path) -> np.ndarray:
    array = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ProtocolError(f"not an RGB image: {path}")
    return array


def _read_label(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() == ".mat":
        value = sio.loadmat(path).get("inst_map")
        if value is None:
            raise ProtocolError(f"missing inst_map in {path}")
        return np.asarray(value, dtype=np.int32)
    if path.suffix.lower() == ".npy":
        return np.asarray(np.load(path), dtype=np.int32)
    raise ProtocolError(f"unsupported label type: {path}")


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _tissue_fraction(rgb: np.ndarray) -> float:
    od = rgb_to_od(rgb, np.full(3, 255.0))
    return float((od.sum(axis=-1) > 0.15).mean())


def _select_reference(records: list[dict[str, Any]], matrices: list[np.ndarray]) -> dict[str, Any]:
    if not records:
        raise ProtocolError("no usable calibration image for reference stain matrix")
    median = np.median(np.stack(matrices), axis=0)
    distances = np.asarray([np.linalg.norm(matrix - median) for matrix in matrices])
    target = float(np.quantile(distances, 0.90))
    index = int(np.argmin(np.abs(distances - target)))
    return {
        "sample_id": records[index]["id"],
        "distance_to_dataset_median": float(distances[index]),
        "target_90pct_distance": target,
        "matrix": matrices[index].round(10).tolist(),
        "fallback_matrix": np.asarray(median).round(10).tolist(),
    }


def _calibrate(records: list[dict[str, Any]], dataset: str) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    usable_records: list[dict[str, Any]] = []
    matrices: list[np.ndarray] = []
    failures: list[dict[str, str]] = []
    for record in records:
        rgb = _read_rgb(record["image"])
        try:
            result = decompose(rgb)
        except ValueError as exc:
            failures.append({"sample_id": record["id"], "reason": str(exc)})
            continue
        usable_records.append(record)
        matrices.append(result.matrix)
    if not matrices:
        raise ProtocolError(f"all {dataset} calibration decompositions failed")
    reference = _select_reference(usable_records, matrices)
    fallback = np.asarray(reference["fallback_matrix"], dtype=np.float64)
    return {
        "dataset": dataset,
        "n_calibration_images": len(records),
        "n_success": len(usable_records),
        "failures": failures,
        "reference": reference,
    }, np.asarray(reference["matrix"], dtype=np.float64), fallback


def _candidate_coordinates(height: int, width: int, side: int) -> list[tuple[int, int]]:
    if height < side or width < side:
        raise ProtocolError(f"MoNuSeg image smaller than required {side}x{side}: {height}x{width}")
    xs = sorted({0, (width - side) // 2, width - side})
    ys = sorted({0, (height - side) // 2, height - side})
    return [(x, y) for y in ys for x in xs]


def select_monuseg_crops(
    image_dir: str | Path,
    label_dir: str | Path,
    organ_map: dict[str, str],
    calibration_stems: set[str],
    count: int = 6,
    side: int = 512,
    min_tissue_fraction: float = 0.50,
) -> list[dict[str, Any]]:
    image_dir, label_dir = Path(image_dir), Path(label_dir)
    assert_open_path(image_dir, "MoNuSeg train images")
    assert_open_path(label_dir, "MoNuSeg train labels")
    candidates: list[dict[str, Any]] = []
    for image_path in sorted(path for path in image_dir.iterdir() if path.suffix.lower() in {".png", ".tif", ".tiff", ".jpg", ".jpeg"}):
        stem = image_path.stem
        if stem in calibration_stems:
            continue
        if stem not in organ_map:
            raise ProtocolError(f"organ map has no entry for MoNuSeg image {stem}")
        label_path = label_dir / f"{stem}.mat"
        if not label_path.exists():
            raise ProtocolError(f"missing MoNuSeg label {label_path}")
        rgb = _read_rgb(image_path)
        label = _read_label(label_path)
        if rgb.shape[:2] != label.shape:
            raise ProtocolError(f"image/GT shape mismatch: {image_path}")
        for x, y in _candidate_coordinates(*rgb.shape[:2], side):
            crop = rgb[y : y + side, x : x + side]
            fraction = _tissue_fraction(crop)
            if fraction >= min_tissue_fraction:
                candidates.append({
                    "id": f"monuseg_{stem}_x{x}_y{y}", "dataset": "monuseg", "organ": str(organ_map[stem]),
                    "image": str(image_path.resolve()), "label": str(label_path.resolve()), "crop": [x, y, side, side],
                    "tissue_foreground_fraction": fraction,
                })
    if not candidates:
        raise ProtocolError("no MoNuSeg crop passed the fixed tissue foreground rule")
    # One strongest fixed-coordinate crop per organ before deterministic global backfill.
    candidates.sort(key=lambda row: (str(row["organ"]), -float(row["tissue_foreground_fraction"]), str(row["id"])))
    selected: list[dict[str, Any]] = []
    used_organs: set[str] = set()
    for candidate in candidates:
        if candidate["organ"] not in used_organs:
            selected.append(candidate)
            used_organs.add(str(candidate["organ"]))
            if len(selected) == count:
                return selected
    for candidate in candidates:
        if candidate not in selected:
            selected.append(candidate)
            if len(selected) == count:
                return selected
    raise ProtocolError(f"only {len(selected)} valid MoNuSeg crops; required {count}")


def _quality_row(sample_id: str, view: str, original: np.ndarray, transformed: np.ndarray, decomposition: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "sample_id": sample_id,
        "view": view,
        "height": int(original.shape[0]),
        "width": int(original.shape[1]),
        "rgb_mae_vs_v0": float(np.abs(transformed.astype(float) - original.astype(float)).mean()),
        "tissue_foreground_fraction": _tissue_fraction(transformed),
        "fallback_used": bool(decomposition.fallback_used),
    }
    if view == "V1":
        row["v1_psnr"] = float(peak_signal_noise_ratio(original, transformed, data_range=255))
        row["v1_ssim"] = float(structural_similarity(original, transformed, channel_axis=2, data_range=255))
        row["v1_pass"] = bool(row["rgb_mae_vs_v0"] <= 3.0 and row["v1_psnr"] >= 35.0 and row["v1_ssim"] >= 0.98)
    else:
        row["v1_psnr"] = ""
        row["v1_ssim"] = ""
        row["v1_pass"] = ""
    return row


def _materialize_sample(
    out_dir: Path,
    sample: dict[str, Any],
    within_matrix: np.ndarray,
    cross_matrix: np.ndarray,
    fallback_matrix: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    rgb, gt = _read_rgb(sample["image"]), _read_label(sample["label"])
    x, y, width, height = sample.get("crop", [0, 0, rgb.shape[1], rgb.shape[0]])
    rgb, gt = rgb[y : y + height, x : x + width], gt[y : y + height, x : x + width]
    if rgb.shape[:2] != gt.shape:
        raise ProtocolError(f"cropped image/GT shape mismatch for {sample['id']}")
    views, decomposition = counterfactual_views(rgb, fallback_matrix, within_matrix, cross_matrix)
    label_path = out_dir / "prepared" / sample["dataset"] / "labels" / f"{sample['id']}.npy"
    label_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(label_path, gt.astype(np.int32))
    prepared_views: dict[str, str] = {}
    quality: list[dict[str, Any]] = []
    for view, image in views.items():
        image_path = out_dir / "prepared" / sample["dataset"] / view / f"{sample['id']}.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image).save(image_path)
        prepared_views[view] = str(image_path.resolve())
        quality.append(_quality_row(sample["id"], view, rgb, image, decomposition))
    materialized = dict(sample)
    materialized.update({
        "image": str(Path(sample["image"]).resolve()), "label": str(Path(sample["label"]).resolve()),
        "prepared_label": str(label_path.resolve()), "views": prepared_views,
        "rgb_sha256": sha256_file(sample["image"]), "gt_sha256": sha256_file(sample["label"]),
        "prepared_gt_sha256": sha256_file(label_path), "shape": list(rgb.shape[:2]),
    })
    return materialized, quality, stain_record(decomposition)


def freeze_audit(
    out_dir: str | Path,
    tnbc_audit_manifest: str | Path,
    tnbc_calibration_manifest: str | Path,
    monuseg_image_dir: str | Path,
    monuseg_label_dir: str | Path,
    monuseg_organ_map: str | Path,
    monuseg_calibration_manifest: str | Path,
) -> Path:
    out_dir = Path(out_dir).resolve()
    if (out_dir / "fixed_audit_manifest.json").exists():
        raise ProtocolError(f"audit manifest already frozen: {out_dir}")
    tnbc_audit, tnbc_calibration = _load_records(tnbc_audit_manifest), _load_records(tnbc_calibration_manifest)
    assert_tnbc_records(tnbc_audit, TNBC_AUDIT_PATIENTS, "audit")
    assert_tnbc_records(tnbc_calibration, TNBC_CALIBRATION_PATIENTS, "calibration")
    if len(tnbc_audit) != 7:
        raise ProtocolError(f"TNBC audit must contain exactly 7 images, got {len(tnbc_audit)}")
    for index, record in enumerate(tnbc_audit):
        record.setdefault("id", f"tnbc_p{int(record['patient'])}_{index:02d}")
        record["dataset"] = "tnbc"
    for index, record in enumerate(tnbc_calibration):
        record.setdefault("id", f"tnbc_cal_p{int(record['patient'])}_{index:03d}")
    monuseg_calibration = _load_records(monuseg_calibration_manifest)
    for index, record in enumerate(monuseg_calibration):
        record.setdefault("id", f"monuseg_cal_{index:03d}")
        assert_open_path(record["image"], "MoNuSeg calibration")
    calibration_stems = {Path(record["image"]).stem for record in monuseg_calibration}
    organ_map = json.loads(Path(monuseg_organ_map).read_text(encoding="utf-8"))
    monuseg_audit = select_monuseg_crops(monuseg_image_dir, monuseg_label_dir, organ_map, calibration_stems)
    tnbc_stats, tnbc_within, tnbc_fallback = _calibrate(tnbc_calibration, "tnbc")
    monuseg_stats, monuseg_within, monuseg_fallback = _calibrate(monuseg_calibration, "monuseg")
    all_samples: list[dict[str, Any]] = []
    all_quality: list[dict[str, Any]] = []
    stains: dict[str, Any] = {"tnbc": tnbc_stats, "monuseg": monuseg_stats, "per_audit_sample": {}}
    for sample in tnbc_audit:
        item, quality, stain = _materialize_sample(out_dir, sample, tnbc_within, monuseg_within, tnbc_fallback)
        all_samples.append(item); all_quality.extend(quality); stains["per_audit_sample"][item["id"]] = stain
    for sample in monuseg_audit:
        item, quality, stain = _materialize_sample(out_dir, sample, monuseg_within, tnbc_within, monuseg_fallback)
        all_samples.append(item); all_quality.extend(quality); stains["per_audit_sample"][item["id"]] = stain
    failed_v1 = [row["sample_id"] for row in all_quality if row["view"] == "V1" and not row["v1_pass"]]
    _write_csv(out_dir / "transform_quality.csv", all_quality)
    write_json(out_dir / "stain_statistics.json", stains)
    write_json(out_dir / "transform_manifest.json", {
        "method": "Macenko-style non-negative OD H&E decomposition",
        "views": {"V0": "original RGB", "V1": "OD identity", "V2": "H x0.8", "V3": "H x1.2", "V4": "same-domain 90th-distance reference matrix", "V5": "cross-dataset fixed reference matrix"},
        "deterministic": True, "geometry_changes": False, "fallback": "dataset calibration median stain matrix",
    })
    manifest = {
        "frozen": True, "seed": 3407, "tnbc_audit_patients": [7, 8], "tnbc_closed_patients": [9, 10, 11],
        "monuseg_split": "train_only", "monuseg_audit_crop_count": 6,
        "sample_count": len(all_samples), "samples": all_samples,
        "quality_gate": {"v1_max_mae": 3.0, "v1_min_psnr": 35.0, "v1_min_ssim": 0.98, "failed_samples": failed_v1},
    }
    write_json(out_dir / "fixed_audit_manifest.json", manifest)
    if failed_v1:
        raise ProtocolError(f"OD identity quality gate failed; model inference is forbidden: {failed_v1}")
    return out_dir / "fixed_audit_manifest.json"
