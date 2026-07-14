"""Audit one authorized training split and create a frozen ResiMix donor bank."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import scipy.io as sio
from skimage import io

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from resimixpms.coverage import StaticCoverageCache  # noqa: E402
from resimixpms.donor import TrainingSample, audit_training_samples, donor_manifest_row, write_donor_bank  # noqa: E402
from resimixpms.experiment import sha256_file, write_json  # noqa: E402
from resimixpms.manifests import (  # noqa: E402
    ManifestPreflightError,
    load_allowed_image_names,
    load_crop_records,
    validate_manifest_patient_isolation,
)
from resimixpms.transplant import (  # noqa: E402
    CONTEXT_FEATURE_NAMES,
    boundary_gradient_energy,
    context_features,
    deterministic_donor_choice,
    deterministic_geometry,
    enumerate_legal_hosts,
    rgb_to_od,
    transform_donor,
)


def _safe_image_path(image_root: Path, manifest_name: str) -> Path:
    relative = Path(str(manifest_name).replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts:
        raise ManifestPreflightError(f"unsafe manifest image name: {manifest_name}")
    candidates = [relative]
    if not relative.suffix:
        candidates.extend(relative.with_suffix(ext) for ext in (".png", ".tif", ".tiff", ".jpg", ".jpeg"))
    root = image_root.resolve()
    found = []
    for candidate in candidates:
        path = (root / candidate).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ManifestPreflightError(f"image escapes root: {manifest_name}") from exc
        if path.is_file():
            found.append(path)
    if len(found) != 1:
        raise ManifestPreflightError(f"image {manifest_name!r} resolves to {len(found)} files")
    return found[0]


def _crop_union(shape, records):
    allowed = np.zeros(shape, dtype=bool)
    for record in records:
        x, y = int(record["x"]), int(record["y"])
        width, height = int(record["width"]), int(record["height"])
        if x + width > shape[1] or y + height > shape[0]:
            raise ManifestPreflightError(f"training crop lies outside image: {record}")
        allowed[y:y + height, x:x + width] = True
    return allowed


def _sliding_boxes(shape, crop_size, overlap):
    height, width = int(shape[0]), int(shape[1])
    stride = int(crop_size) - int(overlap)
    if stride <= 0:
        raise ValueError("crop_size must exceed overlap")
    def starts(size):
        values = [0]
        while values[-1] + crop_size < size:
            candidate = values[-1] + stride
            if candidate + crop_size >= size:
                if values[-1] != size - crop_size:
                    values.append(max(0, size - crop_size))
                break
            values.append(candidate)
        return sorted(set(values))
    return [(x, y, min(width, x + crop_size), min(height, y + crop_size)) for x in starts(width) for y in starts(height)]


def _tissue_mask(rgb):
    return rgb_to_od(rgb).sum(axis=-1) >= 0.15


def build_donor_bank(options):
    train_manifest = Path(options.train_manifest)
    names = load_allowed_image_names(train_manifest)
    patient_by_name = {}
    if options.dataset == "tnbc":
        records = validate_manifest_patient_isolation(
            train_manifest, range(1, 7), {9, 10, 11}
        )
        patient_by_name = {
            Path(str(row.get("image_name", row.get("image", row.get("name"))))).stem: int(row["patient_id"])
            for row in records
        }
    crop_records_by_stem: dict[str, list[dict[str, Any]]] = {}
    if options.train_crop_manifest:
        for record in load_crop_records(options.train_crop_manifest):
            crop_records_by_stem.setdefault(Path(str(record["image_name"])).stem, []).append(record)

    cache_manifest = Path(options.coverage_manifest)
    cache = StaticCoverageCache.open(cache_manifest.parent)
    if cache.manifest.path.resolve() != cache_manifest.resolve():
        raise ValueError("coverage cache does not resolve to the supplied sealed manifest")
    image_root = Path(options.train_image_root)
    label_root = Path(options.train_label_root)
    samples = []
    raw_by_stem = {}
    expected_ids = set()
    for name in names:
        image_path = _safe_image_path(image_root, name)
        relative = image_path.relative_to(image_root)
        stem = relative.stem
        expected_ids.add(stem)
        label_path = label_root / relative.with_suffix(".mat")
        if not label_path.is_file():
            raise ManifestPreflightError(f"missing GT label: {label_path}")
        rgb = io.imread(image_path)[..., :3]
        instance_map = np.asarray(sio.loadmat(label_path)["inst_map"], dtype=np.int32)
        coverage = cache.load(stem, verify_sha256=True)
        if instance_map.shape != rgb.shape[:2] or coverage.shape != instance_map.shape:
            raise ValueError(f"image/GT/static-coverage shape mismatch for {relative}")
        crop_records = crop_records_by_stem.get(stem, [])
        audit_map = instance_map.copy()
        if options.dataset == "monuseg_lite":
            if not crop_records:
                raise ManifestPreflightError(f"MoNuSeg-Lite image lacks frozen training crop: {relative}")
            allowed_pixels = _crop_union(instance_map.shape, crop_records)
            # A donor must be a real nucleus completely present in the frozen
            # training-crop universe, never a favorable fragment outside it.
            for instance_id in np.unique(instance_map):
                if instance_id and not np.all(allowed_pixels[instance_map == instance_id]):
                    audit_map[audit_map == instance_id] = 0
        samples.append(TrainingSample(
            source_id=stem,
            dataset=options.dataset,
            split="train",
            instance_map=audit_map,
            prediction_masks=coverage,
            coverage_map=coverage,
            rgb=rgb,
            patient_id=patient_by_name.get(stem),
            source_metadata={"split": "train", "manifest": str(train_manifest)},
        ))
        raw_by_stem[stem] = (rgb, instance_map, coverage, crop_records)
    if set(cache.image_ids) != expected_ids:
        raise ValueError("static coverage cache image IDs differ from the authorized training manifest")

    bank = audit_training_samples(samples)
    csv_path, summary_path = write_donor_bank(bank, options.output_dir)

    contexts = []
    boundary_energies = []
    for record in bank.audits:
        if record.annulus_mask.any():
            contexts.append(context_features(record.rgb_patch, record.annulus_mask, _tissue_mask(record.rgb_patch)))
        boundary_energies.append(boundary_gradient_energy(record.rgb_patch, record.mask))
    if not contexts or not boundary_energies:
        raise RuntimeError("cannot freeze host statistics without natural training nuclei")
    context_array = np.asarray(contexts, dtype=np.float64)
    context_mean = context_array.mean(axis=0)
    context_std = context_array.std(axis=0)

    manifest_rows = {row["donor_id"]: (row, record) for row, record in (
        (donor_manifest_row(record), record) for record in bank.donors
    )}
    donors_by_category = {"Missed": [], "IoU-Cliff": [], "Low-Quality Matched": []}
    for row, _ in manifest_rows.values():
        donors_by_category[row["category"]].append(row)
    legal_distances = []
    for stem, (rgb, instance_map, coverage, fixed_crops) in raw_by_stem.items():
        boxes = [(int(row["x"]), int(row["y"]), int(row["x"] + row["width"]), int(row["y"] + row["height"])) for row in fixed_crops]
        if not boxes:
            boxes = _sliding_boxes(instance_map.shape, options.crop_size, options.overlap)
        for crop_index, (x1, y1, x2, y2) in enumerate(boxes):
            sample_key = f"host-stats:{stem}:{crop_index}:{x1}:{y1}"
            choice = deterministic_donor_choice(donors_by_category, 3407, sample_key)
            if choice is None:
                continue
            _, donor_row = choice
            _, donor_record = manifest_rows[donor_row["donor_id"]]
            geometry = deterministic_geometry(3407, (sample_key, donor_row["donor_id"]))
            donor = transform_donor(donor_record.rgb_patch, donor_record.mask, donor_record.annulus_mask, geometry)
            host_rgb = rgb[y1:y2, x1:x2]
            host_inst = instance_map[y1:y2, x1:x2]
            host_coverage = coverage[y1:y2, x1:x2]
            candidates = enumerate_legal_hosts(
                host_rgb, donor, host_inst, host_coverage, _tissue_mask(host_rgb),
                context_mean, context_std, seed=3407, sample_key=sample_key, max_candidates=32,
            )
            legal_distances.extend(float(candidate.context_distance) for candidate in candidates)
    if not legal_distances:
        raise RuntimeError("no legal training host candidates; cannot freeze context p95")
    statistics = {
        "context_feature_names": list(CONTEXT_FEATURE_NAMES),
        "context_mean": context_mean.tolist(),
        "context_std": context_std.tolist(),
        "natural_boundary_gradient_p95": float(np.quantile(boundary_energies, 0.95)),
        "legal_context_distance_p95": float(np.quantile(legal_distances, 0.95)),
        "tissue_total_od_threshold": 0.15,
        "training_manifest": str(train_manifest),
        "training_manifest_sha256": sha256_file(train_manifest),
        "coverage_manifest": str(cache_manifest),
        "coverage_manifest_sha256": sha256_file(cache_manifest),
        "legal_context_candidate_count": len(legal_distances),
        "natural_nucleus_count": len(bank.audits),
        "seed": 3407,
    }
    write_json(Path(options.output_dir) / "host_context_statistics.json", statistics)
    write_json(Path(options.output_dir) / "donor_bank_build_manifest.json", {
        "dataset": options.dataset, "train_manifest": str(train_manifest),
        "train_crop_manifest": str(options.train_crop_manifest or ""),
        "coverage_manifest": str(cache_manifest), "seed": 3407,
        "donor_bank_manifest": str(csv_path), "donor_bank_summary": str(summary_path),
    })
    return csv_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("tnbc", "monuseg_lite"))
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--train-image-root", required=True)
    parser.add_argument("--train-label-root", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--coverage-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-crop-manifest", default="")
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--overlap", type=int, required=True)
    options = parser.parse_args()
    result = build_donor_bank(options)
    print(result)


if __name__ == "__main__":
    main()
