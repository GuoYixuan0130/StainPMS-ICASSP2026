"""Read-only TNBC p1--p8 prepared-label provenance audit.

The audit consumes explicit source manifests, never discovers a TNBC directory,
and rejects p9--p11 before an image, MAT label, or raw PNG is opened.  Raw GT
searches use only exact ``GT_01`` ... ``GT_08`` candidate paths derived from
the approved manifests; archive contents and unrelated directory entries are
never enumerated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage as ndi
from scipy.io import loadmat
from skimage import io
from skimage.feature import peak_local_max
from skimage.segmentation import relabel_sequential, watershed


ALLOWED_PATIENTS = set(range(1, 9))
CLOSED_PATIENTS = {9, 10, 11}
SAMPLE_RE = re.compile(r"^(?:patient[_-]?)?0*([0-9]{1,2})[_-]0*([0-9]+)$", re.I)


class ProtocolViolation(RuntimeError):
    """Raised before sample content access when a closed patient is requested."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_entries(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        raise ValueError("source manifest must be a JSON object or list")
    for key in ("samples", "entries", "items", "paths", "files", "images", "records"):
        value = payload.get(key)
        if isinstance(value, list):
            return value
    raise ValueError("source manifest has no supported explicit sample list")


def first_value(record: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = record.get(key)
        if value not in (None, ""):
            return value
    return None


def sample_identity(raw: Any) -> tuple[str, int, int]:
    record = {"image": raw} if isinstance(raw, str) else dict(raw)
    image_value = first_value(
        record,
        ("image", "image_name", "image_path", "img", "path", "file", "filename", "name"),
    )
    if image_value is None:
        raise ValueError(f"sample has no image value: {record}")
    sample_id = str(record.get("sample_id") or Path(str(image_value)).stem)
    match = SAMPLE_RE.fullmatch(sample_id)
    if not match:
        raise ValueError(f"cannot infer TNBC patient/index from {sample_id!r}")
    patient = int(first_value(record, ("patient", "patient_id", "patient_number")) or match.group(1))
    image_index = int(match.group(2))
    return sample_id, patient, image_index


def ordered_samples(source_manifests: list[Path]) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source_path in source_manifests:
        if not source_path.is_file():
            raise FileNotFoundError(f"source manifest not found: {source_path}")
        with source_path.open("r", encoding="utf-8") as handle:
            entries = extract_entries(json.load(handle))
        for raw in entries:
            sample_id, patient, image_index = sample_identity(raw)
            # Closed-patient validation deliberately precedes all data-path I/O.
            if patient in CLOSED_PATIENTS:
                raise ProtocolViolation(
                    f"closed TNBC patient {patient} rejected before file access"
                )
            if patient not in ALLOWED_PATIENTS:
                raise ProtocolViolation(f"TNBC patient {patient} is outside p1--p8")
            if sample_id in seen:
                raise ValueError(f"duplicate sample_id across manifests: {sample_id}")
            seen.add(sample_id)
            samples.append(
                {
                    "sample_id": sample_id,
                    "patient": patient,
                    "image_index": image_index,
                    "source_manifest": str(source_path),
                    "source_manifest_sha256": sha256_file(source_path),
                }
            )
    return samples


def unique_summary(array: np.ndarray, *, limit: int = 64) -> dict[str, Any]:
    values = np.unique(array)
    as_python = values.tolist() if values.size <= limit else values[:limit].tolist()
    return {
        "unique_value_count": int(values.size),
        "unique_values": as_python,
        "unique_values_truncated": bool(values.size > limit),
        "min": values.min().item() if values.size else None,
        "max": values.max().item() if values.size else None,
    }


def load_prepared(path: Path) -> np.ndarray:
    if path.suffix.lower() != ".mat":
        raise ValueError(f"prepared label must be .mat, got {path}")
    payload = loadmat(path)
    if "inst_map" not in payload:
        raise KeyError(f"prepared MAT has no inst_map: {path}")
    array = np.squeeze(np.asarray(payload["inst_map"]))
    if array.ndim != 2:
        raise ValueError(f"prepared inst_map must be 2-D: {path} -> {array.shape}")
    return array


def foreground_from_raw(path: Path) -> np.ndarray:
    raw = np.asarray(io.imread(path))
    if raw.ndim == 2:
        return raw > 0
    if raw.ndim == 3:
        return np.any(raw > 0, axis=-1)
    raise ValueError(f"raw label must be 2-D or RGB-like: {path} -> {raw.shape}")


def watershed_from_binary(binary: np.ndarray, *, min_distance: int, sigma: float) -> np.ndarray:
    binary = np.asarray(binary, dtype=bool)
    if not binary.any():
        return np.zeros(binary.shape, dtype=np.int32)
    distance = ndi.distance_transform_edt(binary)
    if sigma > 0:
        distance = ndi.gaussian_filter(distance, sigma)
    coords = peak_local_max(distance, min_distance=min_distance, labels=binary)
    if len(coords) == 0:
        labels, _ = ndi.label(binary)
    else:
        markers = np.zeros(binary.shape, dtype=np.int32)
        markers[tuple(coords.T)] = np.arange(1, len(coords) + 1)
        labels = watershed(-distance, markers, mask=binary)
    labels, _, _ = relabel_sequential(labels.astype(np.int32))
    return labels.astype(np.int32)


def raw_candidates(raw_roots: list[Path], patient: int, image_index: int) -> list[Path]:
    """Return exact p1--p8 candidate filenames without directory enumeration."""
    candidates: list[Path] = []
    for root in raw_roots:
        gt_dir = root / f"GT_{patient:02d}"
        candidates.extend(
            [
                gt_dir / f"{image_index}.png",
                gt_dir / f"{image_index:02d}.png",
            ]
        )
    return candidates


def disconnected_ids(inst_map: np.ndarray) -> dict[str, int]:
    labels = np.rint(inst_map).astype(np.int64, copy=False)
    result: dict[str, int] = {}
    structure = np.ones((3, 3), dtype=np.uint8)
    for instance_id in np.unique(labels):
        if instance_id <= 0:
            continue
        _, count = ndi.label(labels == instance_id, structure=structure)
        if count > 1:
            result[str(int(instance_id))] = int(count)
    return result


def audit(args: argparse.Namespace) -> dict[str, Any]:
    source_manifests = [Path(value).resolve() for value in args.source_manifest]
    samples = ordered_samples(source_manifests)
    if args.expected_count is not None and len(samples) != int(args.expected_count):
        raise ValueError(f"expected {args.expected_count} approved samples, found {len(samples)}")

    image_root = Path(args.image_root).resolve()
    prepared_root = Path(args.prepared_label_root).resolve()
    raw_roots = [Path(value).resolve() for value in args.raw_root]
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for sample in samples:
        sample_id = sample["sample_id"]
        image_path = image_root / f"{sample_id}.png"
        prepared_path = prepared_root / f"{sample_id}.mat"
        row = {
            **sample,
            "image_path": str(image_path),
            "prepared_label_path": str(prepared_path),
            "prepared_label_suffix": prepared_path.suffix.lower(),
            "raw_search_candidates": [
                str(path)
                for path in raw_candidates(raw_roots, sample["patient"], sample["image_index"])
            ],
            "errors": [],
        }
        if not image_path.is_file():
            row["errors"].append(f"missing image: {image_path}")
        if not prepared_path.is_file():
            row["errors"].append(f"missing prepared label: {prepared_path}")
        if not row["errors"]:
            try:
                prepared = load_prepared(prepared_path)
                prepared_int = np.rint(prepared).astype(np.int64, copy=False)
                if not np.array_equal(prepared, prepared_int):
                    raise ValueError("prepared inst_map is not integer-valued")
                if (prepared_int < 0).any():
                    raise ValueError("prepared inst_map has negative values")
                positive_ids = np.unique(prepared_int)
                positive_ids = positive_ids[positive_ids > 0]
                row["prepared"] = {
                    "dtype": str(prepared.dtype),
                    "shape": list(prepared.shape),
                    **unique_summary(prepared_int),
                    "instance_count": int(len(positive_ids)),
                    "disconnected_instance_ids": disconnected_ids(prepared_int),
                }
            except Exception as exc:  # preserve the remaining p1--p8 audit
                row["errors"].append(f"prepared_label_error: {type(exc).__name__}: {exc}")
                prepared_int = None
        else:
            prepared_int = None

        existing_raw = [Path(value) for value in row["raw_search_candidates"] if Path(value).is_file()]
        row["raw_existing_candidates"] = [str(path) for path in existing_raw]
        if len(existing_raw) > 1:
            row["errors"].append("ambiguous raw GT candidates; none selected")
        elif len(existing_raw) == 1 and prepared_int is not None:
            try:
                raw_binary = foreground_from_raw(existing_raw[0])
                if raw_binary.shape != prepared_int.shape:
                    raise ValueError(
                        f"shape mismatch raw={raw_binary.shape} prepared={prepared_int.shape}"
                    )
                _, cc8_count = ndi.label(raw_binary, structure=np.ones((3, 3), dtype=np.uint8))
                watershed_map = watershed_from_binary(
                    raw_binary,
                    min_distance=int(args.watershed_min_distance),
                    sigma=float(args.watershed_sigma),
                )
                row["raw_vs_prepared"] = {
                    "raw_label_path": str(existing_raw[0]),
                    "raw_foreground_pixels": int(raw_binary.sum()),
                    "prepared_foreground_pixels": int((prepared_int > 0).sum()),
                    "foreground_xor_pixels": int(np.logical_xor(raw_binary, prepared_int > 0).sum()),
                    "foreground_equal": bool(np.array_equal(raw_binary, prepared_int > 0)),
                    "raw_binary_components_8": int(cc8_count),
                    "prepared_instance_count": int((np.unique(prepared_int) > 0).sum()),
                    "prepared_minus_cc8_instance_count": int((np.unique(prepared_int) > 0).sum() - cc8_count),
                    "current_prep_watershed": {
                        "min_distance": int(args.watershed_min_distance),
                        "sigma": float(args.watershed_sigma),
                        "instance_count": int((np.unique(watershed_map) > 0).sum()),
                        "exact_instance_map_equal_to_prepared": bool(
                            np.array_equal(watershed_map, prepared_int)
                        ),
                    },
                }
            except Exception as exc:
                row["errors"].append(f"raw_comparison_error: {type(exc).__name__}: {exc}")
        elif not existing_raw:
            row["raw_vs_prepared"] = {"status": "raw_gt_not_found_at_explicit_candidates"}

        if row["errors"]:
            errors.append({"sample_id": sample_id, "errors": row["errors"]})
        rows.append(row)

    comparisons = [row["raw_vs_prepared"] for row in rows if "raw_label_path" in row.get("raw_vs_prepared", {})]
    aggregate = {
        "approved_sample_count": len(rows),
        "patients": dict(sorted(Counter(str(row["patient"]) for row in rows).items())),
        "prepared_label_suffixes": dict(sorted(Counter(row["prepared_label_suffix"] for row in rows).items())),
        "raw_gt_found_count": len(comparisons),
        "raw_gt_not_found_count": sum(
            row.get("raw_vs_prepared", {}).get("status") == "raw_gt_not_found_at_explicit_candidates"
            for row in rows
        ),
        "prepared_disconnected_id_image_count": sum(
            bool(row.get("prepared", {}).get("disconnected_instance_ids")) for row in rows
        ),
        "foreground_equal_count": sum(bool(row["raw_vs_prepared"].get("foreground_equal")) for row in rows if "raw_label_path" in row.get("raw_vs_prepared", {})),
        "errors": len(errors),
    }
    return {
        "schema_version": 1,
        "phase": "0.5",
        "dataset": "tnbc",
        "protocol": "p1_8_label_provenance_read_only",
        "status": "complete" if not errors else "issues_found",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifests": [
            {"path": str(path), "sha256": sha256_file(path)} for path in source_manifests
        ],
        "closed_patient_attestation": "Only source-manifest p1--p8 sample identities were used; p9--p11 were rejected before sample file access.",
        "prepared_label_policy": "Historical prepared inst_map labels remain the continuity baseline GT; this audit does not select GT by AJI/PQ.",
        "watershed_interpretation": {
            "implementation": "tools/prep_tnbc.py binary_to_instances",
            "min_distance": int(args.watershed_min_distance),
            "sigma": float(args.watershed_sigma),
            "provenance_caveat": "Matching this implementation does not by itself prove how the existing prepared labels were generated."
        },
        "aggregate": aggregate,
        "errors": errors,
        "samples": rows,
    }


def render_summary(report: dict[str, Any]) -> str:
    aggregate = report["aggregate"]
    lines = [
        "# TNBC p1--p8 label provenance audit",
        "",
        f"- Status: `{report['status']}`",
        f"- Approved samples: `{aggregate['approved_sample_count']}`",
        f"- Patients: `{aggregate['patients']}`",
        f"- Prepared label suffixes: `{aggregate['prepared_label_suffixes']}`",
        f"- Raw GT found at exact p1--p8 candidates: `{aggregate['raw_gt_found_count']}`",
        f"- Raw GT not found at exact candidates: `{aggregate['raw_gt_not_found_count']}`",
        f"- Prepared labels with same-ID disconnected regions: `{aggregate['prepared_disconnected_id_image_count']}`",
        "",
        "## Watershed interpretation",
        "",
        f"- Current converter defaults: `min_distance={report['watershed_interpretation']['min_distance']}`, `sigma={report['watershed_interpretation']['sigma']}`.",
        "- Existing prepared labels remain the historical continuity GT; this audit does not choose a label protocol from segmentation metrics.",
    ]
    if aggregate["raw_gt_found_count"]:
        lines.extend(
            [
                "",
                "## Raw/prepared comparison",
                "",
                "| sample | foreground equal | raw CC@8 | prepared instances | delta | watershed exact |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in report["samples"]:
            comparison = row.get("raw_vs_prepared", {})
            if "raw_label_path" not in comparison:
                continue
            watershed_info = comparison["current_prep_watershed"]
            lines.append(
                "| {sample} | {foreground} | {cc8} | {prepared} | {delta} | {watershed} |".format(
                    sample=row["sample_id"],
                    foreground=comparison["foreground_equal"],
                    cc8=comparison["raw_binary_components_8"],
                    prepared=comparison["prepared_instance_count"],
                    delta=comparison["prepared_minus_cc8_instance_count"],
                    watershed=watershed_info["exact_instance_map_equal_to_prepared"],
                )
            )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", action="append", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--prepared-label-root", required=True)
    parser.add_argument(
        "--raw-root",
        action="append",
        default=[],
        help="Parent containing exact GT_01 ... GT_08 paths; never recursively scanned.",
    )
    parser.add_argument("--expected-count", type=int, default=37)
    parser.add_argument("--watershed-min-distance", type=int, default=10)
    parser.add_argument("--watershed-sigma", type=float, default=1.0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = audit(args)
    except (ProtocolViolation, ValueError, FileNotFoundError) as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}), file=sys.stderr)
        return 2
    output = Path(args.output).resolve()
    summary_output = Path(args.summary_output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    summary_output.write_text(render_summary(report), encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(output)}))
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
