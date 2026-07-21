"""Audit MoNuSeg XML regions against legacy prepared instance labels.

Each XML ``Region`` is assigned one candidate instance identity.  The candidate
rasterizer uses coordinates exactly as recorded, ``skimage.draw.polygon`` and a
deterministic last-region-wins overlap policy.  This is an auditable candidate
conversion, not a claim of bitwise equivalence to MATLAB ``poly2mask``.

Legacy labels are never modified.  Optional regenerated MAT files are written
under a separate root and an existing non-identical file is rejected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import xml.etree.ElementTree as ET
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy import ndimage as ndi
from scipy.io import loadmat, savemat
from skimage import io as skio
from skimage.draw import polygon


CONNECTIVITY_8 = np.ones((3, 3), dtype=np.uint8)
REFERENCE_CLASSIC30_REGION_COUNT = 21623


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _children_named(element: ET.Element, name: str) -> list[ET.Element]:
    return [child for child in element.iter() if _local_name(child.tag) == name]


def _region_vertices(region: ET.Element) -> tuple[np.ndarray, list[str]]:
    errors: list[str] = []
    vertices: list[tuple[float, float]] = []
    for vertex in _children_named(region, "Vertex"):
        try:
            x = float(vertex.attrib["X"])
            y = float(vertex.attrib["Y"])
        except (KeyError, TypeError, ValueError):
            errors.append("invalid_vertex")
            continue
        if not math.isfinite(x) or not math.isfinite(y):
            errors.append("nonfinite_vertex")
            continue
        vertices.append((x, y))
    return np.asarray(vertices, dtype=np.float64), sorted(set(errors))


def _orientation(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def _normalize_polygon_vertices(vertices: np.ndarray, eps: float = 1.0e-9) -> np.ndarray:
    """Remove zero-length edges, including an explicit repeated closing vertex."""
    if len(vertices) <= 1:
        return vertices
    kept = [vertices[0]]
    for vertex in vertices[1:]:
        if not np.allclose(vertex, kept[-1], atol=eps, rtol=0.0):
            kept.append(vertex)
    normalized = np.asarray(kept, dtype=np.float64)
    if len(normalized) > 1 and np.allclose(
        normalized[0], normalized[-1], atol=eps, rtol=0.0
    ):
        normalized = normalized[:-1]
    return normalized


def _on_segment(a: np.ndarray, b: np.ndarray, p: np.ndarray, eps: float = 1.0e-9) -> bool:
    return (
        min(a[0], b[0]) - eps <= p[0] <= max(a[0], b[0]) + eps
        and min(a[1], b[1]) - eps <= p[1] <= max(a[1], b[1]) + eps
        and abs(_orientation(a, b, p)) <= eps
    )


def _segments_intersect(
    a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray, eps: float = 1.0e-9
) -> bool:
    o1 = _orientation(a, b, c)
    o2 = _orientation(a, b, d)
    o3 = _orientation(c, d, a)
    o4 = _orientation(c, d, b)
    if ((o1 > eps and o2 < -eps) or (o1 < -eps and o2 > eps)) and (
        (o3 > eps and o4 < -eps) or (o3 < -eps and o4 > eps)
    ):
        return True
    return (
        (abs(o1) <= eps and _on_segment(a, b, c, eps))
        or (abs(o2) <= eps and _on_segment(a, b, d, eps))
        or (abs(o3) <= eps and _on_segment(c, d, a, eps))
        or (abs(o4) <= eps and _on_segment(c, d, b, eps))
    )


def polygon_self_intersects(vertices: np.ndarray) -> bool:
    vertices = _normalize_polygon_vertices(vertices)
    count = len(vertices)
    if count < 4:
        return False
    for left in range(count):
        left_next = (left + 1) % count
        for right in range(left + 1, count):
            right_next = (right + 1) % count
            if left == right or left_next == right or right_next == left:
                continue
            if left == 0 and right_next == 0:
                continue
            if _segments_intersect(
                vertices[left], vertices[left_next], vertices[right], vertices[right_next]
            ):
                return True
    return False


def _rasterize_region(vertices: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    if len(vertices) < 3:
        return mask
    rows, cols = polygon(vertices[:, 1], vertices[:, 0], shape=shape)
    mask[rows, cols] = True
    return mask


def _instance_ids(label: np.ndarray) -> list[int]:
    return [int(value) for value in np.unique(label) if int(value) > 0]


def _disconnected_ids(label: np.ndarray) -> dict[str, int]:
    result: dict[str, int] = {}
    for instance_id in _instance_ids(label):
        _, count = ndi.label(label == instance_id, structure=CONNECTIVITY_8)
        if count > 1:
            result[str(instance_id)] = int(count)
    return result


def _foreground_comparison(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    left_bin = left > 0
    right_bin = right > 0
    intersection = int(np.logical_and(left_bin, right_bin).sum())
    union = int(np.logical_or(left_bin, right_bin).sum())
    total = int(left_bin.sum() + right_bin.sum())
    return {
        "left_foreground_pixels": int(left_bin.sum()),
        "right_foreground_pixels": int(right_bin.sum()),
        "intersection_pixels": intersection,
        "union_pixels": union,
        "xor_pixels": int(np.logical_xor(left_bin, right_bin).sum()),
        "dice": float(2 * intersection / total) if total else None,
        "iou": float(intersection / union) if union else None,
        "foreground_equal": bool(np.array_equal(left_bin, right_bin)),
    }


def _load_legacy(path: Path) -> np.ndarray:
    payload = loadmat(path)
    if "inst_map" not in payload:
        raise KeyError(f"legacy MAT has no inst_map: {path}")
    label = np.asarray(payload["inst_map"])
    label = np.squeeze(label)
    if label.ndim != 2:
        raise ValueError(f"legacy inst_map is not 2-D: {path} {label.shape}")
    return label.astype(np.int32, copy=False)


def _image_identity_comparison(
    source_bytes: bytes,
    prepared_path: Path,
) -> dict[str, Any]:
    source = np.asarray(skio.imread(BytesIO(source_bytes)))
    prepared = np.asarray(skio.imread(prepared_path))

    def rgb_view(value: np.ndarray) -> np.ndarray:
        if value.ndim == 2:
            return np.repeat(value[..., None], 3, axis=-1)
        if value.ndim == 3 and value.shape[-1] >= 3:
            return value[..., :3]
        return value

    source_rgb = rgb_view(source)
    prepared_rgb = rgb_view(prepared)
    comparable = source_rgb.shape == prepared_rgb.shape
    return {
        "source_shape": list(source.shape),
        "source_dtype": str(source.dtype),
        "prepared_shape": list(prepared.shape),
        "prepared_dtype": str(prepared.dtype),
        "spatial_shape_equal": bool(source.shape[:2] == prepared.shape[:2]),
        "rgb_shape_equal": bool(comparable),
        "exact_rgb_pixel_equal": bool(
            comparable and np.array_equal(source_rgb, prepared_rgb)
        ),
        "policy": "identity_and_preprocessing_check_only_no_color_statistics",
    }


def _parse_regions(xml_bytes: bytes) -> list[ET.Element]:
    root = ET.fromstring(xml_bytes)
    return _children_named(root, "Region")


def audit_xml_label(
    xml_bytes: bytes,
    legacy: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray]:
    shape = tuple(int(value) for value in legacy.shape)
    regions = _parse_regions(xml_bytes)
    regenerated = np.zeros(shape, dtype=np.int32)
    coverage = np.zeros(shape, dtype=np.uint16)
    region_masks: list[np.ndarray] = []
    anomalies: list[dict[str, Any]] = []
    empty_count = 0
    out_of_bounds_count = 0
    self_intersection_count = 0
    disconnected_count = 0
    invalid_vertex_regions = 0
    overlap_pixels_written = 0

    for region_index, region in enumerate(regions, start=1):
        vertices, vertex_errors = _region_vertices(region)
        if vertex_errors:
            invalid_vertex_regions += 1
        out_of_bounds = bool(
            len(vertices)
            and (
                (vertices[:, 0] < 0).any()
                or (vertices[:, 0] > shape[1] - 1).any()
                or (vertices[:, 1] < 0).any()
                or (vertices[:, 1] > shape[0] - 1).any()
            )
        )
        self_intersection = polygon_self_intersects(vertices)
        mask = _rasterize_region(vertices, shape)
        _, components = ndi.label(mask, structure=CONNECTIVITY_8)
        empty = not bool(mask.any())
        if empty:
            empty_count += 1
        if out_of_bounds:
            out_of_bounds_count += 1
        if self_intersection:
            self_intersection_count += 1
        if components > 1:
            disconnected_count += 1
        if vertex_errors or empty or out_of_bounds or self_intersection or components > 1:
            anomalies.append(
                {
                    "region_index": region_index,
                    "xml_region_id": region.attrib.get("Id") or region.attrib.get("ID"),
                    "vertex_count": int(len(vertices)),
                    "vertex_errors": vertex_errors,
                    "empty": empty,
                    "out_of_bounds": out_of_bounds,
                    "self_intersection": self_intersection,
                    "raster_components_8": int(components),
                    "raster_area": int(mask.sum()),
                }
            )
        overlap_pixels_written += int(np.logical_and(mask, coverage > 0).sum())
        coverage[mask] += 1
        regenerated[mask] = region_index
        region_masks.append(mask)

    best_legacy_to_regions: dict[int, list[int]] = defaultdict(list)
    regions_spanning_legacy: list[dict[str, Any]] = []
    for region_index, mask in enumerate(region_masks, start=1):
        overlap_ids, overlap_counts = np.unique(legacy[mask], return_counts=True)
        positive = [
            (int(value), int(count))
            for value, count in zip(overlap_ids, overlap_counts)
            if int(value) > 0
        ]
        if not positive:
            continue
        best_id, _ = max(positive, key=lambda item: (item[1], -item[0]))
        best_legacy_to_regions[best_id].append(region_index)
        if len(positive) > 1:
            regions_spanning_legacy.append(
                {
                    "region_index": region_index,
                    "legacy_ids": [value for value, _ in positive],
                    "intersection_pixels": [count for _, count in positive],
                }
            )
    merged_legacy = {
        str(instance_id): region_ids
        for instance_id, region_ids in best_legacy_to_regions.items()
        if len(region_ids) > 1
    }
    legacy_ids = set(_instance_ids(legacy))
    legacy_with_xml = set(best_legacy_to_regions)
    effective_ids = _instance_ids(regenerated)
    report = {
        "shape": list(shape),
        "xml_region_count": len(regions),
        "xml_empty_region_count": empty_count,
        "xml_invalid_vertex_region_count": invalid_vertex_regions,
        "xml_out_of_bounds_region_count": out_of_bounds_count,
        "xml_self_intersection_region_count": self_intersection_count,
        "xml_disconnected_raster_region_count": disconnected_count,
        "xml_region_overlap_pixels_written": overlap_pixels_written,
        "xml_region_overlap_unique_pixels": int((coverage > 1).sum()),
        "candidate_effective_instance_count": len(effective_ids),
        "candidate_fully_occluded_region_count": len(regions) - len(effective_ids),
        "candidate_disconnected_instance_ids": _disconnected_ids(regenerated),
        "legacy_instance_count": len(legacy_ids),
        "legacy_disconnected_instance_ids": _disconnected_ids(legacy),
        "legacy_ids_best_matched_by_multiple_xml_regions": merged_legacy,
        "legacy_ids_without_any_xml_overlap": sorted(legacy_ids - legacy_with_xml),
        "xml_regions_overlapping_multiple_legacy_ids": regions_spanning_legacy,
        "foreground_comparison_legacy_vs_candidate": _foreground_comparison(
            legacy, regenerated
        ),
        "exact_instance_map_equal": bool(np.array_equal(legacy, regenerated)),
        "anomalous_regions": anomalies,
    }
    return report, regenerated


def _save_regenerated(path: Path, label: np.ndarray) -> str:
    if path.exists():
        existing = _load_legacy(path)
        if not np.array_equal(existing, label):
            raise FileExistsError(f"refusing to overwrite non-identical generated label: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        savemat(path, {"inst_map": label.astype(np.int32, copy=False)})
    return _sha256_file(path)


def _sum_field(rows: Iterable[dict[str, Any]], key: str) -> int:
    return sum(int(row["label_audit"][key]) for row in rows)


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    source_shapes: dict[str, int] = defaultdict(int)
    prepared_shapes: dict[str, int] = defaultdict(int)
    for row in rows:
        comparison = row["source_vs_prepared_image"]
        source_shapes[str(comparison["source_shape"])] += 1
        prepared_shapes[str(comparison["prepared_shape"])] += 1
    return {
        "image_count": len(rows),
        "xml_region_count": _sum_field(rows, "xml_region_count"),
        "xml_empty_region_count": _sum_field(rows, "xml_empty_region_count"),
        "xml_invalid_vertex_region_count": _sum_field(
            rows, "xml_invalid_vertex_region_count"
        ),
        "xml_out_of_bounds_region_count": _sum_field(
            rows, "xml_out_of_bounds_region_count"
        ),
        "xml_self_intersection_region_count": _sum_field(
            rows, "xml_self_intersection_region_count"
        ),
        "xml_disconnected_raster_region_count": _sum_field(
            rows, "xml_disconnected_raster_region_count"
        ),
        "candidate_effective_instance_count": _sum_field(
            rows, "candidate_effective_instance_count"
        ),
        "candidate_fully_occluded_region_count": _sum_field(
            rows, "candidate_fully_occluded_region_count"
        ),
        "legacy_instance_count": _sum_field(rows, "legacy_instance_count"),
        "legacy_disconnected_image_count": sum(
            bool(row["label_audit"]["legacy_disconnected_instance_ids"]) for row in rows
        ),
        "candidate_disconnected_image_count": sum(
            bool(row["label_audit"]["candidate_disconnected_instance_ids"]) for row in rows
        ),
        "exact_instance_map_equal_image_count": sum(
            bool(row["label_audit"]["exact_instance_map_equal"]) for row in rows
        ),
        "source_image_shape_counts": dict(sorted(source_shapes.items())),
        "prepared_image_shape_counts": dict(sorted(prepared_shapes.items())),
        "source_prepared_spatial_shape_mismatch_image_count": sum(
            not row["source_vs_prepared_image"]["spatial_shape_equal"] for row in rows
        ),
        "source_prepared_exact_rgb_equal_image_count": sum(
            row["source_vs_prepared_image"]["exact_rgb_pixel_equal"] for row in rows
        ),
    }


def audit_manifest(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = Path(args.manifest)
    manifest = _load_json(manifest_path)
    if manifest.get("dataset") != "monuseg":
        raise ValueError("XML audit requires a MoNuSeg manifest")
    if "test" in str(manifest.get("role", "")).lower() or "test" in str(
        manifest.get("access_policy", "")
    ).lower():
        raise ValueError("sealed test manifest is forbidden for XML/label audit")
    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("manifest has no records")
    source_descriptor = manifest["source_archive"]
    archive_path_value = source_descriptor.get("path")
    archive: zipfile.ZipFile | None = None
    source_access_mode = "local_source_tree" if not archive_path_value else "archive"
    if archive_path_value:
        archive = zipfile.ZipFile(Path(str(archive_path_value)), "r")
    generated_root = Path(args.regenerated_label_root) if args.regenerated_label_root else None
    rows: list[dict[str, Any]] = []
    try:
        for record in records:
            sample_id = str(record["sample_id"])
            image_member = str(record["source_image_member"])
            if archive is None:
                source_image_path = Path(str(record["source_image_path"]))
                xml_source_path = Path(str(record["source_xml_path"]))
                source_image_bytes = source_image_path.read_bytes()
                xml_bytes = xml_source_path.read_bytes()
            else:
                with archive.open(image_member, "r") as handle:
                    source_image_bytes = handle.read()
                xml_member = str(record["source_xml_member"])
                with archive.open(xml_member, "r") as handle:
                    xml_bytes = handle.read()
            source_image_sha = hashlib.sha256(source_image_bytes).hexdigest()
            if source_image_sha != record.get("source_image_sha256"):
                raise ValueError(f"source image hash mismatch for {sample_id}")
            prepared_image_path = Path(str(record["image_path"]))
            if _sha256_file(prepared_image_path) != record.get("image_sha256"):
                raise ValueError(f"prepared image hash mismatch for {sample_id}")
            source_vs_prepared = _image_identity_comparison(
                source_image_bytes, prepared_image_path
            )
            xml_member = str(record["source_xml_member"])
            xml_sha = hashlib.sha256(xml_bytes).hexdigest()
            if xml_sha != record.get("source_xml_sha256"):
                raise ValueError(f"XML hash mismatch for {sample_id}")
            legacy_path = Path(str(record["label_path"]))
            if _sha256_file(legacy_path) != record.get("label_sha256"):
                raise ValueError(f"legacy label hash mismatch for {sample_id}")
            legacy = _load_legacy(legacy_path)
            label_audit, regenerated = audit_xml_label(xml_bytes, legacy)
            generated_path = None
            generated_sha = None
            if generated_root is not None:
                generated_path = generated_root / f"{sample_id}.mat"
                if generated_path.resolve() == legacy_path.resolve():
                    raise ValueError(f"generated path aliases legacy path: {sample_id}")
                generated_sha = _save_regenerated(generated_path, regenerated)
            rows.append(
                {
                    "sample_id": sample_id,
                    "case": record.get("case"),
                    "subset": record.get("subset"),
                    "source_image_member": image_member,
                    "source_image_sha256": source_image_sha,
                    "prepared_image_path": str(prepared_image_path),
                    "prepared_image_sha256": record.get("image_sha256"),
                    "source_vs_prepared_image": source_vs_prepared,
                    "source_xml_member": xml_member,
                    "source_xml_sha256": xml_sha,
                    "legacy_label_path": str(legacy_path),
                    "legacy_label_sha256": record.get("label_sha256"),
                    "candidate_label_path": str(generated_path) if generated_path else None,
                    "candidate_label_sha256": generated_sha,
                    "label_audit": label_audit,
                }
            )
    finally:
        if archive is not None:
            archive.close()

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("subset") or "unknown")].append(row)
    aggregates = {key: _aggregate(value) for key, value in grouped.items()}
    aggregates["download37"] = _aggregate(rows)
    classic_total = aggregates.get("classic30", {}).get("xml_region_count")
    reference = {
        "reported_classic30_nuclei": REFERENCE_CLASSIC30_REGION_COUNT,
        "audited_classic30_xml_regions": classic_total,
        "delta": (
            int(classic_total) - REFERENCE_CLASSIC30_REGION_COUNT
            if classic_total is not None
            else None
        ),
        "policy": "investigate_version_or_conversion_difference; never force counts",
    }
    anomaly_total = sum(
        value
        for key, value in aggregates["download37"].items()
        if key
        in {
            "xml_empty_region_count",
            "xml_invalid_vertex_region_count",
            "xml_out_of_bounds_region_count",
            "xml_self_intersection_region_count",
            "xml_disconnected_raster_region_count",
        }
    )
    status = "complete" if anomaly_total == 0 else "complete_with_xml_anomalies"
    return {
        "schema_version": 1,
        "phase": "0.5",
        "dataset": "monuseg",
        "status": status,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": _sha256_file(manifest_path),
        "source_archive": source_descriptor,
        "source_access_mode": source_access_mode,
        "rasterization_protocol": {
            "id": "xml_region_skimage_polygon_last_wins_v1",
            "identity_unit": "one XML Region per instance ID",
            "coordinate_transform": "coordinates_as_recorded_no_offset",
            "overlap_policy": "later XML Region overwrites earlier Region pixels",
            "connectivity_audit": 8,
            "claim": "candidate audit conversion; not yet locked as official GT",
        },
        "aggregates": aggregates,
        "classic30_reported_count_comparison": reference,
        "samples": rows,
    }


def _write_summary(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# MoNuSeg XML/legacy label audit",
        "",
        f"- Status: `{report['status']}`",
        f"- Rasterizer: `{report['rasterization_protocol']['id']}`",
        "- Legacy labels were read-only and were not overwritten.",
        "",
        "| subset | images | XML Regions | candidate instances | legacy instances | legacy disconnected images |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for subset, aggregate in report["aggregates"].items():
        lines.append(
            f"| {subset} | {aggregate['image_count']} | {aggregate['xml_region_count']} | "
            f"{aggregate['candidate_effective_instance_count']} | {aggregate['legacy_instance_count']} | "
            f"{aggregate['legacy_disconnected_image_count']} |"
        )
    reference = report["classic30_reported_count_comparison"]
    lines.extend(
        [
            "",
            "## Classic30 count check",
            "",
            f"- Reported reference: `{reference['reported_classic30_nuclei']}`",
            f"- Audited XML Regions: `{reference['audited_classic30_xml_regions']}`",
            f"- Delta: `{reference['delta']}`",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--regenerated-label-root", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = audit_manifest(args)
        _write_json(Path(args.output), report)
        _write_summary(Path(args.summary_output), report)
    except Exception as exc:
        print(json.dumps({"status": "issues_found", "error": str(exc)}))
        return 2
    print(json.dumps({"status": report["status"], "output": args.output}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
