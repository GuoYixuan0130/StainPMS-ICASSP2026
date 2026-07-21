"""Audit the XML encoding style of a non-test MoNuSeg source-tree manifest.

This is a source-structure audit only: it does not decode image pixels, create
labels, access test records, or select a ground-truth protocol.
"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

try:  # Supports both ``python tools/script.py`` and ``python -m tools.script``.
    from tools.audit_monuseg_xml_labels import (
        _children_named,
        _local_name,
        _normalize_polygon_vertices,
        _region_vertices,
        polygon_intersection_diagnostics,
    )
except ModuleNotFoundError:  # pragma: no cover - exercised by direct CLI use
    from audit_monuseg_xml_labels import (  # type: ignore[no-redef]
        _children_named,
        _local_name,
        _normalize_polygon_vertices,
        _region_vertices,
        polygon_intersection_diagnostics,
    )


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _semantic_attributes(element: ET.Element) -> str:
    ignored = {"id", "displayid"}
    values = {
        key: value
        for key, value in element.attrib.items()
        if key.lower() not in ignored
    }
    return json.dumps(values, sort_keys=True, separators=(",", ":"))


def _attribute_key_counts(elements: list[ET.Element]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for element in elements:
        counts.update(element.attrib.keys())
    return dict(sorted(counts.items()))


def analyze_xml_bytes(xml_bytes: bytes) -> dict[str, Any]:
    root = ET.fromstring(xml_bytes)
    annotations = _children_named(root, "Annotation")
    regions = _children_named(root, "Region")
    annotation_semantics: Counter[str] = Counter(
        _semantic_attributes(annotation) for annotation in annotations
    )
    region_semantics: Counter[str] = Counter(_semantic_attributes(region) for region in regions)
    explicit_closure = 0
    consecutive_duplicate_vertices = 0
    proper_crossings = 0
    path_touches = 0
    vertex_counts: list[int] = []
    for region in regions:
        vertices, _ = _region_vertices(region)
        vertex_counts.append(int(len(vertices)))
        if len(vertices) > 1 and np.allclose(vertices[0], vertices[-1], atol=1.0e-9, rtol=0.0):
            explicit_closure += 1
        normalized = _normalize_polygon_vertices(vertices)
        consecutive_duplicate_vertices += int(len(vertices) - len(normalized))
        geometry = polygon_intersection_diagnostics(vertices)
        proper_crossings += int(geometry["proper_crossing_count"] > 0)
        path_touches += int(geometry["nonadjacent_touch_count"] > 0)
    return {
        "root_tag": _local_name(root.tag),
        "annotation_count": len(annotations),
        "region_count": len(regions),
        "annotation_attribute_key_counts": _attribute_key_counts(annotations),
        "region_attribute_key_counts": _attribute_key_counts(regions),
        "annotation_semantic_signature_counts": dict(sorted(annotation_semantics.items())),
        "region_semantic_signature_counts": dict(sorted(region_semantics.items())),
        "explicit_closing_vertex_region_count": explicit_closure,
        "consecutive_duplicate_vertices_removed": consecutive_duplicate_vertices,
        "proper_crossing_region_count": proper_crossings,
        "nonadjacent_path_touch_region_count": path_touches,
        "vertex_count": {
            "min": min(vertex_counts) if vertex_counts else None,
            "max": max(vertex_counts) if vertex_counts else None,
            "mean": float(sum(vertex_counts) / len(vertex_counts)) if vertex_counts else None,
        },
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = (
        "annotation_count",
        "region_count",
        "explicit_closing_vertex_region_count",
        "consecutive_duplicate_vertices_removed",
        "proper_crossing_region_count",
        "nonadjacent_path_touch_region_count",
    )
    totals = {key: sum(int(row["schema"][key]) for row in rows) for key in keys}
    annotation_signatures: Counter[str] = Counter()
    region_signatures: Counter[str] = Counter()
    annotation_attribute_keys: Counter[str] = Counter()
    region_attribute_keys: Counter[str] = Counter()
    for row in rows:
        schema = row["schema"]
        annotation_signatures.update(schema["annotation_semantic_signature_counts"])
        region_signatures.update(schema["region_semantic_signature_counts"])
        annotation_attribute_keys.update(schema["annotation_attribute_key_counts"])
        region_attribute_keys.update(schema["region_attribute_key_counts"])
    return {
        "image_count": len(rows),
        **totals,
        "annotation_attribute_key_counts": dict(sorted(annotation_attribute_keys.items())),
        "region_attribute_key_counts": dict(sorted(region_attribute_keys.items())),
        "annotation_semantic_signature_counts": dict(sorted(annotation_signatures.items())),
        "region_semantic_signature_counts": dict(sorted(region_signatures.items())),
    }


def audit_manifest(manifest_path: Path) -> dict[str, Any]:
    manifest = _read_json(manifest_path)
    if manifest.get("dataset") != "monuseg":
        raise ValueError("XML schema audit requires a MoNuSeg manifest")
    if "test" in str(manifest.get("role", "")).lower() or "test" in str(
        manifest.get("access_policy", "")
    ).lower():
        raise ValueError("sealed test manifest is forbidden for XML schema audit")
    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("manifest has no records")
    rows: list[dict[str, Any]] = []
    for record in records:
        if "source_xml_path" not in record:
            raise ValueError("XML schema audit currently requires a local source-tree manifest")
        xml_path = Path(str(record["source_xml_path"]))
        rows.append(
            {
                "sample_id": record.get("sample_id"),
                "subset": record.get("subset"),
                "source_xml_path": str(xml_path),
                "schema": analyze_xml_bytes(xml_path.read_bytes()),
            }
        )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("subset") or "unknown")].append(row)
    aggregates = {subset: _aggregate(value) for subset, value in grouped.items()}
    aggregates["download37"] = _aggregate(rows)
    return {
        "schema_version": 1,
        "phase": "0.5",
        "dataset": "monuseg",
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifest": str(manifest_path),
        "source_access_mode": "local_source_tree_xml_only",
        "policy": "structure audit only; no image decoding, label conversion, or test access",
        "aggregates": aggregates,
        "samples": rows,
    }


def render_summary(report: dict[str, Any]) -> str:
    lines = [
        "# MoNuSeg XML schema audit",
        "",
        "- Scope: local training XML structure only; no image decoding or test access.",
        "",
        "| subset | images | annotations | regions | explicit closures | proper crossings | path touches |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for subset in ("classic30", "extended7", "download37"):
        aggregate = report["aggregates"].get(subset)
        if not aggregate:
            continue
        lines.append(
            f"| {subset} | {aggregate['image_count']} | {aggregate['annotation_count']} | "
            f"{aggregate['region_count']} | {aggregate['explicit_closing_vertex_region_count']} | "
            f"{aggregate['proper_crossing_region_count']} | "
            f"{aggregate['nonadjacent_path_touch_region_count']} |"
        )
    for subset in ("classic30", "extended7"):
        aggregate = report["aggregates"].get(subset)
        if aggregate:
            lines.extend(
                [
                    "",
                    f"## {subset} annotation semantic signatures",
                    "",
                    "```json",
                    json.dumps(aggregate["annotation_semantic_signature_counts"], indent=2),
                    "```",
                    "",
                    f"## {subset} region semantic signatures",
                    "",
                    "```json",
                    json.dumps(aggregate["region_semantic_signature_counts"], indent=2),
                    "```",
                ]
            )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary-output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = audit_manifest(args.manifest)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(render_summary(report), encoding="utf-8")
    except Exception as exc:
        print(json.dumps({"status": "issues_found", "error": str(exc)}))
        return 2
    print(json.dumps({"status": "complete", "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
