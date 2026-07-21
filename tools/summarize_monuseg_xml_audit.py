"""Create a concise, human-readable Phase 0.5 XML-audit findings report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


AGGREGATE_KEYS = (
    "xml_empty_region_count",
    "xml_invalid_vertex_region_count",
    "xml_out_of_bounds_region_count",
    "xml_self_intersection_region_count",
    "xml_nonadjacent_path_touch_region_count",
    "xml_disconnected_raster_region_count",
    "candidate_fully_occluded_region_count",
    "candidate_effective_instance_count",
    "legacy_instance_count",
    "exact_instance_map_equal_image_count",
    "legacy_disconnected_image_count",
)


def _sample_requires_review(audit: dict[str, Any]) -> bool:
    """Keep the human review list focused on instance-affecting differences.

    Boundary-coordinate and path-geometry fields remain in the JSON for every
    image.  They are deliberately excluded here because their interpretation
    depends on the XML coordinate convention and otherwise obscure the small
    set of samples whose instance identities actually differ.
    """
    return bool(
        audit.get("candidate_effective_instance_count")
        != audit.get("legacy_instance_count")
        or audit.get("candidate_fully_occluded_region_count")
        or audit.get("xml_empty_region_count")
        or audit.get("xml_invalid_vertex_region_count")
        or audit.get("xml_disconnected_raster_region_count")
        or audit.get("legacy_disconnected_instance_ids")
    )


def render_summary(report: dict[str, Any]) -> str:
    aggregates = report.get("aggregates", {})
    lines = [
        "# MoNuSeg XML audit key findings",
        "",
        f"- Audit status: `{report.get('status')}`",
        f"- Source access mode: `{report.get('source_access_mode')}`",
        "- This report identifies conversion/label discrepancies; it does not select a GT protocol.",
        "",
        "## Aggregate anomaly and count fields",
        "",
    ]
    for subset in ("classic30", "extended7", "download37"):
        aggregate = aggregates.get(subset)
        if not isinstance(aggregate, dict):
            continue
        lines.append(f"### {subset}")
        lines.append("")
        for key in AGGREGATE_KEYS:
            lines.append(f"- `{key}`: `{aggregate.get(key)}`")
        lines.append("")

    comparison = report.get("classic30_reported_count_comparison", {})
    lines.extend(
        [
            "## Classic30 paper-count comparison",
            "",
            f"- Reported reference: `{comparison.get('reported_classic30_nuclei')}`",
            f"- Audited XML Regions: `{comparison.get('audited_classic30_xml_regions')}`",
            f"- Delta: `{comparison.get('delta')}`",
            "",
            "## Samples requiring source/conversion review",
            "",
            "Per-image coordinate and path-geometry values remain in the JSON; this list is limited to instance-affecting discrepancies.",
            "",
        ]
    )
    rows: list[dict[str, Any]] = []
    for row in report.get("samples", []):
        audit = row.get("label_audit", {})
        if _sample_requires_review(audit):
            rows.append(
                {
                    "sample_id": row.get("sample_id"),
                    "subset": row.get("subset"),
                    "xml_regions": audit.get("xml_region_count"),
                    "candidate_instances": audit.get("candidate_effective_instance_count"),
                    "legacy_instances": audit.get("legacy_instance_count"),
                    "fully_occluded": audit.get("candidate_fully_occluded_region_count"),
                    "empty": audit.get("xml_empty_region_count"),
                    "out_of_bounds": audit.get("xml_out_of_bounds_region_count"),
                    "self_intersections": audit.get("xml_self_intersection_region_count"),
                    "nonadjacent_path_touches": audit.get(
                        "xml_nonadjacent_path_touch_region_count"
                    ),
                    "disconnected_xml": audit.get("xml_disconnected_raster_region_count"),
                    "disconnected_legacy_ids": audit.get("legacy_disconnected_instance_ids"),
                }
            )
    if not rows:
        lines.append("- None.")
    else:
        for row in rows:
            lines.append(f"- `{json.dumps(row, sort_keys=True)}`")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = json.loads(args.input.read_text(encoding="utf-8"))
        if not isinstance(report, dict):
            raise ValueError("input must be a JSON object")
        text = render_summary(report)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    except Exception as exc:
        print(json.dumps({"status": "issues_found", "error": str(exc)}))
        return 2
    print(json.dumps({"status": "complete", "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
