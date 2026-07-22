"""Export compact, auditable Phase 1 tables from completed diagnostics.

The input directories must contain the original ``summary.json``,
``gt_instances.csv`` and ``images.json`` written by the Phase 1 runner.  The
point/final contingency table is computed directly from per-GT rows and is
cross-checked against the mutually exclusive error partition before anything
is written.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ERROR_CLASSES = (
    "final_matched_tp",
    "point_miss",
    "candidate_generation_miss",
    "selection_ranking_miss",
    "assembly_nms_conflict_miss",
)


def _parse_cell(value: str) -> Any:
    if value == "":
        return None
    if value in {"True", "False"}:
        return value == "True"
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            {key: _parse_cell(value) for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _threshold_map(values: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {f"{float(item['threshold']):.1f}": item for item in values}


@dataclass(frozen=True)
class Source:
    directory: Path
    summary: dict[str, Any]
    gt_rows: list[dict[str, Any]]
    image_rows: list[dict[str, Any]]


def read_source(raw_directory: str | Path) -> Source:
    directory = Path(raw_directory).resolve()
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if summary.get("status") != "complete":
        raise ValueError(f"Phase 1 source is not complete: {directory}")
    gt_rows = _read_csv(directory / "gt_instances.csv")
    image_rows = json.loads((directory / "images.json").read_text(encoding="utf-8"))
    expected = int(summary["overall"]["gt_instance_count"])
    if len(gt_rows) != expected:
        raise ValueError(f"{directory}: gt row count {len(gt_rows)} != summary {expected}")
    keys = [(str(row["sample_id"]), int(row["gt_instance_id"])) for row in gt_rows]
    if len(keys) != len(set(keys)):
        raise ValueError(f"{directory}: duplicate (sample_id, gt_instance_id) rows")
    return Source(directory, summary, gt_rows, image_rows)


def contingency(rows: list[dict[str, Any]]) -> dict[str, int]:
    result = {
        "with_own_point_final_matched": 0,
        "with_own_point_final_fn": 0,
        "without_own_point_final_matched": 0,
        "without_own_point_final_fn": 0,
    }
    for row in rows:
        has_point = int(row["auto_point_count"]) > 0
        matched = bool(row["final_matched"])
        if has_point and matched:
            result["with_own_point_final_matched"] += 1
        elif has_point:
            result["with_own_point_final_fn"] += 1
        elif matched:
            result["without_own_point_final_matched"] += 1
        else:
            result["without_own_point_final_fn"] += 1
    result["with_own_point_total"] = (
        result["with_own_point_final_matched"] + result["with_own_point_final_fn"]
    )
    result["without_own_point_total"] = (
        result["without_own_point_final_matched"] + result["without_own_point_final_fn"]
    )
    result["final_matched_total"] = (
        result["with_own_point_final_matched"]
        + result["without_own_point_final_matched"]
    )
    result["final_fn_total"] = (
        result["with_own_point_final_fn"] + result["without_own_point_final_fn"]
    )
    result["gt_total"] = len(rows)
    return result


def validate_partition(
    rows: list[dict[str, Any]], summary_block: dict[str, Any], label: str
) -> dict[str, int]:
    observed_errors = {name: 0 for name in ERROR_CLASSES}
    for row in rows:
        name = str(row["error_class"])
        if name not in observed_errors:
            raise ValueError(f"{label}: unknown error class {name!r}")
        observed_errors[name] += 1
    expected_errors = {
        name: int(summary_block["error_classes"].get(name, 0)) for name in ERROR_CLASSES
    }
    if observed_errors != expected_errors:
        raise ValueError(
            f"{label}: per-GT error counts {observed_errors} != summary {expected_errors}"
        )
    table = contingency(rows)
    point_recall = summary_block["auto_point_recall"]
    if table["with_own_point_total"] != int(point_recall["numerator"]):
        raise ValueError(f"{label}: own-point total does not match auto-point recall")
    if table["without_own_point_final_fn"] != observed_errors["point_miss"]:
        raise ValueError(f"{label}: no-point/final-FN cell does not match point_miss")
    if table["final_matched_total"] != observed_errors["final_matched_tp"]:
        raise ValueError(f"{label}: final-matched column does not match final_matched_tp")
    later_misses = sum(observed_errors[name] for name in ERROR_CLASSES[2:])
    if table["with_own_point_final_fn"] != later_misses:
        raise ValueError(f"{label}: own-point/final-FN cell does not match later misses")
    if table["gt_total"] != int(summary_block["gt_instance_count"]):
        raise ValueError(f"{label}: contingency total does not match summary")
    return observed_errors


def _base(source: Source, cohort: str, group_type: str, group_id: str) -> dict[str, Any]:
    summary = source.summary
    return {
        "dataset": summary["dataset"],
        "cohort": cohort,
        "group_type": group_type,
        "group_id": group_id,
        "metric_spec_sha256": summary["metric_spec"]["sha256"],
        "checkpoint_sha256": summary["checkpoint"]["checkpoint_sha256"],
        "checkpoint_classification": summary["checkpoint"]["classification"],
        "manifest_sha256": summary["manifest"]["sha256"],
        "manifest_protocol_id": summary["manifest"]["protocol_id"],
    }


def _summary_row(base: dict[str, Any], block: dict[str, Any]) -> dict[str, Any]:
    gt_ccr = _threshold_map(block["ccr_gt_point"])
    auto_cond = _threshold_map(block["ccr_auto_given_point"])
    auto_e2e = _threshold_map(block["ccr_auto_e2e"])
    candidate = block["candidate_iou"]
    points = block["auto_points"]
    structural = block["final_structural_errors"]
    row = {
        **base,
        "image_count": int(block["image_count"]),
        "gt_instance_count": int(block["gt_instance_count"]),
        "auto_point_recall_numerator": int(block["auto_point_recall"]["numerator"]),
        "auto_point_recall_denominator": int(block["auto_point_recall"]["denominator"]),
        "auto_point_recall": block["auto_point_recall"]["value"],
        "auto_point_count": int(points["count"]),
        "background_point_count": int(points["background_count"]),
        "background_point_fraction": points["background_fraction"],
        "points_per_gt_mean": points.get("per_gt_count_mean"),
        "best_candidate_iou_mean": candidate["best_mean"],
        "selected_candidate_iou_mean": candidate["selected_standard_candidate_mean"],
        "selection_regret_mean": candidate["selection_regret_mean"],
        "qualified_candidate_count_at_0_5": int(candidate["qualified_candidate_count"]),
        "qualified_but_not_final_count_at_0_5": int(candidate["qualified_but_not_final_count"]),
        "final_tp": int(structural["tp"]),
        "final_fp": int(structural["fp"]),
        "final_fn": int(structural["fn"]),
        "final_split_unmatched_gt": int(structural["split_unmatched_gt_count"]),
        "final_merge_unmatched_pred": int(structural["merge_unmatched_pred_count"]),
        "final_boundary_localization_unmatched_gt": int(
            structural["boundary_localization_unmatched_gt_count"]
        ),
    }
    for threshold in ("0.3", "0.5", "0.7"):
        suffix = threshold.replace(".", "_")
        row[f"gt_point_ccr_{suffix}"] = gt_ccr[threshold]["value"]
        row[f"auto_ccr_given_point_{suffix}"] = auto_cond[threshold]["value"]
        row[f"auto_ccr_e2e_{suffix}"] = auto_e2e[threshold]["value"]
    return row


def _partition_row(
    base: dict[str, Any], rows: list[dict[str, Any]], block: dict[str, Any]
) -> dict[str, Any]:
    counts = validate_partition(rows, block, f"{base['cohort']}:{base['group_id']}")
    total = len(rows)
    result = {**base, "gt_instance_count": total}
    for name in ERROR_CLASSES:
        result[f"{name}_count"] = counts[name]
        result[f"{name}_fraction"] = counts[name] / total if total else None
    return result


def export_tables(sources: list[Source], output_dir: Path) -> dict[str, Any]:
    summary_rows: list[dict[str, Any]] = []
    partition_rows: list[dict[str, Any]] = []
    contingency_rows: list[dict[str, Any]] = []
    provenance_rows: list[dict[str, Any]] = []

    for source in sources:
        scope = str(source.summary["scope_label"])
        groups: list[tuple[str, str, list[dict[str, Any]], dict[str, Any]]] = [
            ("cohort", scope, source.gt_rows, source.summary["overall"])
        ]
        if source.summary["dataset"] == "tnbc":
            patients = sorted({int(row["patient"]) for row in source.gt_rows})
            for patient in patients:
                patient_rows = [row for row in source.gt_rows if int(row["patient"]) == patient]
                groups.append(
                    (
                        "patient",
                        str(patient),
                        patient_rows,
                        source.summary["groups"][f"patient:{patient}"],
                    )
                )

        for group_type, group_id, rows, block in groups:
            base = _base(source, scope, group_type, group_id)
            summary_rows.append(_summary_row(base, block))
            partition_rows.append(_partition_row(base, rows, block))
            table = contingency(rows)
            contingency_rows.append({**base, **table})

        provenance_rows.append(
            {
                **_base(source, scope, "source", scope),
                "source_directory": str(source.directory),
                "source_git_commit": source.summary["repository"]["commit"],
                "gt_instances_csv_row_count": len(source.gt_rows),
                "images_json_row_count": len(source.image_rows),
                "checkpoint_selection_history": source.summary["checkpoint"].get(
                    "selection_history"
                ),
                "checkpoint_training_manifest": source.summary["checkpoint"].get(
                    "training_manifest"
                ),
            }
        )

    paths = {
        "summary": output_dir / "phase1_summary.csv",
        "error_partition": output_dir / "phase1_error_partition.csv",
        "point_final_contingency": output_dir / "phase1_point_final_contingency.csv",
        "provenance": output_dir / "phase1_provenance.csv",
    }
    _write_csv(paths["summary"], summary_rows)
    _write_csv(paths["error_partition"], partition_rows)
    _write_csv(paths["point_final_contingency"], contingency_rows)
    _write_csv(paths["provenance"], provenance_rows)
    return {
        "status": "complete",
        "source_count": len(sources),
        "scope_count": len(summary_rows),
        "outputs": {name: str(path.resolve()) for name, path in paths.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    sources = [read_source(value) for value in args.input_dir]
    report = export_tables(sources, Path(args.output_dir).resolve())
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
