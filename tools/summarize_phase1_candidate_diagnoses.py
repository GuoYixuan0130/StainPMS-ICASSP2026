"""Combine completed Phase 1 read-only diagnostic output directories.

This tool reads only Phase 1 JSON/CSV outputs. It never opens data, models, or
checkpoints, so it is safe for the p1--p6/p7--p8 report consolidation step.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stainpms.phase1_metrics import summarize_gt_rows


def parse_cell(value: str):
    if value == "":
        return None
    if value in {"True", "False"}:
        return value == "True"
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        try:
            return float(value) if "." in value else int(value)
        except ValueError:
            return value


def read_rows(directory: Path) -> tuple[dict, list[dict], list[dict]]:
    summary_path = directory / "summary.json"
    rows_path = directory / "gt_instances.csv"
    images_path = directory / "images.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "complete":
        raise ValueError(f"only complete Phase 1 outputs can be consolidated: {directory}")
    with rows_path.open(newline="", encoding="utf-8") as handle:
        rows = [{key: parse_cell(value) for key, value in row.items()} for row in csv.DictReader(handle)]
    images = json.loads(images_path.read_text(encoding="utf-8"))
    return summary, rows, images


def summarize(rows: list[dict], images: list[dict], thresholds: list[float], match_iou: float) -> dict:
    report = summarize_gt_rows(rows, thresholds=thresholds, match_iou=match_iou)
    structural_keys = ("tp", "fp", "fn", "split_unmatched_gt_count", "merge_unmatched_pred_count", "boundary_localization_unmatched_gt_count")
    points = sum(int(image["auto_decoder_point_count"]) for image in images)
    backgrounds = sum(int(image["background_auto_point_count"]) for image in images)
    report["auto_points"] = {
        "count": points,
        "background_count": backgrounds,
        "background_fraction": backgrounds / points if points else None,
    }
    report["final_structural_errors"] = {
        key: sum(int(image["structural_errors"][key]) for image in images) for key in structural_keys
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    summaries, rows, images = [], [], []
    for raw in args.input_dir:
        summary, child_rows, child_images = read_rows(Path(raw).resolve())
        summaries.append(summary)
        rows.extend(child_rows)
        images.extend(child_images)
    datasets = {summary["dataset"] for summary in summaries}
    if len(datasets) != 1:
        raise ValueError(f"all input directories must be one dataset, got {sorted(datasets)}")
    spec_ids = {summary["metric_spec"]["sha256"] for summary in summaries}
    if len(spec_ids) != 1:
        raise ValueError("metric-spec hashes differ across inputs")
    checkpoints = {summary["checkpoint"]["checkpoint_sha256"] for summary in summaries}
    if len(checkpoints) != 1:
        raise ValueError("checkpoint hashes differ across inputs")
    match_iou = float(summaries[0]["metric_spec"].get("main_match_iou", 0.5))
    thresholds = [0.3, 0.5, 0.7]
    report = {
        "schema_version": 1,
        "phase": 1,
        "status": "complete_consolidated",
        "dataset": summaries[0]["dataset"],
        "source_outputs": [str(Path(value).resolve()) for value in args.input_dir],
        "metric_spec_sha256": summaries[0]["metric_spec"]["sha256"],
        "checkpoint_sha256": summaries[0]["checkpoint"]["checkpoint_sha256"],
        "checkpoint_classification": summaries[0]["checkpoint"]["classification"],
        "overall": summarize(rows, images, thresholds=thresholds, match_iou=match_iou),
        "by_split": {},
    }
    for split in sorted({str(row.get("split")) for row in rows}):
        report["by_split"][split] = summarize(
            [row for row in rows if str(row.get("split")) == split],
            [image for image in images if str(image.get("split")) == split],
            thresholds=thresholds,
            match_iou=match_iou,
        )
    for patient in sorted({row.get("patient") for row in rows if row.get("patient") is not None}):
        report.setdefault("by_patient", {})[str(patient)] = summarize(
            [row for row in rows if row.get("patient") == patient],
            [image for image in images if image.get("patient") == patient],
            thresholds=thresholds,
            match_iou=match_iou,
        )
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "output": str(output), "gt_instances": len(rows), "images": len(images)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
