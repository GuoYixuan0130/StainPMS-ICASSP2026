"""Summarize the retained C0/C1 screen with low-storage PQ-best ablations.

This is reporting only: all inputs are already-written p7/p8 diagnosis
outputs.  It retains both fixed-epoch-5 comparisons and the owner-approved
development-PQ-best selections, including paired GT coverage flips.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stainpms.phase2a_pqbest import choose_pq_best, selected_vs_reference_report
from stainpms.phase2a_tnbc_screen import MECHANISM_METRICS, TASK_METRICS, metric_deltas


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def selected_record(selection: dict[str, Any]) -> dict[str, Any]:
    return selection["record"]


def fixed_delta(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    if int(reference["epoch"]) != 5 or int(candidate["epoch"]) != 5:
        raise ValueError("fixed comparison requires epoch-5 records")
    return metric_deltas(reference, candidate)


def interaction(records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Single-seed descriptive interaction: full - coverage - quality + C0."""

    c0, full, coverage, quality = (records[name] for name in ("c0", "c1_full", "coverage_only", "quality_only"))
    return {
        "definition": "M_full - M_coverage_only - M_quality_only + M_C0",
        "interpretation": "descriptive single-seed quantity only; not evidence of proven synergy",
        "task_metrics_image_macro": {
            metric: full["patient_macro"]["task_metrics_image_macro"][metric]
            - coverage["patient_macro"]["task_metrics_image_macro"][metric]
            - quality["patient_macro"]["task_metrics_image_macro"][metric]
            + c0["patient_macro"]["task_metrics_image_macro"][metric]
            for metric in TASK_METRICS
        },
        "mechanism": {
            metric: full["patient_macro"]["mechanism"][metric]
            - coverage["patient_macro"]["mechanism"][metric]
            - quality["patient_macro"]["mechanism"][metric]
            + c0["patient_macro"]["mechanism"][metric]
            for metric in MECHANISM_METRICS
        },
    }


def summary_rows(selections: dict[str, dict[str, Any]], fixed: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, selection in selections.items():
        record = selected_record(selection)
        for patient in ("7", "8"):
            row = {"comparison": name, "selection": "development_pq_best", "epoch": record["epoch"], "level": f"patient_{patient}"}
            row.update(record["patients"][patient]["task_metrics_image_macro"])
            row.update(record["patients"][patient]["mechanism"])
            rows.append(row)
        row = {"comparison": name, "selection": "development_pq_best", "epoch": record["epoch"], "level": "patient_macro"}
        row.update(record["patient_macro"]["task_metrics_image_macro"])
        row.update(record["patient_macro"]["mechanism"])
        rows.append(row)
    for name, delta in fixed.items():
        row = {"comparison": name, "selection": "fixed_epoch_5_delta", "epoch": 5, "level": "patient_macro"}
        row.update(delta["patient_macro"]["task_metrics_image_macro"])
        row.update(delta["patient_macro"]["mechanism"])
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["comparison", "selection", "epoch", "level", *TASK_METRICS, *MECHANISM_METRICS]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def markdown(selections: dict[str, dict[str, Any]], fixed: dict[str, dict[str, Any]]) -> str:
    lines = [
        "# TNBC PQ-best loss ablation summary",
        "",
        "All selection uses the frozen equal-patient p7/p8 macro PQ rule; exact PQ ties retain the earlier epoch. Fixed epoch 5 is retained as the equal-training-length comparison.",
        "",
        "| arm | PQ-best epoch | macro PQ | macro AJI | best CCR@0.5 | selected CCR@0.5 | regret |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, selection in selections.items():
        record = selected_record(selection)
        task = record["patient_macro"]["task_metrics_image_macro"]
        mechanism = record["patient_macro"]["mechanism"]
        lines.append(
            f"| {arm} | {record['epoch']} | {task['pq']:.6f} | {task['aji']:.6f} | "
            f"{mechanism['best_candidate_ccr_at_0_5']:.6f} | {mechanism['selected_candidate_ccr_at_0_5']:.6f} | {mechanism['selection_regret']:.6f} |"
        )
    lines.extend(["", "## Fixed epoch-5 deltas versus C0", "", "| arm - C0 | AJI | PQ | best CCR@0.5 | selected CCR@0.5 | regret |", "|---|---:|---:|---:|---:|---:|"])
    for arm, delta in fixed.items():
        task = delta["patient_macro"]["task_metrics_image_macro"]
        mechanism = delta["patient_macro"]["mechanism"]
        lines.append(
            f"| {arm} | {task['aji']:+.6f} | {task['pq']:+.6f} | {mechanism['best_candidate_ccr_at_0_5']:+.6f} | "
            f"{mechanism['selected_candidate_ccr_at_0_5']:+.6f} | {mechanism['selection_regret']:+.6f} |"
        )
    lines.extend(["", "C0 is retained as the continued-training control. Comparison to the historical checkpoint describes overall warm-start change; only comparison to C0 can support an added-loss contribution.", ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c0-c1-metrics", required=True)
    parser.add_argument("--coverage-summary", required=True)
    parser.add_argument("--quality-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output_dir}")
    c0_c1 = read_json(Path(args.c0_c1_metrics))
    coverage = read_json(Path(args.coverage_summary))
    quality = read_json(Path(args.quality_summary))
    if coverage.get("training_configuration", {}).get("arm") != "coverage_only":
        raise ValueError("coverage summary does not declare coverage_only")
    if quality.get("training_configuration", {}).get("arm") != "quality_only":
        raise ValueError("quality summary does not declare quality_only")
    records = {
        "c0": list(c0_c1["c0"]),
        "c1_full": list(c0_c1["c1"]),
        "coverage_only": [record["diagnosis"] for record in coverage["epochs"]],
        "quality_only": [record["diagnosis"] for record in quality["epochs"]],
    }
    selections = {name: choose_pq_best(value) for name, value in records.items()}
    fixed = {
        name: fixed_delta(records["c0"][-1], value[-1])
        for name, value in records.items()
        if name != "c0"
    }
    pqbest_vs_c0 = {
        name: selected_vs_reference_report(selections["c0"], selection)
        for name, selection in selections.items()
        if name != "c0"
    }
    selected_records = {name: selected_record(selection) for name, selection in selections.items()}
    payload = {
        "schema_version": 1,
        "protocol": "tnbc_loss_ablation_pqbest_v1",
        "development_model_selection": {
            "metric": "equal_patient_macro_PQ",
            "tie_break": "earlier_epoch_on_exact_equal_PQ",
            "p7_p8_role": "development_checkpoint_selection_only",
            "fixed_epoch_5_retained": True,
        },
        "pq_best_selection": selections,
        "fixed_epoch_5_vs_c0": fixed,
        "pq_best_vs_c0": pqbest_vs_c0,
        "descriptive_interaction": interaction(selected_records),
        "interpretation_boundary": {
            "historical_checkpoint_comparison": "overall exploratory warm-start change",
            "c0_comparison": "required control for the candidate-loss contribution",
            "not_allowed": "attribute improvement to an added loss when it exceeds only the historical checkpoint but not C0",
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output_dir / "pqbest_ablation_summary.json", payload)
    write_csv(output_dir / "pqbest_ablation_summary.csv", summary_rows(selections, fixed))
    (output_dir / "pqbest_ablation_summary.md").write_text(markdown(selections, fixed), encoding="utf-8")
    print(json.dumps({"status": "complete", "output_dir": str(output_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
