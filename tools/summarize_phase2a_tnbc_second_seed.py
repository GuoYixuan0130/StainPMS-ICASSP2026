"""Summarize the owner-approved TNBC C0/C1 second-seed reproduction.

This tool is read-only: it consumes the two completed low-storage training
summaries and their frozen p7/p8 diagnosis outputs. The primary comparison is
fixed epoch 5; PQ-best records remain secondary development bookkeeping.
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


PROTOCOL = "tnbc_c0_c1_second_seed_2027_v1"


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


def diagnosis_records(summary: dict[str, Any], *, expected_arm: str) -> list[dict[str, Any]]:
    if summary.get("protocol") != PROTOCOL:
        raise ValueError(f"summary protocol mismatch: {summary.get('protocol')!r}")
    observed_arm = summary.get("training_configuration", {}).get("arm")
    if observed_arm != expected_arm:
        raise ValueError(f"expected {expected_arm} summary, got {observed_arm!r}")
    records = [record.get("diagnosis") for record in summary.get("epochs", [])]
    if len(records) != 5 or any(not isinstance(record, dict) for record in records):
        raise ValueError("summary must contain five diagnosis records")
    if [int(record.get("epoch", -1)) for record in records] != [1, 2, 3, 4, 5]:
        raise ValueError("diagnosis epoch records must be contiguous from one through five")
    return records


def decision(delta: dict[str, Any]) -> str:
    metrics = delta["patient_macro"]["task_metrics_image_macro"]
    aji_positive = float(metrics["aji"]) > 0.0
    pq_positive = float(metrics["pq"]) > 0.0
    if aji_positive and pq_positive:
        return "second_seed_repeat_final_task_signal"
    if not aji_positive and not pq_positive:
        return "not_reproducible_under_current_route"
    return "mixed_or_unstable_signal"


def rows(c0: list[dict[str, Any]], c1: list[dict[str, Any]], delta: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for arm, record in (("c0", c0[-1]), ("c1_full", c1[-1])):
        for patient in ("7", "8"):
            row = {"comparison": arm, "selection": "fixed_epoch_5", "epoch": 5, "level": f"patient_{patient}"}
            row.update(record["patients"][patient]["task_metrics_image_macro"])
            row.update(record["patients"][patient]["mechanism"])
            result.append(row)
        row = {"comparison": arm, "selection": "fixed_epoch_5", "epoch": 5, "level": "patient_macro"}
        row.update(record["patient_macro"]["task_metrics_image_macro"])
        row.update(record["patient_macro"]["mechanism"])
        result.append(row)
    for patient in ("7", "8"):
        row = {"comparison": "c1_full_minus_c0", "selection": "fixed_epoch_5_delta", "epoch": 5, "level": f"patient_{patient}"}
        row.update(delta["patients"][patient]["task_metrics_image_macro"])
        row.update(delta["patients"][patient]["mechanism"])
        result.append(row)
    row = {"comparison": "c1_full_minus_c0", "selection": "fixed_epoch_5_delta", "epoch": 5, "level": "patient_macro"}
    row.update(delta["patient_macro"]["task_metrics_image_macro"])
    row.update(delta["patient_macro"]["mechanism"])
    result.append(row)
    return result


def write_csv(path: Path, values: list[dict[str, Any]]) -> None:
    fields = ["comparison", "selection", "epoch", "level", *TASK_METRICS, *MECHANISM_METRICS]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(values)


def markdown(c0_selection: dict[str, Any], c1_selection: dict[str, Any], delta: dict[str, Any], outcome: str) -> str:
    c0_task = c0_selection["record"]["patient_macro"]["task_metrics_image_macro"]
    c1_task = c1_selection["record"]["patient_macro"]["task_metrics_image_macro"]
    fixed = delta["patient_macro"]
    fixed_task = fixed["task_metrics_image_macro"]
    fixed_mechanism = fixed["mechanism"]
    return "\n".join(
        [
            "# TNBC C0/C1 second-seed summary",
            "",
            "Primary comparison: paired C1-full minus C0 at fixed epoch 5. PQ-best is recorded only as the frozen development checkpoint-selection result.",
            "",
            f"- Fixed epoch-5 decision: `{outcome}`",
            f"- Fixed epoch-5 patient-macro delta: AJI `{fixed_task['aji']:+.6f}`, PQ `{fixed_task['pq']:+.6f}`, best CCR@0.5 `{fixed_mechanism['best_candidate_ccr_at_0_5']:+.6f}`, selected CCR@0.5 `{fixed_mechanism['selected_candidate_ccr_at_0_5']:+.6f}`, regret `{fixed_mechanism['selection_regret']:+.6f}`.",
            "",
            "| arm | PQ-best epoch | macro AJI | macro PQ |",
            "|---|---:|---:|---:|",
            f"| C0 | {c0_selection['selected_epoch']} | {c0_task['aji']:.6f} | {c0_task['pq']:.6f} |",
            f"| C1-full | {c1_selection['selected_epoch']} | {c1_task['aji']:.6f} | {c1_task['pq']:.6f} |",
            "",
            "Candidate metrics are descriptive mechanism context only; they are not used as an advancement gate and do not establish stable candidate-generation improvement.",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c0-summary", required=True)
    parser.add_argument("--c1-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output_dir}")
    c0 = diagnosis_records(read_json(Path(args.c0_summary)), expected_arm="c0")
    c1 = diagnosis_records(read_json(Path(args.c1_summary)), expected_arm="c1")
    fixed = metric_deltas(c0[-1], c1[-1])
    c0_selection = choose_pq_best(c0)
    c1_selection = choose_pq_best(c1)
    payload = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "primary_comparison": "fixed_epoch_5_C1_full_minus_C0",
        "development_model_selection": {
            "metric": "equal_patient_macro_PQ",
            "tie_break": "earlier_epoch_on_exact_equal_PQ",
            "pq_best_is_secondary": True,
        },
        "fixed_epoch_5": {"c0": c0[-1], "c1_full": c1[-1], "c1_full_minus_c0": fixed},
        "pq_best_selection": {"c0": c0_selection, "c1_full": c1_selection},
        "pq_best_vs_c0": selected_vs_reference_report(c0_selection, c1_selection),
        "owner_interpretation": decision(fixed),
        "interpretation_boundary": {
            "candidate_metrics": "reported as mechanism context only; not a method advancement gate",
            "not_allowed": "claim that coverage loss has stably improved candidate generation",
            "sealed_data": "TNBC patients 9--11 are not accessed",
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output_dir / "second_seed_summary.json", payload)
    write_csv(output_dir / "second_seed_summary.csv", rows(c0, c1, fixed))
    (output_dir / "second_seed_summary.md").write_text(
        markdown(c0_selection, c1_selection, fixed, payload["owner_interpretation"]),
        encoding="utf-8",
    )
    print(json.dumps({"status": "complete", "output_dir": str(output_dir), "decision": payload["owner_interpretation"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
