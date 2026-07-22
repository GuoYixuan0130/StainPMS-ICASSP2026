"""Create machine-readable C0/C1 TNBC five-epoch screening tables.

Inputs are immutable no-gradient Phase 1 diagnosis directories.  The tool
never opens training data, checkpoints, or test identities; it aggregates
already-written p7/p8 outputs and applies the owner-frozen epoch-5 rule.
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

from stainpms.phase2a_tnbc_screen import (
    MECHANISM_METRICS,
    TASK_METRICS,
    assess_epoch5,
    build_epoch_record,
    metric_deltas,
    read_json,
)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def read_gt_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_epoch_path(value: str) -> tuple[int, Path]:
    raw_epoch, sep, raw_path = value.partition("=")
    if not sep:
        raise argparse.ArgumentTypeError("epoch directory must have EPOCH=PATH form")
    try:
        epoch = int(raw_epoch)
    except ValueError as error:
        raise argparse.ArgumentTypeError("epoch must be an integer") from error
    if epoch < 1 or not raw_path:
        raise argparse.ArgumentTypeError("epoch must be >=1 and path must be non-empty")
    return epoch, Path(raw_path).resolve()


def diagnosis_record(arm: str, epoch: int, directory: Path) -> dict[str, Any]:
    required = ("summary.json", "images.json", "gt_instances.csv")
    missing = [name for name in required if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(f"diagnosis directory missing {missing}: {directory}")
    summary = read_json(directory / "summary.json")
    # `summary.json` is an object, whereas the Phase-1 diagnosis writes
    # `images.json` as an ordered list of per-image records.
    images = json.loads((directory / "images.json").read_text(encoding="utf-8"))
    if not isinstance(images, list):
        raise ValueError(f"images.json must be a list: {directory}")
    record = build_epoch_record(
        arm=arm,
        epoch=epoch,
        summary=summary,
        images=images,
        gt_rows=read_gt_rows(directory / "gt_instances.csv"),
        source_dir=directory,
    )
    record["source_provenance"] = {
        "manifest": summary.get("manifest"),
        "checkpoint": summary.get("checkpoint"),
        "metric_spec": summary.get("metric_spec"),
        "frozen_inference": summary.get("frozen_inference"),
    }
    return record


def flatten_rows(records: list[dict[str, Any]], deltas: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        for patient in ("7", "8"):
            row = {"arm": record["arm"], "epoch": record["epoch"], "level": f"patient_{patient}", "patient": patient}
            row.update(record["patients"][patient]["task_metrics_image_macro"])
            row.update(record["patients"][patient]["mechanism"])
            rows.append(row)
        row = {"arm": record["arm"], "epoch": record["epoch"], "level": "patient_macro", "patient": "7_8_equal"}
        row.update(record["patient_macro"]["task_metrics_image_macro"])
        row.update(record["patient_macro"]["mechanism"])
        rows.append(row)
    for epoch, delta in sorted(deltas.items()):
        for patient in ("7", "8"):
            row = {"arm": "C1_minus_C0", "epoch": epoch, "level": f"patient_{patient}", "patient": patient}
            row.update(delta["patients"][patient]["task_metrics_image_macro"])
            row.update(delta["patients"][patient]["mechanism"])
            rows.append(row)
        row = {"arm": "C1_minus_C0", "epoch": epoch, "level": "patient_macro", "patient": "7_8_equal"}
        row.update(delta["patient_macro"]["task_metrics_image_macro"])
        row.update(delta["patient_macro"]["mechanism"])
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["arm", "epoch", "level", "patient", *TASK_METRICS, *MECHANISM_METRICS]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def markdown(records: list[dict[str, Any]], deltas: dict[int, dict[str, Any]], decision: dict[str, Any]) -> str:
    lines = [
        "# TNBC C0/C1 fixed five-epoch screen",
        "",
        "All task metrics are strict complete-image metrics. `dice1` and `dice2` are both retained; legacy reporting's `Dice` corresponds to `dice1`.",
        "Candidate metrics use equal p7/p8 patient macro after within-patient aggregation. The primary comparison is fixed epoch 5 only.",
        "",
        "| epoch | arm | AJI | PQ | best CCR@0.5 | selected CCR@0.5 | regret |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for record in sorted(records, key=lambda item: (item["epoch"], item["arm"])):
        task = record["patient_macro"]["task_metrics_image_macro"]
        mechanism = record["patient_macro"]["mechanism"]
        lines.append(
            f"| {record['epoch']} | {record['arm']} | {task['aji']:.6f} | {task['pq']:.6f} | "
            f"{mechanism['best_candidate_ccr_at_0_5']:.6f} | {mechanism['selected_candidate_ccr_at_0_5']:.6f} | {mechanism['selection_regret']:.6f} |"
        )
        if record["epoch"] in deltas and record["arm"] == "c1":
            delta = deltas[record["epoch"]]["patient_macro"]
            lines.append(
                f"| {record['epoch']} | C1−C0 | {delta['task_metrics_image_macro']['aji']:+.6f} | "
                f"{delta['task_metrics_image_macro']['pq']:+.6f} | {delta['mechanism']['best_candidate_ccr_at_0_5']:+.6f} | "
                f"{delta['mechanism']['selected_candidate_ccr_at_0_5']:+.6f} | {delta['mechanism']['selection_regret']:+.6f} |"
            )
    lines.extend(["", "## Frozen epoch-5 decision", "", f"- Decision: `{decision['decision']}`"])
    for name, passed in decision["checks"].items():
        lines.append(f"- `{name}`: `{passed}`")
    lines.extend(["", decision["interpretation_boundary"], ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epoch0-dir", required=True)
    parser.add_argument("--c0-epoch-dir", action="append", required=True, type=parse_epoch_path)
    parser.add_argument("--c1-epoch-dir", action="append", required=True, type=parse_epoch_path)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty summary directory: {output_dir}")
    c0_paths = dict(args.c0_epoch_dir)
    c1_paths = dict(args.c1_epoch_dir)
    if sorted(c0_paths) != [1, 2, 3, 4, 5] or sorted(c1_paths) != [1, 2, 3, 4, 5]:
        raise ValueError("exactly one C0 and C1 diagnosis directory is required for each epoch 1..5")
    epoch0 = diagnosis_record("shared_epoch0", 0, Path(args.epoch0_dir).resolve())
    c0 = [diagnosis_record("c0", epoch, directory) for epoch, directory in sorted(c0_paths.items())]
    c1 = [diagnosis_record("c1", epoch, directory) for epoch, directory in sorted(c1_paths.items())]
    deltas = {epoch: metric_deltas(c0[epoch - 1], c1[epoch - 1]) for epoch in range(1, 6)}
    decision = assess_epoch5(c0[-1], c1[-1])
    payload = {
        "schema_version": 1,
        "protocol": "tnbc_c0_c1_5epoch_exploratory_v1",
        "primary_epoch": 5,
        "shared_epoch0": epoch0,
        "c0": c0,
        "c1": c1,
        "c1_minus_c0": deltas,
        "screening_decision": decision,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output_dir / "epoch_metrics.json", payload)
    write_json_atomic(output_dir / "screening_decision.json", decision)
    write_csv(output_dir / "epoch_metrics.csv", flatten_rows(c0 + c1, deltas))
    (output_dir / "screening_summary.md").write_text(markdown(c0 + c1, deltas, decision), encoding="utf-8")
    print(json.dumps({"status": "complete", "output_dir": str(output_dir), "decision": decision["decision"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
