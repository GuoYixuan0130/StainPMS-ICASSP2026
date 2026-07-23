"""Combine six completed TNBC zero-training oracle runs without inference."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stainpms.zero_training_oracle import summarize_numeric


SEEDS = (2027, 1337)
ARMS = ("c0", "c1")
EXCLUDED_SEED_3407 = {
    "seed": 3407,
    "reason": "C0 fixed epoch-5 complete checkpoint was deleted during prior retention compaction and cannot be recovered; no substitute checkpoint is permitted.",
}
SCOPES = (("p7", "7"), ("p8", "8"), ("patient_macro", "patient_macro"))
STAGES = ("native_final", "final_pool_oracle", "native_selected_pool_oracle", "all_candidate_pool_oracle")
STAGE_FIELDS = (
    "tp",
    "fp",
    "fn",
    "dq",
    "sq",
    "pq",
    "coverage_recall_at_0_5",
    "raw_prediction_group_count",
    "raw_prediction_mask_count",
)
TASK_FIELDS = ("dice1", "dice2", "aji", "dq", "sq", "pq")
MECHANISM_FIELDS = (
    "all_candidate_coverage_recall_at_0_5",
    "native_selected_coverage_recall_at_0_5",
    "selection_regret_mean",
    "all_candidate_best_iou_mean",
    "all_candidate_best_iou_median",
    "all_candidate_best_iou_q10",
    "all_candidate_best_iou_q25",
    "all_candidate_best_iou_q75",
    "all_candidate_best_iou_q90",
    "selected_candidate_iou_mean",
    "selected_candidate_iou_median",
    "selected_candidate_iou_q10",
    "selected_candidate_iou_q25",
    "selected_candidate_iou_q75",
    "selected_candidate_iou_q90",
)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def parse_assignment(value: str) -> tuple[int, str, Path]:
    try:
        left, raw_path = value.split("=", 1)
        raw_seed, arm = left.split(":", 1)
        seed = int(raw_seed)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--input must be SEED:ARM=/absolute/path/to/summary.json") from exc
    if seed not in SEEDS or arm not in ARMS:
        raise argparse.ArgumentTypeError("only paired epoch-5 diagnostic seeds 2027/1337 and arms c0/c1 are permitted")
    return seed, arm, Path(raw_path).resolve()


def validate_summary(summary: dict[str, Any], *, seed: int, arm: str, path: Path) -> None:
    if summary.get("status") != "complete" or int(summary.get("seed", -1)) != seed or summary.get("arm") != arm:
        raise ValueError(f"invalid completed oracle summary: {path}")
    reproduction = summary.get("reference_reproduction", {})
    if reproduction.get("status") != "pass":
        raise ValueError(f"native reproduction gate did not pass: {path}")
    patients = summary.get("summary", {}).get("patients", {})
    if set(patients) != {"7", "8"}:
        raise ValueError(f"summary lacks exactly p7/p8: {path}")


def scope_record(summary: dict[str, Any], scope_key: str) -> dict[str, Any]:
    if scope_key == "patient_macro":
        return summary["summary"]["patient_macro"]
    return summary["summary"]["patients"][scope_key]


def stage_values(scope: dict[str, Any], stage: str) -> dict[str, float | int | None]:
    record = scope["stages"][stage]
    output = {key: record.get(key) for key in STAGE_FIELDS}
    task = record.get("task_metrics_image_macro", {})
    for key in ("dice1", "dice2", "aji"):
        output[key] = task.get(key)
    return output


def difference(left: dict[str, Any], right: dict[str, Any], fields: tuple[str, ...]) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for field in fields:
        lvalue = left.get(field)
        rvalue = right.get(field)
        result[field] = None if lvalue is None or rvalue is None else float(lvalue) - float(rvalue)
    return result


def aggregate_paired(records: list[dict[str, Any]], fields: tuple[str, ...]) -> dict[str, Any]:
    return {field: summarize_numeric(record.get(field) for record in records) for field in fields}


def stage_gaps(scope: dict[str, Any]) -> dict[str, dict[str, float | None]]:
    stages = scope["stages"]
    return {
        "all_candidate_oracle_minus_native_selected_oracle": difference(stages["all_candidate_pool_oracle"], stages["native_selected_pool_oracle"], STAGE_FIELDS),
        "native_selected_oracle_minus_final_pool_oracle": difference(stages["native_selected_pool_oracle"], stages["final_pool_oracle"], STAGE_FIELDS),
        "final_pool_oracle_minus_native_final": difference(stages["final_pool_oracle"], stages["native_final"], STAGE_FIELDS),
    }


def c1_minus_c0(c1: dict[str, Any], c0: dict[str, Any]) -> dict[str, Any]:
    stages = {stage: difference(c1["stages"][stage], c0["stages"][stage], (*STAGE_FIELDS, "dice1", "dice2", "aji")) for stage in STAGES}
    mechanism = difference(c1["candidate_quality"], c0["candidate_quality"], MECHANISM_FIELDS)
    error_fields = sorted({*c0["errors"], *c1["errors"]})
    errors = difference(c1["errors"], c0["errors"], tuple(error_fields))
    gaps = {name: difference(c1_gap, c0_gap, STAGE_FIELDS) for name, (c1_gap, c0_gap) in {
        name: (stage_gaps(c1)[name], stage_gaps(c0)[name]) for name in stage_gaps(c0)
    }.items()}
    return {"stages": stages, "mechanism": mechanism, "errors": errors, "stage_gaps": gaps}


def mean_stage_gap_by_seed(per_seed: list[dict[str, Any]], gap_name: str) -> float:
    values = [record["c1_minus_c0"]["stage_gaps"][gap_name]["pq"] for record in per_seed]
    numeric = [float(value) for value in values if value is not None]
    return float(np.mean(numeric)) if numeric else float("nan")


def format_stat(value: float | None, *, signed: bool = False) -> str:
    if value is None:
        return "NA"
    return f"{float(value):+.6f}" if signed else f"{float(value):.6f}"


def markdown(payload: dict[str, Any]) -> str:
    aggregate = payload["two_seed_patient_macro"]
    lines = [
        "# TNBC two-seed zero-training four-level oracle diagnosis",
        "",
        "- Scope: fixed epoch-5 C0/C1, paired seeds 2027/1337, TNBC development p7/p8 only.",
        "- Excluded seed 3407: its C0 fixed epoch-5 complete checkpoint was deleted and cannot be recovered; no surrogate is used.",
        "- Native-final reproduction gate: passed for all four runs before oracle attribution.",
        "- Oracle pools remove unmatched predictions and are ideal upper bounds, not deployable performance.",
        "",
        "## Paired C1-full minus C0 at native final",
        "",
        "| metric | mean | sample std | positive seeds |",
        "|---|---:|---:|---:|",
    ]
    for field in ("aji", "pq", "dq", "sq", "dice1"):
        values = aggregate["c1_minus_c0"]["stages"]["native_final"][field]
        lines.append(f"| {field} | {format_stat(values['mean'], signed=True)} | {format_stat(values['std_sample'])} | {values['positive_count']}/{values['count']} |")
    lines.extend(["", "## Oracle PQ gaps within each arm", "", "Values are pooled-stage upper-bound differences, patient-macro then two-seed mean.", "", "| arm | gap | PQ mean | sample std |", "|---|---|---:|---:|"])
    for arm in ARMS:
        for name, values in aggregate["within_arm_gaps"][arm].items():
            pq = values["pq"]
            lines.append(f"| {arm} | {name} | {format_stat(pq['mean'], signed=True)} | {format_stat(pq['std_sample'])} |")
    lines.extend(["", "## Paired C1-full minus C0 changes in stage gaps", "", "| gap | PQ mean | sample std | positive seeds |", "|---|---:|---:|---:|"])
    for name, values in aggregate["c1_minus_c0"]["stage_gaps"].items():
        pq = values["pq"]
        lines.append(f"| {name} | {format_stat(pq['mean'], signed=True)} | {format_stat(pq['std_sample'])} | {pq['positive_count']}/{pq['count']} |")
    lines.extend(["", "## Continuous candidate IoU", "", "| metric, C1-full minus C0 | mean | sample std | positive seeds |", "|---|---:|---:|---:|"])
    for field in ("all_candidate_best_iou_mean", "all_candidate_best_iou_median", "selected_candidate_iou_mean", "selected_candidate_iou_median", "selection_regret_mean"):
        values = aggregate["c1_minus_c0"]["mechanism"][field]
        lines.append(f"| {field} | {format_stat(values['mean'], signed=True)} | {format_stat(values['std_sample'])} | {values['positive_count']}/{values['count']} |")
    lines.extend(["", "## Evidence boundary", "", "This report locates frozen-pool losses only. It does not select a C2 module or authorise training. Candidate RLEs, quality scores, prompt-group IDs, selected pre-assembly masks, final masks, and per-image oracle matches are retained under each input run directory.", ""])
    return "\n".join(lines)


def write_csv(path: Path, payload: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    for record in payload["per_seed"]:
        seed = record["seed"]
        for scope_name, scope in record["scopes"].items():
            for arm in ARMS:
                for stage, values in scope[arm]["stages"].items():
                    rows.append({"seed": seed, "scope": scope_name, "comparison": arm, "section": "stage", "name": stage, **values})
                rows.append({"seed": seed, "scope": scope_name, "comparison": arm, "section": "mechanism", "name": "candidate_quality", **scope[arm]["candidate_quality"]})
                rows.append({"seed": seed, "scope": scope_name, "comparison": arm, "section": "errors", "name": "error_counts", **scope[arm]["errors"]})
                for name, values in scope[arm]["stage_gaps"].items():
                    rows.append({"seed": seed, "scope": scope_name, "comparison": arm, "section": "within_arm_gap", "name": name, **values})
            for stage, values in scope["c1_minus_c0"]["stages"].items():
                rows.append({"seed": seed, "scope": scope_name, "comparison": "c1_minus_c0", "section": "stage", "name": stage, **values})
            rows.append({"seed": seed, "scope": scope_name, "comparison": "c1_minus_c0", "section": "mechanism", "name": "candidate_quality", **scope["c1_minus_c0"]["mechanism"]})
            rows.append({"seed": seed, "scope": scope_name, "comparison": "c1_minus_c0", "section": "errors", "name": "error_counts", **scope["c1_minus_c0"]["errors"]})
            for name, values in scope["c1_minus_c0"]["stage_gaps"].items():
                rows.append({"seed": seed, "scope": scope_name, "comparison": "c1_minus_c0", "section": "stage_gap", "name": name, **values})
    fields = sorted({field for row in rows for field in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, type=parse_assignment, help="SEED:ARM=/path/to/summary.json; provide all four paired runs")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    inputs: dict[tuple[int, str], Path] = {}
    for seed, arm, path in args.input:
        if (seed, arm) in inputs:
            raise ValueError(f"duplicate input for seed={seed}, arm={arm}")
        inputs[(seed, arm)] = path
    expected = {(seed, arm) for seed in SEEDS for arm in ARMS}
    if set(inputs) != expected:
        raise ValueError("all four paired epoch-5 seed/arm summaries are required")
    summaries: dict[tuple[int, str], dict[str, Any]] = {}
    for key, path in inputs.items():
        summary = read_json(path)
        validate_summary(summary, seed=key[0], arm=key[1], path=path)
        summaries[key] = summary

    per_seed: list[dict[str, Any]] = []
    for seed in SEEDS:
        scopes: dict[str, Any] = {}
        for label, key in SCOPES:
            c0_source = scope_record(summaries[(seed, "c0")], key)
            c1_source = scope_record(summaries[(seed, "c1")], key)
            c0 = {
                "stages": {stage: stage_values(c0_source, stage) for stage in STAGES},
                "candidate_quality": c0_source["candidate_quality"],
                "errors": c0_source["errors"],
            }
            c1 = {
                "stages": {stage: stage_values(c1_source, stage) for stage in STAGES},
                "candidate_quality": c1_source["candidate_quality"],
                "errors": c1_source["errors"],
            }
            scopes[label] = {
                "c0": {**c0, "stage_gaps": stage_gaps(c0)},
                "c1": {**c1, "stage_gaps": stage_gaps(c1)},
                "c1_minus_c0": c1_minus_c0(c1, c0),
            }
        per_seed.append({"seed": seed, "scopes": scopes, "input_summaries": {arm: str(inputs[(seed, arm)]) for arm in ARMS}})

    macro_records = [record["scopes"]["patient_macro"] for record in per_seed]
    within_arm_gaps = {
        arm: {
            name: aggregate_paired([record[arm]["stage_gaps"][name] for record in macro_records], STAGE_FIELDS)
            for name in macro_records[0][arm]["stage_gaps"]
        }
        for arm in ARMS
    }
    aggregate_c1_c0 = {
        "stages": {stage: aggregate_paired([record["c1_minus_c0"]["stages"][stage] for record in macro_records], (*STAGE_FIELDS, "dice1", "dice2", "aji")) for stage in STAGES},
        "mechanism": aggregate_paired([record["c1_minus_c0"]["mechanism"] for record in macro_records], MECHANISM_FIELDS),
        "errors": aggregate_paired([record["c1_minus_c0"]["errors"] for record in macro_records], tuple(sorted(macro_records[0]["c1_minus_c0"]["errors"]))),
        "stage_gaps": {name: aggregate_paired([record["c1_minus_c0"]["stage_gaps"][name] for record in macro_records], STAGE_FIELDS) for name in macro_records[0]["c1_minus_c0"]["stage_gaps"]},
    }
    payload = {
        "schema_version": 1,
        "protocol": "tnbc_zero_training_oracle_diagnosis_two_seed_v1",
        "status": "complete",
        "scope": "TNBC p7/p8 only; paired seeds 2027/1337 at fixed epoch 5; no training",
        "excluded_seed": EXCLUDED_SEED_3407,
        "per_seed": per_seed,
        "two_seed_patient_macro": {"within_arm_gaps": within_arm_gaps, "c1_minus_c0": aggregate_c1_c0},
    }
    output_dir = Path(args.output_dir).resolve()
    write_json_atomic(output_dir / "zero_training_diagnosis.json", payload)
    write_csv(output_dir / "zero_training_diagnosis.csv", payload)
    (output_dir / "zero_training_diagnosis.md").write_text(markdown(payload), encoding="utf-8")
    print(json.dumps({"status": "complete", "output_dir": str(output_dir)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
