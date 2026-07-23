"""Summarize fixed-epoch C2-AR versus the paired C0/C1 development results.

This tool is intentionally post-hoc reporting only: it reads six completed
zero-training oracle summaries (two seeds x C0/C1/C2-AR), performs no model
loading, and applies the five owner-frozen C2 promotion conditions.
"""

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
ARMS = ("c0", "c1", "c2_ar")
SCOPES = (("p7", "7"), ("p8", "8"), ("patient_macro", "patient_macro"))
STAGES = (
    "native_final",
    "final_pool_oracle",
    "native_selected_pool_oracle",
    "all_candidate_pool_oracle",
)
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
    "dice1",
    "dice2",
    "aji",
)
MECHANISM_FIELDS = (
    "all_candidate_coverage_recall_at_0_5",
    "native_selected_coverage_recall_at_0_5",
    "selection_regret_mean",
    "all_candidate_best_iou_mean",
    "all_candidate_best_iou_median",
    "selected_candidate_iou_mean",
    "selected_candidate_iou_median",
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
        raise argparse.ArgumentTypeError("C2 report requires seed 2027/1337 and arm c0/c1/c2_ar")
    return seed, arm, Path(raw_path).resolve()


def validate_summary(summary: dict[str, Any], *, seed: int, arm: str, path: Path) -> None:
    if summary.get("status") != "complete" or int(summary.get("seed", -1)) != seed or summary.get("arm") != arm:
        raise ValueError(f"invalid completed oracle summary: {path}")
    reproduction = summary.get("reference_reproduction", {})
    if arm in {"c0", "c1"} and reproduction.get("status") != "pass":
        raise ValueError(f"C0/C1 reproduction gate did not pass: {path}")
    if arm == "c2_ar" and reproduction.get("status") not in {"pass", "not_applicable_new_c2_checkpoint"}:
        raise ValueError(f"invalid C2 reference state: {path}")
    patients = summary.get("summary", {}).get("patients", {})
    if set(patients) != {"7", "8"}:
        raise ValueError(f"summary lacks exactly p7/p8: {path}")


def scope_source(summary: dict[str, Any], key: str) -> dict[str, Any]:
    return summary["summary"]["patient_macro"] if key == "patient_macro" else summary["summary"]["patients"][key]


def stage_values(scope: dict[str, Any], stage: str) -> dict[str, float | int | None]:
    source = scope["stages"][stage]
    task = source.get("task_metrics_image_macro", {})
    values = {field: source.get(field) for field in STAGE_FIELDS if field not in {"dice1", "dice2", "aji"}}
    values.update({field: task.get(field) for field in ("dice1", "dice2", "aji")})
    return values


def difference(left: dict[str, Any], right: dict[str, Any], fields: tuple[str, ...]) -> dict[str, float | None]:
    return {
        field: None if left.get(field) is None or right.get(field) is None else float(left[field]) - float(right[field])
        for field in fields
    }


def stage_gaps(arm: dict[str, Any]) -> dict[str, dict[str, float | None]]:
    stages = arm["stages"]
    return {
        "all_candidate_oracle_minus_native_selected_oracle": difference(
            stages["all_candidate_pool_oracle"], stages["native_selected_pool_oracle"], STAGE_FIELDS
        ),
        "native_selected_oracle_minus_final_pool_oracle": difference(
            stages["native_selected_pool_oracle"], stages["final_pool_oracle"], STAGE_FIELDS
        ),
        "final_pool_oracle_minus_native_final": difference(
            stages["final_pool_oracle"], stages["native_final"], STAGE_FIELDS
        ),
    }


def normalized_arm(scope: dict[str, Any]) -> dict[str, Any]:
    return {
        "stages": {stage: stage_values(scope, stage) for stage in STAGES},
        "mechanism": {field: scope["candidate_quality"].get(field) for field in MECHANISM_FIELDS},
        "errors": dict(scope["errors"]),
    }


def arm_difference(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_gaps, right_gaps = stage_gaps(left), stage_gaps(right)
    error_fields = tuple(sorted({*left["errors"], *right["errors"]}))
    return {
        "stages": {stage: difference(left["stages"][stage], right["stages"][stage], STAGE_FIELDS) for stage in STAGES},
        "mechanism": difference(left["mechanism"], right["mechanism"], MECHANISM_FIELDS),
        "errors": difference(left["errors"], right["errors"], error_fields),
        "stage_gaps": {name: difference(left_gaps[name], right_gaps[name], STAGE_FIELDS) for name in left_gaps},
    }


def aggregate_rows(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> dict[str, Any]:
    return {field: summarize_numeric(row.get(field) for row in rows) for field in fields}


def aggregate_comparison(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "stages": {
            stage: aggregate_rows([record["stages"][stage] for record in records], STAGE_FIELDS)
            for stage in STAGES
        },
        "mechanism": aggregate_rows([record["mechanism"] for record in records], MECHANISM_FIELDS),
        "errors": aggregate_rows(
            [record["errors"] for record in records],
            tuple(sorted({key for record in records for key in record["errors"]})),
        ),
        "stage_gaps": {
            name: aggregate_rows([record["stage_gaps"][name] for record in records], STAGE_FIELDS)
            for name in records[0]["stage_gaps"]
        },
    }


def format_stat(value: float | None, *, signed: bool = False) -> str:
    if value is None:
        return "NA"
    return f"{float(value):+.6f}" if signed else f"{float(value):.6f}"


def promotion_gate(per_seed: list[dict[str, Any]], aggregate: dict[str, Any]) -> dict[str, Any]:
    macro = [record["scopes"]["patient_macro"] for record in per_seed]
    pq_values = [record["c2_ar_minus_c0"]["stages"]["native_final"]["pq"] for record in macro]
    aji = aggregate["c2_ar_minus_c0"]["stages"]["native_final"]["aji"]["mean"]
    selected_oracle = aggregate["c2_ar_minus_c0"]["stages"]["native_selected_pool_oracle"]["pq"]["mean"]
    assembly_gap = aggregate["c2_ar_minus_c1"]["stage_gaps"]["native_selected_oracle_minus_final_pool_oracle"]["pq"]["mean"]
    fp_penalty = aggregate["c2_ar_minus_c1"]["stage_gaps"]["final_pool_oracle_minus_native_final"]["pq"]["mean"]
    conditions = {
        "both_seed_native_pq_positive_vs_c0": all(value is not None and float(value) > 0.0 for value in pq_values),
        "mean_native_aji_positive_vs_c0": aji is not None and float(aji) > 0.0,
        "mean_assembly_gap_smaller_vs_c1": assembly_gap is not None and float(assembly_gap) < 0.0,
        "mean_final_fp_penalty_smaller_vs_c1": fp_penalty is not None and float(fp_penalty) < 0.0,
        "mean_selected_pool_oracle_pq_positive_vs_c0": selected_oracle is not None and float(selected_oracle) > 0.0,
    }
    return {
        "status": "pass" if all(conditions.values()) else "do_not_promote",
        "conditions": conditions,
        "values": {
            "native_pq_delta_by_seed_vs_c0": pq_values,
            "mean_native_aji_delta_vs_c0": aji,
            "mean_selected_pool_oracle_pq_delta_vs_c0": selected_oracle,
            "mean_assembly_gap_delta_vs_c1": assembly_gap,
            "mean_final_fp_penalty_delta_vs_c1": fp_penalty,
        },
    }


def markdown(payload: dict[str, Any]) -> str:
    aggregate = payload["two_seed_patient_macro"]
    lines = [
        "# C2-AR two-seed TNBC development report",
        "",
        "- Scope: fixed epoch-5 C2-AR, paired seed 2027 and 1337, TNBC p7/p8 development only.",
        "- C2-AR uses no inference-time change: native token-0 selection, NMS, and assembly remain frozen.",
        "- C0/C1 are read-only paired references; C2 is compared against both.",
        "",
        "## Native final paired deltas",
        "",
        "| comparison | metric | mean | sample std | positive seeds |",
        "|---|---|---:|---:|---:|",
    ]
    for comparison in ("c2_ar_minus_c0", "c2_ar_minus_c1"):
        for metric in ("dice1", "aji", "dq", "sq", "pq"):
            values = aggregate[comparison]["stages"]["native_final"][metric]
            lines.append(
                f"| {comparison} | {metric} | {format_stat(values['mean'], signed=True)} | "
                f"{format_stat(values['std_sample'])} | {values['positive_count']}/{values['count']} |"
            )
    lines.extend(["", "## C2-AR mechanism deltas", "", "| comparison | quantity | mean | sample std |", "|---|---|---:|---:|"])
    for comparison in ("c2_ar_minus_c0", "c2_ar_minus_c1"):
        for gap in ("native_selected_oracle_minus_final_pool_oracle", "final_pool_oracle_minus_native_final"):
            values = aggregate[comparison]["stage_gaps"][gap]["pq"]
            lines.append(f"| {comparison} | {gap} PQ | {format_stat(values['mean'], signed=True)} | {format_stat(values['std_sample'])} |")
        for metric in ("all_candidate_coverage_recall_at_0_5", "native_selected_coverage_recall_at_0_5", "selection_regret_mean"):
            values = aggregate[comparison]["mechanism"][metric]
            lines.append(f"| {comparison} | {metric} | {format_stat(values['mean'], signed=True)} | {format_stat(values['std_sample'])} |")
    gate = payload["promotion_gate"]
    lines.extend(["", "## Pre-registered promotion gate", ""])
    for name, value in gate["conditions"].items():
        lines.append(f"- {name}: {'pass' if value else 'fail'}")
    lines.extend(["", f"- Decision: `{gate['status']}`.", ""])
    return "\n".join(lines)


def write_csv(path: Path, payload: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    for record in payload["per_seed"]:
        for scope_name, scope in record["scopes"].items():
            for arm in ARMS:
                for stage, values in scope[arm]["stages"].items():
                    rows.append({"seed": record["seed"], "scope": scope_name, "comparison": arm, "section": "stage", "name": stage, **values})
                rows.append({"seed": record["seed"], "scope": scope_name, "comparison": arm, "section": "mechanism", "name": "candidate_quality", **scope[arm]["mechanism"]})
                rows.append({"seed": record["seed"], "scope": scope_name, "comparison": arm, "section": "errors", "name": "counts", **scope[arm]["errors"]})
                for name, values in stage_gaps(scope[arm]).items():
                    rows.append({"seed": record["seed"], "scope": scope_name, "comparison": arm, "section": "within_arm_gap", "name": name, **values})
            for comparison in ("c2_ar_minus_c0", "c2_ar_minus_c1"):
                values = scope[comparison]
                for stage, stage_values_ in values["stages"].items():
                    rows.append({"seed": record["seed"], "scope": scope_name, "comparison": comparison, "section": "stage", "name": stage, **stage_values_})
                rows.append({"seed": record["seed"], "scope": scope_name, "comparison": comparison, "section": "mechanism", "name": "candidate_quality", **values["mechanism"]})
                rows.append({"seed": record["seed"], "scope": scope_name, "comparison": comparison, "section": "errors", "name": "counts", **values["errors"]})
                for name, gap_values in values["stage_gaps"].items():
                    rows.append({"seed": record["seed"], "scope": scope_name, "comparison": comparison, "section": "stage_gap", "name": name, **gap_values})
    fields = sorted({field for row in rows for field in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, type=parse_assignment)
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
        raise ValueError("six epoch-5 summaries (2027/1337 x c0/c1/c2_ar) are required")
    summaries = {}
    for key, path in inputs.items():
        summary = read_json(path)
        validate_summary(summary, seed=key[0], arm=key[1], path=path)
        summaries[key] = summary

    per_seed: list[dict[str, Any]] = []
    for seed in SEEDS:
        scopes: dict[str, Any] = {}
        for label, source_key in SCOPES:
            arms = {arm: normalized_arm(scope_source(summaries[(seed, arm)], source_key)) for arm in ARMS}
            scopes[label] = {
                **{arm: {**value, "stage_gaps": stage_gaps(value)} for arm, value in arms.items()},
                "c2_ar_minus_c0": arm_difference(arms["c2_ar"], arms["c0"]),
                "c2_ar_minus_c1": arm_difference(arms["c2_ar"], arms["c1"]),
            }
        per_seed.append({"seed": seed, "scopes": scopes, "input_summaries": {arm: str(inputs[(seed, arm)]) for arm in ARMS}})

    macro = [record["scopes"]["patient_macro"] for record in per_seed]
    aggregate = {
        comparison: aggregate_comparison([record[comparison] for record in macro])
        for comparison in ("c2_ar_minus_c0", "c2_ar_minus_c1")
    }
    payload = {
        "schema_version": 1,
        "protocol": "tnbc_c2_ar_two_seed_v1",
        "status": "complete",
        "scope": "TNBC p7/p8 only; fixed epoch 5; paired seeds 2027/1337; no sealed-test access",
        "per_seed": per_seed,
        "two_seed_patient_macro": aggregate,
    }
    payload["promotion_gate"] = promotion_gate(per_seed, aggregate)
    output_dir = Path(args.output_dir).resolve()
    write_json_atomic(output_dir / "c2_ar_results.json", payload)
    write_csv(output_dir / "c2_ar_results.csv", payload)
    (output_dir / "c2_ar_results.md").write_text(markdown(payload), encoding="utf-8")
    print(json.dumps({"status": "complete", "output_dir": str(output_dir), "promotion": payload["promotion_gate"]["status"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
