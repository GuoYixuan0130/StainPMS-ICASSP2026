"""Summarize the pre-registered C2-E/C2-U TNBC component attribution.

Reads completed epoch-5, p7/p8-only oracle reports and their read-only
mechanism audits.  It never loads a checkpoint or evaluates data.
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
ARMS = ("c0", "c1", "c2_eu", "c2_e", "c2_u")
SCOPES = (("p7", "7"), ("p8", "8"), ("patient_macro", "patient_macro"))
STAGES = ("native_final", "final_pool_oracle", "native_selected_pool_oracle", "all_candidate_pool_oracle")
TASK_FIELDS = ("dice1", "dice2", "aji", "dq", "sq", "pq")
ERROR_FIELDS = (
    "generation_miss", "selection_miss", "assembly_loss", "native_final_tp",
    "native_final_false_positive_count", "native_final_false_negative_count",
    "duplicate_unmatched_prediction_count", "merge_overlap_fraction_gt_or_pred_gt_0",
    "split_overlap_fraction_gt_or_pred_gt_0",
)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def parse_assignment(value: str) -> tuple[int, str, Path]:
    try:
        left, raw_path = value.split("=", 1)
        raw_seed, arm = left.split(":", 1)
        seed = int(raw_seed)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("assignment must be SEED:ARM=/absolute/path") from exc
    if seed not in SEEDS or arm not in ARMS:
        raise argparse.ArgumentTypeError("requires seeds 2027/1337 and arms c0/c1/c2_eu/c2_e/c2_u")
    return seed, arm, Path(raw_path).resolve()


def oracle_arm(arm: str) -> str:
    return "c2_ar" if arm == "c2_eu" else arm


def validate_oracle(summary: dict[str, Any], seed: int, arm: str, path: Path) -> None:
    expected = oracle_arm(arm)
    if summary.get("status") != "complete" or int(summary.get("seed", -1)) != seed or summary.get("arm") != expected:
        raise ValueError(f"oracle identity mismatch: {path}")
    if expected in {"c0", "c1"} and summary.get("reference_reproduction", {}).get("status") != "pass":
        raise ValueError(f"C0/C1 reproduction failed: {path}")
    if expected not in {"c0", "c1"} and summary.get("reference_reproduction", {}).get("status") not in {"not_applicable_new_c2_checkpoint", "pass"}:
        raise ValueError(f"new C2 state must not claim a failed reproduction check: {path}")


def _scope(summary: dict[str, Any], source: str) -> dict[str, Any]:
    return summary["summary"]["patient_macro"] if source == "patient_macro" else summary["summary"]["patients"][source]


def normalized(scope: dict[str, Any]) -> dict[str, Any]:
    stages: dict[str, dict[str, float | int | None]] = {}
    for stage in STAGES:
        raw = scope["stages"][stage]
        strict = raw.get("task_metrics_image_macro", {})
        stages[stage] = {
            **{field: raw.get(field) for field in ("tp", "fp", "fn", "dq", "sq", "pq", "coverage_recall_at_0_5")},
            **{field: strict.get(field) for field in ("dice1", "dice2", "aji")},
        }
    return {"stages": stages, "errors": {key: scope["errors"].get(key) for key in ERROR_FIELDS}, "candidate_quality": dict(scope["candidate_quality"])}


def numerical_difference(left: dict[str, Any], right: dict[str, Any]) -> dict[str, float | None]:
    keys = sorted({*left, *right})
    out: dict[str, float | None] = {}
    for key in keys:
        a, b = left.get(key), right.get(key)
        out[key] = None if a is None or b is None else float(a) - float(b)
    return out


def gaps(arm: dict[str, Any]) -> dict[str, dict[str, float | None]]:
    return {
        "selected_to_final_assembly_gap": numerical_difference(
            arm["stages"]["native_selected_pool_oracle"], arm["stages"]["final_pool_oracle"]
        ),
        "final_fp_penalty": numerical_difference(
            arm["stages"]["final_pool_oracle"], arm["stages"]["native_final"]
        ),
    }


def compare(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    return {
        "stages": {stage: numerical_difference(left["stages"][stage], right["stages"][stage]) for stage in STAGES},
        "errors": numerical_difference(left["errors"], right["errors"]),
        "candidate_quality": numerical_difference(left["candidate_quality"], right["candidate_quality"]),
        "gaps": {key: numerical_difference(gaps(left)[key], gaps(right)[key]) for key in gaps(left)},
    }


def mechanism_normalized(payload: dict[str, Any], source: str) -> dict[str, Any]:
    raw = payload["patient_macro"]["patients"][source] if source in {"7", "8"} else payload["patient_macro"]["image_macro"]
    if source in {"7", "8"}:
        score = raw["score_calibration_image_macro"]
        excl = raw["exclusivity"]
        intervention = raw["oracle_score_intervention_image_macro"]
    else:
        score = raw["score_calibration"]
        excl = raw["exclusivity"]
        intervention = raw["oracle_score_intervention"]
    return {"score": score, "exclusivity": excl, "oracle_score_intervention": intervention}


def aggregate_diffs(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {key: summarize_numeric(row.get(key) for row in rows) for key in sorted({k for row in rows for k in row})}


def summary_stats(per_seed: list[dict[str, Any]], comparison_key: str) -> dict[str, Any]:
    macro = [record["scopes"]["patient_macro"][comparison_key] for record in per_seed]
    return {
        "native": aggregate_diffs([row["stages"]["native_final"] for row in macro]),
        "selected_oracle": aggregate_diffs([row["stages"]["native_selected_pool_oracle"] for row in macro]),
        "errors": aggregate_diffs([row["errors"] for row in macro]),
        "gaps": {name: aggregate_diffs([row["gaps"][name] for row in macro]) for name in macro[0]["gaps"]},
    }


def component_gate(per_seed: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    rows = [record["scopes"]["patient_macro"] for record in per_seed]
    versus_c0 = [row[f"{arm}_minus_c0"] for row in rows]
    versus_c1 = [row[f"{arm}_minus_c1"] for row in rows]
    native_pq_c0 = [float(row["stages"]["native_final"]["pq"]) for row in versus_c0]
    aji_c0 = [float(row["stages"]["native_final"]["aji"]) for row in versus_c0]
    selected_c0 = [float(row["stages"]["native_selected_pool_oracle"]["pq"]) for row in versus_c0]
    native_pq_c1 = [float(row["stages"]["native_final"]["pq"]) for row in versus_c1]
    if arm == "c2_e":
        direct = [float(row["gaps"]["selected_to_final_assembly_gap"]["pq"]) for row in versus_c1]
        leakage_delta = [
            float(record["mechanisms"]["c2_e"]["exclusivity"].get("hard_foreign_gt_fraction_mean"))
            - float(record["mechanisms"]["c1"]["exclusivity"].get("hard_foreign_gt_fraction_mean"))
            for record in per_seed
        ]
        merge_delta = [float(row["errors"].get("merge_overlap_fraction_gt_or_pred_gt_0", 0.0)) for row in versus_c1]
        hard_overlap_delta = [
            float(record["mechanisms"]["c2_e"]["exclusivity"].get("hard_overlap_positive_pair_fraction"))
            - float(record["mechanisms"]["c1"]["exclusivity"].get("hard_overlap_positive_pair_fraction"))
            for record in per_seed
        ]
        conditions = {
            "both_seed_assembly_gap_smaller_vs_c1": all(value < 0.0 for value in direct),
            "both_seed_foreign_leakage_not_higher_vs_c1": all(value <= 0.0 for value in leakage_delta),
            "both_seed_at_least_one_exclusivity_measure_improved_vs_c1": all(
                leakage < 0.0 or merge < 0.0 or overlap < 0.0
                for leakage, merge, overlap in zip(leakage_delta, merge_delta, hard_overlap_delta, strict=True)
            ),
            "selected_pool_oracle_not_damaged_vs_c1": all(float(row["stages"]["native_selected_pool_oracle"]["pq"]) >= 0.0 for row in versus_c1),
            "native_pq_not_lower_vs_c1": all(value >= 0.0 for value in native_pq_c1),
        }
    else:
        direct = [float(row["gaps"]["final_fp_penalty"]["pq"]) for row in versus_c1]
        auroc_delta = [
            float(record["mechanisms"]["c2_u"]["score"].get("auroc"))
            - float(record["mechanisms"]["c1"]["score"].get("auroc"))
            for record in per_seed
        ]
        conditions = {
            "both_seed_final_fp_penalty_smaller_vs_c1": all(value < 0.0 for value in direct),
            "both_seed_utility_score_auroc_not_lower_vs_c1": all(value >= 0.0 for value in auroc_delta),
            "native_dq_not_lower_vs_c1": all(float(row["stages"]["native_final"]["dq"]) >= 0.0 for row in versus_c1),
            "native_pq_not_lower_vs_c1": all(value >= 0.0 for value in native_pq_c1),
        }
    conditions.update({
        "both_seed_native_pq_positive_vs_c0": all(value > 0.0 for value in native_pq_c0),
        "mean_native_aji_positive_vs_c0": float(np.mean(aji_c0)) > 0.0,
        "mean_selected_pool_oracle_pq_positive_vs_c0": float(np.mean(selected_c0)) > 0.0,
    })
    return {"status": "mechanism_supported" if all(conditions.values()) else "not_supported", "conditions": conditions,
            "evidence": {"native_pq_delta_vs_c0": native_pq_c0, "native_pq_delta_vs_c1": native_pq_c1, "aji_delta_vs_c0": aji_c0, "selected_oracle_pq_delta_vs_c0": selected_c0, "direct_gap_delta_vs_c1": direct, **({"foreign_leakage_delta_vs_c1": leakage_delta, "merge_delta_vs_c1": merge_delta, "hard_overlap_delta_vs_c1": hard_overlap_delta} if arm == "c2_e" else {"utility_score_auroc_delta_vs_c1": auroc_delta})}}


def fmt(value: float | None, signed: bool = False) -> str:
    return "NA" if value is None else (f"{value:+.6f}" if signed else f"{value:.6f}")


def markdown(payload: dict[str, Any]) -> str:
    lines = ["# C2 component attribution: TNBC p7/p8 development", "", "- Scope: fixed epoch 5; paired seeds 2027 and 1337; no p9-p11 or MoNuSeg access.", "- C2-E and C2-U retain C1 coverage/quality and change only the specified C2 component.", "- Oracle-score intervention is GT-only and not a model-performance result.", "", "## Patient-macro native final deltas", "", "| arm vs C1 | AJI mean ± std | DQ mean ± std | PQ mean ± std | selected-pool oracle PQ Δ | positive PQ seeds |", "|---|---:|---:|---:|---:|---:|"]
    for arm in ("c2_eu", "c2_e", "c2_u"):
        stats = payload["two_seed_patient_macro"][f"{arm}_minus_c1"]["native"]
        selected = payload["two_seed_patient_macro"][f"{arm}_minus_c1"]["selected_oracle"]["pq"]
        lines.append(f"| {arm} | {fmt(stats['aji']['mean'], True)} ± {fmt(stats['aji']['std_sample'])} | {fmt(stats['dq']['mean'], True)} ± {fmt(stats['dq']['std_sample'])} | {fmt(stats['pq']['mean'], True)} ± {fmt(stats['pq']['std_sample'])} | {fmt(selected['mean'], True)} | {stats['pq']['positive_count']}/{stats['pq']['count']} |")
    lines += ["", "## Assembly and FP decomposition vs C1", "", "| arm | selected→final PQ gap Δ | final-FP penalty Δ | merge Δ | duplicate Δ |", "|---|---:|---:|---:|---:|"]
    for arm in ("c2_eu", "c2_e", "c2_u"):
        stats = payload["two_seed_patient_macro"][f"{arm}_minus_c1"]
        lines.append(f"| {arm} | {fmt(stats['gaps']['selected_to_final_assembly_gap']['pq']['mean'], True)} | {fmt(stats['gaps']['final_fp_penalty']['pq']['mean'], True)} | {fmt(stats['errors']['merge_overlap_fraction_gt_or_pred_gt_0']['mean'], True)} | {fmt(stats['errors']['duplicate_unmatched_prediction_count']['mean'], True)} |")
    lines += ["", "## Direct component mechanism statistics", "", "| arm | seed | hard foreign leakage | soft foreign leakage | soft overlap | AUROC | AUPRC | Brier | ECE | oracle-score PQ |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for record in payload["per_seed"]:
        for arm in ("c1", "c2_eu", "c2_e", "c2_u"):
            mechanism = record["mechanisms"][arm]
            ex, score, oracle = mechanism["exclusivity"], mechanism["score"], mechanism["oracle_score_intervention"]
            lines.append(f"| {arm} | {record['seed']} | {fmt(ex.get('hard_foreign_gt_fraction_mean'))} | {fmt(ex.get('soft_foreign_gt_probability_mean'))} | {fmt(ex.get('soft_selected_overlap_mean'))} | {fmt(score.get('auroc'))} | {fmt(score.get('auprc'))} | {fmt(score.get('brier'))} | {fmt(score.get('ece'))} | {fmt(oracle.get('pq'))} |")
    lines += ["", "## Component decisions", ""]
    for arm in ("c2_e", "c2_u"):
        gate = payload["component_gates"][arm]
        lines.append(f"- {arm}: `{gate['status']}`.")
        for name, passed in gate["conditions"].items():
            lines.append(f"  - {name}: {'pass' if passed else 'fail'}")
    lines += ["", "## Train-only utility-label audit", "", "| arm | seed | valid selected prompts | unique TP | unmatched FP | duplicate | merge risk | positive fraction |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for arm in ("c2_eu", "c2_e", "c2_u"):
        for seed in SEEDS:
            audit = payload["training_loss_audits"][f"{seed}:{arm}"]
            count = audit["counts"]
            lines.append(f"| {arm} | {seed} | {count['valid_prompt_count']} | {count['unique_tp_count']} | {count['unmatched_fp_count']} | {count['duplicate_count']} | {count['merge_risk_count']} | {fmt(audit['positive_fraction'])} |")
    lines += [""]
    return "\n".join(lines)


def write_csv(path: Path, payload: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    for record in payload["per_seed"]:
        for scope, values in record["scopes"].items():
            for arm in ARMS:
                for stage, metric in values[arm]["stages"].items():
                    rows.append({"seed": record["seed"], "scope": scope, "comparison": arm, "section": "stage", "name": stage, **metric})
            for arm in ("c2_eu", "c2_e", "c2_u"):
                for comparator in ("c0", "c1"):
                    diff = values[f"{arm}_minus_{comparator}"]
                    rows.append({"seed": record["seed"], "scope": scope, "comparison": f"{arm}_minus_{comparator}", "section": "errors", "name": "errors", **diff["errors"]})
                    for name, gap in diff["gaps"].items():
                        rows.append({"seed": record["seed"], "scope": scope, "comparison": f"{arm}_minus_{comparator}", "section": "gap", "name": name, **gap})
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", action="append", required=True, type=parse_assignment)
    parser.add_argument("--mechanism", action="append", required=True, type=parse_assignment)
    parser.add_argument("--training-summary", action="append", required=True, type=parse_assignment)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    oracles = {(seed, arm): path for seed, arm, path in args.oracle}
    mechanisms = {(seed, arm): path for seed, arm, path in args.mechanism}
    training = {(seed, arm): path for seed, arm, path in args.training_summary}
    expected_oracles = {(seed, arm) for seed in SEEDS for arm in ARMS}
    expected_mechanisms = {(seed, arm) for seed in SEEDS for arm in ("c1", "c2_eu", "c2_e", "c2_u")}
    expected_training = {(seed, arm) for seed in SEEDS for arm in ("c2_eu", "c2_e", "c2_u")}
    if set(oracles) != expected_oracles or set(mechanisms) != expected_mechanisms or set(training) != expected_training:
        raise ValueError("require 10 oracle reports, 8 mechanism reports, and 6 C2 training summaries")
    oracle_payloads = {}
    for (seed, arm), path in oracles.items():
        value = read_json(path); validate_oracle(value, seed, arm, path); oracle_payloads[(seed, arm)] = value
    mechanism_payloads = {}
    for (seed, arm), path in mechanisms.items():
        value = read_json(path)
        if value.get("status") != "complete" or int(value.get("seed", -1)) != seed or value.get("arm") != oracle_arm(arm):
            raise ValueError(f"mechanism identity mismatch: {path}")
        mechanism_payloads[(seed, arm)] = value
    training_audits = {}
    for (seed, arm), path in training.items():
        value = read_json(path)
        audit = value.get("runtime", {}).get("c2_ar_loss_audit", {})
        counts = {key: int(audit.get(key, 0)) for key in ("valid_prompt_count", "unique_tp_count", "unmatched_fp_count", "duplicate_count", "merge_risk_count")}
        # C2-EU was completed before the explicit valid_prompt_count field was
        # added.  The three mutually exclusive detached utility labels already
        # exhaust that denominator, so this is an exact backward-compatible
        # recovery rather than an estimate.
        if counts["valid_prompt_count"] == 0:
            counts["valid_prompt_count"] = counts["unique_tp_count"] + counts["unmatched_fp_count"] + counts["duplicate_count"]
        if value.get("status") != "complete" or counts["valid_prompt_count"] <= 0:
            raise ValueError(f"missing complete C2 train-only utility audit: {path}")
        training_audits[f"{seed}:{arm}"] = {"path": str(path), "counts": counts, "positive_fraction": float(counts["unique_tp_count"] / counts["valid_prompt_count"]), "negative_fraction": float((counts["unmatched_fp_count"] + counts["duplicate_count"]) / counts["valid_prompt_count"]), "weighted_extra_ratio": value.get("runtime", {}).get("c2_ar_loss_audit", {}).get("means", {}).get("extra_to_total_ratio")}
    per_seed = []
    for seed in SEEDS:
        scopes = {}
        for label, source in SCOPES:
            arms = {arm: normalized(_scope(oracle_payloads[(seed, arm)], source)) for arm in ARMS}
            scopes[label] = {**arms}
            for arm in ("c2_eu", "c2_e", "c2_u"):
                scopes[label][f"{arm}_minus_c0"] = compare(arms[arm], arms["c0"])
                scopes[label][f"{arm}_minus_c1"] = compare(arms[arm], arms["c1"])
        per_seed.append({"seed": seed, "scopes": scopes, "mechanisms": {arm: mechanism_normalized(mechanism_payloads[(seed, arm)], "patient_macro") for arm in ("c1", "c2_eu", "c2_e", "c2_u")}})
    aggregate = {f"{arm}_minus_{base}": summary_stats(per_seed, f"{arm}_minus_{base}") for arm in ("c2_eu", "c2_e", "c2_u") for base in ("c0", "c1")}
    payload = {"schema_version": 1, "protocol": "tnbc_c2_component_ablation_v1", "status": "complete", "scope": "TNBC p7/p8 development only; epoch 5 fixed; paired seeds 2027/1337", "per_seed": per_seed, "two_seed_patient_macro": aggregate, "training_loss_audits": training_audits}
    payload["component_gates"] = {arm: component_gate(per_seed, arm) for arm in ("c2_e", "c2_u")}
    output = Path(args.output_dir).resolve(); output.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output / "c2_component_ablation.json", payload)
    write_csv(output / "c2_component_ablation.csv", payload)
    (output / "c2_component_ablation.md").write_text(markdown(payload), encoding="utf-8")
    print(json.dumps({"status": "complete", "output_dir": str(output), "gates": {key: value["status"] for key, value in payload["component_gates"].items()}}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
