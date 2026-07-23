#!/usr/bin/env python3
"""Run the no-training C3 assembly-score control feasibility audit.

Only compact C1 p7/p8 zero-training artifacts are read.  No checkpoint,
dataset loader, model, optimizer, or mask decoder is constructed.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stainpms.c2_component_audit import deserialize_gt, deserialize_selected
from stainpms.c3_score_control_audit import OPERATIONS, audit_image


SEEDS = (2027, 1337)
METRICS = ("tp", "fp", "fn", "dq", "sq", "pq")
SINGLE_OPERATIONS = (
    "fp_demotion_oracle",
    "duplicate_order_oracle",
    "conflict_order_oracle",
    "merge_risk_demotion_oracle",
)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def read_gzip_json(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_assignment(value: str) -> tuple[int, Path]:
    try:
        raw_seed, raw_path = value.split("=", 1)
        seed = int(raw_seed)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("assignment must be SEED=/absolute/path") from exc
    if seed not in SEEDS:
        raise argparse.ArgumentTypeError("only seeds 2027 and 1337 are allowed")
    return seed, Path(raw_path).resolve()


def average(rows: list[dict[str, Any]], fields: tuple[str, ...] = METRICS) -> dict[str, float | None]:
    return {
        field: (float(np.mean([float(row[field]) for row in rows])) if rows else None)
        for field in fields
    }


def delta(row: dict[str, Any], native: dict[str, Any]) -> dict[str, float]:
    return {field: float(row[field]) - float(native[field]) for field in METRICS}


def aggregate_conflicts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = (
        "component_count",
        "non_singleton_component_count",
        "edge_count",
        "singleton_unmatched_fp_count",
        "conflicting_unmatched_fp_count",
        "duplicate_count",
        "merge_risk_count",
    )
    component_sizes = [size for row in rows for size in row["component_sizes"]]
    top_correct = sum(int(row["unique_tp_native_top1"]["numerator"]) for row in rows)
    top_total = sum(int(row["unique_tp_native_top1"]["denominator"]) for row in rows)
    pairwise: dict[str, Any] = {}
    for name in ("all_negative", "unmatched_fp", "duplicate"):
        values = [row["pairwise_ordering"][name] for row in rows]
        correct = sum(int(value["correct"]) for value in values)
        total = sum(int(value["count"]) for value in values)
        ties = sum(int(value["ties"]) for value in values)
        margins = [margin for value in values for margin in value["margin_values"]]
        pairwise[name] = {
            "correct": correct,
            "count": total,
            "ties": ties,
            "accuracy": float(correct / total) if total else None,
            "positive_minus_negative_margin": quantiles(margins),
        }
    edge_reasons = {
        key: sum(int(row["edge_reason_counts"].get(key, 0)) for row in rows)
        for key in ("prompt_group", "nms_box_iou", "paint_mask_overlap")
    }
    return {
        **{name: sum(int(row[name]) for row in rows) for name in counts},
        "component_size": quantiles(component_sizes),
        "edge_reason_counts": edge_reasons,
        "unique_tp_native_top1": {
            "numerator": top_correct,
            "denominator": top_total,
            "accuracy": float(top_correct / top_total) if top_total else None,
        },
        "pairwise_ordering": pairwise,
    }


def quantiles(values: list[float | int]) -> dict[str, float | int | None]:
    array = np.asarray([float(value) for value in values], dtype=np.float64)
    if not array.size:
        return {key: None for key in ("count", "mean", "q10", "q25", "median", "q75", "q90")}
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "q10": float(np.quantile(array, 0.10)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.quantile(array, 0.50)),
        "q75": float(np.quantile(array, 0.75)),
        "q90": float(np.quantile(array, 0.90)),
    }


def aggregate_patient(rows: list[dict[str, Any]]) -> dict[str, Any]:
    stages = {name: average([row["stages"][name] for row in rows]) for name in OPERATIONS}
    return {
        "image_count": len(rows),
        "stages_image_macro": stages,
        "deltas_vs_native_image_macro": {name: delta(stages[name], stages["native"]) for name in OPERATIONS},
        "conflicts": aggregate_conflicts([row["conflicts"] for row in rows]),
        "target_counts": {
            name: sum(int(row["targets"][name]) for row in rows)
            for name in (
                "unmatched_fp_score_demoted_count",
                "duplicate_competition_pair_count",
                "duplicate_score_reordered_count",
                "conflict_component_reordered_count",
                "harmful_merge_risk_score_demoted_count",
            )
        },
        "retention_count_preservation": {
            operation: {
                "preserved_image_count": sum(bool(row["retention_count_preserved"][operation]) for row in rows),
                "image_count": len(rows),
                "all_images_preserved": all(bool(row["retention_count_preserved"][operation]) for row in rows),
            }
            for operation in OPERATIONS
        },
    }


def patient_macro(patients: dict[str, dict[str, Any]], image_rows: list[dict[str, Any]]) -> dict[str, Any]:
    stages = {
        operation: {
            field: float(np.mean([float(patients[str(patient)]["stages_image_macro"][operation][field]) for patient in (7, 8)]))
            for field in METRICS
        }
        for operation in OPERATIONS
    }
    # Aggregate the original per-image conflict records.  Patient summaries
    # already collapse distributions, so feeding those back into this helper
    # would lose the component-size and score-margin samples.
    conflict_rows = [row["conflicts"] for row in image_rows]
    # Conflict counts are reported as sums over both patients; scalar rates are
    # also preserved in the constituent per-patient records.
    return {
        "aggregation": "equal patient macro for TP/FP/FN/DQ/SQ/PQ; conflict counts summed across p7/p8",
        "stages_patient_macro": stages,
        "deltas_vs_native_patient_macro": {name: delta(stages[name], stages["native"]) for name in OPERATIONS},
        "conflicts_both_patients": aggregate_conflicts(conflict_rows),
        "target_counts_both_patients": {
            name: sum(int(patients[str(patient)]["target_counts"][name]) for patient in (7, 8))
            for name in patients["7"]["target_counts"]
        },
        "retention_count_preservation": {
            operation: {
                "preserved_image_count": sum(
                    int(patients[str(patient)]["retention_count_preservation"][operation]["preserved_image_count"])
                    for patient in (7, 8)
                ),
                "image_count": sum(
                    int(patients[str(patient)]["retention_count_preservation"][operation]["image_count"])
                    for patient in (7, 8)
                ),
                "all_images_preserved": all(
                    bool(patients[str(patient)]["retention_count_preservation"][operation]["all_images_preserved"])
                    for patient in (7, 8)
                ),
            }
            for operation in OPERATIONS
        },
    }


def close(a: float, b: float, *, atol: float = 1.0e-7) -> bool:
    return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=atol)


def validate_source(root: Path, seed: int) -> list[dict[str, Any]]:
    summary = read_json(root / "summary.json")
    if summary.get("status") != "complete" or int(summary.get("seed", -1)) != seed or summary.get("arm") != "c1":
        raise ValueError(f"C3 requires a completed C1 source for seed {seed}: {root}")
    if summary.get("reference_reproduction", {}).get("status") != "pass":
        raise ValueError(f"C1 source reproduction did not pass: {root}")
    payloads = [read_gzip_json(path) for path in sorted((root / "completed_images").glob("*.json.gz"))]
    artifacts = [payload["artifact"] for payload in payloads]
    if len(artifacts) != 7 or {int(row["patient"]) for row in artifacts} != {7, 8}:
        raise ValueError(f"C3 source must contain exactly seven p7/p8 artifacts: {root}")
    # Keep the outer payload as well: its image_record is the frozen native
    # reference that must be reproduced before any GT-only attribution runs.
    return payloads


def audit_seed(root: Path, seed: int, *, nms_iou: float) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for payload in validate_source(root, seed):
        artifact = payload["artifact"]
        gt_map = deserialize_gt(artifact)
        selected = deserialize_selected(artifact)
        result = audit_image(selected, gt_map, nms_iou=nms_iou)
        expected = payload["image_record"]["stages"]["native_final"]
        actual = result["stages"]["native"]
        mismatch = {
            field: {"expected": expected.get(field), "actual": actual.get(field)}
            for field in METRICS
            if expected.get(field) is None or not close(actual[field], expected[field])
        }
        if mismatch:
            raise RuntimeError(
                f"native assembly reproduction mismatch for seed={seed}, sample={artifact['sample_id']}: {mismatch}"
            )
        rows.append(
            {
                "sample_id": str(artifact["sample_id"]),
                "patient": int(artifact["patient"]),
                "stages": result["stages"],
                "deltas_vs_native": result["deltas_vs_native"],
                "targets": result["targets"],
                "retention_count_preserved": result["retention_count_preserved"],
                "conflicts": result["conflicts"],
            }
        )
    by_patient = {
        str(patient): aggregate_patient([row for row in rows if int(row["patient"]) == patient])
        for patient in (7, 8)
    }
    return {
        "seed": seed,
        "source_c1_oracle_directory": str(root),
        "per_image": rows,
        "patients": by_patient,
        "patient_macro": patient_macro(by_patient, rows),
    }


def reference_full_oracle(path: Path, seed: int) -> float:
    payload = read_json(path)
    if payload.get("status") != "complete" or int(payload.get("seed", -1)) != seed or payload.get("arm") != "c1":
        raise ValueError(f"invalid C1 full-oracle reference: {path}")
    return float(payload["patient_macro"]["image_macro"]["oracle_score_intervention"]["pq"])


def c3_gate(per_seed: list[dict[str, Any]]) -> dict[str, Any]:
    gains = {
        int(record["seed"]): {
            name: float(record["patient_macro"]["deltas_vs_native_patient_macro"][name]["pq"])
            for name in OPERATIONS
            if name != "native"
        }
        for record in per_seed
    }
    full = {seed: values["full_score_oracle"] for seed, values in gains.items()}
    candidates: dict[str, Any] = {}
    for operation in SINGLE_OPERATIONS:
        final_count_condition = (
            all(
                bool(record["patient_macro"]["retention_count_preservation"][operation]["all_images_preserved"])
                for record in per_seed
            )
            if operation in {"duplicate_order_oracle", "conflict_order_oracle"}
            else True
        )
        conditions = {
            "both_seed_absolute_pq_gain_at_least_0_003": all(gains[seed][operation] >= 0.003 for seed in SEEDS),
            "both_seed_recovers_at_least_30_percent_of_full": all(
                full[seed] > 0.0 and gains[seed][operation] >= 0.30 * full[seed]
                for seed in SEEDS
            ),
            "same_operation_is_largest_independent_gain_each_seed": all(
                gains[seed][operation] >= max(gains[seed][name] for name in SINGLE_OPERATIONS)
                for seed in SEEDS
            ),
            "fixed_final_instance_count_for_ordering_intervention": final_count_condition,
        }
        candidates[operation] = {
            "status": "supported" if all(conditions.values()) else "not_supported",
            "conditions": conditions,
            "gains": {str(seed): gains[seed][operation] for seed in SEEDS},
            "full_gains": {str(seed): full[seed] for seed in SEEDS},
        }
    supported = [name for name, detail in candidates.items() if detail["status"] == "supported"]
    direction = {
        "duplicate_order_oracle": "conflict-set structured ranking",
        "conflict_order_oracle": "conflict-set structured ranking",
        "fp_demotion_oracle": "context-aware keep/reject scoring",
        "merge_risk_demotion_oracle": "close score route and return to candidate generation or local correction",
    }
    return {
        "status": "one_direction_supported" if len(supported) == 1 else "close_assembly_scoring_route",
        "single_supported_operation": supported[0] if len(supported) == 1 else None,
        "proposed_direction": direction[supported[0]] if len(supported) == 1 else None,
        "candidates": candidates,
        "note": "independent oracle gains are non-additive and are never summed",
    }


def markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# C3 score-control feasibility audit",
        "",
        "- Scope: fixed C1 epoch-5 selected masks and native scores; TNBC p7/p8 only; seeds 2027 and 1337.",
        "- Every score operation is GT-only, keeps masks/candidate pool/NMS/assembly fixed, and is an upper bound rather than model performance.",
        "- Native assembly has no fixed keep/reject score threshold. `fp_demotion_oracle` is therefore score demotion, not literal threshold filtering.",
        "",
        "## Frozen C2 decision",
        "",
        "- C2-E and C2-U failed their pre-registered mechanism gates.",
        "- C2-EU had a small two-seed PQ gain versus C1 but did not stably exceed C0 and yielded no independently retainable component.",
        "- The rejected object is the current exclusivity plus pointwise utility-MSE training form, not assembly score as a control variable.",
        "- Training unique-TP labels were about 78%; the failure is not attributed to supervision scarcity.",
        "",
        "## Independent score-oracle gain decomposition",
        "",
        "| seed | native PQ | FP demotion Δ | duplicate order Δ | conflict order Δ | merge-risk demotion Δ | full score oracle Δ |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in payload["per_seed"]:
        stage = record["patient_macro"]["stages_patient_macro"]
        gains = record["patient_macro"]["deltas_vs_native_patient_macro"]
        lines.append(
            f"| {record['seed']} | {stage['native']['pq']:.6f} | {gains['fp_demotion_oracle']['pq']:+.6f} | {gains['duplicate_order_oracle']['pq']:+.6f} | {gains['conflict_order_oracle']['pq']:+.6f} | {gains['merge_risk_demotion_oracle']['pq']:+.6f} | {gains['full_score_oracle']['pq']:+.6f} |"
        )
    lines += [
        "",
        "Independent interventions are intentionally non-additive; their gains must not be summed.",
        "",
        "## Patient-wise score-control results",
        "",
        "| seed | patient | native TP/FP/FN | native DQ | native SQ | native PQ | FP demotion ΔPQ | duplicate order ΔPQ | conflict order ΔPQ | merge-risk demotion ΔPQ | full oracle ΔPQ |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in payload["per_seed"]:
        for patient in (7, 8):
            patient_record = record["patients"][str(patient)]
            stage = patient_record["stages_image_macro"]
            gains = patient_record["deltas_vs_native_image_macro"]
            native = stage["native"]
            lines.append(
                f"| {record['seed']} | {patient} | {native['tp']:.3f}/{native['fp']:.3f}/{native['fn']:.3f} | "
                f"{native['dq']:.6f} | {native['sq']:.6f} | {native['pq']:.6f} | "
                f"{gains['fp_demotion_oracle']['pq']:+.6f} | {gains['duplicate_order_oracle']['pq']:+.6f} | "
                f"{gains['conflict_order_oracle']['pq']:+.6f} | {gains['merge_risk_demotion_oracle']['pq']:+.6f} | "
                f"{gains['full_score_oracle']['pq']:+.6f} |"
            )
    lines += [
        "",
        "## Native conflict-set ordering",
        "",
        "| seed | components | non-singleton | singleton FP | conflicting FP | duplicate | merge risk | unique-TP top-1 accuracy | unique-vs-negative order accuracy |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in payload["per_seed"]:
        conflict = record["patient_macro"]["conflicts_both_patients"]
        top = conflict["unique_tp_native_top1"]
        pair = conflict["pairwise_ordering"]["all_negative"]
        lines.append(
            f"| {record['seed']} | {conflict['component_count']} | {conflict['non_singleton_component_count']} | {conflict['singleton_unmatched_fp_count']} | {conflict['conflicting_unmatched_fp_count']} | {conflict['duplicate_count']} | {conflict['merge_risk_count']} | {top['accuracy'] if top['accuracy'] is not None else 'NA'} ({top['numerator']}/{top['denominator']}) | {pair['accuracy'] if pair['accuracy'] is not None else 'NA'} ({pair['correct']}/{pair['count']}) |"
        )
    lines += ["", "### Fixed final-instance count for score permutations", ""]
    for record in payload["per_seed"]:
        retention = record["patient_macro"]["retention_count_preservation"]
        lines.append(
            f"- seed {record['seed']}: duplicate ordering preserved {retention['duplicate_order_oracle']['preserved_image_count']}/{retention['duplicate_order_oracle']['image_count']} image counts; "
            f"conflict ordering preserved {retention['conflict_order_oracle']['preserved_image_count']}/{retention['conflict_order_oracle']['image_count']}."
        )
    lines += ["", "## C3 feasibility gate", ""]
    gate = payload["c3_gate"]
    lines.append(f"- Status: `{gate['status']}`.")
    if gate["single_supported_operation"]:
        lines.append(f"- The only supported direction is `{gate['proposed_direction']}` from `{gate['single_supported_operation']}`.")
    else:
        lines.append("- No single score-only intervention met both-seed direction, 30%-of-full-oracle, and +0.003 absolute-PQ requirements; close the assembly-scoring route.")
    for name, detail in gate["candidates"].items():
        lines.append(f"- {name}: `{detail['status']}`.")
        for condition, passed in detail["conditions"].items():
            lines.append(f"  - {condition}: {'pass' if passed else 'fail'}")
    lines.append("")
    return "\n".join(lines)


def write_csv(path: Path, payload: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    for record in payload["per_seed"]:
        seed = record["seed"]
        for image in record["per_image"]:
            for operation in OPERATIONS:
                rows.append({
                    "seed": seed,
                    "patient": image["patient"],
                    "sample_id": image["sample_id"],
                    "section": "stage",
                    "operation": operation,
                    **image["stages"][operation],
                    **{f"delta_{name}": value for name, value in image["deltas_vs_native"][operation].items()},
                })
            conflict = image["conflicts"]
            rows.append({
                "seed": seed,
                "patient": image["patient"],
                "sample_id": image["sample_id"],
                "section": "conflict",
                "operation": "native",
                "component_count": conflict["component_count"],
                "non_singleton_component_count": conflict["non_singleton_component_count"],
                "singleton_unmatched_fp_count": conflict["singleton_unmatched_fp_count"],
                "conflicting_unmatched_fp_count": conflict["conflicting_unmatched_fp_count"],
                "duplicate_count": conflict["duplicate_count"],
                "merge_risk_count": conflict["merge_risk_count"],
                "unique_tp_top1_accuracy": conflict["unique_tp_native_top1"]["accuracy"],
                "pairwise_all_negative_accuracy": conflict["pairwise_ordering"]["all_negative"]["accuracy"],
            })
    fields = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, type=parse_assignment, help="SEED=C1_ORACLE_DIR")
    parser.add_argument("--reference-full-oracle", action="append", required=True, type=parse_assignment, help="SEED=C1_COMPONENT_MECHANISMS_JSON")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--instance-nms-iou", type=float, default=0.5)
    args = parser.parse_args()
    if not 0.0 < float(args.instance_nms_iou) < 1.0:
        raise ValueError("instance NMS IoU must be in (0, 1)")
    inputs = dict(args.input)
    references = dict(args.reference_full_oracle)
    if set(inputs) != set(SEEDS) or set(references) != set(SEEDS):
        raise ValueError("requires exactly seeds 2027 and 1337 for inputs and references")
    output = args.output_dir.resolve()
    if output.exists():
        raise ValueError(f"refusing to overwrite output directory: {output}")
    config_path = args.config.resolve()
    config = read_json(config_path)
    if config.get("protocol_id") != "tnbc_c3_score_control_feasibility_audit_v1":
        raise ValueError(f"unexpected C3 configuration: {config_path}")
    per_seed = [audit_seed(inputs[seed], seed, nms_iou=float(args.instance_nms_iou)) for seed in SEEDS]
    reference_validation: dict[str, Any] = {}
    for record in per_seed:
        seed = int(record["seed"])
        actual = float(record["patient_macro"]["stages_patient_macro"]["full_score_oracle"]["pq"])
        expected = reference_full_oracle(references[seed], seed)
        if not close(actual, expected):
            raise RuntimeError(f"full score oracle failed to reproduce prior C1 result for seed {seed}: {actual} != {expected}")
        reference_validation[str(seed)] = {"status": "pass", "expected_full_oracle_pq": expected, "reproduced_full_oracle_pq": actual}
    payload = {
        "schema_version": 1,
        "protocol": "tnbc_c3_score_control_feasibility_audit_v1",
        "status": "complete",
        "scope": "TNBC p7/p8 only; C1 epoch-5 selected masks and native scores; seeds 2027/1337; no model/checkpoint/data-loader access",
        "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
        "frozen_c2_closure": {
            "commit": "8dbe0fcf0a11a546d1bc1d08a72f34fed093074d",
            "c2_e_and_c2_u_mechanism_gates": "not_supported",
            "c2_eu": "small two-seed PQ gain vs C1 but not stable over C0 and no independently retainable component",
            "existing_full_oracle_pq_gain": {"2027": 0.010989, "1337": 0.009476},
            "interpretation": "current exclusivity plus pointwise utility-MSE training is closed; assembly score remains a control variable",
            "training_unique_tp_fraction": "about 78 percent; not supervision scarcity",
        },
        "native_assembly": {
            "implementation": "run.run_on_epoch._assemble_instance_map",
            "instance_nms_iou": float(args.instance_nms_iou),
            "score_keep_reject_threshold": None,
            "score_semantics": "same-group prefilter, NMS ordering, and paint-order conflict resolution",
        },
        "full_oracle_reproduction": reference_validation,
        "per_seed": per_seed,
        "c3_gate": c3_gate(per_seed),
    }
    output.mkdir(parents=True, exist_ok=False)
    write_json_atomic(output / "c3_score_control_audit.json", payload)
    (output / "c3_score_control_audit.md").write_text(markdown(payload), encoding="utf-8")
    write_csv(output / "c3_score_control_audit.csv", payload)
    print(json.dumps({"status": payload["status"], "output_dir": str(output), "c3_gate": payload["c3_gate"]["status"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
