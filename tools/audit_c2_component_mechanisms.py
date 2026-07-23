"""Compute read-only C2 component mechanism statistics from oracle artifacts.

No model is loaded and no mask is regenerated.  The input must be one completed
TNBC p7/p8 zero-training diagnosis directory.  GT-only score intervention is
reported solely as an assembly-control upper bound.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stainpms.c2_component_audit import image_component_audit


def read_gzip_json(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def numeric_mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return float(np.mean(values)) if values else None


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    utility = rows[0]["utility_labels"] if rows else {}
    label_keys = ("unique_tp", "unmatched_fp", "duplicate", "merge_risk", "valid_prediction_count")
    label_sums = {key: sum(int(row["utility_labels"].get(key, 0)) for row in rows) for key in label_keys}
    valid = label_sums["valid_prediction_count"]
    score_keys = ("auroc", "auprc", "brier", "ece", "positive_fraction")
    scores = {key: numeric_mean([row["score_calibration"] for row in rows], key) for key in score_keys}
    label_score_means = {
        label: numeric_mean(
            [row["score_calibration"]["scores_by_utility_label"].get(label, {}) for row in rows], "mean"
        )
        for label in ("unique_tp", "unmatched_fp", "duplicate")
    }
    negative_score_position = {
        label: {
            key: sum(
                int(row["score_calibration"]["negative_score_position_relative_to_unique_tp"].get(label, {}).get(key, 0) or 0)
                for row in rows
            )
            for key in ("count", "below_minimum_unique_tp_score_count", "below_median_unique_tp_score_count")
        }
        for label in ("unmatched_fp", "duplicate")
    }
    exclusivity = {
        "hard_foreign_gt_fraction_mean": numeric_mean(
            [row["exclusivity"]["hard_foreign_gt_fraction"] for row in rows], "mean"
        ),
        "soft_foreign_gt_probability_mean": numeric_mean(
            [row["exclusivity"]["soft_foreign_gt_probability"] for row in rows], "mean"
        ),
        "soft_selected_overlap_mean": numeric_mean(
            [row["exclusivity"]["soft_selected_overlap"] for row in rows], "mean"
        ),
        "hard_overlap_candidate_pair_count": sum(
            int(row["exclusivity"]["hard_overlap_candidate_pair_count"]) for row in rows
        ),
        "hard_overlap_positive_pair_count": sum(
            int(row["exclusivity"]["hard_overlap_positive_pair_count"]) for row in rows
        ),
    }
    pair_count = exclusivity["hard_overlap_candidate_pair_count"]
    exclusivity["hard_overlap_positive_pair_fraction"] = (
        float(exclusivity["hard_overlap_positive_pair_count"] / pair_count) if pair_count else 0.0
    )
    intervention = {
        key: numeric_mean([row["oracle_score_intervention"] for row in rows], key)
        for key in ("tp", "fp", "fn", "dq", "sq", "pq")
    }
    return {
        "image_count": len(rows),
        "utility_labels": {
            **label_sums,
            "positive_unique_tp_fraction": float(label_sums["unique_tp"] / valid) if valid else 0.0,
            "negative_fraction": float((label_sums["unmatched_fp"] + label_sums["duplicate"]) / valid) if valid else 0.0,
            "utility_effective_sample_fraction": float(
                sum(float(row["utility_labels"]["utility_effective_sample_fraction"]) for row in rows) / len(rows)
            ) if rows else 0.0,
        },
        "score_calibration_image_macro": {
            **scores,
            "assembly_score_threshold": None,
            "threshold_note": "native assembly has no fixed score threshold; score controls within-group retention and NMS ordering",
            "mean_assembly_score_by_utility_label": label_score_means,
            "negative_score_position_relative_to_unique_tp": negative_score_position,
        },
        "exclusivity": exclusivity,
        "oracle_score_intervention_image_macro": intervention,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    flat: list[dict[str, Any]] = []
    for row in rows:
        item = {
            "sample_id": row["sample_id"],
            "patient": row["patient"],
            **row["utility_labels"],
            **{f"score_{key}": value for key, value in row["score_calibration"].items() if isinstance(value, (float, int)) or value is None},
            **{f"excl_{key}": value for key, value in row["exclusivity"].items() if isinstance(value, (float, int)) or value is None},
            **{f"oracle_{key}": value for key, value in row["oracle_score_intervention"].items() if isinstance(value, (float, int)) or value is None},
        }
        flat.append(item)
    fields = sorted({key for row in flat for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flat)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", required=True, type=int, choices=(2027, 1337))
    parser.add_argument("--arm", required=True, choices=("c1", "c2_ar", "c2_e", "c2_u"))
    args = parser.parse_args()
    root = Path(args.oracle_dir).resolve()
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if summary.get("status") != "complete" or int(summary.get("seed", -1)) != args.seed or summary.get("arm") != args.arm:
        raise ValueError("oracle input identity does not match --seed/--arm")
    artifacts = [read_gzip_json(path)["artifact"] for path in sorted((root / "completed_images").glob("*.json.gz"))]
    if len(artifacts) != 7 or {int(row["patient"]) for row in artifacts} != {7, 8}:
        raise ValueError("C2 component audit requires exactly seven p7/p8 artifacts")
    image_rows = [image_component_audit(artifact) for artifact in artifacts]
    by_patient = {
        str(patient): aggregate([row for row in image_rows if int(row["patient"]) == patient])
        for patient in (7, 8)
    }
    patient_macro = {
        "aggregation": "equal-weight mean of patient-7 and patient-8 image-macro statistics; label counts are reported separately",
        "patients": by_patient,
        "image_macro": {
            "score_calibration": {
                key: numeric_mean([by_patient[str(patient)]["score_calibration_image_macro"] for patient in (7, 8)], key)
                for key in ("auroc", "auprc", "brier", "ece", "positive_fraction")
            },
            "exclusivity": {
                key: numeric_mean([by_patient[str(patient)]["exclusivity"] for patient in (7, 8)], key)
                for key in ("hard_foreign_gt_fraction_mean", "soft_foreign_gt_probability_mean", "soft_selected_overlap_mean", "hard_overlap_positive_pair_fraction")
            },
            "oracle_score_intervention": {
                key: numeric_mean([by_patient[str(patient)]["oracle_score_intervention_image_macro"] for patient in (7, 8)], key)
                for key in ("tp", "fp", "fn", "dq", "sq", "pq")
            },
        },
    }
    payload = {
        "schema_version": 1,
        "protocol": "tnbc_c2_component_ablation_v1",
        "status": "complete",
        "scope": "TNBC p7/p8 only; read-only artifacts; no model, training, or sealed-test access",
        "seed": args.seed,
        "arm": args.arm,
        "oracle_dir": str(root),
        "per_image": image_rows,
        "patients": by_patient,
        "patient_macro": patient_macro,
        "oracle_score_intervention_note": "GT-only intervention with masks fixed and the frozen native assembly; not model performance",
    }
    output = Path(args.output_dir).resolve()
    write_json_atomic(output / "component_mechanisms.json", payload)
    write_csv(output / "component_mechanisms.csv", image_rows)
    print(json.dumps({"status": "complete", "output": str(output / "component_mechanisms.json")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
