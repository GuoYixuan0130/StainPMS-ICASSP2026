"""Development PQ-best selection and paired coverage accounting for Phase 2A.

This module is deliberately model-free.  It consumes only the machine-readable
per-epoch diagnosis outputs written by the frozen Phase-1 evaluator.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from stainpms.phase2a_tnbc_screen import build_epoch_record, metric_deltas, read_json


def read_images(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"JSON list required: {path}")
    return payload


def read_gt_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def diagnosis_epoch_record(*, arm: str, epoch: int, directory: Path) -> dict[str, Any]:
    required = ("summary.json", "images.json", "gt_instances.csv")
    missing = [name for name in required if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(f"diagnosis directory missing {missing}: {directory}")
    return build_epoch_record(
        arm=arm,
        epoch=epoch,
        summary=read_json(directory / "summary.json"),
        images=read_images(directory / "images.json"),
        gt_rows=read_gt_rows(directory / "gt_instances.csv"),
        source_dir=directory,
    )


def development_patient_macro_pq(record: dict[str, Any]) -> float:
    value = record.get("patient_macro", {}).get("task_metrics_image_macro", {}).get("pq")
    if value is None:
        raise ValueError("epoch record lacks patient-macro PQ")
    return float(value)


def choose_pq_best(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Select maximum equal-patient macro PQ; exact ties retain earlier epoch."""

    if not records:
        raise ValueError("at least one epoch record is required")
    ordered = sorted(records, key=lambda item: int(item["epoch"]))
    expected = list(range(1, len(ordered) + 1))
    observed = [int(item["epoch"]) for item in ordered]
    if observed != expected:
        raise ValueError(f"epoch records must be contiguous from one: {observed}")
    selected = ordered[0]
    selected_pq = development_patient_macro_pq(selected)
    for record in ordered[1:]:
        pq = development_patient_macro_pq(record)
        if pq > selected_pq:
            selected = record
            selected_pq = pq
    return {
        "selection_metric": "development_equal_patient_macro_pq",
        "tie_break": "earlier_epoch_on_exact_equal_PQ",
        "selected_epoch": int(selected["epoch"]),
        "selected_patient_macro_pq": selected_pq,
        "record": selected,
        "epoch_patient_macro_pq": [
            {"epoch": int(record["epoch"]), "pq": development_patient_macro_pq(record)}
            for record in ordered
        ],
    }


def _as_threshold_success(value: Any, threshold: float) -> bool:
    if value is None or str(value).strip() == "":
        return False
    try:
        return float(value) >= float(threshold)
    except (TypeError, ValueError) as error:
        raise ValueError(f"candidate IoU must be numeric, null, or empty; got {value!r}") from error


def paired_coverage_flips(
    reference_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    *,
    field: str,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Compare the same dev GT identities without treating nuclei as IID tests."""

    key_fields = ("sample_id", "gt_instance_id")

    def index(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
        result: dict[tuple[str, str], dict[str, Any]] = {}
        for row in rows:
            key = tuple(str(row.get(name, "")) for name in key_fields)
            if not all(key) or key in result:
                raise ValueError(f"invalid or duplicate GT identity in paired flip input: {key}")
            result[key] = row
        return result

    reference = index(reference_rows)
    candidate = index(candidate_rows)
    if set(reference) != set(candidate):
        raise ValueError("paired flip inputs do not contain identical GT identities")
    counts = {"success_to_failure": 0, "failure_to_success": 0, "success_to_success": 0, "failure_to_failure": 0}
    numerator_reference = 0
    numerator_candidate = 0
    per_patient: dict[str, dict[str, int]] = {}
    for key in sorted(reference):
        left = _as_threshold_success(reference[key].get(field), threshold)
        right = _as_threshold_success(candidate[key].get(field), threshold)
        numerator_reference += int(left)
        numerator_candidate += int(right)
        label = (
            "success_to_success" if left and right else "success_to_failure" if left else "failure_to_success" if right else "failure_to_failure"
        )
        counts[label] += 1
        patient = str(reference[key].get("patient", ""))
        patient_counts = per_patient.setdefault(patient, {name: 0 for name in counts})
        patient_counts[label] += 1
    return {
        "field": field,
        "threshold": float(threshold),
        "denominator": len(reference),
        "reference_numerator": numerator_reference,
        "candidate_numerator": numerator_candidate,
        "reference_ccr": numerator_reference / len(reference),
        "candidate_ccr": numerator_candidate / len(reference),
        "flips": counts,
        "per_patient_flips": per_patient,
    }


def selected_vs_reference_report(
    reference_selection: dict[str, Any], candidate_selection: dict[str, Any]
) -> dict[str, Any]:
    """Report selected-model deltas and best/selected paired threshold flips."""

    reference = reference_selection["record"]
    candidate = candidate_selection["record"]
    reference_rows = read_gt_rows(Path(reference["source_dir"]) / "gt_instances.csv")
    candidate_rows = read_gt_rows(Path(candidate["source_dir"]) / "gt_instances.csv")
    # `metric_deltas` intentionally enforces equal epochs for fixed-budget
    # comparisons.  PQ-best selection can choose different epochs per arm, so
    # align only its guard field while preserving the selected epoch explicitly
    # in the output.  The patient metrics themselves have no epoch dependency.
    aligned_candidate = dict(candidate)
    aligned_candidate["epoch"] = reference["epoch"]
    return {
        "reference_selected_epoch": int(reference["epoch"]),
        "candidate_selected_epoch": int(candidate["epoch"]),
        "delta_candidate_minus_reference": metric_deltas(reference, aligned_candidate),
        "paired_best_candidate_ccr_at_0_5": paired_coverage_flips(
            reference_rows, candidate_rows, field="auto_best_candidate_iou"
        ),
        "paired_selected_candidate_ccr_at_0_5": paired_coverage_flips(
            reference_rows, candidate_rows, field="auto_selected_candidate_iou"
        ),
    }
