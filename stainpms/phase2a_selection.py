"""Frozen TNBC Phase 2A patient-macro checkpoint selection."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

import numpy as np


def tnbc_patient_macro_score(
    image_records: Iterable[dict[str, Any]], patient_by_sample: dict[str, int]
) -> dict[str, Any]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in image_records:
        sample_id = str(record["sample_id"])
        if sample_id not in patient_by_sample:
            raise ValueError(f"evaluation sample is absent from manifest: {sample_id}")
        if not bool(record.get("included_in_macro")):
            raise ValueError(f"TNBC development image excluded from strict macro: {sample_id}")
        grouped[int(patient_by_sample[sample_id])].append(record)
    if sorted(grouped) != [7, 8]:
        raise ValueError(f"TNBC development must contain patients 7 and 8, got {sorted(grouped)}")
    by_patient: dict[str, dict[str, Any]] = {}
    for patient in (7, 8):
        records = grouped[patient]
        metrics = {}
        for metric in ("dice1", "dice2", "aji", "aji_p", "dq", "sq", "pq"):
            values = [float(record["metrics"][metric]) for record in records]
            metrics[metric] = float(np.mean(values))
        by_patient[str(patient)] = {"image_count": len(records), "metrics_macro": metrics}
    macro_patient_aji = float(np.mean([by_patient[str(p)]["metrics_macro"]["aji"] for p in (7, 8)]))
    macro_patient_pq = float(np.mean([by_patient[str(p)]["metrics_macro"]["pq"] for p in (7, 8)]))
    return {
        "by_patient": by_patient,
        "macro_patient_aji": macro_patient_aji,
        "macro_patient_pq": macro_patient_pq,
        "selection_score": 0.5 * macro_patient_aji + 0.5 * macro_patient_pq,
        "selection_formula": "0.5 * macro_patient_AJI + 0.5 * macro_patient_PQ",
    }


def choose_tnbc_checkpoint(
    candidates: Iterable[dict[str, Any]], tie_tolerance: float = 0.001
) -> dict[str, Any]:
    ordered = sorted(candidates, key=lambda item: int(item["optimizer_updates"]))
    if not ordered:
        raise ValueError("at least one checkpoint candidate is required")
    chosen = ordered[0]
    for candidate in ordered[1:]:
        score_delta = float(candidate["selection_score"]) - float(chosen["selection_score"])
        if score_delta >= float(tie_tolerance):
            chosen = candidate
    return {
        **chosen,
        "tie_tolerance": float(tie_tolerance),
        "tie_policy": "when score difference is less than 0.001, choose the earlier checkpoint",
    }
