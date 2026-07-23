#!/usr/bin/env python3
"""Gate C2-AR train-only smokes before any formal C2 epoch is run."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def numeric_difference(left: Any, right: Any) -> tuple[float | None, float | None]:
    if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
        return None, None
    absolute = abs(float(left) - float(right))
    relative = absolute / max(abs(float(right)), 1.0e-12)
    return absolute, relative


def regression_check(c1: dict[str, Any], c2_zero: dict[str, Any], *, absolute_tolerance: float, relative_tolerance: float) -> dict[str, Any]:
    mismatches: dict[str, Any] = {}
    for name in sorted(set(c1.get("losses", {})) & set(c2_zero.get("losses", {}))):
        absolute, relative = numeric_difference(c1["losses"][name], c2_zero["losses"][name])
        if absolute is not None and absolute > absolute_tolerance and relative > relative_tolerance:
            mismatches[f"loss:{name}"] = {"absolute": absolute, "relative": relative}
    for name in ("crop_batches_seen", "optimizer_steps", "native_candidate_decoder_calls", "native_candidate_prompt_count"):
        left = c1.get("runtime", {}).get(name)
        right = c2_zero.get("runtime", {}).get(name)
        if left != right:
            mismatches[f"runtime:{name}"] = {"c1": left, "c2_zero": right}
    for name in ("point_head", "mask_decoder", "quality_head"):
        left = c1.get("runtime", {}).get("gradient_audit", {}).get("group_l2_mean", {}).get(name)
        right = c2_zero.get("runtime", {}).get("gradient_audit", {}).get("group_l2_mean", {}).get(name)
        absolute, relative = numeric_difference(left, right)
        if absolute is None or (absolute > absolute_tolerance and relative > relative_tolerance):
            mismatches[f"gradient:{name}"] = {"c1": left, "c2_zero": right, "absolute": absolute, "relative": relative}
    zero_losses = {
        key: c2_zero.get("losses", {}).get(key)
        for key in ("loss_c2_ar_exclusivity", "loss_c2_ar_utility")
    }
    if any(value is None or abs(float(value)) > absolute_tolerance for value in zero_losses.values()):
        mismatches["c2_zero_losses"] = zero_losses
    return {
        "status": "pass" if not mismatches else "fail",
        "absolute_tolerance": absolute_tolerance,
        "relative_tolerance": relative_tolerance,
        "zero_c2_losses": zero_losses,
        "mismatches": mismatches,
    }


def scale_check(c2: dict[str, Any]) -> dict[str, Any]:
    runtime = c2.get("runtime", {})
    audit = runtime.get("c2_ar_loss_audit", {})
    means = audit.get("means", {})
    ratio = means.get("extra_to_total_ratio")
    raw_terms = {
        "exclusivity": means.get("exclusivity_loss_before_lambda"),
        "utility": means.get("utility_loss_before_lambda"),
    }
    valid_numbers = [ratio, *raw_terms.values(), means.get("total_loss")]
    finite = all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in valid_numbers)
    nontrivial = all(float(value) > 0.0 for value in raw_terms.values() if value is not None)
    non_dominating = isinstance(ratio, (int, float)) and 0.0 < float(ratio) < 0.5
    counts = {
        key: audit.get(key)
        for key in ("foreign_valid_prompt_count", "neighbor_pair_count", "unique_tp_count", "unmatched_fp_count", "duplicate_count", "merge_risk_count")
    }
    count_valid = all(isinstance(value, int) and value >= 0 for value in counts.values())
    return {
        "status": "pass" if finite and nontrivial and non_dominating and count_valid else "fail",
        "finite": finite,
        "nontrivial_raw_terms": nontrivial,
        "extra_to_total_ratio_in_open_interval_0_0_5": non_dominating,
        "means": means,
        "detached_match_counts": counts,
    }


def assert_arm(report: dict[str, Any], expected: str, label: str) -> None:
    if report.get("status") != "complete":
        raise ValueError(f"{label} smoke is not complete")
    if report.get("training_configuration", {}).get("arm") != expected:
        raise ValueError(f"{label} does not declare arm {expected}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c1", required=True, type=Path)
    parser.add_argument("--c2-zero", required=True, type=Path)
    parser.add_argument("--c2", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--absolute-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--relative-tolerance", type=float, default=1.0e-5)
    args = parser.parse_args()
    c1 = read_json(args.c1.resolve())
    c2_zero = read_json(args.c2_zero.resolve())
    c2 = read_json(args.c2.resolve())
    assert_arm(c1, "c1", "C1")
    assert_arm(c2_zero, "c2_ar", "C2 zero")
    assert_arm(c2, "c2_ar", "C2")
    regression = regression_check(c1, c2_zero, absolute_tolerance=args.absolute_tolerance, relative_tolerance=args.relative_tolerance)
    scale = scale_check(c2)
    payload = {
        "schema_version": 1,
        "protocol": "tnbc_c2_ar_smoke_gate_v1",
        "status": "pass" if regression["status"] == "pass" and scale["status"] == "pass" else "fail",
        "inputs": {"c1": str(args.c1.resolve()), "c2_zero": str(args.c2_zero.resolve()), "c2": str(args.c2.resolve())},
        "c1_regression_with_zero_c2_coefficients": regression,
        "c2_loss_and_detached_match_scale": scale,
    }
    write_json_atomic(args.output.resolve(), payload)
    print(json.dumps({"status": payload["status"], "output": str(args.output.resolve())}, ensure_ascii=False))
    return 0 if payload["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
