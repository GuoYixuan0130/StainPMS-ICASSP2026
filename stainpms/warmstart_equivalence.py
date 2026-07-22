"""Machine-readable numerical gates for Phase 2A C0/C1 smoke reports."""

from __future__ import annotations

import math
from typing import Any


def _difference(left: float, right: float) -> dict[str, float]:
    absolute = abs(float(left) - float(right))
    denominator = max(abs(float(left)), abs(float(right)), 1e-12)
    return {
        "left": float(left),
        "right": float(right),
        "absolute_error": absolute,
        "relative_error": absolute / denominator,
    }


def _within(record: dict[str, float], absolute_tolerance: float, relative_tolerance: float) -> bool:
    return bool(
        record["absolute_error"] <= absolute_tolerance
        or record["relative_error"] <= relative_tolerance
    )


def _original_losses(report: dict[str, Any]) -> dict[str, float]:
    return {
        name: float(value)
        for name, value in report["losses"].items()
        if not name.startswith("loss_candidate_")
    }


def compare_c0_reference(
    legacy: dict[str, Any],
    c0: dict[str, Any],
    *,
    absolute_tolerance: float = 1e-6,
    relative_tolerance: float = 1e-5,
) -> dict[str, Any]:
    """Compare one-update legacy and explicit-four-token C0 reports."""
    shared_identity = {
        "dataset": legacy["data"]["protocol_id"] == c0["data"]["protocol_id"],
        "manifest_sha256": legacy["data"]["manifest_sha256"]
        == c0["data"]["manifest_sha256"],
        "coverage_manifest_sha256": legacy["data"]["coverage"]["sha256"]
        == c0["data"]["coverage"]["sha256"],
        "checkpoint_sha256": legacy["initialization"]["checkpoint_sha256"]
        == c0["initialization"]["checkpoint_sha256"],
        "seed": legacy["determinism"]["seed"] == c0["determinism"]["seed"],
        "optimizer_updates": legacy["runtime"]["optimizer_steps"]
        == c0["runtime"]["optimizer_steps"],
        "crop_batches_seen": legacy["runtime"]["crop_batches_seen"]
        == c0["runtime"]["crop_batches_seen"],
        "optimizer": legacy["training_configuration"]["optimizer"]
        == c0["training_configuration"]["optimizer"],
        "scheduler": legacy["training_configuration"]["scheduler"]
        == c0["training_configuration"]["scheduler"],
        "data_order": legacy["training_configuration"]["data_order"]
        == c0["training_configuration"]["data_order"],
    }

    left_losses = _original_losses(legacy)
    right_losses = _original_losses(c0)
    if set(left_losses) != set(right_losses):
        raise ValueError("legacy and C0 original loss component names differ")
    loss_differences = {
        name: _difference(left_losses[name], right_losses[name])
        for name in sorted(left_losses)
    }

    legacy_gradient = legacy["runtime"]["gradient_audit"]
    c0_gradient = c0["runtime"]["gradient_audit"]
    gradient_norm_differences = {
        name: _difference(
            legacy_gradient["group_l2_mean"][name],
            c0_gradient["group_l2_mean"][name],
        )
        for name in sorted(legacy_gradient["group_l2_mean"])
    }
    key_gradient_differences = {}
    for name, left in legacy_gradient["key_gradients"].items():
        right = c0_gradient["key_gradients"].get(name)
        if right is None or left["name"] != right["name"] or left["shape"] != right["shape"]:
            raise ValueError(f"key gradient identity differs for {name}")
        if len(left["values"]) != len(right["values"]):
            raise ValueError(f"key gradient length differs for {name}")
        element_records = [
            _difference(a, b) for a, b in zip(left["values"], right["values"], strict=True)
        ]
        key_gradient_differences[name] = {
            "parameter": left["name"],
            "shape": left["shape"],
            "element_count": len(element_records),
            "max_absolute_error": max(
                (item["absolute_error"] for item in element_records), default=0.0
            ),
            "max_relative_error": max(
                (item["relative_error"] for item in element_records), default=0.0
            ),
        }

    scalar_records = list(loss_differences.values()) + list(
        gradient_norm_differences.values()
    )
    scalar_ok = all(
        _within(item, absolute_tolerance, relative_tolerance) for item in scalar_records
    )
    key_ok = all(
        item["max_absolute_error"] <= absolute_tolerance
        or item["max_relative_error"] <= relative_tolerance
        for item in key_gradient_differences.values()
    )
    c0_forward = c0["runtime"]
    forward_mapping_ok = bool(
        c0_forward.get("native_mask_token_count") == 4
        and c0_forward.get("original_supervised_mask_token") == 0
        and c0_forward.get("native_candidate_decoder_calls", 0) > 0
    )
    passed = (
        all(shared_identity.values())
        and scalar_ok
        and key_ok
        and forward_mapping_ok
        and legacy.get("status") == "complete"
        and c0.get("status") == "complete"
    )
    return {
        "status": "pass" if passed else "fail",
        "absolute_tolerance": absolute_tolerance,
        "relative_tolerance": relative_tolerance,
        "shared_identity": shared_identity,
        "forward_mapping": {
            "legacy": "decoder forward(multimask_output=False) -> training token0",
            "c0": "decoder predict_masks(tokens0..3) -> training token0",
            "verified": forward_mapping_ok,
        },
        "loss_components": loss_differences,
        "gradient_group_norms": gradient_norm_differences,
        "key_parameter_gradients": key_gradient_differences,
    }


def summarize_c1_scale(c0: dict[str, Any], c1: dict[str, Any]) -> dict[str, Any]:
    """Report C1 scale/gradient ratios without treating them as tuning criteria."""
    if c1.get("status") != "complete":
        return {"status": "fail", "reason": "C1 smoke report is not complete"}
    c0_runtime = c0["runtime"]
    c1_runtime = c1["runtime"]
    forward_identity = {
        "optimizer_updates": c0_runtime["optimizer_steps"] == c1_runtime["optimizer_steps"],
        "crop_batches_seen": c0_runtime["crop_batches_seen"]
        == c1_runtime["crop_batches_seen"],
        "decoder_calls": c0_runtime.get("native_candidate_decoder_calls")
        == c1_runtime.get("native_candidate_decoder_calls"),
        "prompt_count": c0_runtime.get("native_candidate_prompt_count")
        == c1_runtime.get("native_candidate_prompt_count"),
        "native_mask_token_count": c0_runtime.get("native_mask_token_count")
        == c1_runtime.get("native_mask_token_count")
        == 4,
        "supervised_token": c0_runtime.get("original_supervised_mask_token")
        == c1_runtime.get("original_supervised_mask_token")
        == 0,
        "checkpoint_sha256": c0["initialization"]["checkpoint_sha256"]
        == c1["initialization"]["checkpoint_sha256"],
        "manifest_sha256": c0["data"]["manifest_sha256"]
        == c1["data"]["manifest_sha256"],
        "coverage_manifest_sha256": c0["data"]["coverage"]["sha256"]
        == c1["data"]["coverage"]["sha256"],
        "seed": c0["determinism"]["seed"] == c1["determinism"]["seed"],
        "optimizer": c0["training_configuration"]["optimizer"]
        == c1["training_configuration"]["optimizer"],
        "scheduler": c0["training_configuration"]["scheduler"]
        == c1["training_configuration"]["scheduler"],
        "data_order": c0["training_configuration"]["data_order"]
        == c1["training_configuration"]["data_order"],
    }
    c0_norms = c0_runtime["gradient_audit"]["group_l2_mean"]
    c1_norms = c1_runtime["gradient_audit"]["group_l2_mean"]
    ratios = {
        name: (
            float(c1_norms[name]) / float(c0_norms[name])
            if float(c0_norms[name]) != 0.0
            else (math.inf if float(c1_norms[name]) != 0.0 else 1.0)
        )
        for name in sorted(c0_norms)
    }
    audit = c1_runtime.get("candidate_loss_audit")
    return {
        "status": "complete" if all(forward_identity.values()) and audit else "fail",
        "forward_identity": forward_identity,
        "gradient_norm_ratio_C1_over_C0": ratios,
        "candidate_loss_audit": audit,
        "interpretation_boundary": (
            "Scale audit only. Coefficients are frozen and must not be changed from this report."
        ),
    }
