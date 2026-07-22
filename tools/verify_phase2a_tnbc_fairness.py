"""Fail closed unless formal TNBC C0/C1 crop-batch execution is identical."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def stable_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def equivalent_value(c0: Any, c1: Any, label: str, failures: list[dict[str, Any]]) -> None:
    if c0 != c1:
        failures.append({"field": label, "c0": c0, "c1": c1})


def verify(c0: dict[str, Any], c1: dict[str, Any]) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    for field in ("protocol", "dataset", "screen_config"):
        equivalent_value(c0.get(field), c1.get(field), field, failures)
    equivalent_value(c0.get("data", {}).get("manifest_sha256"), c1.get("data", {}).get("manifest_sha256"), "train_manifest_sha256", failures)
    equivalent_value(c0.get("data", {}).get("coverage", {}).get("sha256"), c1.get("data", {}).get("coverage", {}).get("sha256"), "coverage_manifest_sha256", failures)
    equivalent_value(c0.get("determinism", {}).get("seed"), c1.get("determinism", {}).get("seed"), "seed", failures)
    c0_epochs = c0.get("epochs", [])
    c1_epochs = c1.get("epochs", [])
    if len(c0_epochs) != 5 or len(c1_epochs) != 5:
        failures.append({"field": "epoch_record_count", "c0": len(c0_epochs), "c1": len(c1_epochs)})
    checks = []
    for index, (left, right) in enumerate(zip(c0_epochs, c1_epochs, strict=False), start=1):
        item_failures: list[dict[str, Any]] = []
        for field in (
            "epoch",
            "attempted_crop_batches",
            "effective_optimizer_updates",
            "no_prompt_batch_count",
            "no_prompt_batch_indices_sha256",
            "optimizer_updates",
            "learning_rate_after_scheduler_step",
            "scheduler_state_after_step",
        ):
            if left.get(field) != right.get(field):
                item_failures.append({"field": field, "c0": left.get(field), "c1": right.get(field)})
        if int(left.get("attempted_crop_batches", -1)) != 270:
            item_failures.append({"field": "c0_attempted_crop_batches", "observed": left.get("attempted_crop_batches"), "expected": 270})
        if int(right.get("attempted_crop_batches", -1)) != 270:
            item_failures.append({"field": "c1_attempted_crop_batches", "observed": right.get("attempted_crop_batches"), "expected": 270})
        for arm, value in (("c0", left), ("c1", right)):
            no_prompt = int(value.get("no_prompt_batch_count", -1))
            updates = int(value.get("effective_optimizer_updates", -1))
            attempted = int(value.get("attempted_crop_batches", -1))
            positions = value.get("no_prompt_batch_indices", [])
            if updates + no_prompt != attempted:
                item_failures.append({"field": f"{arm}_update_plus_no_prompt", "updates": updates, "no_prompt": no_prompt, "attempted": attempted})
            if stable_sha256(positions) != value.get("no_prompt_batch_indices_sha256"):
                item_failures.append({"field": f"{arm}_no_prompt_index_hash_self_check"})
        checks.append({"epoch": index, "status": "pass" if not item_failures else "fail", "failures": item_failures})
        failures.extend({"epoch": index, **failure} for failure in item_failures)
    for arm, payload in (("c0", c0), ("c1", c1)):
        if int(payload.get("planned_attempted_crop_batches", -1)) != 1350:
            failures.append({"field": f"{arm}_planned_attempted_crop_batches", "observed": payload.get("planned_attempted_crop_batches"), "expected": 1350})
        if int(payload.get("runtime", {}).get("crop_batches_seen", -1)) != 1350:
            failures.append({"field": f"{arm}_actual_attempted_crop_batches", "observed": payload.get("runtime", {}).get("crop_batches_seen"), "expected": 1350})
    return {
        "schema_version": 1,
        "protocol": "tnbc_c0_c1_5epoch_exploratory_v1",
        "status": "pass" if not failures else "fail",
        "checks": checks,
        "failures": failures,
        "attestation": {
            "C0_C1_same_attempted_crop_batches": not any("attempted_crop_batches" in item.get("field", "") for item in failures),
            "C0_C1_same_no_prompt_positions": not any("no_prompt" in item.get("field", "") for item in failures),
            "C0_C1_same_effective_optimizer_updates": not any("effective_optimizer_updates" in item.get("field", "") for item in failures),
        },
    }


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c0-summary", required=True)
    parser.add_argument("--c1-summary", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = verify(read_json(Path(args.c0_summary)), read_json(Path(args.c1_summary)))
    write_json_atomic(Path(args.output), result)
    print(json.dumps({"status": result["status"], "output": args.output, "failure_count": len(result["failures"])}))
    return 0 if result["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
