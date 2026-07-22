"""Fail closed unless one new PQ-best ablation arm matches preserved C0 training.

The existing C0 was produced under the former fixed-epoch retention protocol;
the approved coverage-only/quality-only arms use the new low-storage PQ-best
protocol.  This verifier therefore compares only the invariant execution
contract, not the intentionally different retention/model-selection fields.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def verify(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []

    def same(label: str, left: Any, right: Any) -> None:
        if left != right:
            failures.append({"field": label, "c0": left, "candidate": right})

    same("dataset", reference.get("dataset"), candidate.get("dataset"))
    same(
        "train_manifest_sha256",
        reference.get("data", {}).get("manifest_sha256"),
        candidate.get("data", {}).get("manifest_sha256"),
    )
    same(
        "coverage_manifest_sha256",
        reference.get("data", {}).get("coverage", {}).get("sha256"),
        candidate.get("data", {}).get("coverage", {}).get("sha256"),
    )
    same("seed", reference.get("determinism", {}).get("seed"), candidate.get("determinism", {}).get("seed"))
    reference_epochs = reference.get("epochs", [])
    candidate_epochs = candidate.get("epochs", [])
    if len(reference_epochs) != 5 or len(candidate_epochs) != 5:
        failures.append(
            {
                "field": "epoch_record_count",
                "c0": len(reference_epochs),
                "candidate": len(candidate_epochs),
            }
        )
    checks: list[dict[str, Any]] = []
    for epoch, (left, right) in enumerate(zip(reference_epochs, candidate_epochs, strict=False), start=1):
        local: list[dict[str, Any]] = []
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
                local.append({"field": field, "c0": left.get(field), "candidate": right.get(field)})
        for label, record in (("c0", left), ("candidate", right)):
            attempted = int(record.get("attempted_crop_batches", -1))
            updates = int(record.get("effective_optimizer_updates", -1))
            no_prompt = int(record.get("no_prompt_batch_count", -1))
            if attempted != 270:
                local.append({"field": f"{label}_attempted_crop_batches", "observed": attempted, "expected": 270})
            if updates + no_prompt != attempted:
                local.append(
                    {
                        "field": f"{label}_update_plus_no_prompt",
                        "updates": updates,
                        "no_prompt": no_prompt,
                        "attempted": attempted,
                    }
                )
            if stable_hash(record.get("no_prompt_batch_indices", [])) != record.get("no_prompt_batch_indices_sha256"):
                local.append({"field": f"{label}_no_prompt_index_hash_self_check"})
        checks.append({"epoch": epoch, "status": "pass" if not local else "fail", "failures": local})
        failures.extend({"epoch": epoch, **item} for item in local)
    for label, report in (("c0", reference), ("candidate", candidate)):
        if int(report.get("planned_attempted_crop_batches", -1)) != 1350:
            failures.append(
                {
                    "field": f"{label}_planned_attempted_crop_batches",
                    "observed": report.get("planned_attempted_crop_batches"),
                    "expected": 1350,
                }
            )
        if int(report.get("actual_attempted_crop_batches", -1)) != 1350:
            failures.append(
                {
                    "field": f"{label}_actual_attempted_crop_batches",
                    "observed": report.get("actual_attempted_crop_batches"),
                    "expected": 1350,
                }
            )
    return {
        "schema_version": 1,
        "protocol": "tnbc_loss_ablation_pqbest_v1",
        "status": "pass" if not failures else "fail",
        "reference": "preserved_C0_continued_training_control",
        "candidate_arm": candidate.get("training_configuration", {}).get("arm"),
        "checks": checks,
        "failures": failures,
        "attestation": {
            "same_attempted_crop_batches": not any("attempted_crop_batches" in item["field"] for item in failures),
            "same_no_prompt_positions": not any("no_prompt" in item["field"] for item in failures),
            "same_effective_optimizer_updates": not any("effective_optimizer_updates" in item["field"] for item in failures),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c0-summary", required=True)
    parser.add_argument("--candidate-summary", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = verify(read_json(Path(args.c0_summary)), read_json(Path(args.candidate_summary)))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps({"status": result["status"], "output": str(output), "failure_count": len(result["failures"])}))
    return 0 if result["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
