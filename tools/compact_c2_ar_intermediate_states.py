"""Remove only superseded C2-EU epoch-1--4 recovery states.

The completed C2-EU two-seed runs have already been diagnosed at their fixed
epoch-5 state.  This tool preserves that epoch-5 full state, its declaration,
the five-epoch metrics, manifests, logs, and SHA256 records.  It never touches
new C2-E/U runs, C0/C1 references, or any dataset file.

Default mode is read-only.  ``--apply`` is required for deletion and repeats
all safety checks before unlinking the explicitly listed epoch-1--4 state and
declaration pairs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def build_plan(run_roots: list[Path]) -> dict[str, Any]:
    plans: list[dict[str, Any]] = []
    for raw_root in run_roots:
        root = raw_root.resolve()
        arm_root = root / "c2_ar"
        summary_path = arm_root / "training_summary.json"
        summary = read_json(summary_path)
        if summary.get("status") != "complete" or summary.get("stage") != "formal_tnbc_c2_ar_5epoch":
            raise ValueError(f"not a completed formal C2-EU run: {root}")
        # The completed C2-EU runs predate the explicit
        # ``checkpoint_retention`` field.  A missing value is therefore a
        # legacy-schema case, not evidence that the run used a different
        # retention policy.  Infer all-state retention only after the stricter
        # checks below confirm five contiguous, hash-verified full states.
        declared_retention = summary.get("checkpoint_retention")
        if declared_retention not in (None, "all_full_states"):
            raise ValueError(f"C2-EU source does not declare all-full-state retention: {root}")
        records = summary.get("epochs", [])
        if not isinstance(records, list) or len(records) != 5:
            raise ValueError(f"C2-EU summary must contain five epochs: {root}")
        checkpoints = (arm_root / "checkpoints").resolve()
        declarations = (arm_root / "checkpoint_declarations").resolve()
        retained: dict[str, Any] | None = None
        remove: list[dict[str, Any]] = []
        for expected_epoch, record in enumerate(records, start=1):
            if int(record.get("epoch", -1)) != expected_epoch:
                raise ValueError(f"non-contiguous C2-EU epoch record at {root}")
            checkpoint = Path(str(record.get("checkpoint_path", ""))).resolve()
            declaration = Path(str(record.get("checkpoint_declaration", ""))).resolve()
            if checkpoint.parent != checkpoints or declaration.parent != declarations:
                raise ValueError(f"unsafe checkpoint/declaration target under {root}")
            if not checkpoint.is_file() or not declaration.is_file():
                raise FileNotFoundError(f"missing formal C2-EU state/declaration for epoch {expected_epoch}: {root}")
            declared = read_json(declaration)
            actual_sha = sha256_file(checkpoint)
            if declared.get("checkpoint_sha256") != actual_sha or record.get("checkpoint_sha256") != actual_sha:
                raise ValueError(f"checkpoint SHA256 mismatch at {checkpoint}")
            if declared.get("phase") != "2A-warmstart-c2-ar" or declared.get("protocol") != "tnbc_c2_ar_two_seed_v1" or declared.get("arm") != "c2_ar" or int(declared.get("epoch", -1)) != expected_epoch:
                raise ValueError(f"invalid C2-EU checkpoint declaration: {declaration}")
            payload = {"epoch": expected_epoch, "checkpoint": str(checkpoint), "declaration": str(declaration), "checkpoint_sha256": actual_sha, "checkpoint_bytes": checkpoint.stat().st_size, "declaration_bytes": declaration.stat().st_size}
            if expected_epoch == 5:
                retained = payload
            else:
                remove.append(payload)
        assert retained is not None
        plans.append(
            {
                "run_root": str(root),
                "source_checkpoint_retention": (
                    declared_retention
                    if declared_retention is not None
                    else "legacy_inferred_from_five_complete_states"
                ),
                "retained_epoch5": retained,
                "remove_epoch1_to4": remove,
            }
        )
    reclaim = sum(item["checkpoint_bytes"] + item["declaration_bytes"] for plan in plans for item in plan["remove_epoch1_to4"])
    return {"schema_version": 1, "protocol": "c2_eu_epoch5_retention_compaction_v1", "status": "planned_read_only", "runs": plans, "projected_reclaimable_bytes": reclaim, "projected_reclaimable_gib": reclaim / (1024 ** 3), "deletion_scope": "only validated C2-EU epoch-1--4 .pth and matching declaration JSON; fixed epoch-5 state and all summaries/logs/manifests are retained"}


def apply(plan: dict[str, Any]) -> dict[str, Any]:
    if plan.get("status") != "planned_read_only":
        raise ValueError("--apply requires a fresh planned_read_only plan")
    removed: list[dict[str, Any]] = []
    for run in plan["runs"]:
        retained = run["retained_epoch5"]
        retained_checkpoint = Path(retained["checkpoint"])
        retained_declaration = Path(retained["declaration"])
        if not retained_checkpoint.is_file() or not retained_declaration.is_file() or sha256_file(retained_checkpoint) != retained["checkpoint_sha256"]:
            raise RuntimeError(f"epoch-5 retention verification failed: {run['run_root']}")
        for item in run["remove_epoch1_to4"]:
            checkpoint, declaration = Path(item["checkpoint"]), Path(item["declaration"])
            if not checkpoint.is_file() or not declaration.is_file() or checkpoint.stat().st_size != item["checkpoint_bytes"] or sha256_file(checkpoint) != item["checkpoint_sha256"]:
                raise RuntimeError(f"deletion target changed after planning: {checkpoint}")
        for item in run["remove_epoch1_to4"]:
            checkpoint, declaration = Path(item["checkpoint"]), Path(item["declaration"])
            checkpoint.unlink(); declaration.unlink()
            removed.append({"checkpoint": str(checkpoint), "declaration": str(declaration), "bytes": item["checkpoint_bytes"] + item["declaration_bytes"]})
    output = dict(plan); output["status"] = "complete_epoch1_to4_removed"; output["removed"] = removed; output["removed_bytes"] = sum(item["bytes"] for item in removed); return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", action="append", required=True)
    parser.add_argument("--plan-output", required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    roots = [Path(value) for value in args.run_root]
    plan_path = Path(args.plan_output).resolve()
    plan = build_plan(roots)
    if args.apply:
        plan = apply(plan)
    write_json_atomic(plan_path, plan)
    print(json.dumps({"status": plan["status"], "plan": str(plan_path), "reclaimable_gib": plan.get("projected_reclaimable_gib")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
