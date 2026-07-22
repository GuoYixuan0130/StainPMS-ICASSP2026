"""Safely compact completed C0/C1 full states under the superseding PQ-best rule.

Default invocation is read-only: it identifies each arm's equal-patient-macro
PQ-best epoch from the already frozen p7/p8 diagnosis table and calculates
reclaimable bytes.  `--materialize-best` writes and validates weights-only
model/model1 copies but does not delete anything.  Deletion is a separate,
explicit `--delete-verified-source-states` action and is restricted to the
validated `epoch_*.pth` files inside the supplied screen root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stainpms.phase2a_pqbest import choose_pq_best


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _screen_records(screen_root: Path) -> dict[str, list[dict[str, Any]]]:
    metrics_path = screen_root / "summary" / "epoch_metrics.json"
    metrics = read_json(metrics_path)
    records: dict[str, list[dict[str, Any]]] = {}
    for arm in ("c0", "c1"):
        value = metrics.get(arm)
        if not isinstance(value, list) or len(value) != 5:
            raise ValueError(f"screen metrics must contain exactly five {arm} diagnosis records")
        records[arm] = value
    return records


def build_plan(screen_root: Path) -> dict[str, Any]:
    screen_root = screen_root.resolve()
    records = _screen_records(screen_root)
    arms: dict[str, dict[str, Any]] = {}
    for arm in ("c0", "c1"):
        training_summary_path = screen_root / arm / "training_summary.json"
        training_summary = read_json(training_summary_path)
        selection = choose_pq_best(records[arm])
        epoch = int(selection["selected_epoch"])
        epoch_records = training_summary.get("epochs", [])
        if len(epoch_records) != 5:
            raise ValueError(f"{arm} training summary does not retain five epoch records")
        source_record = next(
            (record for record in epoch_records if int(record.get("epoch", -1)) == epoch), None
        )
        if source_record is None:
            raise ValueError(f"{arm} selected epoch {epoch} has no training-state record")
        source = Path(str(source_record.get("checkpoint_path", ""))).resolve()
        expected_parent = (screen_root / arm / "checkpoints").resolve()
        if source.parent != expected_parent or not source.name.startswith("epoch_") or source.suffix != ".pth":
            raise ValueError(f"{arm} selected source checkpoint is outside its validated epoch-state directory: {source}")
        if not source.is_file():
            raise FileNotFoundError(f"{arm} selected source checkpoint is missing: {source}")
        source_sha = sha256_file(source)
        if source_sha != str(source_record.get("checkpoint_sha256", "")):
            raise ValueError(f"{arm} selected source SHA256 differs from the training summary")
        source_paths = sorted(expected_parent.glob("epoch_*.pth"))
        if len(source_paths) != 5:
            raise ValueError(f"{arm} checkpoint directory must contain exactly five original epoch states")
        source_sizes = {str(path): path.stat().st_size for path in source_paths}
        arms[arm] = {
            "selected_epoch": epoch,
            "selected_patient_macro_pq": selection["selected_patient_macro_pq"],
            "pq_by_epoch": selection["epoch_patient_macro_pq"],
            "source_checkpoint": str(source),
            "source_checkpoint_sha256": source_sha,
            "source_epoch_states": [str(path) for path in source_paths],
            "source_epoch_state_bytes": source_sizes,
            "weights_only_target": str((screen_root / arm / "best_pq" / "model_model1_weights.pth").resolve()),
            "declaration_target": str((screen_root / arm / "best_pq" / "declaration.json").resolve()),
            "training_summary": str(training_summary_path.resolve()),
        }
    all_source_bytes = sum(sum(item["source_epoch_state_bytes"].values()) for item in arms.values())
    return {
        "schema_version": 1,
        "protocol": "tnbc_c0_c1_pqbest_retention_compaction_v1",
        "screen_root": str(screen_root),
        "screen_metrics_path": str((screen_root / "summary" / "epoch_metrics.json").resolve()),
        "screen_metrics_sha256": sha256_file(screen_root / "summary" / "epoch_metrics.json"),
        "arms": arms,
        "source_full_epoch_state_bytes": all_source_bytes,
        "source_full_epoch_state_gib": all_source_bytes / (1024**3),
        "status": "planned_read_only",
        "deletion_authority": (
            "No source state is deleted by default. Only --delete-verified-source-states may "
            "remove the validated epoch_*.pth source states after weights-only validation."
        ),
    }


def materialize_best(plan: dict[str, Any]) -> dict[str, Any]:
    import torch

    materialized: dict[str, Any] = {}
    for arm, item in plan["arms"].items():
        source = Path(item["source_checkpoint"])
        target = Path(item["weights_only_target"])
        declaration = Path(item["declaration_target"])
        # The full source was produced by this local formal screen and its hash
        # was verified in build_plan.  It contains RNG/optimizer metadata, so
        # the explicit broader loader is required on PyTorch >=2.6.
        payload = torch.load(source, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or "model" not in payload or "model1" not in payload:
            raise ValueError(f"{arm} source checkpoint lacks model/model1 state")
        weights = {
            "schema_version": 1,
            "phase": "2A-warmstart-retention-compaction",
            "protocol": plan["protocol"],
            "dataset": "tnbc",
            "arm": arm,
            "model": payload["model"],
            "model1": payload["model1"],
            "selected_epoch": item["selected_epoch"],
            "selected_patient_macro_pq": item["selected_patient_macro_pq"],
            "source_checkpoint": str(source),
            "source_checkpoint_sha256": item["source_checkpoint_sha256"],
            "screen_metrics_sha256": plan["screen_metrics_sha256"],
            "texture_memory_bank_list": [],
            "embedded_texture_bank_loaded": False,
        }
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".tmp")
        torch.save(weights, temporary)
        os.replace(temporary, target)
        target_sha = sha256_file(target)
        verified = torch.load(target, map_location="cpu", weights_only=True)
        for key in ("model", "model1"):
            if set(verified[key]) != set(payload[key]):
                raise RuntimeError(f"{arm} weights-only {key} keys differ from validated source")
            for name in payload[key]:
                if not torch.equal(verified[key][name], payload[key][name]):
                    raise RuntimeError(f"{arm} weights-only tensor differs from source: {key}.{name}")
        declaration_payload = {
            "schema_version": 1,
            "dataset": "tnbc",
            "classification": "historical_exploratory",
            "checkpoint_path": str(target),
            "checkpoint_sha256": target_sha,
            "checkpoint_kind": "weights_only_model_and_model1",
            "selection_history": "development equal-patient-macro PQ-best; exact tie retained earlier epoch",
            "selected_epoch": item["selected_epoch"],
            "selected_patient_macro_pq": item["selected_patient_macro_pq"],
            "source_checkpoint": str(source),
            "source_checkpoint_sha256": item["source_checkpoint_sha256"],
            "screen_metrics_path": plan["screen_metrics_path"],
            "screen_metrics_sha256": plan["screen_metrics_sha256"],
            "p7_p8_exposure": "development checkpoint selection only",
            "p9_p11_exposure": "none",
            "embedded_texture_bank_loaded": False,
        }
        write_json_atomic(declaration, declaration_payload)
        materialized[arm] = {
            "weights_only_target": str(target),
            "weights_only_sha256": target_sha,
            "weights_only_bytes": target.stat().st_size,
            "declaration": str(declaration),
            "tensor_identity_verified": True,
        }
        del verified, weights, payload
    plan = dict(plan)
    plan["status"] = "materialized_verified_not_deleted"
    plan["materialized"] = materialized
    retained = sum(item["weights_only_bytes"] for item in materialized.values())
    plan["retained_weights_only_bytes"] = retained
    plan["projected_reclaimable_bytes"] = plan["source_full_epoch_state_bytes"] - retained
    plan["projected_reclaimable_gib"] = plan["projected_reclaimable_bytes"] / (1024**3)
    return plan


def delete_verified_sources(plan: dict[str, Any]) -> dict[str, Any]:
    if plan.get("status") != "materialized_verified_not_deleted":
        raise ValueError("source deletion requires a materialized_verified_not_deleted plan")
    removed: list[dict[str, Any]] = []
    for arm, item in plan["arms"].items():
        result = plan.get("materialized", {}).get(arm, {})
        target = Path(result.get("weights_only_target", ""))
        if not target.is_file() or sha256_file(target) != result.get("weights_only_sha256"):
            raise RuntimeError(f"{arm} weights-only target is absent or changed; source deletion refused")
        expected_parent = Path(plan["screen_root"]) / arm / "checkpoints"
        for raw_source in item["source_epoch_states"]:
            source = Path(raw_source).resolve()
            if source.parent != expected_parent.resolve() or not source.name.startswith("epoch_"):
                raise RuntimeError(f"unsafe source deletion target rejected: {source}")
            if not source.is_file():
                raise FileNotFoundError(f"validated source state disappeared before deletion: {source}")
            expected_size = item["source_epoch_state_bytes"].get(str(source))
            if source.stat().st_size != expected_size:
                raise RuntimeError(f"source state size changed after planning: {source}")
        for raw_source in item["source_epoch_states"]:
            source = Path(raw_source).resolve()
            size = source.stat().st_size
            source.unlink()
            removed.append({"path": str(source), "bytes": size})
    plan = dict(plan)
    plan["status"] = "complete_sources_removed"
    plan["removed_source_epoch_states"] = removed
    plan["removed_source_epoch_state_bytes"] = sum(item["bytes"] for item in removed)
    return plan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen-root", required=True)
    parser.add_argument("--plan-output", default="")
    parser.add_argument("--materialize-best", action="store_true")
    parser.add_argument("--delete-verified-source-states", action="store_true")
    args = parser.parse_args()
    root = Path(args.screen_root).resolve()
    plan_path = Path(args.plan_output).resolve() if args.plan_output else root / "summary" / "pqbest_retention_plan.json"
    if args.delete_verified_source_states:
        if not plan_path.is_file():
            raise FileNotFoundError("deletion requires the prior materialized plan JSON")
        plan = read_json(plan_path)
        plan = delete_verified_sources(plan)
    else:
        plan = build_plan(root)
        if args.materialize_best:
            plan = materialize_best(plan)
    write_json_atomic(plan_path, plan)
    print(
        json.dumps(
            {
                "status": plan["status"],
                "plan": str(plan_path),
                "projected_reclaimable_gib": plan.get("projected_reclaimable_gib"),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
