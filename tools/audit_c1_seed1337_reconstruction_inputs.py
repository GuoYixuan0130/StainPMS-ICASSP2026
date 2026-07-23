#!/usr/bin/env python3
"""Fail closed before reconstructing the missing seed-1337 C1 epoch-5 state.

This is intentionally read-only.  It compares the original run metadata with
the frozen reconstruction configuration and train-only inputs; it neither
constructs a dataset nor reads p7/p8 images or results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_PROTOCOL = "tnbc_c1_seed1337_reconstructed_epoch5_v1"
EXPECTED_TRAIN_PROTOCOL = "tnbc_stainpms_prepared_continuity_v1_phase1_train"
EXPECTED_INIT_SHA = "44a3cb3e93051301d789e44f93769588abfa727d7a174b0270f55305ef023781"


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


def json_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def command_value(command: list[Any], flag: str) -> str | None:
    try:
        index = [str(value) for value in command].index(flag)
    except ValueError:
        return None
    return str(command[index + 1]) if index + 1 < len(command) else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-training-summary", required=True, type=Path)
    parser.add_argument("--screen-config", required=True, type=Path)
    parser.add_argument("--train-manifest", required=True, type=Path)
    parser.add_argument("--coverage-manifest", required=True, type=Path)
    parser.add_argument("--initialization-checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite input audit: {output}")
    summary_path = args.source_training_summary.resolve()
    screen_path = args.screen_config.resolve()
    train_path = args.train_manifest.resolve()
    coverage_path = args.coverage_manifest.resolve()
    init_path = args.initialization_checkpoint.resolve()
    for path in (summary_path, screen_path, train_path, coverage_path, init_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    summary = read_json(summary_path)
    screen = read_json(screen_path)
    train = read_json(train_path)
    coverage = read_json(coverage_path)
    config = summary.get("training_configuration", {})
    objective = config.get("objective", {})
    # The accepted original C1 summaries predate C2-AR and therefore omit the
    # `c2_ar` object entirely.  Absence is an auditable zero here: C2 did not
    # exist in that code path, rather than being an unknown nonzero setting.
    source_c2 = objective.get("c2_ar", {})
    if not isinstance(source_c2, dict):
        source_c2 = {}
    optimizer = config.get("optimizer", {})
    scheduler = config.get("scheduler", {})
    determinism = summary.get("determinism", {})
    data = summary.get("data", {})
    init = summary.get("initialization", {})
    command = list(summary.get("command", []))
    expected_values = {
        "--seed": "1337", "--epochs": "5", "--lr": "1e-5", "--weight_decay": "1e-4",
        "--clip-grad": "0.1", "--crop_size": "256", "--out_size": "256", "--overlap": "32",
        "--b": "1", "--pms_start_epoch": "0", "--iterative_baseline_refresh_every": "20",
        "--pms_loss_coef": "0.5", "--pms_object_weight": "1.0", "--pms_residual_mask_weight": "0.3",
        "--pms_preserve_loss_coef": "1.0", "--pms_gt_match_radius": "8", "--pms_preserve_max_prompts": "20",
        "--stain_min_distance": "12", "--stain_top_k": "20", "--candidate_coverage_tau": "0.1",
        "--candidate_coverage_coefficient": "1.0", "--candidate_quality_coefficient": "1.0",
    }
    expected_flags = {"--verify_manifest_hashes", "--texture", "--context", "--use_pms", "--pms_self_bootstrap", "--coverage_accumulate", "--pms_preserve_covered"}
    checks = {
        "source_summary_complete_c1_seed1337": (
            summary.get("status") == "complete"
            and config.get("arm") == "c1"
            and int(determinism.get("seed", -1)) == 1337
        ),
        "source_train_is_p1_p6": (
            data.get("protocol_id") == EXPECTED_TRAIN_PROTOCOL
            and int(data.get("record_count", -1)) == 30
        ),
        "source_init_sha_is_approved": init.get("checkpoint_sha256") == EXPECTED_INIT_SHA,
        "frozen_config_protocol": screen.get("protocol_id") == EXPECTED_PROTOCOL,
        "frozen_config_seed": int(screen.get("determinism", {}).get("seed", -1)) == 1337,
        "frozen_config_budget": (
            int(screen.get("optimization", {}).get("epochs", -1)) == 5
            and int(screen.get("optimization", {}).get("attempted_crop_batches_per_epoch", -1)) == 270
            and int(screen.get("optimization", {}).get("planned_attempted_crop_batches", -1)) == 1350
        ),
        "frozen_config_c1_objective": (
            float(screen.get("c1_objective", {}).get("candidate_coverage_tau", float("nan"))) == 0.1
            and float(screen.get("c1_objective", {}).get("coverage_coefficient", float("nan"))) == 1.0
            and float(screen.get("c1_objective", {}).get("quality_coefficient", float("nan"))) == 1.0
            and float(screen.get("c1_objective", {}).get("c2_exclusivity_coefficient", float("nan"))) == 0.0
            and float(screen.get("c1_objective", {}).get("c2_utility_coefficient", float("nan"))) == 0.0
        ),
        "source_optimizer_matches_frozen": (
            optimizer.get("type") == "AdamW"
            and float(optimizer.get("learning_rate", float("nan"))) == 1e-5
            and float(optimizer.get("weight_decay", float("nan"))) == 1e-4
            and scheduler.get("type") == "MultiStepLR"
            and list(scheduler.get("milestones", [])) == [80, 140, 200]
        ),
        "source_c1_objective_matches_frozen": (
            float(objective.get("candidate_coverage_tau", float("nan"))) == 0.1
            and float(objective.get("candidate_coverage_coefficient", float("nan"))) == 1.0
            and float(objective.get("candidate_quality_coefficient", float("nan"))) == 1.0
            and float(source_c2.get("selected_mask_exclusivity_coefficient", 0.0)) == 0.0
            and float(source_c2.get("unique_tp_utility_coefficient", 0.0)) == 0.0
        ),
        "source_command_preserves_original_core_recipe": (
            all(command_value(command, flag) == value for flag, value in expected_values.items())
            and all(flag in command for flag in expected_flags)
            and command_value(command, "--load") == "unclockwise"
            and command_value(command, "--sam_config") == "sam2_hiera_l"
        ),
        "train_manifest_matches_source": sha256_file(train_path) == data.get("manifest_sha256"),
        "train_manifest_scope": (
            train.get("protocol_id") == EXPECTED_TRAIN_PROTOCOL
            and len(train.get("records", [])) == 30
            and sorted({int(row.get("patient", -1)) for row in train.get("records", [])}) == [1, 2, 3, 4, 5, 6]
        ),
        "coverage_manifest_is_train_only": (
            coverage.get("dataset") == "tnbc"
            and coverage.get("train_manifest", {}).get("sha256") == sha256_file(train_path)
        ),
        "initialization_file_sha_matches_approved": sha256_file(init_path) == EXPECTED_INIT_SHA,
    }
    payload = {
        "schema_version": 1,
        "protocol": "tnbc_c1_seed1337_reconstruction_input_audit_v1",
        "status": "pass" if all(checks.values()) else "fail",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "read-only pre-training provenance audit; no dataset construction, p7/p8 reading, inference, or training",
        "checks": checks,
        "source_original_c1": {"training_summary": str(summary_path), "sha256": sha256_file(summary_path)},
        "frozen_reconstruction_config": {"path": str(screen_path), "sha256": sha256_file(screen_path)},
        "train_manifest": {"path": str(train_path), "sha256": sha256_file(train_path)},
        "coverage_manifest": {"path": str(coverage_path), "sha256": sha256_file(coverage_path)},
        "initialization": {"path": str(init_path), "sha256": sha256_file(init_path)},
        "source_c2_compatibility": {
            "c2_object_present_in_original_c1_summary": bool(source_c2),
            "missing_c2_fields_interpreted_as": 0.0,
            "reason": "the original C1 run predates the C2-AR implementation",
        },
    }
    payload["input_fingerprint_sha256"] = json_sha256(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "output": str(output)}, ensure_ascii=False))
    return 0 if payload["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
