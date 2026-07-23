#!/usr/bin/env python3
"""Freeze and attest a train-only reconstructed C1 seed-1337 epoch-5 state.

This tool is the mandatory barrier between p1--6 reconstruction training and
any p7/p8 inference.  It derives a weights-only inference checkpoint from the
already retained complete state, copies immutable input records, and writes a
canonical tensor identity manifest.  It never constructs a dataset or runs a
model forward pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROTOCOL = "tnbc_c1_seed1337_reconstructed_epoch5_v1"
INIT_SHA = "44a3cb3e93051301d789e44f93769588abfa727d7a174b0270f55305ef023781"


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


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_torch_save(path: Path, payload: Any, torch_module) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch_module.save(payload, temporary)
    os.replace(temporary, path)


def canonical_tensor_hash(state: dict[str, Any], torch_module) -> tuple[str, dict[str, Any]]:
    digest = hashlib.sha256()
    summary: dict[str, Any] = {}
    for section in ("model", "model1"):
        values = state.get(section)
        if not isinstance(values, dict):
            raise ValueError(f"checkpoint has no {section} state dict")
        names: list[str] = []
        elements = 0
        for name in sorted(values):
            tensor = values[name]
            if not torch_module.is_tensor(tensor):
                raise ValueError(f"{section}.{name} is not a tensor")
            raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch_module.uint8).numpy().tobytes()
            header = json.dumps(
                {"section": section, "name": str(name), "dtype": str(tensor.dtype), "shape": list(tensor.shape)},
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            digest.update(len(header).to_bytes(8, "little")); digest.update(header)
            digest.update(len(raw).to_bytes(8, "little")); digest.update(raw)
            names.append(str(name)); elements += int(tensor.numel())
        summary[section] = {
            "tensor_count": len(names), "element_count": elements,
            "first_keys": names[:12], "last_keys": names[-12:],
        }
    return digest.hexdigest(), summary


def git_value(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--training-summary", required=True, type=Path)
    parser.add_argument("--full-state", required=True, type=Path)
    parser.add_argument("--full-declaration", required=True, type=Path)
    parser.add_argument("--screen-config", required=True, type=Path)
    parser.add_argument("--train-manifest", required=True, type=Path)
    parser.add_argument("--coverage-manifest", required=True, type=Path)
    parser.add_argument("--initialization-checkpoint", required=True, type=Path)
    parser.add_argument("--input-audit", required=True, type=Path)
    args = parser.parse_args()

    import torch

    repo = args.repo_root.resolve()
    run_root = args.run_root.resolve()
    summary_path = args.training_summary.resolve()
    state_path = args.full_state.resolve()
    declaration_path = args.full_declaration.resolve()
    screen_path = args.screen_config.resolve()
    train_path = args.train_manifest.resolve()
    coverage_path = args.coverage_manifest.resolve()
    init_path = args.initialization_checkpoint.resolve()
    input_audit_path = args.input_audit.resolve()
    for path in (summary_path, state_path, declaration_path, screen_path, train_path, coverage_path, init_path, input_audit_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if run_root != summary_path.parent:
        raise ValueError("training summary must live directly in the reconstruction run root")
    output_dir = run_root / "epoch5_frozen"
    if output_dir.exists():
        raise FileExistsError(f"epoch-5 frozen output already exists: {output_dir}")
    summary = read_json(summary_path)
    declaration = read_json(declaration_path)
    input_audit = read_json(input_audit_path)
    full_sha = sha256_file(state_path)
    if input_audit.get("status") != "pass":
        raise ValueError("reconstruction input audit did not pass")
    checks = {
        "summary_complete": summary.get("status") == "complete",
        "summary_stage": summary.get("stage") == "formal_tnbc_c1_seed1337_reconstruction_5epoch",
        "summary_protocol": summary.get("protocol") == PROTOCOL,
        "summary_seed": int(summary.get("determinism", {}).get("seed", -1)) == 1337,
        "summary_arm": summary.get("training_configuration", {}).get("arm") == "c1",
        "summary_train_only_attestation": summary.get("sealed_data_attestation", {}).get("TNBC_p7_p11_accessed") is False,
        "summary_no_development_before_freeze": summary.get("evaluation_plan", {}).get("epochs_1_to_5", "").startswith("strict p7-p8 diagnosis is prohibited"),
        "budget": (
            int(summary.get("actual_attempted_crop_batches", -1)) == 1350
            and int(summary.get("planned_attempted_crop_batches", -1)) == 1350
            and len(summary.get("epochs", [])) == 5
            and all(int(row.get("epoch", -1)) == index for index, row in enumerate(summary.get("epochs", []), 1))
        ),
        "only_epoch5_full_state": (
            summary.get("checkpoint_retention") == "epoch5_full_state_only"
            and sum(bool(row.get("full_checkpoint_retained")) for row in summary.get("epochs", [])) == 1
            and bool(summary.get("epochs", [])[-1].get("full_checkpoint_retained"))
        ),
        "full_state_epoch5": int(declaration.get("epoch", -1)) == 5 and int(summary.get("epochs", [])[-1].get("epoch", -1)) == 5,
        "full_state_hash": declaration.get("checkpoint_sha256") == full_sha,
        "full_state_protocol": declaration.get("protocol") == PROTOCOL and declaration.get("arm") == "c1",
        "full_state_reconstructed_classification": declaration.get("classification") == "historical_exploratory_reconstructed",
        "initialization_sha": sha256_file(init_path) == INIT_SHA and summary.get("initialization", {}).get("checkpoint_sha256") == INIT_SHA,
        "screen_config_hash": summary.get("screen_config", {}).get("sha256") == sha256_file(screen_path),
        "train_manifest_hash": summary.get("data", {}).get("manifest_sha256") == sha256_file(train_path),
    }
    if not all(checks.values()):
        raise ValueError("reconstructed epoch-5 freeze validation failed: " + json.dumps(checks, sort_keys=True))
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    if not all(key in state for key in ("model", "model1", "optimizer", "scheduler", "rng_state")):
        raise ValueError("epoch-5 complete state lacks model/model1/optimizer/scheduler/RNG")
    canonical_hash, tensor_summary = canonical_tensor_hash(state, torch)
    weights = {
        "schema_version": 1,
        "phase": state.get("phase"),
        "protocol": PROTOCOL,
        "dataset": "tnbc",
        "arm": "c1_reconstructed",
        "lineage": "reconstructed C1 seed-1337 lineage",
        "epoch": 5,
        "model": state["model"],
        "model1": state["model1"],
        "source_full_state_path": str(state_path),
        "source_full_state_sha256": full_sha,
        "canonical_model_model1_tensor_sha256": canonical_hash,
        "train_manifest": state.get("train_manifest"),
        "coverage": state.get("coverage"),
        "screen_config": state.get("screen_config"),
        "texture_memory_bank_list": [],
        "embedded_texture_bank_loaded": False,
    }
    weights_path = output_dir / "epoch5_model_model1_weights.pth"
    atomic_torch_save(weights_path, weights, torch)
    weights_sha = sha256_file(weights_path)
    copied_dir = output_dir / "immutable_inputs"
    copies = {}
    for label, source in {
        "screen_config": screen_path,
        "train_manifest": train_path,
        "coverage_manifest": coverage_path,
        "input_audit": input_audit_path,
        "full_state_declaration": declaration_path,
    }.items():
        target = copied_dir / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copies[label] = {"source": str(source), "copy": str(target), "sha256": sha256_file(target)}
    manifest = {
        "schema_version": 1,
        "protocol": "tnbc_c1_seed1337_reconstructed_epoch5_freeze_v1",
        "status": "frozen_before_development_access",
        "lineage": "reconstructed C1 seed-1337 lineage",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "training": {"summary_path": str(summary_path), "summary_sha256": sha256_file(summary_path), "epoch": 5, "attempted_crop_batches": 1350, "optimizer_updates": summary.get("actual_optimizer_updates"), "no_prompt_batches": summary.get("actual_no_prompt_batch_count")},
        "complete_state": {"path": str(state_path), "sha256": full_sha, "bytes": state_path.stat().st_size, "declaration_path": str(declaration_path), "declaration_sha256": sha256_file(declaration_path)},
        "weights_only": {"path": str(weights_path), "sha256": weights_sha, "bytes": weights_path.stat().st_size, "canonical_model_model1_tensor_sha256": canonical_hash, "state_dict": tensor_summary},
        "initialization": {"path": str(init_path), "sha256": sha256_file(init_path), "approved_sha256": INIT_SHA},
        "immutable_input_copies": copies,
        "rng_state": {"present": isinstance(state.get("rng_state"), dict), "keys": sorted(state.get("rng_state", {}).keys()) if isinstance(state.get("rng_state"), dict) else []},
        "repository": {"branch": git_value(repo, "branch", "--show-current"), "commit": git_value(repo, "rev-parse", "HEAD")},
        "environment": {"python": sys.version, "torch": torch.__version__, "torch_cuda": torch.version.cuda, "platform": platform.platform()},
        "checks": checks,
        "development_access": "not performed by this tool; only permitted after this manifest exists",
    }
    atomic_json(output_dir / "frozen_epoch5_manifest.json", manifest)
    atomic_json(
        output_dir / "epoch5_weights_declaration.json",
        {
            "schema_version": 1,
            "dataset": "tnbc",
            "classification": "historical_exploratory_reconstructed",
            "checkpoint_kind": "weights_only_model_and_model1",
            "checkpoint_path": str(weights_path),
            "checkpoint_sha256": weights_sha,
            "canonical_model_model1_tensor_sha256": canonical_hash,
            "lineage": "reconstructed C1 seed-1337 lineage",
            "epoch": 5,
            "protocol": PROTOCOL,
            "arm": "c1_reconstructed",
            "source_full_state_sha256": full_sha,
            "training_manifest": state.get("train_manifest"),
            "p7_p8_exposure": "permitted only after this frozen manifest was written; read-only diagnosis",
            "p9_p11_exposure": "none",
            "texture_memory_bank_loaded": False,
        },
    )
    print(json.dumps({"status": "frozen_before_development_access", "manifest": str(output_dir / "frozen_epoch5_manifest.json"), "weights": str(weights_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
