#!/usr/bin/env python3
"""Fail-closed provenance audit for recovery of C1 fixed epoch-5 weights.

This tool is read-only: it loads no dataset, invokes no model inference and
does not alter declarations.  A PQ-best weights-only file is accepted only if
its bytes, embedded provenance, epoch log, save-code commit and C3 source
hash form one explicit epoch-5 chain.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SEEDS = (2027, 1337)


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


def parse_assignment(value: str) -> tuple[int, Path]:
    try:
        raw_seed, raw_path = value.split("=", 1)
        seed = int(raw_seed)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("assignment must be SEED=/absolute/path") from exc
    if seed not in SEEDS:
        raise argparse.ArgumentTypeError("only C1 seeds 2027 and 1337 are allowed")
    return seed, Path(raw_path).resolve()


def canonical_tensor_hash(state: dict[str, Any], torch_module) -> tuple[str, dict[str, Any]]:
    """Hash model/model1 tensors independent of pickle/container bytes."""

    digest = hashlib.sha256()
    summary: dict[str, Any] = {}
    for section in ("model", "model1"):
        values = state.get(section)
        if not isinstance(values, dict):
            raise ValueError(f"weights-only checkpoint has no {section} state dict")
        tensor_names = []
        elements = 0
        for name in sorted(values):
            tensor = values[name]
            if not torch_module.is_tensor(tensor):
                raise ValueError(f"{section}.{name} is not a tensor")
            # State dicts can contain scalar counters (0-D tensors).  Flatten
            # first so the byte reinterpretation has a concrete dimension on
            # every PyTorch version.
            raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch_module.uint8).numpy().tobytes()
            header = json.dumps({"section": section, "name": str(name), "dtype": str(tensor.dtype), "shape": list(tensor.shape)}, sort_keys=True, separators=(",", ":")).encode("utf-8")
            digest.update(len(header).to_bytes(8, "little")); digest.update(header)
            digest.update(len(raw).to_bytes(8, "little")); digest.update(raw)
            tensor_names.append(str(name)); elements += int(tensor.numel())
        summary[section] = {"tensor_count": len(tensor_names), "element_count": elements, "first_keys": tensor_names[:12], "last_keys": tensor_names[-12:]}
    return digest.hexdigest(), summary


def training_seed(summary: dict[str, Any]) -> int | None:
    candidates = [
        summary.get("determinism", {}).get("seed"),
        summary.get("training_configuration", {}).get("seed"),
        summary.get("seed"),
    ]
    for value in candidates:
        if isinstance(value, (int, float)):
            return int(value)
    return None


def extract_epoch_chain(summary: dict[str, Any], epoch_metrics: dict[str, Any], best_sha: str) -> dict[str, Any]:
    records = list(summary.get("epochs", []) or epoch_metrics.get("epochs", []))
    if len(records) != 5 or [int(row.get("epoch", -1)) for row in records] != [1, 2, 3, 4, 5]:
        return {"status": "fail", "reason": "five contiguous epoch records are unavailable", "records": len(records)}
    parsed = []
    for row in records:
        diagnosis = row.get("diagnosis", {})
        try:
            pq = float(diagnosis["patient_macro"]["task_metrics_image_macro"]["pq"])
        except (KeyError, TypeError, ValueError):
            return {"status": "fail", "reason": "epoch patient-macro PQ is absent", "epoch": row.get("epoch")}
        parsed.append({"epoch": int(row["epoch"]), "pq": pq, "selected_as_best_pq": bool(row.get("selected_as_best_pq", False)), "best_pq_weights_sha256": row.get("best_pq_weights_sha256"), "last_state_sha256": row.get("last_state_sha256")})
    winning_pq = max(row["pq"] for row in parsed)
    winning_epoch = next(row["epoch"] for row in parsed if row["pq"] == winning_pq)  # exact tie: earlier wins
    matching = [row for row in parsed if row["best_pq_weights_sha256"] == best_sha]
    selected = matching[-1] if matching else None
    conditions = {
        "tie_rule_recomputed_epoch_is_5": winning_epoch == 5,
        "current_best_sha_appears_in_epoch_log": selected is not None,
        "current_best_sha_was_saved_at_epoch_5": selected is not None and selected["epoch"] == 5 and selected["selected_as_best_pq"],
    }
    return {"status": "pass" if all(conditions.values()) else "fail", "epochs": parsed, "tie_rule_winning_epoch": winning_epoch, "tie_rule_winning_pq": winning_pq, "current_best_epoch_record": selected, "conditions": conditions}


def code_evidence(repo: Path, commit: str | None) -> dict[str, Any]:
    if not commit:
        return {"status": "fail", "reason": "training summary does not record a repository commit"}
    try:
        source = subprocess.check_output(["git", "show", f"{commit}:main.py"], cwd=repo, text=True, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        return {"status": "fail", "commit": commit, "reason": "recorded training commit is unavailable in local git history"}
    required = {
        "strict_improvement": "current_pq > prior_selection" in source,
        "weights_written_from_current_state": '"model": state["model"]' in source and '"model1": state["model1"]' in source,
        "selected_epoch_embedded": '"selected_epoch": epoch + 1' in source,
        "source_state_hash_embedded": '"source_last_state_sha256": state_sha' in source,
        "atomic_best_write": "_torch_save_atomic(best_weights_path" in source,
    }
    return {"status": "pass" if all(required.values()) else "fail", "commit": commit, "checks": required}


def c3_evidence(c1_oracle_root: Path, c3_audit: dict[str, Any], seed: int, source_last_sha: str | None) -> dict[str, Any]:
    summary_path = c1_oracle_root / f"seed{seed}_c1" / "summary.json"
    if not summary_path.is_file():
        return {"status": "fail", "reason": "C3 C1 oracle summary is missing", "path": str(summary_path)}
    summary = read_json(summary_path)
    checkpoint = summary.get("checkpoint", {})
    oracle_sha = checkpoint.get("checkpoint_sha256")
    c3_seed = next((row for row in c3_audit.get("per_seed", []) if int(row.get("seed", -1)) == seed), None)
    c3_points_to_source = c3_seed is not None and Path(str(c3_seed.get("source_c1_oracle_directory", ""))).resolve() == (c1_oracle_root / f"seed{seed}_c1").resolve()
    conditions = {
        "oracle_summary_is_completed_c1": summary.get("status") == "complete" and summary.get("arm") == "c1" and int(summary.get("seed", -1)) == seed,
        "oracle_checkpoint_is_declared_epoch_5": int(checkpoint.get("epoch", checkpoint.get("declaration_epoch", -1))) == 5 or int(checkpoint.get("epoch", -1)) == 5,
        "weights_payload_source_last_state_sha_matches_c3_source": bool(source_last_sha) and source_last_sha == oracle_sha,
        "c3_audit_references_this_oracle_directory": c3_points_to_source,
    }
    return {"status": "pass" if all(conditions.values()) else "fail", "oracle_summary_path": str(summary_path), "oracle_checkpoint_sha256": oracle_sha, "conditions": conditions}


def inventory(root: Path, seed: int, limit: int = 250) -> list[dict[str, Any]]:
    if not root.is_dir():
        return []
    found = []
    seed_text = str(seed)
    for directory, dirnames, names in os.walk(root):
        dirnames[:] = [name for name in dirnames if name not in {"data", "torch_cache", ".git"}]
        for name in names:
            lower = name.lower()
            if not (lower.endswith((".pth", ".pt", ".json", ".log", ".csv")) and (seed_text in str(directory) or "c1" in lower or "epoch_0005" in lower or "last_complete" in lower)):
                continue
            path = Path(directory) / name
            try:
                info = {"path": str(path), "bytes": path.stat().st_size}
            except OSError:
                continue
            found.append(info)
            if len(found) >= limit:
                return found
    return found


def audit_seed(seed: int, root: Path, c1_oracle_root: Path, c3_audit: dict[str, Any], repo: Path, search_root: Path) -> dict[str, Any]:
    base = root / "c1"
    best = base / "best_pq" / "model_model1_weights.pth"
    declaration_path = base / "best_pq" / "declaration.json"
    summary_path = base / "training_summary.json"
    metrics_path = base / "epoch_metrics.json"
    last_declaration_path = base / "checkpoints" / "last_complete_state.json"
    required = [best, declaration_path, summary_path, metrics_path, last_declaration_path]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        return {"seed": seed, "status": "reconstruction_required", "missing": missing}
    declaration, summary, metrics, last_declaration = (read_json(path) for path in (declaration_path, summary_path, metrics_path, last_declaration_path))
    actual_sha = sha256_file(best)
    import torch
    try:
        state = torch.load(best, map_location="cpu", weights_only=True)
        load_mode = "weights_only"
    except Exception:
        # The declaration hash is checked before this explicit trusted local
        # fallback; this does not execute a model or training code.
        if declaration.get("checkpoint_sha256") != actual_sha:
            raise ValueError("best_pq declaration SHA mismatch prevents trusted checkpoint inspection")
        state = torch.load(best, map_location="cpu", weights_only=False)
        load_mode = "trusted_hash_verified_pickle_fallback"
    canonical_hash, state_summary = canonical_tensor_hash(state, torch)
    embedded = {key: state.get(key) for key in ("schema_version", "phase", "protocol", "dataset", "arm", "selected_epoch", "selected_patient_macro_pq", "source_last_state_sha256", "train_manifest", "development_manifest", "coverage", "screen_config", "texture_memory_bank_list", "embedded_texture_bank_loaded")}
    epoch_chain = extract_epoch_chain(summary, metrics, actual_sha)
    run_commit = summary.get("repository", {}).get("commit")
    code = code_evidence(repo, str(run_commit) if run_commit else None)
    c3 = c3_evidence(c1_oracle_root, c3_audit, seed, embedded.get("source_last_state_sha256"))
    data = summary.get("data", {})
    initialization = summary.get("initialization", {})
    train_identity = embedded.get("train_manifest") if isinstance(embedded.get("train_manifest"), dict) else {}
    development_identity = embedded.get("development_manifest") if isinstance(embedded.get("development_manifest"), dict) else {}
    summary_development = summary.get("development_manifest") if isinstance(summary.get("development_manifest"), dict) else {}
    def identity_sha(value: dict[str, Any]) -> str | None:
        for key in ("manifest_sha256", "sha256", "hashes_sha256"):
            if isinstance(value.get(key), str):
                return str(value[key])
        return None
    config_conditions = {
        "summary_complete": summary.get("status") == "complete",
        "summary_seed_matches": training_seed(summary) == seed,
        "summary_arm_is_c1": summary.get("training_configuration", {}).get("arm") == "c1" or summary.get("arm") == "c1",
        "summary_initialization_has_sha256": isinstance(initialization.get("checkpoint_sha256"), str) and len(str(initialization.get("checkpoint_sha256"))) == 64,
        "summary_train_manifest_is_p1_p6": data.get("protocol_id") == "tnbc_stainpms_prepared_continuity_v1_phase1_train" and int(data.get("record_count", -1)) == 30,
        "weights_train_manifest_matches_summary": identity_sha(train_identity) is not None and identity_sha(train_identity) == data.get("manifest_sha256"),
        "weights_development_manifest_matches_summary": bool(summary_development) and identity_sha(development_identity) is not None and identity_sha(development_identity) == identity_sha(summary_development),
        "weights_embedded_dataset_arm": embedded.get("dataset") == "tnbc" and embedded.get("arm") == "c1",
        "weights_embedded_selected_epoch_is_5": int(embedded.get("selected_epoch", -1)) == 5,
        "best_file_sha_matches_declaration": declaration.get("checkpoint_sha256") == actual_sha,
        "last_declaration_is_epoch_5": int(last_declaration.get("epoch", -1)) == 5,
        "weights_have_model_and_model1": all(name in state for name in ("model", "model1")),
        "weights_texture_bank_empty": not bool(state.get("texture_memory_bank_list", [])) and state.get("embedded_texture_bank_loaded") is False,
    }
    recovered = all(config_conditions.values()) and epoch_chain.get("status") == "pass" and code.get("status") == "pass" and c3.get("status") == "pass"
    recovery_manifest = {
        "schema_version": 1, "protocol": "tnbc_c1_epoch5_recovery_audit_v1", "seed": seed,
        "status": "recovered_epoch5_weights_only" if recovered else "reconstruction_required",
        "best_pq": {"path": str(best), "bytes": best.stat().st_size, "sha256": actual_sha, "load_mode": load_mode, "canonical_model_model1_tensor_sha256": canonical_hash, "state_dict": state_summary, "embedded_provenance": embedded},
        "declaration_path": str(declaration_path), "declaration": declaration,
        "epoch_log_evidence": epoch_chain, "save_code_evidence": code, "c3_consistency_evidence": c3,
        "configuration_identity_conditions": config_conditions,
        "other_candidate_file_inventory_read_only": inventory(search_root, seed),
        "conclusion": "All required direct provenance links pass; this weights-only file is accepted solely as the original C1 fixed epoch-5 inference state." if recovered else "At least one required direct provenance link is absent or fails. Do not use best_pq for C4; use the frozen reconstruction protocol.",
    }
    return recovery_manifest


def markdown(payload: dict[str, Any]) -> str:
    lines = ["# C1 epoch-5 recovery audit", "", "- Scope: read-only provenance inspection only; no dataset construction, inference, training or declaration mutation.", "- A PQ-best file is accepted only with a direct C1 epoch-5 log, save-code, configuration, hash and C3-source chain.", ""]
    for row in payload["seeds"]:
        lines += [f"## Seed {row['seed']}", "", f"- Status: `{row['status']}`."]
        best = row.get("best_pq", {})
        if best:
            lines.append(f"- Best-PQ SHA256: `{best['sha256']}`; canonical model/model1 tensor SHA256: `{best['canonical_model_model1_tensor_sha256']}`.")
        for label, checks in (("configuration", row.get("configuration_identity_conditions", {})), ("epoch log", row.get("epoch_log_evidence", {}).get("conditions", {})), ("save code", row.get("save_code_evidence", {}).get("checks", {})), ("C3 consistency", row.get("c3_consistency_evidence", {}).get("conditions", {}))):
            if checks:
                lines.append(f"- {label}: " + ", ".join(f"{key}={'pass' if value else 'fail'}" for key, value in checks.items()))
        lines.append(f"- Conclusion: {row.get('conclusion', 'required evidence missing')}")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-root", action="append", required=True, type=parse_assignment)
    parser.add_argument("--c1-oracle-root", required=True, type=Path)
    parser.add_argument("--c3-audit", required=True, type=Path)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--search-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    roots = dict(args.seed_root)
    if set(roots) != set(SEEDS):
        raise ValueError("both seed roots are required")
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite audit output: {output}")
    c3 = read_json(args.c3_audit.resolve())
    if c3.get("c3_gate", {}).get("single_supported_operation") != "conflict_order_oracle":
        raise ValueError("audit requires the accepted C3 conflict-order report")
    rows = [audit_seed(seed, roots[seed], args.c1_oracle_root.resolve(), c3, args.repo_root.resolve(), args.search_root.resolve()) for seed in SEEDS]
    status = "recovered_epoch5_weights_only" if all(row["status"] == "recovered_epoch5_weights_only" for row in rows) else "reconstruction_required"
    payload = {"schema_version": 1, "protocol": "tnbc_c1_epoch5_recovery_audit_v1", "status": status, "created_at_utc": datetime.now(timezone.utc).isoformat(), "scope": "read-only C1 checkpoint provenance/recovery audit; TNBC p1-p6/p7-p8 data were not read", "seeds": rows, "next_action": "C4 may resume only with these recovered fixed epoch-5 weights" if status.startswith("recovered") else "Do not start C4. Follow the project-lead reconstructed C1 lineage protocol for every non-recovered seed."}
    output.mkdir(parents=True, exist_ok=False)
    write_json = lambda path, value: path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_json(output / "c1_epoch5_recovery_audit.json", payload)
    (output / "c1_epoch5_recovery_audit.md").write_text(markdown(payload), encoding="utf-8")
    for row in rows:
        write_json(output / f"seed{row['seed']}_recovery_manifest.json", row)
    print(json.dumps({"status": status, "output_dir": str(output)}, ensure_ascii=False))
    return 0 if status.startswith("recovered") else 2


if __name__ == "__main__":
    raise SystemExit(main())
