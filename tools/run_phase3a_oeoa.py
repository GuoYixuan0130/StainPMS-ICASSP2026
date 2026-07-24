#!/usr/bin/env python3
"""Read-only Phase 3A Orthogonal Error-Oracle Audit (OEOA).

``prepare`` freezes the source identities, hashes and preregistration before
any real oracle aggregate is calculated. ``run`` consumes only that frozen
prepared directory and the existing compact p7/p8 C0/C1 artifacts.  This file
intentionally has no model, trainer, optimiser, checkpoint, or GPU dependency.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stainpms.c2_component_audit import deserialize_gt, deserialize_selected
from stainpms.oeoa import (
    ACTION_CLASSES,
    METRIC_FIELDS,
    ROUTES,
    action_mask,
    actions_for_mask,
    all_action_masks,
    apply_component_oracle,
    average_metrics,
    build_overlap_components,
    candidate_pool_ceiling,
    compact_metrics,
    localize_final_fns,
    map_metrics,
    map_sha256,
    metric_delta,
    pairwise_interactions,
    relabel_contiguously,
    shapley_contributions,
    summarize_localizations,
)
from stainpms.zero_training_oracle import ORACLE_MATCH_IOU, decode_binary_rle


SEEDS = (2027, 1337)
DEV_PATIENTS = (7, 8)
BASELINE_COMMIT = "bb8d7ba4f9394ed82845e788bdd21a489d3a8da2"
PROTOCOL = "tnbc_phase3a_orthogonal_error_oracle_audit_v1"
CONFIG_PATH = ROOT / "configs" / "phase3a" / "tnbc_oeoa_v1.json"
DESIGN_PATH = ROOT / "docs" / "phase3a_oeoa_design.md"
FLOAT_TOLERANCE = 1.0e-7


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def read_gzip_json(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def run_git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout


def require_clean_repository(repository: Path, expected_commit: str) -> dict[str, str]:
    if not (repository / "configs").is_dir():
        raise ValueError(f"repository does not look like StainPMS: {repository}")
    if run_git(repository, "rev-parse", "--is-inside-work-tree").strip() != "true":
        raise ValueError(f"not a Git worktree: {repository}")
    branch = run_git(repository, "branch", "--show-current").strip()
    head = run_git(repository, "rev-parse", "HEAD").strip()
    status = run_git(repository, "status", "--short")
    if branch != "research/f3c-stainpms":
        raise ValueError(f"Phase 3A requires research/f3c-stainpms, got {branch!r}")
    if head != expected_commit:
        raise ValueError(f"Phase 3A expected commit {expected_commit}, got {head}")
    if status.strip():
        raise ValueError("Phase 3A requires a clean worktree before any source read")
    return {"branch": branch, "commit": head, "worktree_status": status}


def parse_assignment(value: str) -> tuple[int, Path]:
    try:
        raw_seed, raw_path = value.split("=", 1)
        seed = int(raw_seed)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("assignment must be SEED=/absolute/path") from exc
    if seed not in SEEDS:
        raise argparse.ArgumentTypeError("only seeds 2027 and 1337 are permitted")
    return seed, Path(raw_path).resolve()


def parse_assignments(values: Iterable[tuple[int, Path]], *, label: str) -> dict[int, Path]:
    output = dict(values)
    if set(output) != set(SEEDS):
        raise ValueError(f"{label} must provide exactly seeds {SEEDS}")
    return output


def _summary_identity(summary: Mapping[str, Any]) -> dict[str, Any]:
    checkpoint = summary.get("checkpoint") or {}
    manifest = summary.get("manifest") or {}
    return {
        "arm": summary.get("arm"),
        "seed": summary.get("seed"),
        "checkpoint_sha256": checkpoint.get("checkpoint_sha256"),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "manifest_sha256": manifest.get("sha256"),
        "manifest_patients": manifest.get("patients"),
        "manifest_record_count": manifest.get("record_count"),
        "frozen_epoch5_manifest": summary.get("frozen_epoch5_manifest"),
    }


def load_compact_source(root: Path, *, seed: int, expected_arm: str, label: str) -> dict[str, Any]:
    summary_path = root / "summary.json"
    if not summary_path.is_file():
        raise ValueError(f"missing {label} summary: {summary_path}")
    summary = read_json(summary_path)
    if summary.get("status") != "complete" or int(summary.get("seed", -1)) != seed or summary.get("arm") != expected_arm:
        raise ValueError(f"invalid {label} summary at {root}: {summary.get('status')}/{summary.get('seed')}/{summary.get('arm')}")
    manifest = summary.get("manifest") or {}
    if int(manifest.get("record_count", -1)) != 7 or set(int(value) for value in manifest.get("patients", [])) != set(DEV_PATIENTS):
        raise ValueError(f"{label} is not exactly the TNBC p7/p8 development scope: {root}")
    paths = sorted((root / "completed_images").glob("*.json.gz"))
    if len(paths) != 7:
        raise ValueError(f"{label} needs exactly seven compact images, got {len(paths)} at {root}")
    payloads = [read_gzip_json(path) for path in paths]
    records: list[dict[str, Any]] = []
    for path, payload in zip(paths, payloads, strict=True):
        artifact = payload.get("artifact")
        image_record = payload.get("image_record")
        if not isinstance(artifact, dict) or not isinstance(image_record, dict):
            raise ValueError(f"malformed compact artifact: {path}")
        sample_id = str(artifact.get("sample_id", ""))
        patient = int(artifact.get("patient", -1))
        if not sample_id or patient not in DEV_PATIENTS:
            raise ValueError(f"out-of-scope compact artifact: {path}")
        records.append({"path": path, "payload": payload, "sample_id": sample_id, "patient": patient})
    sample_ids = [row["sample_id"] for row in records]
    if len(set(sample_ids)) != 7 or {row["patient"] for row in records} != set(DEV_PATIENTS):
        raise ValueError(f"{label} does not contain seven unique p7/p8 cases")
    return {
        "root": root,
        "summary": summary,
        "summary_path": summary_path,
        "records": sorted(records, key=lambda row: row["sample_id"]),
        "files": [
            {"path": str(summary_path), "sha256": sha256_file(summary_path), "bytes": int(summary_path.stat().st_size)},
            *[
                {"path": str(row["path"]), "sha256": sha256_file(row["path"]), "bytes": int(row["path"].stat().st_size)}
                for row in records
            ],
        ],
        "identity": _summary_identity(summary),
    }


def deserialize_all(artifact: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for encoded in artifact.get("all_native_candidates", []):
        row = dict(encoded)
        row["mask"] = decode_binary_rle(row.pop("mask_rle"))
        row["record_index"] = int(row["record_index"])
        row["prompt_group_id"] = int(row["prompt_group_id"])
        row["token"] = int(row.get("token", -1))
        row["crop_index"] = int(row.get("crop_index", -1))
        rows.append(row)
    return rows


def final_map_from_artifact(artifact: Mapping[str, Any]) -> np.ndarray:
    output = np.zeros(tuple(int(value) for value in artifact["image_shape"]), dtype=np.int32)
    for row in artifact.get("native_final_instances", []):
        mask = decode_binary_rle(row["mask_rle"])
        if bool((output[mask] != 0).any()):
            raise ValueError("native final compact artifact contains overlapping instances")
        output[mask] = int(row["final_instance_id"])
    return output


def full_sha(value: Any, *, label: str) -> str:
    text = str(value or "")
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{label} must be a complete lowercase SHA256, got {text!r}")
    return text


def validate_c3_binding(c3_path: Path, c1_sources: Mapping[int, dict[str, Any]]) -> dict[str, Any]:
    payload = read_json(c3_path)
    if (
        payload.get("status") != "complete"
        or payload.get("lineage", {}).get("seed2027") != "historical verified C1 epoch-5 lineage"
        or payload.get("lineage", {}).get("seed1337") != "reconstructed C1 seed-1337 lineage"
    ):
        raise ValueError("C3 audit does not preserve the approved two-lineage contract")
    by_seed = {int(row.get("seed", -1)): row for row in payload.get("per_seed", []) if isinstance(row, dict)}
    if set(by_seed) != set(SEEDS):
        raise ValueError("C3 audit must contain exactly seed 2027 and reconstructed seed 1337")
    for seed in SEEDS:
        source = Path(str(by_seed[seed].get("source_c1_oracle_directory", ""))).resolve()
        if source != c1_sources[seed]["root"]:
            raise ValueError(f"C3 source C1 mismatch for seed {seed}: {source} != {c1_sources[seed]['root']}")
    identity = by_seed[1337].get("source_identity") or {}
    if identity.get("lineage") != "reconstructed C1 seed-1337 lineage":
        raise ValueError("seed-1337 C3 source identity is not reconstructed")
    return {"path": str(c3_path), "sha256": sha256_file(c3_path), "per_seed": by_seed}


def validate_lineages(
    *,
    c1_sources: Mapping[int, dict[str, Any]],
    recovery_manifest: Path,
    frozen_manifest: Path,
    c3_path: Path,
) -> dict[str, Any]:
    recovery = read_json(recovery_manifest)
    if recovery.get("status") != "recovered_epoch5_weights_only" or int(recovery.get("seed", -1)) != 2027:
        raise ValueError("seed-2027 recovery manifest is not the approved epoch-5 weights-only audit")
    best = recovery.get("best_pq") or {}
    embedded = best.get("embedded_provenance") or {}
    recovery_weights_sha = full_sha(best.get("sha256"), label="seed-2027 recovery weights-only SHA256")
    recovery_canonical_sha = full_sha(best.get("canonical_model_model1_tensor_sha256"), label="seed-2027 canonical tensor SHA256")
    recovery_state_sha = full_sha(embedded.get("source_last_state_sha256"), label="seed-2027 source last-state SHA256")
    c1_2027 = c1_sources[2027]["summary"]
    c1_2027_checkpoint = c1_2027.get("checkpoint") or {}
    if (
        c1_2027.get("arm") != "c1"
        or int(c1_2027_checkpoint.get("epoch", -1)) != 5
        or c1_2027_checkpoint.get("checkpoint_sha256") != recovery_state_sha
    ):
        raise ValueError("seed-2027 compact C1 is not bound to the recovery-audited epoch-5 lineage")

    frozen = read_json(frozen_manifest)
    if frozen.get("status") != "frozen_before_development_access" or frozen.get("lineage") != "reconstructed C1 seed-1337 lineage":
        raise ValueError("seed-1337 frozen manifest is not the approved reconstructed lineage")
    complete = frozen.get("complete_state") or {}
    weights = frozen.get("weights_only") or {}
    complete_sha = full_sha(complete.get("sha256"), label="seed-1337 complete-state SHA256")
    weights_sha = full_sha(weights.get("sha256"), label="seed-1337 weights-only SHA256")
    canonical_sha = full_sha(weights.get("canonical_model_model1_tensor_sha256"), label="seed-1337 canonical tensor SHA256")
    frozen_sha = sha256_file(frozen_manifest)
    c1_1337 = c1_sources[1337]["summary"]
    c1_1337_checkpoint = c1_1337.get("checkpoint") or {}
    c1_1337_frozen = c1_1337.get("frozen_epoch5_manifest") or {}
    if (
        c1_1337.get("arm") != "c1_reconstructed"
        or int(c1_1337_checkpoint.get("epoch", -1)) != 5
        or c1_1337_checkpoint.get("checkpoint_sha256") != complete_sha
        or c1_1337_frozen.get("status") != "frozen_before_development_access"
        or c1_1337_frozen.get("lineage") != "reconstructed C1 seed-1337 lineage"
        or c1_1337_frozen.get("sha256") != frozen_sha
        or c1_1337_frozen.get("canonical_model_model1_tensor_sha256") != canonical_sha
    ):
        raise ValueError("seed-1337 compact C1 is not bound solely to reconstructed frozen epoch 5")

    c3 = validate_c3_binding(c3_path, c1_sources)
    c3_identity = c3["per_seed"][1337].get("source_identity") or {}
    if (
        c3_identity.get("checkpoint_sha256") != complete_sha
        or c3_identity.get("frozen_epoch5_manifest_sha256") != frozen_sha
    ):
        raise ValueError("reconstructed seed-1337 C3 identity does not match the original frozen manifest")
    return {
        "seed2027": {
            "kind": "recovery_audited_original_c1_epoch5_weights_only",
            "recovery_manifest_path": str(recovery_manifest),
            "recovery_manifest_sha256": sha256_file(recovery_manifest),
            "weights_only_sha256": recovery_weights_sha,
            "canonical_model_model1_tensor_sha256": recovery_canonical_sha,
            "source_last_state_sha256": recovery_state_sha,
        },
        "seed1337": {
            "kind": "reconstructed_c1_epoch5_full_state",
            "frozen_manifest_path": str(frozen_manifest),
            "frozen_manifest_sha256": frozen_sha,
            "complete_state_sha256": complete_sha,
            "weights_only_sha256": weights_sha,
            "canonical_model_model1_tensor_sha256": canonical_sha,
        },
        "c3_binding": {"path": c3["path"], "sha256": c3["sha256"]},
    }


def source_file_manifest(sources: Mapping[str, Mapping[int, dict[str, Any]]], extra_files: Iterable[Path]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for source_kind, by_seed in sorted(sources.items()):
        for seed, source in sorted(by_seed.items()):
            for row in source["files"]:
                result.append({"kind": source_kind, "seed": int(seed), **row})
    for path in extra_files:
        result.append({"kind": "lineage_or_preregistration", "seed": None, "path": str(path), "sha256": sha256_file(path), "bytes": int(path.stat().st_size)})
    return result


def write_sha_sums(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    lines = [f"{row['sha256']}  {row['path']}" for row in rows]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def copy_audit_code(destination: Path) -> list[str]:
    destination.mkdir(parents=True, exist_ok=True)
    files = [ROOT / "stainpms" / "oeoa.py", ROOT / "tools" / "run_phase3a_oeoa.py", CONFIG_PATH, DESIGN_PATH]
    copied: list[str] = []
    for source in files:
        target = destination / source.name
        shutil.copy2(source, target)
        copied.append(str(target))
    return copied


def load_all_sources(args: argparse.Namespace) -> dict[str, Any]:
    c1_paths = parse_assignments(args.c1_source, label="--c1-source")
    c0_paths = parse_assignments(args.c0_source, label="--c0-source")
    c1 = {
        2027: load_compact_source(c1_paths[2027], seed=2027, expected_arm="c1", label="seed-2027 C1"),
        1337: load_compact_source(c1_paths[1337], seed=1337, expected_arm="c1_reconstructed", label="seed-1337 reconstructed C1"),
    }
    c0 = {seed: load_compact_source(c0_paths[seed], seed=seed, expected_arm="c0", label=f"seed-{seed} C0") for seed in SEEDS}
    lineage = validate_lineages(
        c1_sources=c1,
        recovery_manifest=Path(args.recovery_manifest).resolve(),
        frozen_manifest=Path(args.frozen_manifest).resolve(),
        c3_path=Path(args.c3_audit).resolve(),
    )
    return {"c1": c1, "c0": c0, "lineage": lineage}


def prepared_manifest_path(output: Path) -> Path:
    return output / "prepared" / "phase3a_oeoa_input_manifest.json"


def prepare(args: argparse.Namespace) -> int:
    repository = Path(args.repository).resolve()
    repository_state = require_clean_repository(repository, args.expected_commit)
    if not CONFIG_PATH.is_file() or not DESIGN_PATH.is_file():
        raise FileNotFoundError("Phase 3A static preregistration files are missing")
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Phase 3A output must be a new empty directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    sources = load_all_sources(args)
    prepared = output / "prepared"
    prepared.mkdir()
    config_copy = prepared / "phase3a_oeoa_config.json"
    design_copy = prepared / "phase3a_oeoa_design.md"
    shutil.copy2(CONFIG_PATH, config_copy)
    shutil.copy2(DESIGN_PATH, design_copy)
    files = source_file_manifest(
        {"c0_compact": sources["c0"], "c1_compact": sources["c1"]},
        [Path(args.recovery_manifest).resolve(), Path(args.frozen_manifest).resolve(), Path(args.c3_audit).resolve(), CONFIG_PATH, DESIGN_PATH],
    )
    input_manifest = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": "frozen_before_real_p7_p8_oracle_aggregation",
        "baseline_commit": BASELINE_COMMIT,
        "repository": {"branch": repository_state["branch"], "commit": repository_state["commit"]},
        "scope": "TNBC p7/p8 compact C0/C1 artifacts only; no neural inference or training",
        "c1_lineage": sources["lineage"],
        "sources": {
            name: {
                str(seed): {
                    "directory": str(source["root"]),
                    "summary_identity": source["identity"],
                    "sample_ids": [row["sample_id"] for row in source["records"]],
                    "patients": [row["patient"] for row in source["records"]],
                }
                for seed, source in sorted(by_seed.items())
            }
            for name, by_seed in (("c0", sources["c0"]), ("c1", sources["c1"]))
        },
        "input_files": files,
        "static_config": {"path": str(CONFIG_PATH), "sha256": sha256_file(CONFIG_PATH)},
        "static_design": {"path": str(DESIGN_PATH), "sha256": sha256_file(DESIGN_PATH)},
    }
    write_json_atomic(prepared_manifest_path(output), input_manifest)
    write_sha_sums(prepared / "input_SHA256SUMS", files)
    copy_audit_code(prepared / "audit_code")
    (prepared / "preparation_commit.txt").write_text(repository_state["commit"] + "\n", encoding="utf-8")
    (prepared / "preparation_worktree_status.txt").write_text(repository_state["worktree_status"], encoding="utf-8")
    print(json.dumps({"status": "prepared", "output_dir": str(output), "prepared_manifest": str(prepared_manifest_path(output))}, ensure_ascii=False))
    return 0


def validate_prepared_inputs(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    """Rehash every declared input and reject any change after preregistration."""

    repository = Path(args.repository).resolve()
    repository_state = require_clean_repository(repository, args.expected_commit)
    output = Path(args.output_dir).resolve()
    manifest_path = prepared_manifest_path(output)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Phase 3A prepared input manifest is absent: {manifest_path}")
    manifest_sha = sha256_file(manifest_path)
    if full_sha(args.confirmed_input_manifest_sha256, label="confirmed input manifest SHA256") != manifest_sha:
        raise ValueError("real OEOA aggregation requires the exact manually confirmed prepared input-manifest SHA256")
    prepared = read_json(manifest_path)
    if prepared.get("status") != "frozen_before_real_p7_p8_oracle_aggregation" or prepared.get("protocol") != PROTOCOL:
        raise ValueError("prepared input manifest does not have the frozen OEOA preregistration status")
    if prepared.get("baseline_commit") != BASELINE_COMMIT:
        raise ValueError("prepared input manifest has the wrong Phase 3A baseline commit")
    if prepared.get("repository", {}).get("commit") != repository_state["commit"]:
        raise ValueError("repository commit changed after OEOA preregistration")
    if sha256_file(CONFIG_PATH) != prepared.get("static_config", {}).get("sha256") or sha256_file(DESIGN_PATH) != prepared.get("static_design", {}).get("sha256"):
        raise ValueError("static OEOA design/config changed after preregistration")
    sources = load_all_sources(args)
    current_files = source_file_manifest(
        {"c0_compact": sources["c0"], "c1_compact": sources["c1"]},
        [Path(args.recovery_manifest).resolve(), Path(args.frozen_manifest).resolve(), Path(args.c3_audit).resolve(), CONFIG_PATH, DESIGN_PATH],
    )
    expected_files = prepared.get("input_files")
    if not isinstance(expected_files, list) or current_files != expected_files:
        raise ValueError("an OEOA input path, size, or SHA256 changed after preregistration")
    test_log = Path(args.test_log).resolve()
    if not test_log.is_file() or not test_log.read_text(encoding="utf-8", errors="replace").strip():
        raise ValueError("Phase 3A real aggregation requires a non-empty synthetic-test log")
    return prepared, sources, {"manifest_sha256": manifest_sha, "test_log": str(test_log), **repository_state}


def assert_metrics_close(actual: Mapping[str, float], expected: Mapping[str, Any], *, label: str) -> None:
    mismatches = {
        field: {"actual": float(actual[field]), "expected": float(expected[field])}
        for field in METRIC_FIELDS
        if not math.isclose(float(actual[field]), float(expected[field]), rel_tol=0.0, abs_tol=FLOAT_TOLERANCE)
    }
    if mismatches:
        raise RuntimeError(f"formal strict metric reproduction failed for {label}: {mismatches}")


def expected_compact_metrics(payload: Mapping[str, Any], *, label: str) -> Mapping[str, Any]:
    image_record = payload.get("image_record") or {}
    stage = (image_record.get("stages") or {}).get("native_final") or {}
    metrics = stage.get("strict_metrics") or {}
    if any(field not in metrics for field in METRIC_FIELDS):
        raise ValueError(f"compact native-final strict metrics missing for {label}")
    return metrics


def build_cases(sources: Mapping[str, Any]) -> tuple[dict[int, dict[str, dict[str, Any]]], list[dict[str, Any]], dict[str, Any]]:
    """Decode the compact inputs and establish all input/reproduction invariants."""

    cases: dict[int, dict[str, dict[str, Any]]] = {}
    inventory_rows: list[dict[str, Any]] = []
    qc: dict[str, Any] = {
        "c0_c1_case_set_identical": True,
        "c0_c1_gt_map_identical": True,
        "native_c1_metric_reproduction": True,
        "native_c0_metric_reproduction": True,
        "zero_action_exact_identity": True,
        "all_actions_exact_gt": True,
        "component_categories_exhaustive": True,
        "action_order_invariant_all_128": True,
        "relabel_metric_invariant": True,
        "all_oracle_outputs_nonoverlapping": True,
    }
    for seed in SEEDS:
        c1_by_sample = {row["sample_id"]: row for row in sources["c1"][seed]["records"]}
        c0_by_sample = {row["sample_id"]: row for row in sources["c0"][seed]["records"]}
        if sources["c1"][seed]["identity"]["manifest_sha256"] != sources["c0"][seed]["identity"]["manifest_sha256"]:
            raise RuntimeError(f"strict C0/C1 development-manifest mismatch for seed {seed}")
        if set(c1_by_sample) != set(c0_by_sample):
            raise RuntimeError(f"strict C0/C1 case-set mismatch for seed {seed}")
        cases[seed] = {}
        for sample_id in sorted(c1_by_sample):
            c1_payload = c1_by_sample[sample_id]["payload"]
            c0_payload = c0_by_sample[sample_id]["payload"]
            c1_artifact = c1_payload["artifact"]
            c0_artifact = c0_payload["artifact"]
            patient = int(c1_artifact["patient"])
            if patient != int(c0_artifact["patient"]):
                raise RuntimeError(f"strict C0/C1 patient mismatch for seed {seed}, {sample_id}")
            gt = deserialize_gt(c1_artifact)
            c0_gt = deserialize_gt(c0_artifact)
            if not np.array_equal(gt, c0_gt):
                raise RuntimeError(f"strict C0/C1 GT map mismatch for seed {seed}, {sample_id}")
            c1_map = final_map_from_artifact(c1_artifact)
            c0_map = final_map_from_artifact(c0_artifact)
            c1_evaluation = map_metrics(gt, c1_map, sample_id=sample_id)
            c0_evaluation = map_metrics(gt, c0_map, sample_id=sample_id)
            c1_metrics = compact_metrics(c1_evaluation)
            c0_metrics = compact_metrics(c0_evaluation)
            assert_metrics_close(c1_metrics, expected_compact_metrics(c1_payload, label=f"C1/{seed}/{sample_id}"), label=f"C1/{seed}/{sample_id}")
            assert_metrics_close(c0_metrics, expected_compact_metrics(c0_payload, label=f"C0/{seed}/{sample_id}"), label=f"C0/{seed}/{sample_id}")
            components, graph = build_overlap_components(gt, c1_map, sample_id=sample_id)
            if sum(int(value) for value in graph["category_counts"].values()) != int(graph["component_count"]):
                raise RuntimeError(f"component category inventory is not exhaustive for {seed}/{sample_id}")
            zero = apply_component_oracle(gt, c1_map, components, ())
            if not np.array_equal(zero, c1_map):
                raise RuntimeError(f"zero-action oracle map is not exact C1 for {seed}/{sample_id}")
            assert_metrics_close(compact_metrics(map_metrics(gt, zero, sample_id=sample_id)), c1_metrics, label=f"zero/{seed}/{sample_id}")
            all_actions = apply_component_oracle(gt, c1_map, components, ACTION_CLASSES)
            if not np.array_equal(all_actions, gt):
                raise RuntimeError(f"all-action oracle is not exact GT for {seed}/{sample_id}")
            cases[seed][sample_id] = {
                "seed": seed,
                "sample_id": sample_id,
                "patient": patient,
                "gt": gt,
                "c1_map": c1_map,
                "c0_map": c0_map,
                "c1_metrics": c1_metrics,
                "c0_metrics": c0_metrics,
                "components": components,
                "graph": graph,
                "selected_records": deserialize_selected(c1_artifact),
                "all_records": deserialize_all(c1_artifact),
                "c1_artifact_sha256": sha256_file(c1_by_sample[sample_id]["path"]),
                "c0_artifact_sha256": sha256_file(c0_by_sample[sample_id]["path"]),
                "gt_map_sha256": map_sha256(gt),
                "c1_final_map_sha256": map_sha256(c1_map),
            }
            for component in components:
                inventory_rows.append(
                    {
                        "seed": seed,
                        "patient": patient,
                        "sample_id": sample_id,
                        "component_id": component["component_id"],
                        "category": component["category"],
                        "pred_count": component["pred_count"],
                        "gt_count": component["gt_count"],
                        "pred_ids": json.dumps(component["pred_ids"], separators=(",", ":")),
                        "gt_ids": json.dumps(component["gt_ids"], separators=(",", ":")),
                        "pred_area": component["pred_area"],
                        "gt_area": component["gt_area"],
                        "overlap_pixel_count": component["overlap_pixel_count"],
                        "standard_pq_tp": component["standard_pq_tp"],
                        "one_to_one_iou": component["one_to_one_iou"],
                        "graph_component_count": graph["component_count"],
                    }
                )
    return cases, inventory_rows, qc


def evaluate_all_combinations(cases: Mapping[int, Mapping[str, Mapping[str, Any]]]) -> tuple[dict[int, dict[str, dict[int, dict[str, float]]]], dict[str, Any]]:
    values: dict[int, dict[str, dict[int, dict[str, float]]]] = {}
    qc = {"combination_count": len(all_action_masks()), "maps_checked": 0, "relabel_checks": 0}
    for seed in SEEDS:
        values[seed] = {}
        for sample_id, case in sorted(cases[seed].items()):
            gt = case["gt"]
            c1_map = case["c1_map"]
            components = case["components"]
            values[seed][sample_id] = {}
            for mask in all_action_masks():
                actions = actions_for_mask(mask)
                forward = apply_component_oracle(gt, c1_map, components, actions)
                reverse = apply_component_oracle(gt, c1_map, components, tuple(reversed(actions)))
                if not np.array_equal(forward, reverse):
                    raise RuntimeError(f"action-order invariance failed for {seed}/{sample_id}/mask={mask}")
                evaluation = map_metrics(gt, forward, sample_id=sample_id)
                metrics = compact_metrics(evaluation)
                reindexed = compact_metrics(map_metrics(gt, relabel_contiguously(forward), sample_id=sample_id))
                assert_metrics_close(reindexed, metrics, label=f"relabel/{seed}/{sample_id}/mask={mask}")
                values[seed][sample_id][mask] = metrics
                qc["maps_checked"] += 1
                qc["relabel_checks"] += 1
            assert_metrics_close(values[seed][sample_id][0], case["c1_metrics"], label=f"all-combo-zero/{seed}/{sample_id}")
            print(f"[phase3a-oeoa] seed={seed} sample={sample_id} combinations=128 complete", flush=True)
        print(f"[phase3a-oeoa] seed={seed} all p7/p8 combinations complete", flush=True)
    return values, qc


def scope_descriptors() -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for seed in SEEDS:
        for patient in DEV_PATIENTS:
            result.append({"level": "patient_image_macro", "seed": seed, "patient": patient, "sample_id": ""})
        result.append({"level": "seed_patient_macro", "seed": seed, "patient": "", "sample_id": ""})
    result.append({"level": "two_seed_macro", "seed": "all", "patient": "", "sample_id": ""})
    return result


def aggregate_case_metric(
    cases: Mapping[int, Mapping[str, Mapping[str, Any]]],
    combo_values: Mapping[int, Mapping[str, Mapping[int, Mapping[str, float]]]],
    descriptor: Mapping[str, Any],
    mask: int,
    *,
    arm: str,
) -> dict[str, float]:
    level = str(descriptor["level"])
    if level == "patient_image_macro":
        seed, patient = int(descriptor["seed"]), int(descriptor["patient"])
        rows = []
        for sample_id, case in cases[seed].items():
            if int(case["patient"]) == patient:
                rows.append(case[f"{arm}_metrics"] if arm in {"c0", "c1"} else combo_values[seed][sample_id][mask])
        return average_metrics(rows)
    if level == "seed_patient_macro":
        seed = int(descriptor["seed"])
        patient_rows = [
            aggregate_case_metric(cases, combo_values, {"level": "patient_image_macro", "seed": seed, "patient": patient}, mask, arm=arm)
            for patient in DEV_PATIENTS
        ]
        return average_metrics(patient_rows)
    if level == "two_seed_macro":
        seed_rows = [
            aggregate_case_metric(cases, combo_values, {"level": "seed_patient_macro", "seed": seed}, mask, arm=arm)
            for seed in SEEDS
        ]
        return average_metrics(seed_rows)
    raise ValueError(f"unknown aggregate level {level!r}")


def with_metric_columns(row: dict[str, Any], prefix: str, metrics: Mapping[str, float]) -> None:
    for field in METRIC_FIELDS:
        row[f"{prefix}_{field}"] = float(metrics[field])


def oracle_row(
    *,
    descriptor: Mapping[str, Any],
    mask: int,
    c0: Mapping[str, float],
    c1: Mapping[str, float],
    oracle: Mapping[str, float],
) -> dict[str, Any]:
    actions = actions_for_mask(mask)
    row: dict[str, Any] = {
        "aggregation_level": descriptor["level"],
        "seed": descriptor["seed"],
        "patient": descriptor["patient"],
        "sample_id": descriptor.get("sample_id", ""),
        "combination_id": int(mask),
        "action_count": len(actions),
        "enabled_actions": "+".join(actions) if actions else "zero_action",
        "is_atomic": len(actions) == 1,
        "is_all_actions": len(actions) == len(ACTION_CLASSES),
    }
    with_metric_columns(row, "c0", c0)
    with_metric_columns(row, "c1", c1)
    with_metric_columns(row, "oracle", oracle)
    with_metric_columns(row, "oracle_minus_c1", metric_delta(oracle, c1))
    with_metric_columns(row, "oracle_minus_c0", metric_delta(oracle, c0))
    return row


def build_combination_rows(
    cases: Mapping[int, Mapping[str, Mapping[str, Any]]],
    combo_values: Mapping[int, Mapping[str, Mapping[int, Mapping[str, float]]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    all_rows: list[dict[str, Any]] = []
    per_case_rows: list[dict[str, Any]] = []
    for mask in all_action_masks():
        for seed in SEEDS:
            for sample_id, case in sorted(cases[seed].items()):
                descriptor = {"level": "image", "seed": seed, "patient": case["patient"], "sample_id": sample_id}
                row = oracle_row(descriptor=descriptor, mask=mask, c0=case["c0_metrics"], c1=case["c1_metrics"], oracle=combo_values[seed][sample_id][mask])
                all_rows.append(row)
                per_case_rows.append(row)
        for descriptor in scope_descriptors():
            c0 = aggregate_case_metric(cases, combo_values, descriptor, mask, arm="c0")
            c1 = aggregate_case_metric(cases, combo_values, descriptor, mask, arm="c1")
            oracle = aggregate_case_metric(cases, combo_values, descriptor, mask, arm="oracle")
            all_rows.append(oracle_row(descriptor=descriptor, mask=mask, c0=c0, c1=c1, oracle=oracle))
    return all_rows, per_case_rows


def route_rows(
    cases: Mapping[int, Mapping[str, Mapping[str, Any]]],
    combo_values: Mapping[int, Mapping[str, Mapping[int, Mapping[str, float]]]],
    *,
    include_atomic: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    definitions: list[tuple[str, tuple[str, ...]]] = []
    if include_atomic:
        definitions.extend((name, (name,)) for name in ACTION_CLASSES)
    else:
        definitions.extend(ROUTES.items())
    for name, actions in definitions:
        mask = action_mask(actions)
        for descriptor in scope_descriptors():
            c0 = aggregate_case_metric(cases, combo_values, descriptor, mask, arm="c0")
            c1 = aggregate_case_metric(cases, combo_values, descriptor, mask, arm="c1")
            oracle = aggregate_case_metric(cases, combo_values, descriptor, mask, arm="oracle")
            row = oracle_row(descriptor=descriptor, mask=mask, c0=c0, c1=c1, oracle=oracle)
            row["analysis_name"] = name
            row["declared_actions"] = "+".join(actions)
            rows.append(row)
    return rows


def interaction_rows(
    cases: Mapping[int, Mapping[str, Mapping[str, Any]]],
    combo_values: Mapping[int, Mapping[str, Mapping[int, Mapping[str, float]]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for descriptor in scope_descriptors():
        per_metric = {
            field: pairwise_interactions(
                {
                    mask: aggregate_case_metric(cases, combo_values, descriptor, mask, arm="oracle")[field]
                    for mask in all_action_masks()
                }
            )
            for field in METRIC_FIELDS
        }
        for left_index in range(len(ACTION_CLASSES)):
            for right_index in range(left_index + 1, len(ACTION_CLASSES)):
                pair = (ACTION_CLASSES[left_index], ACTION_CLASSES[right_index])
                row: dict[str, Any] = {
                    "aggregation_level": descriptor["level"],
                    "seed": descriptor["seed"],
                    "patient": descriptor["patient"],
                    "action_a": pair[0],
                    "action_b": pair[1],
                }
                for field in METRIC_FIELDS:
                    row[f"interaction_{field}"] = float(per_metric[field][pair])
                rows.append(row)
    return rows


def shapley_rows(
    cases: Mapping[int, Mapping[str, Mapping[str, Any]]],
    combo_values: Mapping[int, Mapping[str, Mapping[int, Mapping[str, float]]]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    checks: dict[str, Any] = {"all_scopes_pass": True, "scopes": []}
    for descriptor in scope_descriptors():
        values_by_metric = {
            field: {
                mask: aggregate_case_metric(cases, combo_values, descriptor, mask, arm="oracle")[field]
                for mask in all_action_masks()
            }
            for field in ("aji", "pq")
        }
        contributions = {field: shapley_contributions(values) for field, values in values_by_metric.items()}
        scope_check = {"aggregation_level": descriptor["level"], "seed": descriptor["seed"], "patient": descriptor["patient"]}
        for field in ("aji", "pq"):
            expected = float(values_by_metric[field][action_mask(ACTION_CLASSES)] - values_by_metric[field][0])
            observed = float(sum(contributions[field].values()))
            passed = math.isclose(expected, observed, rel_tol=0.0, abs_tol=1.0e-12)
            scope_check[f"{field}_sum_equals_all_minus_c1"] = passed
            scope_check[f"{field}_all_minus_c1"] = expected
            scope_check[f"{field}_shapley_sum"] = observed
            checks["all_scopes_pass"] = bool(checks["all_scopes_pass"] and passed)
        checks["scopes"].append(scope_check)
        for action in ACTION_CLASSES:
            rows.append(
                {
                    "aggregation_level": descriptor["level"],
                    "seed": descriptor["seed"],
                    "patient": descriptor["patient"],
                    "action": action,
                    "shapley_aji": float(contributions["aji"][action]),
                    "shapley_pq": float(contributions["pq"][action]),
                    "all_oracle_minus_c1_aji": scope_check["aji_all_minus_c1"],
                    "all_oracle_minus_c1_pq": scope_check["pq_all_minus_c1"],
                }
            )
    if not checks["all_scopes_pass"]:
        raise RuntimeError("Shapley conservation quality control failed")
    return rows, checks


def target_recovery_rows(
    cases: Mapping[int, Mapping[str, Mapping[str, Any]]],
    combo_values: Mapping[int, Mapping[str, Mapping[int, Mapping[str, float]]]],
    *,
    target: float = 0.020,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for route, actions in ROUTES.items():
        mask = action_mask(actions)
        for descriptor in scope_descriptors():
            c0 = aggregate_case_metric(cases, combo_values, descriptor, mask, arm="c0")
            c1 = aggregate_case_metric(cases, combo_values, descriptor, mask, arm="c1")
            oracle = aggregate_case_metric(cases, combo_values, descriptor, mask, arm="oracle")
            for field in ("aji", "pq"):
                current = float(c1[field] - c0[field])
                oracle_gain = float(oracle[field] - c1[field])
                total = float(oracle[field] - c0[field])
                if current >= target:
                    status, recovery = "already_at_target_before_oracle", 0.0
                elif oracle_gain <= 0.0:
                    status, recovery = "impossible", None
                else:
                    recovery = float((target - current) / oracle_gain)
                    status = "oracle_cannot_reach_target" if recovery > 1.0 else "theoretical_reach_possible"
                rows.append(
                    {
                        "route": route,
                        "declared_actions": "+".join(actions),
                        "metric": field,
                        "aggregation_level": descriptor["level"],
                        "seed": descriptor["seed"],
                        "patient": descriptor["patient"],
                        "c1_minus_c0": current,
                        "oracle_minus_c1": oracle_gain,
                        "oracle_minus_c0": total,
                        "distance_to_c0_relative_plus_0_020": float(target - total),
                        "required_recovery": recovery,
                        "status": status,
                    }
                )
    return rows


def minimal_subset_rows(
    cases: Mapping[int, Mapping[str, Mapping[str, Any]]],
    combo_values: Mapping[int, Mapping[str, Mapping[int, Mapping[str, float]]]],
    *,
    target: float = 0.020,
) -> list[dict[str, Any]]:
    criteria: dict[str, list[int]] = {}
    for field in ("aji", "pq"):
        two_seed_descriptor = {"level": "two_seed_macro", "seed": "all", "patient": ""}
        candidates = [
            mask
            for mask in all_action_masks()
            if aggregate_case_metric(cases, combo_values, two_seed_descriptor, mask, arm="oracle")[field]
            - aggregate_case_metric(cases, combo_values, two_seed_descriptor, mask, arm="c0")[field]
            >= target
        ]
        criteria[f"two_seed_macro_{field}_c0_relative_at_least_0.020"] = candidates
        candidates = [
            mask
            for mask in all_action_masks()
            if all(
                aggregate_case_metric(cases, combo_values, {"level": "seed_patient_macro", "seed": seed, "patient": ""}, mask, arm="oracle")[field]
                - aggregate_case_metric(cases, combo_values, {"level": "seed_patient_macro", "seed": seed, "patient": ""}, mask, arm="c0")[field]
                >= target
                for seed in SEEDS
            )
        ]
        criteria[f"both_seed_patient_macro_{field}_c0_relative_at_least_0.020"] = candidates
    rows: list[dict[str, Any]] = []
    for criterion, candidates in criteria.items():
        if not candidates:
            rows.append({"criterion": criterion, "status": "no_subset_reaches_target", "combination_id": "", "action_count": "", "enabled_actions": ""})
            continue
        minimum = min(mask.bit_count() for mask in candidates)
        for mask in sorted(mask for mask in candidates if mask.bit_count() == minimum):
            rows.append(
                {
                    "criterion": criterion,
                    "status": "minimal_subsets_reported",
                    "combination_id": mask,
                    "action_count": minimum,
                    "enabled_actions": "+".join(actions_for_mask(mask)),
                }
            )
    return rows


def fn_and_candidate_rows(cases: Mapping[int, Mapping[str, Mapping[str, Any]]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    localization_rows: list[dict[str, Any]] = []
    ceiling_rows: list[dict[str, Any]] = []
    per_case_ceiling: list[dict[str, Any]] = []
    for seed in SEEDS:
        for sample_id, case in sorted(cases[seed].items()):
            localized = localize_final_fns(
                gt_map=case["gt"],
                final_map=case["c1_map"],
                selected_records=case["selected_records"],
                all_records=case["all_records"],
                sample_id=sample_id,
            )
            for row in localized:
                localization_rows.append({"seed": seed, "patient": case["patient"], "sample_id": sample_id, **row})
            for pool_name, records in (("selected_candidate_pool", case["selected_records"]), ("all_candidate_pool", case["all_records"])):
                ceiling = candidate_pool_ceiling(records, case["gt"])
                row = {
                    "aggregation_level": "image",
                    "seed": seed,
                    "patient": case["patient"],
                    "sample_id": sample_id,
                    "pool": pool_name,
                    **{key: value for key, value in ceiling.items() if key != "matched"},
                    "interpretation": "GT-only theoretical candidate-set ceiling; not an executable inference result",
                }
                ceiling_rows.append(row)
                per_case_ceiling.append({**row, "matched": ceiling["matched"]})
    return localization_rows, ceiling_rows, per_case_ceiling


def _aggregate_ceiling(rows: Iterable[Mapping[str, Any]], *, descriptor: Mapping[str, Any], pool: str) -> dict[str, Any]:
    selected = [row for row in rows if row["pool"] == pool]
    if not selected:
        raise ValueError("cannot aggregate an empty candidate ceiling set")
    tp = int(sum(int(row["maximum_attainable_tp"]) for row in selected))
    fn = int(sum(int(row["remaining_fn"]) for row in selected))
    iou_sum = float(sum(float(row["matched_iou_sum"]) for row in selected))
    sq_pooled = float(iou_sum / tp) if tp else 0.0
    dq_pooled = float(tp / (tp + 0.5 * fn)) if tp or fn else 0.0
    return {
        "aggregation_level": descriptor["level"],
        "seed": descriptor["seed"],
        "patient": descriptor["patient"],
        "sample_id": "",
        "pool": pool,
        "matching_unit": "native_prompt_group; maximum IoU>0.5 cardinality, then maximum total IoU",
        "raw_candidate_mask_count": int(sum(int(row["raw_candidate_mask_count"]) for row in selected)),
        "raw_prompt_group_count": int(sum(int(row["raw_prompt_group_count"]) for row in selected)),
        "gt_instance_count": int(sum(int(row["gt_instance_count"]) for row in selected)),
        "maximum_attainable_tp": tp,
        "remaining_fn": fn,
        "mean_matched_iou": sq_pooled if tp else None,
        "dq_ceiling": dq_pooled,
        "sq_ceiling": sq_pooled,
        "candidate_set_pq_ceiling": float(np.mean([float(row["candidate_set_pq_ceiling"]) for row in selected])),
        "matched_iou_sum": iou_sum,
        "covered_gt_count": int(sum(int(row["covered_gt_count"]) for row in selected)),
        "one_to_one_conflict_gt_count": int(sum(int(row["one_to_one_conflict_gt_count"]) for row in selected)),
        "metric_aggregation": "image mean at patient scope; later seed/two-seed rows explicitly use equal patient/seed means",
        "interpretation": "GT-only theoretical candidate-set ceiling; counts are pooled across the independent images in this row",
    }


def append_candidate_aggregates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    image_rows = list(rows)
    aggregate_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        for patient in DEV_PATIENTS:
            descriptor = {"level": "patient_image_macro", "seed": seed, "patient": patient}
            selected = [row for row in image_rows if int(row["seed"]) == seed and int(row["patient"]) == patient]
            for pool in ("selected_candidate_pool", "all_candidate_pool"):
                aggregate_rows.append(_aggregate_ceiling(selected, descriptor=descriptor, pool=pool))
        descriptor = {"level": "seed_patient_macro", "seed": seed, "patient": ""}
        selected = [row for row in image_rows if int(row["seed"]) == seed]
        for pool in ("selected_candidate_pool", "all_candidate_pool"):
            aggregate = _aggregate_ceiling(selected, descriptor=descriptor, pool=pool)
            patient_rows = [
                row
                for row in aggregate_rows
                if row["aggregation_level"] == "patient_image_macro" and int(row["seed"]) == seed and row["pool"] == pool
            ]
            for field in ("dq_ceiling", "sq_ceiling", "candidate_set_pq_ceiling"):
                aggregate[field] = float(np.mean([float(row[field]) for row in patient_rows]))
            aggregate["metric_aggregation"] = "equal mean of patient-7 and patient-8 image-macro candidate ceilings; counts remain pooled"
            aggregate_rows.append(aggregate)
    descriptor = {"level": "two_seed_macro", "seed": "all", "patient": ""}
    for pool in ("selected_candidate_pool", "all_candidate_pool"):
        aggregate = _aggregate_ceiling(image_rows, descriptor=descriptor, pool=pool)
        seed_rows = [row for row in aggregate_rows if row["aggregation_level"] == "seed_patient_macro" and row["pool"] == pool]
        for field in ("dq_ceiling", "sq_ceiling", "candidate_set_pq_ceiling"):
            aggregate[field] = float(np.mean([float(row[field]) for row in seed_rows]))
        aggregate["metric_aggregation"] = "equal mean of the two seed patient-macro candidate ceilings; counts remain pooled"
        aggregate_rows.append(aggregate)
    return [*image_rows, *aggregate_rows]


def localization_summary_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    values = list(rows)
    output: list[dict[str, Any]] = []
    descriptors: list[tuple[str, Any, Any, list[Mapping[str, Any]]]] = []
    for seed in SEEDS:
        for patient in DEV_PATIENTS:
            descriptors.append(("patient", seed, patient, [row for row in values if int(row["seed"]) == seed and int(row["patient"]) == patient]))
        descriptors.append(("seed_all_patients", seed, "", [row for row in values if int(row["seed"]) == seed]))
    descriptors.append(("two_seed_all_patients", "all", "", values))
    for level, seed, patient, selected in descriptors:
        for item in summarize_localizations(selected):
            output.append({"aggregation_level": level, "seed": seed, "patient": patient, **item})
        for near_bin in ("(0,0.1)", "[0.1,0.3)", "[0.3,0.5]"):
            selected_bin = [row for row in selected if row.get("candidate_mask_near_miss_bin") == near_bin]
            output.append(
                {
                    "aggregation_level": level,
                    "seed": seed,
                    "patient": patient,
                    "fn_localization": "candidate_mask_near_miss_bin",
                    "near_miss_bin": near_bin,
                    "fn_count": len(selected_bin),
                    "total_fn_count": len(selected),
                    "fn_proportion": float(len(selected_bin) / len(selected)) if selected else None,
                    "total_gt_area": int(sum(int(row["gt_area"]) for row in selected_bin)),
                    "mean_gt_area": float(np.mean([float(row["gt_area"]) for row in selected_bin])) if selected_bin else None,
                }
            )
    return output


def write_csv_rows(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    values = [dict(row) for row in rows]
    fields = sorted({key for row in values for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(values)


def two_seed_route_values(route_rows_value: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Mapping[str, Any]]]:
    output: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in route_rows_value:
        if row["aggregation_level"] == "two_seed_macro":
            output[str(row["analysis_name"])]["row"] = row
    return output


def describe_key_findings(
    atomic_rows: Iterable[Mapping[str, Any]],
    route_rows_value: Iterable[Mapping[str, Any]],
    interaction_rows_value: Iterable[Mapping[str, Any]],
    target_rows_value: Iterable[Mapping[str, Any]],
    localization_summary: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    atomic_two = [row for row in atomic_rows if row["aggregation_level"] == "two_seed_macro"]
    route_two = [row for row in route_rows_value if row["aggregation_level"] == "two_seed_macro"]
    interaction_two = [row for row in interaction_rows_value if row["aggregation_level"] == "two_seed_macro"]
    target_two = [row for row in target_rows_value if row["aggregation_level"] == "two_seed_macro"]
    localization_two = [row for row in localization_summary if row["aggregation_level"] == "two_seed_all_patients" and row["fn_localization"] != "candidate_mask_near_miss_bin"]
    largest_atomic = {
        field: max(atomic_two, key=lambda row: float(row[f"oracle_minus_c1_{field}"])) if atomic_two else None
        for field in ("aji", "pq")
    }
    interaction_extremes = {}
    for field in ("aji", "pq"):
        interaction_extremes[field] = {
            "largest_positive": max(interaction_two, key=lambda row: float(row[f"interaction_{field}"])) if interaction_two else None,
            "largest_negative": min(interaction_two, key=lambda row: float(row[f"interaction_{field}"])) if interaction_two else None,
        }
    review_pool = [row for row in route_two if row["analysis_name"] in {"mask_quality_total", "coverage", "topology", "precision"}]
    review_candidates = sorted(
        review_pool,
        key=lambda row: (
            -max(float(row["oracle_minus_c1_aji"]), float(row["oracle_minus_c1_pq"])),
            -float(row["oracle_minus_c1_aji"]) - float(row["oracle_minus_c1_pq"]),
            str(row["analysis_name"]),
        ),
    )[:2]
    route_patient_consistency: list[dict[str, Any]] = []
    for route in sorted({str(row["analysis_name"]) for row in route_rows_value}):
        patient_rows = [
            row
            for row in route_rows_value
            if row["analysis_name"] == route and row["aggregation_level"] == "patient_image_macro"
        ]
        patient_rows.sort(key=lambda row: (int(row["seed"]), int(row["patient"])))
        for field in ("aji", "pq"):
            deltas = [float(row[f"oracle_minus_c1_{field}"]) for row in patient_rows]
            route_patient_consistency.append(
                {
                    "route": route,
                    "metric": field,
                    "seed_patient_deltas_vs_c1": {
                        f"seed{row['seed']}_p{row['patient']}": float(row[f"oracle_minus_c1_{field}"])
                        for row in patient_rows
                    },
                    "positive_count_out_of_4": int(sum(value > 0.0 for value in deltas)),
                    "all_four_positive": bool(len(deltas) == 4 and all(value > 0.0 for value in deltas)),
                    "all_four_nonnegative": bool(len(deltas) == 4 and all(value >= 0.0 for value in deltas)),
                }
            )
    return {
        "largest_atomic_upper_bound": {
            field: None if row is None else {"action": row["analysis_name"], "oracle_minus_c1": row[f"oracle_minus_c1_{field}"], "oracle_minus_c0": row[f"oracle_minus_c0_{field}"]}
            for field, row in largest_atomic.items()
        },
        "route_two_seed_macro": {str(row["analysis_name"]): row for row in route_two},
        "interaction_extremes": interaction_extremes,
        "target_two_seed_macro": target_two,
        "fn_localization_two_seed": localization_two,
        "route_seed_patient_consistency": route_patient_consistency,
        "review_candidates_for_project_lead_only": review_candidates,
    }


def _format_float(value: Any) -> str:
    return "NA" if value is None else f"{float(value):+.6f}"


def markdown_report(summary: Mapping[str, Any]) -> str:
    findings = summary["key_findings"]
    lines = [
        "# Phase 3A — Orthogonal Error-Oracle Audit (OEOA)",
        "",
        "## Scope and interpretation",
        "",
        "- TNBC p7/p8 compact artifacts only; seeds 2027 and reconstructed 1337; strictly paired C0/C1 baseline.",
        "- Every result below is a GT-only upper bound on the native C1 final-instance map, not implemented model performance.",
        "- No model was trained, no C4 setting was changed, and p9–11/MoNuSeg were not accessed.",
        "",
        "## 1. Largest remaining atomic error upper bound",
        "",
        "| metric | atomic action | oracle−C1 | oracle−C0 |",
        "|---|---|---:|---:|",
    ]
    for field in ("aji", "pq"):
        row = findings["largest_atomic_upper_bound"][field]
        lines.append(f"| {field.upper()} | {row['action']} | {_format_float(row['oracle_minus_c1'])} | {_format_float(row['oracle_minus_c0'])} |")
    lines += [
        "",
        "## 2. Declared route upper bounds (two-seed macro)",
        "",
        "| route | AJI oracle−C1 | AJI oracle−C0 | PQ oracle−C1 | PQ oracle−C0 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, row in findings["route_two_seed_macro"].items():
        lines.append(f"| {name} | {_format_float(row['oracle_minus_c1_aji'])} | {_format_float(row['oracle_minus_c0_aji'])} | {_format_float(row['oracle_minus_c1_pq'])} | {_format_float(row['oracle_minus_c0_pq'])} |")
    lines += ["", "## 3. Redundancy and synergy", ""]
    for field in ("aji", "pq"):
        extreme = findings["interaction_extremes"][field]
        positive, negative = extreme["largest_positive"], extreme["largest_negative"]
        lines.append(f"- {field.upper()} largest positive interaction: `{positive['action_a']} + {positive['action_b']}` = {_format_float(positive[f'interaction_{field}'])}.")
        lines.append(f"- {field.upper()} largest negative interaction: `{negative['action_a']} + {negative['action_b']}` = {_format_float(negative[f'interaction_{field}'])}.")
    lines += ["", "## 4–5. +0.020 target alignment and required oracle recovery", "", "| route | metric | C1−C0 | oracle−C1 | oracle−C0 | required recovery | status |", "|---|---|---:|---:|---:|---:|---|"]
    for row in findings["target_two_seed_macro"]:
        lines.append(f"| {row['route']} | {row['metric'].upper()} | {_format_float(row['c1_minus_c0'])} | {_format_float(row['oracle_minus_c1'])} | {_format_float(row['oracle_minus_c0'])} | {_format_float(row['required_recovery'])} | {row['status']} |")
    lines += [
        "",
        "## 6. Cross-seed and patient consistency",
        "",
        "- The complete seed×patient (2027/p7, 2027/p8, reconstructed-1337/p7, reconstructed-1337/p8) tables are retained in `route_oracles.csv`, `atomic_oracles.csv`, and `target_recovery_analysis.csv`; no conclusion here uses a two-seed mean alone.",
        "",
    ]
    lines += [
        "",
        "| route | metric | 2027/p7 | 2027/p8 | reconstructed-1337/p7 | reconstructed-1337/p8 | positive / 4 | all positive |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in findings["route_seed_patient_consistency"]:
        deltas = row["seed_patient_deltas_vs_c1"]
        lines.append(
            f"| {row['route']} | {row['metric'].upper()} | {_format_float(deltas['seed2027_p7'])} | {_format_float(deltas['seed2027_p8'])} | {_format_float(deltas['seed1337_p7'])} | {_format_float(deltas['seed1337_p8'])} | {row['positive_count_out_of_4']}/4 | {row['all_four_positive']} |"
        )
    lines += ["", "## 7. Native-final FN localization (two-seed aggregate)", "", "| FN category | count | proportion | mean GT area |", "|---|---:|---:|---:|"]
    for row in findings["fn_localization_two_seed"]:
        area = "NA" if row["mean_gt_area"] is None else f"{float(row['mean_gt_area']):.1f}"
        lines.append(f"| {row['fn_localization']} | {row['fn_count']} | {float(row['fn_proportion'] or 0):.4f} | {area} |")
    lines += [
        "",
        "## 8. Items for project-lead review only",
        "",
        "The following two declared routes are mechanically ranked by their two-seed macro GT-only upper-bound magnitude. This is a review queue, not authorization to implement or train any module:",
        "",
    ]
    for index, row in enumerate(findings["review_candidates_for_project_lead_only"], start=1):
        lines.append(f"{index}. `{row['analysis_name']}` — AJI oracle−C1 {_format_float(row['oracle_minus_c1_aji'])}; PQ oracle−C1 {_format_float(row['oracle_minus_c1_pq'])}.")
    lines += [
        "",
        "## Quality control",
        "",
        f"- Input manifest SHA256: `{summary['prepared_input_manifest_sha256']}`.",
        f"- All quality checks passed: `{summary['quality_control']['all_passed']}`.",
        "- C4-CSR remains `do_not_promote`; this audit does not alter it or start a next stage.",
    ]
    return "\n".join(lines) + "\n"


def archive_sha_sums(output: Path) -> None:
    target = output / "SHA256SUMS"
    paths = sorted(path for path in output.rglob("*") if path.is_file() and path != target)
    lines = [f"{sha256_file(path)}  {path.relative_to(output).as_posix()}" for path in paths]
    target.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def audit_code_is_read_only() -> dict[str, Any]:
    """A lightweight source-level guard against accidental training APIs."""

    import ast

    files = [ROOT / "stainpms" / "oeoa.py", ROOT / "tools" / "run_phase3a_oeoa.py"]
    prohibited_modules = {"torch", "tensorflow", "pytorch_lightning", "transformers"}
    prohibited_calls = {"backward", "zero_grad", "load_state_dict", "save_state_dict", "fit", "train", "step"}
    issues: list[str] = []
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                modules = [alias.name.split(".")[0] for alias in node.names] if isinstance(node, ast.Import) else [str(node.module or "").split(".")[0]]
                if prohibited_modules.intersection(modules):
                    issues.append(f"prohibited import in {path.name}: {modules}")
            if isinstance(node, ast.Call):
                terminal = ""
                if isinstance(node.func, ast.Name):
                    terminal = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    terminal = node.func.attr
                if terminal in prohibited_calls:
                    issues.append(f"prohibited call in {path.name}: {terminal}")
    if issues:
        raise RuntimeError("OEOA read-only code guard failed: " + "; ".join(issues))
    return {"status": "pass", "files": [str(path) for path in files], "prohibited_modules": sorted(prohibited_modules), "prohibited_calls": sorted(prohibited_calls)}


def run(args: argparse.Namespace) -> int:
    prepared, sources, environment = validate_prepared_inputs(args)
    output = Path(args.output_dir).resolve()
    forbidden_existing = [
        output / name
        for name in (
            "per_case_metrics.csv",
            "component_inventory.csv",
            "atomic_oracles.csv",
            "route_oracles.csv",
            "all_128_combinations.csv",
            "summary.json",
            "final_report.md",
        )
        if (output / name).exists()
    ]
    if forbidden_existing:
        raise ValueError(f"refusing to overwrite an OEOA result archive: {forbidden_existing}")
    read_only_guard = audit_code_is_read_only()
    cases, inventory_rows, input_qc = build_cases(sources)
    combo_values, combination_qc = evaluate_all_combinations(cases)
    all_combination_rows, per_case_rows = build_combination_rows(cases, combo_values)
    atomic_rows = route_rows(cases, combo_values, include_atomic=True)
    declared_route_rows = route_rows(cases, combo_values, include_atomic=False)
    interactions = interaction_rows(cases, combo_values)
    shapley, shapley_qc = shapley_rows(cases, combo_values)
    target_rows = target_recovery_rows(cases, combo_values)
    minimal_rows = minimal_subset_rows(cases, combo_values)
    localization_rows, ceiling_image_rows, _ = fn_and_candidate_rows(cases)
    ceiling_rows = append_candidate_aggregates(ceiling_image_rows)
    localization_summary = localization_summary_rows(localization_rows)
    findings = describe_key_findings(atomic_rows, declared_route_rows, interactions, target_rows, localization_summary)

    quality_control = {
        **input_qc,
        "all_128_combinations_order_invariant": True,
        "shapley_conservation": shapley_qc,
        "read_only_audit_code": read_only_guard,
        "seed1337_reconstructed_lineage_only": True,
        "no_c4_baseline": True,
        "only_p7_p8_compact_artifacts": True,
    }
    quality_control["all_passed"] = bool(
        all(value is True for key, value in quality_control.items() if isinstance(value, bool))
        and shapley_qc["all_scopes_pass"]
        and read_only_guard["status"] == "pass"
    )
    if not quality_control["all_passed"]:
        raise RuntimeError("Phase 3A quality control failed")

    write_csv_rows(output / "per_case_metrics.csv", per_case_rows)
    write_csv_rows(output / "component_inventory.csv", inventory_rows)
    write_csv_rows(output / "atomic_oracles.csv", atomic_rows)
    write_csv_rows(output / "route_oracles.csv", declared_route_rows)
    write_csv_rows(output / "all_128_combinations.csv", all_combination_rows)
    write_csv_rows(output / "pairwise_interactions.csv", interactions)
    write_csv_rows(output / "shapley_contributions.csv", shapley)
    write_csv_rows(output / "fn_candidate_localization.csv", localization_rows)
    write_csv_rows(output / "fn_candidate_localization_summary.csv", localization_summary)
    write_csv_rows(output / "candidate_pool_ceiling.csv", ceiling_rows)
    write_csv_rows(output / "target_recovery_analysis.csv", target_rows)
    write_csv_rows(output / "minimal_action_subsets.csv", minimal_rows)
    shutil.copy2(Path(environment["test_log"]), output / "phase3a_oeoa_synthetic_tests.log")
    shutil.copy2(output / "prepared" / "input_SHA256SUMS", output / "input_SHA256SUMS")
    copy_audit_code(output / "audit_code")
    (output / "repository_commit.txt").write_text(environment["commit"] + "\n", encoding="utf-8")
    (output / "worktree_status.txt").write_text(environment["worktree_status"], encoding="utf-8")

    per_seed_patient = {
        str(seed): {
            str(patient): {
                "c0": aggregate_case_metric(cases, combo_values, {"level": "patient_image_macro", "seed": seed, "patient": patient}, 0, arm="c0"),
                "c1": aggregate_case_metric(cases, combo_values, {"level": "patient_image_macro", "seed": seed, "patient": patient}, 0, arm="c1"),
                "all_oracle": aggregate_case_metric(cases, combo_values, {"level": "patient_image_macro", "seed": seed, "patient": patient}, action_mask(ACTION_CLASSES), arm="oracle"),
            }
            for patient in DEV_PATIENTS
        }
        for seed in SEEDS
    }
    summary = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": "complete",
        "scope": "TNBC p7/p8 compact C0/C1 artifacts only; GT-only final-instance component and candidate-pool oracle audit",
        "interpretation": "diagnostic upper bounds only; not implemented or deployable performance",
        "baseline_commit": BASELINE_COMMIT,
        "repository": {"branch": environment["branch"], "commit": environment["commit"], "worktree_clean": True},
        "prepared_input_manifest": str(prepared_manifest_path(output)),
        "prepared_input_manifest_sha256": environment["manifest_sha256"],
        "c1_lineage": prepared["c1_lineage"],
        "strict_pairing": "C0/C1 paired per seed, per exact p7/p8 compact sample ID and GT map",
        "per_seed_patient_baselines_and_all_oracle": per_seed_patient,
        "quality_control": quality_control,
        "key_findings": findings,
        "deliverables": [
            "per_case_metrics.csv",
            "component_inventory.csv",
            "atomic_oracles.csv",
            "route_oracles.csv",
            "all_128_combinations.csv",
            "pairwise_interactions.csv",
            "shapley_contributions.csv",
            "fn_candidate_localization.csv",
            "candidate_pool_ceiling.csv",
            "target_recovery_analysis.csv",
            "minimal_action_subsets.csv",
            "final_report.md",
        ],
    }
    write_json_atomic(output / "summary.json", summary)
    (output / "final_report.md").write_text(markdown_report(summary), encoding="utf-8")
    archive_sha_sums(output)
    print(json.dumps({"status": "complete", "output_dir": str(output), "summary": str(output / "summary.json"), "quality_control": "pass"}, ensure_ascii=False))
    return 0


def parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser(description=__doc__)
    subparsers = argument_parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "run"):
        current = subparsers.add_parser(command)
        current.add_argument("--repository", required=True)
        current.add_argument("--expected-commit", required=True)
        current.add_argument("--recovery-manifest", required=True)
        current.add_argument("--frozen-manifest", required=True)
        current.add_argument("--c3-audit", required=True)
        current.add_argument("--c1-source", required=True, action="append", type=parse_assignment)
        current.add_argument("--c0-source", required=True, action="append", type=parse_assignment)
        current.add_argument("--output-dir", required=True)
        if command == "run":
            current.add_argument("--confirmed-input-manifest-sha256", required=True)
            current.add_argument("--test-log", required=True)
    return argument_parser


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    return prepare(args) if args.command == "prepare" else run(args)


if __name__ == "__main__":
    raise SystemExit(main())
