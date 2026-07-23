"""Read-only preflight audit for the TNBC p7/p8 zero-training diagnosis."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DIAGNOSIS_SEEDS = (2027, 1337)
EXCLUDED_SEED_3407 = {
    "seed": 3407,
    "reason": "C0 fixed epoch-5 complete checkpoint was deleted during prior retention compaction and cannot be recovered; no substitute checkpoint is permitted.",
}


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


def git_value(*args: str) -> str | None:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def parse_root(value: str) -> tuple[int, Path]:
    try:
        raw_seed, raw_path = value.split("=", 1)
        seed = int(raw_seed)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--seed-root must be SEED=/absolute/root") from exc
    if seed not in DIAGNOSIS_SEEDS:
        raise argparse.ArgumentTypeError("only paired epoch-5 diagnostic seeds 2027 and 1337 are permitted")
    return seed, Path(raw_path).resolve()


def inspect_arm(seed: int, arm: str, root: Path) -> dict[str, Any]:
    base = root / arm
    checkpoint = base / "checkpoints" / "last_complete_state.pth"
    declaration = base / "checkpoints" / "last_complete_state.json"
    training_summary = base / "training_summary.json"
    fields = {"checkpoint": checkpoint, "declaration": declaration, "training_summary": training_summary}
    missing = [name for name, path in fields.items() if not path.is_file()]
    result: dict[str, Any] = {"seed": seed, "arm": arm, "root": str(base), "paths": {name: str(path) for name, path in fields.items()}, "missing": missing}
    if missing:
        result["status"] = "missing_required_artifact"
        return result
    payload = read_json(declaration)
    observed_sha = sha256_file(checkpoint)
    training = read_json(training_summary)
    result.update(
        {
            "status": "complete" if payload.get("checkpoint_sha256") == observed_sha else "checkpoint_declaration_sha_mismatch",
            "checkpoint_sha256": observed_sha,
            "checkpoint_bytes": checkpoint.stat().st_size,
            "declaration_protocol": payload.get("protocol"),
            "declaration_epoch": payload.get("epoch"),
            "declaration_classification": payload.get("classification"),
            "training_status": training.get("status"),
            "training_protocol": training.get("protocol"),
            "training_seed": training.get("determinism", {}).get("seed"),
            "training_arm": training.get("training_configuration", {}).get("arm"),
            "fixed_epoch_5_available_as_last_complete_state": int(payload.get("epoch", -1)) == 5,
        }
    )
    if result["training_status"] != "complete" or result["training_seed"] != seed or result["training_arm"] != arm:
        result["status"] = "training_summary_identity_mismatch"
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-root", action="append", required=True, type=parse_root)
    parser.add_argument("--development-manifest", required=True)
    parser.add_argument("--reference-performance-summary", "--reference-three-seed-summary", dest="reference_performance_summary", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    roots = dict(args.seed_root)
    if set(roots) != set(DIAGNOSIS_SEEDS):
        raise ValueError("both paired epoch-5 diagnostic seed roots (2027 and 1337) are required")
    manifest_path = Path(args.development_manifest).resolve()
    reference_path = Path(args.reference_performance_summary).resolve()
    manifest = read_json(manifest_path)
    reference = read_json(reference_path)
    allowed = {int(value) for value in manifest.get("allowed_patients", [])}
    records = manifest.get("records", [])
    record_patients = {int(row.get("patient", -1)) for row in records}
    arms = [inspect_arm(seed, arm, roots[seed]) for seed in DIAGNOSIS_SEEDS for arm in ("c0", "c1")]
    failures = [row for row in arms if row["status"] != "complete"]
    report = {
        "schema_version": 1,
        "protocol": "tnbc_zero_training_oracle_diagnosis_two_seed_v1",
        "status": "complete" if not failures and allowed == {7, 8} and record_patients == {7, 8} else "issues_found",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository": {"branch": git_value("branch", "--show-current"), "commit": git_value("rev-parse", "HEAD"), "dirty_files": (git_value("status", "--short") or "").splitlines()},
        "development_manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path), "protocol_id": manifest.get("protocol_id"), "allowed_patients": sorted(allowed), "record_patients": sorted(record_patients), "record_count": len(records), "sealed_patients_absent": not bool(({9, 10, 11} & record_patients))},
        "reference_fixed_epoch_performance_summary": {"path": str(reference_path), "sha256": sha256_file(reference_path), "protocol": reference.get("protocol"), "seeds": reference.get("seeds")},
        "excluded_seed": EXCLUDED_SEED_3407,
        "epoch5_artifacts": arms,
        "export_capability": {
            "runner": str((ROOT / "tools" / "run_zero_training_oracle_diagnosis.py").resolve()),
            "runner_sha256": sha256_file(ROOT / "tools" / "run_zero_training_oracle_diagnosis.py"),
            "all_four_candidate_masks": "RLE export per automatic prompt group and native token",
            "candidate_quality": "exported per candidate",
            "native_selected_pool": "exported before global group/NMS/conflict assembly",
            "native_final_pool": "exported after original assembly/NMS",
            "prompt_group_id": "exported on every candidate",
            "inference_only": "no optimizer, backward, or training dataset construction",
        },
        "failures": failures,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(output)}, ensure_ascii=False))
    return 0 if report["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
