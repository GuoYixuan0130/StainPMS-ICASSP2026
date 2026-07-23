#!/usr/bin/env python3
"""Write the pre-registered C3 joint gate for old seed-2027 plus reconstructed seed-1337."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c3-audit", required=True, type=Path)
    parser.add_argument("--reconstructed-freeze-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    c3_path = args.c3_audit.resolve()
    frozen_path = args.reconstructed_freeze_manifest.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite joint C3 gate: {output}")
    c3 = read_json(c3_path)
    frozen = read_json(frozen_path)
    seeds = {int(row.get("seed", -1)): row for row in c3.get("per_seed", [])}
    gate = c3.get("c3_gate", {})
    seed2027_reused = c3.get("full_oracle_reproduction", {}).get("2027", {}).get("status") == "reused_historical_c3_without_rerun"
    seed1337_reconstructed = (
        c3.get("full_oracle_reproduction", {}).get("1337", {}).get("status") == "not_applicable_reconstructed_lineage"
        and frozen.get("status") == "frozen_before_development_access"
        and frozen.get("lineage") == "reconstructed C1 seed-1337 lineage"
    )
    checks = {
        "exactly_two_seeds": set(seeds) == {2027, 1337},
        "seed2027_historical_c3_reused_without_rerun": seed2027_reused,
        "seed1337_reconstructed_epoch5_frozen_before_development": seed1337_reconstructed,
        "same_supported_operation_conflict_order": (
            gate.get("status") == "one_direction_supported"
            and gate.get("single_supported_operation") == "conflict_order_oracle"
            and gate.get("proposed_direction") == "conflict-set structured ranking"
        ),
    }
    status = "eligible_for_C4_review" if all(checks.values()) else "close_assembly_scoring_route"
    payload = {
        "schema_version": 1,
        "protocol": "tnbc_c3_reconstructed_seed1337_joint_gate_v1",
        "status": status,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "joint gate only: retained historical seed-2027 C3 plus new reconstructed seed-1337 C3; no C4 implementation or training is authorized by this report",
        "checks": checks,
        "sources": {
            "c3_audit": {"path": str(c3_path), "sha256": sha256_file(c3_path)},
            "reconstructed_seed1337_freeze_manifest": {"path": str(frozen_path), "sha256": sha256_file(frozen_path)},
        },
        "seed_summaries": {
            str(seed): {
                "native_pq": seeds[seed]["patient_macro"]["stages_patient_macro"]["native"]["pq"],
                "conflict_order_oracle_delta_pq": seeds[seed]["patient_macro"]["deltas_vs_native_patient_macro"]["conflict_order_oracle"]["pq"],
                "full_score_oracle_delta_pq": seeds[seed]["patient_macro"]["deltas_vs_native_patient_macro"]["full_score_oracle"]["pq"],
                "source_identity": seeds[seed].get("source_identity"),
            }
            for seed in (2027, 1337)
        },
        "c3_gate": gate,
        "next_action": (
            "submit for project-lead C4 review; do not start C4 automatically"
            if status == "eligible_for_C4_review"
            else "stop the assembly-scoring route; do not tune or retrain as remediation"
        ),
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "c3_reconstructed_joint_gate.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# C3 reconstructed seed-1337 joint gate", "", f"- Status: `{status}`.", "", "## Checks", ""]
    lines += [f"- {name}: {'pass' if value else 'fail'}" for name, value in checks.items()]
    lines += ["", "## Conflict-order oracle", "", "| seed | native PQ | conflict-order delta PQ | full-score delta PQ |", "|---:|---:|---:|---:|"]
    for seed in (2027, 1337):
        item = payload["seed_summaries"][str(seed)]
        lines.append(f"| {seed} | {item['native_pq']:.6f} | {item['conflict_order_oracle_delta_pq']:+.6f} | {item['full_score_oracle_delta_pq']:+.6f} |")
    lines += ["", f"- Next action: {payload['next_action']}", ""]
    (output / "c3_reconstructed_joint_gate.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": status, "output_dir": str(output)}, ensure_ascii=False))
    # A closed route is a valid completed diagnostic outcome, not a launcher
    # failure.  The caller must inspect the explicit status before any C4 work.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
