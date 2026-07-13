"""Protocol constants and fail-closed integrity helpers for StainCF-PMS."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from . import PROTOCOL_VERSION


BASE_SHA = "2a1348cb7a1158a6f77aae2f92c168f9552d8068"
SEED = 3407
PQ_IOU_THRESHOLD = 0.5
POINT_MATCH_DIAGONAL_FRACTION = 0.02
RESIDUAL_PEAK_MIN_DISTANCE = 12
RESIDUAL_PEAK_THRESHOLD = 0.50
BOOTSTRAP_REPLICATES = 2000
VIEWS = ("V0", "V1", "V2", "V3", "V4", "V5")
VIEW_NAMES = {
    "V0": "Original",
    "V1": "OD-Identity",
    "V2": "H-Weak",
    "V3": "H-Strong",
    "V4": "Within-Domain-Style",
    "V5": "Cross-Dataset-Style",
}
TNBC_AUDIT_PATIENTS = frozenset({7, 8})
TNBC_CALIBRATION_PATIENTS = frozenset(range(1, 7))
TNBC_CLOSED_PATIENTS = frozenset({9, 10, 11})


class ProtocolError(RuntimeError):
    """Raised when an input would violate the pre-registered Phase 0 protocol."""


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(canonical_json(value), encoding="utf-8")


def contains_closed_split_token(value: str | Path) -> bool:
    tokens = {part.lower() for part in Path(value).parts}
    return "test" in tokens or "test_" in tokens or "testset" in tokens


def assert_open_path(path: str | Path, purpose: str) -> None:
    if contains_closed_split_token(path):
        raise ProtocolError(f"closed test split path rejected for {purpose}: {path}")


def require_exact_sha256(path: str | Path, expected: str, label: str) -> str:
    observed = sha256_file(path)
    if observed.lower() != expected.lower():
        raise ProtocolError(
            f"{label} checksum mismatch: expected {expected}, observed {observed}"
        )
    return observed


def assert_tnbc_records(records: Iterable[dict[str, Any]], allowed: frozenset[int], purpose: str) -> None:
    records = list(records)
    if not records:
        raise ProtocolError(f"no TNBC records supplied for {purpose}")
    observed = {int(record["patient"]) for record in records}
    if not observed <= allowed:
        raise ProtocolError(
            f"TNBC {purpose} has patients {sorted(observed)}, allowed={sorted(allowed)}"
        )
    for record in records:
        assert_open_path(record["image"], purpose)
        assert_open_path(record["label"], purpose)


def baseline_selection_payload() -> dict[str, Any]:
    """The immutable baseline rationale, duplicated into each run artifact."""
    return {
        "protocol_version": PROTOCOL_VERSION,
        "candidate_shas": [BASE_SHA],
        "selected_base_sha": BASE_SHA,
        "selection_reason": (
            "Priority candidate is the last mainline inclusive-IoU>=0.5 alignment "
            "commit. It contains standard StainPMS evaluation parameters and checkpoint "
            "paths; no post-candidate retired-route training commit is inherited."
        ),
        "priority_candidate_parent_diff": [
            "sam2_train/modeling/stats_utils.py",
            "stainroute/oracle.py",
            "tests/test_stainroute_metrics.py",
            "tools/analyze_eval_artifacts.py",
        ],
        "retired_route_isolation": {
            "post_candidate_retired_training_commits_inherited": False,
            "new_audit_imports_retired_route_modules": False,
            "note": (
                "The historical baseline tree contains legacy packages, including a "
                "stainroute file touched by the inclusive-metric alignment commit. "
                "StainCF-PMS neither imports nor executes those packages."
            ),
        },
        "historical_baseline_values": {
            "monuseg_stainpms_pq": 0.657768,
            "tnbc_e156_stainpms_pq": 0.668077,
            "verification_status": (
                "Provided reference values; not re-run on prohibited formal test splits."
            ),
        },
        "metric_contract": {
            "pq_match_iou": PQ_IOU_THRESHOLD,
            "threshold_semantics": "inclusive (IoU >= 0.5)",
            "primary_metrics": ["AJI", "AJI+", "PQ"],
        },
    }


def selection_payload_with_implementation_sha(repo_root: str | Path) -> dict[str, Any]:
    payload = baseline_selection_payload()
    try:
        payload["implementation_git_sha"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(repo_root), text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        payload["implementation_git_sha"] = "unavailable"
    return payload


@dataclass(frozen=True)
class CheckpointSpec:
    dataset: str
    path: Path
    sha256: str
