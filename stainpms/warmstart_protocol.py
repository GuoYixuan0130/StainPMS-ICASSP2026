"""Safety and reporting helpers for exploratory C0/C1 warm-start runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


MONUSEG_TRAIN37_SAMPLE_IDS = frozenset(
    {
        "TCGA-18-5592-01Z-00-DX1",
        "TCGA-21-5784-01Z-00-DX1",
        "TCGA-21-5786-01Z-00-DX1",
        "TCGA-38-6178-01Z-00-DX1",
        "TCGA-49-4488-01Z-00-DX1",
        "TCGA-50-5931-01Z-00-DX1",
        "TCGA-A7-A13E-01Z-00-DX1",
        "TCGA-A7-A13F-01Z-00-DX1",
        "TCGA-AR-A1AK-01Z-00-DX1",
        "TCGA-AR-A1AS-01Z-00-DX1",
        "TCGA-AY-A8YK-01A-01-TS1",
        "TCGA-B0-5698-01Z-00-DX1",
        "TCGA-B0-5710-01Z-00-DX1",
        "TCGA-B0-5711-01Z-00-DX1",
        "TCGA-BC-A217-01Z-00-DX1",
        "TCGA-CH-5767-01Z-00-DX1",
        "TCGA-DK-A2I6-01A-01-TS1",
        "TCGA-E2-A14V-01Z-00-DX1",
        "TCGA-E2-A1B5-01Z-00-DX1",
        "TCGA-F9-A8NY-01Z-00-DX1",
        "TCGA-FG-A87N-01Z-00-DX1",
        "TCGA-G2-A2EK-01A-02-TSB",
        "TCGA-G9-6336-01Z-00-DX1",
        "TCGA-G9-6348-01Z-00-DX1",
        "TCGA-G9-6356-01Z-00-DX1",
        "TCGA-G9-6362-01Z-00-DX1",
        "TCGA-G9-6363-01Z-00-DX1",
        "TCGA-HE-7128-01Z-00-DX1",
        "TCGA-HE-7129-01Z-00-DX1",
        "TCGA-HE-7130-01Z-00-DX1",
        "TCGA-KB-A93J-01A-01-TS1",
        "TCGA-MH-A561-01Z-00-DX1",
        "TCGA-NH-A8F7-01A-01-TS1",
        "TCGA-RD-A8N9-01A-01-TS1",
        "TCGA-UZ-A9PJ-01Z-00-DX1",
        "TCGA-UZ-A9PN-01Z-00-DX1",
        "TCGA-XS-A8TJ-01Z-00-DX1",
    }
)


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


def validate_train_manifest_identity(
    manifest_path: Path, dataset: str
) -> dict[str, Any]:
    """Reject forbidden identities before dataset files are opened or hashed."""
    payload = read_json(manifest_path.resolve())
    if str(payload.get("dataset", "")).lower() != str(dataset).lower():
        raise ValueError("warm-start manifest dataset mismatch")
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("warm-start manifest requires records")
    if int(payload.get("record_count", len(records))) != len(records):
        raise ValueError("warm-start manifest record_count mismatch")
    sample_ids = [str(record.get("sample_id", "")) for record in records]
    if any(not sample_id for sample_id in sample_ids):
        raise ValueError("warm-start manifest contains an empty sample_id")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("warm-start manifest contains duplicate sample_id values")

    if dataset == "tnbc":
        patients = []
        for record in records:
            patient = int(record.get("patient", -1))
            if patient not in {1, 2, 3, 4, 5, 6}:
                raise ValueError(
                    f"TNBC warm-start rejects non-training patient before file access: {patient}"
                )
            patients.append(patient)
        if len(records) != 30:
            raise ValueError("TNBC warm-start requires the frozen 30-image p1-p6 manifest")
        declared = {int(value) for value in payload.get("allowed_patients", [])}
        if declared and declared != {1, 2, 3, 4, 5, 6}:
            raise ValueError("TNBC allowed_patients must be exactly p1-p6")
        scope = {"patients": sorted(set(patients)), "record_count": len(records)}
    elif dataset == "monuseg":
        if len(records) != 37:
            raise ValueError("MoNuSeg warm-start requires exactly train37")
        if set(sample_ids) != MONUSEG_TRAIN37_SAMPLE_IDS:
            raise ValueError("MoNuSeg warm-start identity differs from frozen train37")
        for record in records:
            for field in ("image_path", "label_path"):
                value = str(record.get(field, ""))
                parts = [part.lower() for part in Path(value).parts]
                if any(part == "test" or "test14" in part for part in parts):
                    raise ValueError(
                        f"MoNuSeg test identity rejected before file access: {value}"
                    )
        scope = {"role": "train37_only", "record_count": len(records)}
    else:
        raise ValueError(f"unsupported warm-start dataset: {dataset}")

    return {
        "path": str(manifest_path.resolve()),
        "sha256": sha256_file(manifest_path.resolve()),
        "protocol_id": payload.get("protocol_id"),
        "sample_ids": sample_ids,
        "scope": scope,
    }


def build_coverage_manifest(
    *,
    cache_dir: Path,
    train_manifest_identity: dict[str, Any],
    dataset: str,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    wall_seconds: float,
    repository: dict[str, str],
    command: list[str],
) -> dict[str, Any]:
    cache_dir = cache_dir.resolve()
    expected_names = list(train_manifest_identity["sample_ids"])
    expected_paths = [cache_dir / f"{sample_id}.npy" for sample_id in expected_names]
    missing = [str(path) for path in expected_paths if not path.is_file()]
    if missing:
        raise ValueError(f"coverage cache is incomplete: {missing[:3]}")
    actual_paths = sorted(cache_dir.glob("*.npy"), key=lambda path: path.name)
    if {path.name for path in actual_paths} != {path.name for path in expected_paths}:
        raise ValueError("coverage cache contains missing or extra NPY files")
    records = [
        {
            "sample_id": sample_id,
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for sample_id, path in zip(expected_names, expected_paths, strict=True)
    ]
    return {
        "schema_version": 1,
        "phase": "2A-warmstart-feasibility",
        "status": "complete",
        "protocol": "shared_train_only_initial_coverage_v1",
        "dataset": dataset,
        "train_manifest": train_manifest_identity,
        "checkpoint": {
            "path": str(checkpoint_path.resolve()),
            "sha256": checkpoint_sha256,
            "loaded_fields": ["model", "model1"],
            "embedded_texture_bank_loaded": False,
        },
        "cache_dir": str(cache_dir),
        "record_count": len(records),
        "records": records,
        "runtime": {"wall_seconds": float(wall_seconds)},
        "repository": repository,
        "command": command,
        "sealed_data_attestation": {
            "evaluation_loader_constructed": False,
            "TNBC_p7_p11_accessed": False,
            "MoNuSeg_test14_accessed": False,
        },
    }


def verify_coverage_manifest(
    coverage_manifest_path: Path,
    *,
    train_manifest_identity: dict[str, Any],
    checkpoint_sha256: str,
    dataset: str,
) -> dict[str, Any]:
    path = coverage_manifest_path.resolve()
    payload = read_json(path)
    if payload.get("status") != "complete" or payload.get("dataset") != dataset:
        raise ValueError("coverage manifest status/dataset mismatch")
    if payload.get("train_manifest", {}).get("sha256") != train_manifest_identity["sha256"]:
        raise ValueError("coverage was generated from a different train manifest")
    if payload.get("checkpoint", {}).get("sha256") != checkpoint_sha256:
        raise ValueError("coverage was generated from a different checkpoint")
    if payload.get("checkpoint", {}).get("embedded_texture_bank_loaded") is not False:
        raise ValueError("coverage manifest does not attest texture-bank isolation")
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("coverage manifest records are missing")
    if [str(record.get("sample_id")) for record in records] != train_manifest_identity["sample_ids"]:
        raise ValueError("coverage record order/identity differs from train manifest")
    cache_dir = Path(payload["cache_dir"]).resolve()
    for record in records:
        record_path = Path(record["path"]).resolve()
        if record_path.parent != cache_dir or not record_path.is_file():
            raise ValueError(f"invalid coverage path: {record_path}")
        if sha256_file(record_path) != str(record["sha256"]).lower():
            raise ValueError(f"coverage SHA256 mismatch: {record_path}")
    actual_names = {item.name for item in cache_dir.glob("*.npy")}
    expected_names = {f"{sample_id}.npy" for sample_id in train_manifest_identity["sample_ids"]}
    if actual_names != expected_names:
        raise ValueError("coverage directory changed after manifest creation")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "cache_dir": str(cache_dir),
        "record_count": len(records),
        "source_runtime": payload.get("runtime"),
    }


def finalize_runtime_audits(runtime_stats: dict[str, Any]) -> dict[str, Any]:
    """Convert accumulated loss/gradient sums into report-ready summaries."""
    result = dict(runtime_stats)
    candidate = result.get("candidate_loss_audit")
    if isinstance(candidate, dict) and int(candidate.get("step_count", 0)) > 0:
        steps = int(candidate["step_count"])
        candidate["means"] = {
            "stainpms_loss": candidate["stainpms_loss_sum"] / steps,
            "coverage_loss_before_lambda": candidate["coverage_loss_sum"] / steps,
            "quality_loss_before_lambda": candidate["quality_loss_sum"] / steps,
            "weighted_extra": candidate["weighted_extra_sum"] / steps,
            "total_loss": candidate["total_loss_sum"] / steps,
            "extra_to_total_ratio": candidate["extra_to_total_ratio_sum"] / steps,
        }
        for group in candidate.get("groups", {}).values():
            count = int(group["valid_prompt_count"])
            group["coverage_mean"] = (
                group["coverage_prompt_weighted_sum"] / count if count else 0.0
            )
            group["quality_mean"] = (
                group["quality_prompt_weighted_sum"] / count if count else 0.0
            )
            group["best_softmin_gradient_weight_mean"] = (
                group["best_softmin_weight_prompt_weighted_sum"] / count
                if count
                else None
            )
            group["effective_candidate_count_mean"] = (
                group["effective_candidate_count_prompt_weighted_sum"] / count
                if count
                else None
            )
    gradient = result.get("gradient_audit")
    if isinstance(gradient, dict) and int(gradient.get("step_count", 0)) > 0:
        steps = int(gradient["step_count"])
        gradient["group_l2_mean"] = {
            name: float(value) / steps
            for name, value in gradient.get("group_l2_sum", {}).items()
        }
    return result
