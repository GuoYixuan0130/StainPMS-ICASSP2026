"""Freeze a p1--p6 TNBC prepared-label manifest for a Phase 0.5 smoke.

The tool consumes an already-authorized explicit source manifest.  It never
walks ``image_root`` and rejects a closed patient before opening any image or
label.  The result is loader-runnable and records content SHA256 values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CLOSED_PATIENTS = {9, 10, 11}
SAMPLE_RE = re.compile(r"^(?:patient[_-]?)?0*([0-9]{1,2})[_-]0*([0-9]+)$", re.I)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_entries(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        raise ValueError("source manifest must be a JSON object or list")
    for key in ("samples", "entries", "items", "paths", "files", "images", "records"):
        value = payload.get(key)
        if isinstance(value, list):
            return value
    raise ValueError("source manifest has no supported explicit sample list")


def first_value(record: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = record.get(key)
        if value not in (None, ""):
            return value
    return None


def infer_patient(sample_id: str) -> int:
    match = SAMPLE_RE.fullmatch(sample_id)
    if not match:
        raise ValueError(f"cannot infer TNBC patient from sample_id={sample_id!r}")
    return int(match.group(1))


def resolve_sample(raw: Any, image_root: Path, label_root: Path) -> tuple[str, int, Path, Path]:
    record = {"image": raw} if isinstance(raw, str) else dict(raw)
    image_value = first_value(
        record,
        ("image", "image_name", "image_path", "img", "path", "file", "filename", "name"),
    )
    if image_value is None:
        raise ValueError(f"sample has no image value: {record}")
    image_path = Path(str(image_value))
    if not image_path.suffix:
        image_path = image_path.with_suffix(".png")
    if not image_path.is_absolute():
        image_path = image_root / image_path
    sample_id = str(record.get("sample_id") or image_path.stem)
    patient_value = first_value(record, ("patient", "patient_id", "patient_number", "subject"))
    patient = int(patient_value) if patient_value is not None else infer_patient(sample_id)
    label_value = first_value(record, ("label", "label_path", "gt", "mask"))
    label_path = Path(str(label_value)) if label_value is not None else label_root / f"{image_path.stem}.mat"
    if not label_path.is_absolute():
        label_path = label_root / label_path
    return sample_id, patient, image_path.resolve(), label_path.resolve()


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    source_path = Path(args.source_manifest).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"source manifest not found: {source_path}")
    image_root = Path(args.image_root).resolve()
    label_root = Path(args.label_root).resolve()
    allowed_patients = {int(value) for value in args.allowed_patients}
    if not allowed_patients or allowed_patients & CLOSED_PATIENTS:
        raise ValueError("allowed patients must be non-empty and exclude closed p9--p11")

    with source_path.open("r", encoding="utf-8") as handle:
        raw_entries = extract_entries(json.load(handle))
    resolved = [resolve_sample(raw, image_root, label_root) for raw in raw_entries]

    # This validation happens before any sample content is opened or hashed.
    sample_ids: set[str] = set()
    for sample_id, patient, _, _ in resolved:
        if sample_id in sample_ids:
            raise ValueError(f"duplicate sample_id: {sample_id}")
        sample_ids.add(sample_id)
        if patient in CLOSED_PATIENTS:
            raise ValueError(f"closed TNBC patient {patient} rejected before file access")
        if patient not in allowed_patients:
            raise ValueError(
                f"patient {patient} is not allowed by --allowed-patients={sorted(allowed_patients)}"
            )
    if args.expected_count is not None and len(resolved) != args.expected_count:
        raise ValueError(f"expected {args.expected_count} records, found {len(resolved)}")

    records = []
    for sample_id, patient, image_path, label_path in resolved:
        if not image_path.is_file():
            raise FileNotFoundError(f"missing image for {sample_id}: {image_path}")
        if not label_path.is_file():
            raise FileNotFoundError(f"missing prepared label for {sample_id}: {label_path}")
        records.append(
            {
                "sample_id": sample_id,
                "patient": patient,
                "image_path": str(image_path),
                "image_sha256": sha256_file(image_path),
                "label_path": str(label_path),
                "label_sha256": sha256_file(label_path),
            }
        )

    return {
        "schema_version": 1,
        "dataset": "tnbc",
        "protocol_id": args.protocol_id,
        "status": "smoke_only_prepared_labels_pending_raw_binary_gt_audit",
        "role": "phase05_train_only_smoke",
        "record_count": len(records),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifest": str(source_path),
        "source_manifest_sha256": sha256_file(source_path),
        "allowed_patients": sorted(allowed_patients),
        "sealed_patient_policy": "p9-p11 rejected before sample file access",
        "prepared_label_policy": "smoke_only_pending_raw_binary_gt_audit",
        "records": records,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--label-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--allowed-patients", type=int, nargs="+", required=True)
    parser.add_argument("--expected-count", type=int, default=None)
    parser.add_argument("--protocol-id", default="tnbc_p1_6_smoke_prepared_labels_v1")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_manifest(args)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"status": "complete", "output": str(output), "records": report["record_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
