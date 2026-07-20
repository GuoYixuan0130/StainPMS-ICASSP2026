"""Verify official TNBC v1.1 and selectively extract patients 1--8 only.

The archive checksum necessarily reads the archive as an opaque byte stream.
ZIP members for patients 9--11 are never opened, extracted, named in output or
counted.  The central directory is examined only to select explicitly allowed
patient folders, as authorized by the Phase 0.5 protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any


OFFICIAL_RECORD_URL = "https://zenodo.org/records/2579118"
OFFICIAL_ARCHIVE_URL = (
    "https://zenodo.org/records/2579118/files/"
    "TNBC_NucleiSegmentation.zip?download=1"
)
OFFICIAL_FILENAME = "TNBC_NucleiSegmentation.zip"
OFFICIAL_SIZE_BYTES = 25232361
OFFICIAL_MD5 = "1605712a752b201b57eacc8f866adb4f"
ALLOWED_PATIENTS = set(range(1, 9))
CLOSED_PATIENTS = {9, 10, 11}
EXPECTED_IMAGES_PER_PATIENT = {1: 7, 2: 3, 3: 5, 4: 8, 5: 4, 6: 3, 7: 3, 8: 4}
FOLDER_RE = re.compile(r"^(SLIDE|GT)[_ -]*0*([0-9]+)$", re.I)
SAMPLE_RE = re.compile(r"^(?:0*([0-9]+)[_-])?0*([0-9]+)$")


def _hash_file(path: Path) -> tuple[str, str]:
    md5 = hashlib.md5()  # noqa: S324 - required to verify the publisher checksum
    sha256 = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            md5.update(block)
            sha256.update(block)
    return md5.hexdigest(), sha256.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _validate_utc(value: str) -> str:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"invalid UTC timestamp: {value!r}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError("--downloaded-at-utc must contain an explicit UTC offset")
    return value


def _folder_identity(info: zipfile.ZipInfo) -> tuple[str, int] | None:
    parts = PurePosixPath(info.filename.replace("\\", "/")).parts[:-1]
    matches: list[tuple[str, int]] = []
    for part in parts:
        match = FOLDER_RE.fullmatch(part)
        if match:
            matches.append((match.group(1).upper(), int(match.group(2))))
    if not matches:
        return None
    if len(set(matches)) != 1:
        raise ValueError("ambiguous patient folder in TNBC archive member")
    return matches[0]


def _sample_identity(patient: int, filename: str) -> tuple[str, int]:
    path = PurePosixPath(filename)
    if path.suffix.lower() != ".png":
        raise ValueError("allowed TNBC patient folder contains a non-PNG file")
    match = SAMPLE_RE.fullmatch(path.stem)
    if not match:
        raise ValueError("cannot map an allowed TNBC member to a sample index")
    declared_patient = int(match.group(1)) if match.group(1) is not None else patient
    if declared_patient != patient:
        raise ValueError("TNBC filename patient does not match its folder")
    image_index = int(match.group(2))
    return f"{patient:02d}_{image_index}", image_index


def extract_allowed(args: argparse.Namespace) -> dict[str, Any]:
    archive_path = Path(args.archive)
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    if archive_path.name != OFFICIAL_FILENAME:
        raise ValueError(
            f"official v1.1 filename must remain {OFFICIAL_FILENAME!r}, got {archive_path.name!r}"
        )
    if archive_path.stat().st_size != OFFICIAL_SIZE_BYTES:
        raise ValueError(
            f"official archive size mismatch: {archive_path.stat().st_size} != {OFFICIAL_SIZE_BYTES}"
        )
    actual_md5, actual_sha256 = _hash_file(archive_path)
    if actual_md5 != OFFICIAL_MD5:
        raise ValueError(f"official archive MD5 mismatch: {actual_md5}")
    downloaded_at = _validate_utc(args.downloaded_at_utc)
    output_root = Path(args.output_root).resolve()
    records: dict[str, dict[str, dict[str, Any]]] = {"SLIDE": {}, "GT": {}}

    with zipfile.ZipFile(archive_path, "r") as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            identity = _folder_identity(info)
            if identity is None:
                continue
            kind, patient = identity
            if patient not in ALLOWED_PATIENTS:
                # Do not open, extract, count or record closed-patient content.
                continue
            sample_id, image_index = _sample_identity(patient, info.filename)
            if sample_id in records[kind]:
                raise ValueError(f"duplicate allowed TNBC {kind} sample: {sample_id}")
            with archive.open(info, "r") as handle:
                payload = handle.read()
            payload_sha = _sha256_bytes(payload)
            target = output_root / f"{kind}_{patient:02d}" / f"{image_index}.png"
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if _sha256_bytes(target.read_bytes()) != payload_sha:
                    raise FileExistsError(
                        f"refusing to overwrite non-identical allowed file: {target}"
                    )
            else:
                with target.open("xb") as handle:
                    handle.write(payload)
            records[kind][sample_id] = {
                "sample_id": sample_id,
                "patient": patient,
                "image_index": image_index,
                "source_member": info.filename,
                "source_size_bytes": info.file_size,
                "source_crc32": f"{info.CRC:08x}",
                "source_sha256": payload_sha,
                "extracted_path": str(target),
                "extracted_sha256": payload_sha,
            }

    expected_samples = {
        f"{patient:02d}_{index}"
        for patient, count in EXPECTED_IMAGES_PER_PATIENT.items()
        for index in range(1, count + 1)
    }
    for kind in ("SLIDE", "GT"):
        found = set(records[kind])
        if found != expected_samples:
            raise ValueError(
                f"allowed {kind} sample mismatch: "
                f"missing={sorted(expected_samples - found)}, "
                f"unexpected={sorted(found - expected_samples)}"
            )

    created_at = datetime.now(timezone.utc).isoformat()
    return {
        "schema_version": 1,
        "phase": "0.5",
        "dataset": "tnbc",
        "protocol_id": "tnbc_official_v1_1_p1_8_selective_extract_v1",
        "status": "complete",
        "created_at_utc": created_at,
        "source": {
            "record_url": OFFICIAL_RECORD_URL,
            "archive_url": OFFICIAL_ARCHIVE_URL,
            "version": "1.1",
            "archive_path": str(archive_path.resolve()),
            "filename": archive_path.name,
            "size_bytes": archive_path.stat().st_size,
            "publisher_md5": OFFICIAL_MD5,
            "verified_md5": actual_md5,
            "sha256": actual_sha256,
            "downloaded_at_utc": downloaded_at,
        },
        "allowed_patients": sorted(ALLOWED_PATIENTS),
        "expected_sample_count": len(expected_samples),
        "counts_by_allowed_patient": {
            str(key): value for key, value in EXPECTED_IMAGES_PER_PATIENT.items()
        },
        "access_attestation": {
            "archive_read_as_opaque_bytes_for_checksum": True,
            "central_directory_examined_for_selective_extraction": True,
            "closed_patient_member_content_opened": False,
            "closed_patient_members_extracted": False,
            "closed_patient_identities_or_counts_recorded": False,
        },
        "records": {
            kind.lower(): [records[kind][sample] for sample in sorted(records[kind])]
            for kind in ("SLIDE", "GT")
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--downloaded-at-utc", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = extract_allowed(args)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    except Exception as exc:
        print(json.dumps({"status": "issues_found", "error": str(exc)}))
        return 2
    print(
        json.dumps(
            {
                "status": report["status"],
                "allowed_samples": report["expected_sample_count"],
                "output": str(args.output),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
