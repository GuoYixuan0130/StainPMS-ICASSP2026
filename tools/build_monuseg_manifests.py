"""Materialize versioned MoNuSeg manifests without decoding sealed test images.

The official page, archive identities and expected case sets are declared in
``configs/manifests/monuseg_release_v1.json``.  This tool verifies the actual
downloaded ZIP files, hashes train image/XML members, and hashes only raw image
members from the sealed test ZIP.  Test annotations are never opened.

The generated manifests are machine-readable audit artifacts.  They are not
automatically copied into a training configuration or treated as a locked
development split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Iterable


IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
SOURCE_IMAGE_SUFFIXES = {".tif", ".tiff"}
ANNOTATION_PATH_TOKENS = {
    "annotation",
    "annotations",
    "ground truth",
    "ground_truth",
    "label",
    "labels",
    "mask",
    "masks",
}
TCGA_SAMPLE_RE = re.compile(r"^(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}(?:-[A-Z0-9]{2,3}){3})", re.I)


def _sha256_stream(handle: BinaryIO) -> str:
    digest = hashlib.sha256()
    for block in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return _sha256_stream(handle)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)
        handle.write("\n")


def _git_value(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _sample_id(stem: str) -> str:
    match = TCGA_SAMPLE_RE.match(stem.upper())
    if not match:
        raise ValueError(f"cannot derive TCGA sample identity from {stem!r}")
    return match.group(1).upper()


def _case_id(sample_id: str) -> str:
    parts = sample_id.split("-")
    if len(parts) < 3:
        raise ValueError(f"invalid TCGA sample id: {sample_id}")
    return "-".join(parts[:3])


def _archive_identity(path: Path, *, downloaded_at_utc: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path.resolve()),
        "filename": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        "downloaded_at_utc": downloaded_at_utc,
    }


def _validate_utc_timestamp(value: str) -> str:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"invalid --downloaded-at-utc timestamp: {value!r}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError("--downloaded-at-utc must include an explicit UTC offset")
    return value


def _zip_members_by_stem(
    archive: zipfile.ZipFile,
    suffixes: Iterable[str],
) -> dict[str, zipfile.ZipInfo]:
    suffix_set = {value.lower() for value in suffixes}
    result: dict[str, zipfile.ZipInfo] = {}
    for info in archive.infolist():
        if info.is_dir():
            continue
        member_path = Path(info.filename)
        if member_path.suffix.lower() not in suffix_set:
            continue
        if member_path.name.startswith("._") or "__MACOSX" in member_path.parts:
            continue
        stem = member_path.stem.upper()
        if stem in result:
            raise ValueError(
                f"duplicate ZIP member stem {stem}: {result[stem].filename}, {info.filename}"
            )
        result[stem] = info
    return result


def _zip_source_images_by_stem(
    archive: zipfile.ZipFile,
) -> dict[str, zipfile.ZipInfo]:
    """Select source TIFFs while rejecting annotation-like ZIP paths.

    This is especially important for the sealed test archive: callers may hash
    raw source image bytes, but must not open image-formatted annotations.
    """

    result: dict[str, zipfile.ZipInfo] = {}
    for info in archive.infolist():
        if info.is_dir():
            continue
        member_path = Path(info.filename)
        if member_path.suffix.lower() not in SOURCE_IMAGE_SUFFIXES:
            continue
        if member_path.name.startswith("._") or "__MACOSX" in member_path.parts:
            continue
        path_parts = [part.casefold() for part in Path(info.filename).parts[:-1]]
        if any(
            token in part
            for part in path_parts
            for token in ANNOTATION_PATH_TOKENS
        ):
            continue
        stem = member_path.stem.upper()
        if stem in result:
            raise ValueError(
                "duplicate source-image ZIP member stem "
                f"{stem}: {result[stem].filename}, {info.filename}"
            )
        result[stem] = info
    return result


def _hash_zip_member(archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> str:
    with archive.open(info, "r") as handle:
        return _sha256_stream(handle)


def _files_by_stem(root: Path, suffixes: Iterable[str]) -> dict[str, Path]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    suffix_set = {value.lower() for value in suffixes}
    result: dict[str, Path] = {}
    for path in sorted(root.iterdir(), key=lambda value: value.name):
        if not path.is_file() or path.suffix.lower() not in suffix_set:
            continue
        stem = path.stem.upper()
        if stem in result:
            raise ValueError(f"duplicate prepared stem {stem} under {root}")
        result[stem] = path
    return result


def _file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "filename": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _auxiliary_identity(path_value: str | None) -> dict[str, Any] | None:
    if not path_value:
        return None
    return _file_record(Path(path_value))


def _metadata_by_sample(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = list(config.get("classic30", [])) + list(config.get("extended7", []))
    return {str(row["sample_id"]).upper(): dict(row) for row in rows}


def _manifest(
    *,
    protocol_id: str,
    role: str,
    source_config: Path,
    source_config_sha256: str,
    archive: dict[str, Any],
    records: list[dict[str, Any]],
    access_policy: str,
    created_at: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "dataset": "monuseg",
        "protocol_id": protocol_id,
        "role": role,
        "status": "identity_complete",
        "access_policy": access_policy,
        "created_at_utc": created_at,
        "source_case_config": str(source_config),
        "source_case_config_sha256": source_config_sha256,
        "source_archive": archive,
        "record_count": len(records),
        "records": records,
    }


def build_manifests(args: argparse.Namespace) -> dict[str, Any]:
    downloaded_at_utc = _validate_utc_timestamp(args.downloaded_at_utc)
    config_path = Path(args.release_config)
    config = _load_json(config_path)
    config_sha = _sha256_file(config_path)
    train_archive_path = Path(args.train_archive)
    test_archive_path = Path(args.test_archive)
    prepared_images = _files_by_stem(
        Path(args.prepared_image_root), IMAGE_SUFFIXES
    )
    legacy_labels = _files_by_stem(Path(args.legacy_label_root), {".mat"})

    classic_order = [str(row["sample_id"]).upper() for row in config["classic30"]]
    extended_order = [str(row["sample_id"]).upper() for row in config["extended7"]]
    test_order = [
        str(row["sample_id"]).upper() for row in config["test14_expected_identities"]
    ]
    expected_train = set(classic_order) | set(extended_order)
    expected_test = set(test_order)
    if set(classic_order) & set(extended_order):
        raise ValueError("classic30 and extended7 overlap in the release config")
    if len(classic_order) != 30 or len(extended_order) != 7 or len(test_order) != 14:
        raise ValueError("release config does not contain exact 30/7/14 identities")

    train_archive_identity = _archive_identity(
        train_archive_path, downloaded_at_utc=downloaded_at_utc
    )
    test_archive_identity = _archive_identity(
        test_archive_path, downloaded_at_utc=downloaded_at_utc
    )
    metadata = _metadata_by_sample(config)

    train_records_by_id: dict[str, dict[str, Any]] = {}
    with zipfile.ZipFile(train_archive_path, "r") as archive:
        image_members = _zip_source_images_by_stem(archive)
        xml_members = _zip_members_by_stem(archive, {".xml"})
        image_ids = {_sample_id(stem): (stem, info) for stem, info in image_members.items()}
        xml_ids = {_sample_id(stem): (stem, info) for stem, info in xml_members.items()}
        if set(image_ids) != expected_train:
            raise ValueError(
                "training archive case mismatch: "
                f"missing={sorted(expected_train - set(image_ids))}, "
                f"unexpected={sorted(set(image_ids) - expected_train)}"
            )
        if set(xml_ids) != expected_train:
            raise ValueError(
                "training XML case mismatch: "
                f"missing={sorted(expected_train - set(xml_ids))}, "
                f"unexpected={sorted(set(xml_ids) - expected_train)}"
            )
        for sample_id in classic_order + extended_order:
            image_stem, image_info = image_ids[sample_id]
            _, xml_info = xml_ids[sample_id]
            if image_stem not in prepared_images or image_stem not in legacy_labels:
                raise ValueError(
                    f"prepared image/label missing for {sample_id}: stem={image_stem}"
                )
            row = dict(metadata[sample_id])
            row.update(
                {
                    "subset": "classic30" if sample_id in set(classic_order) else "extended7",
                    "source_image_member": image_info.filename,
                    "source_image_size_bytes": image_info.file_size,
                    "source_image_crc32": f"{image_info.CRC:08x}",
                    "source_image_sha256": _hash_zip_member(archive, image_info),
                    "source_xml_member": xml_info.filename,
                    "source_xml_size_bytes": xml_info.file_size,
                    "source_xml_crc32": f"{xml_info.CRC:08x}",
                    "source_xml_sha256": _hash_zip_member(archive, xml_info),
                    "image_path": str(prepared_images[image_stem].resolve()),
                    "image_sha256": _sha256_file(prepared_images[image_stem]),
                    "label_path": str(legacy_labels[image_stem].resolve()),
                    "label_sha256": _sha256_file(legacy_labels[image_stem]),
                    "label_version": "legacy_prepared_pending_xml_audit",
                }
            )
            train_records_by_id[sample_id] = row

    test_records_by_id: dict[str, dict[str, Any]] = {}
    with zipfile.ZipFile(test_archive_path, "r") as archive:
        # Deliberately enumerate and hash image members only.  XML, MAT and
        # other annotation-like members are not opened by this code path.
        image_members = _zip_source_images_by_stem(archive)
        image_ids = {_sample_id(stem): info for stem, info in image_members.items()}
        if set(image_ids) != expected_test:
            raise ValueError(
                "sealed test image identity mismatch: "
                f"missing={sorted(expected_test - set(image_ids))}, "
                f"unexpected={sorted(set(image_ids) - expected_test)}"
            )
        for configured in config["test14_expected_identities"]:
            sample_id = str(configured["sample_id"]).upper()
            info = image_ids[sample_id]
            test_records_by_id[sample_id] = {
                "sample_id": sample_id,
                "case": _case_id(sample_id),
                "subset": "sealed_test14_identity_only",
                "source_image_member": info.filename,
                "source_image_size_bytes": info.file_size,
                "source_image_crc32": f"{info.CRC:08x}",
                "source_image_sha256": _hash_zip_member(archive, info),
            }

    train_cases = {_case_id(value) for value in train_records_by_id}
    test_cases = {_case_id(value) for value in test_records_by_id}
    train_image_hashes = {
        row["source_image_sha256"] for row in train_records_by_id.values()
    }
    test_image_hashes = {
        row["source_image_sha256"] for row in test_records_by_id.values()
    }
    overlap = {
        "case_ids": sorted(train_cases & test_cases),
        "source_image_sha256": sorted(train_image_hashes & test_image_hashes),
    }
    if overlap["case_ids"] or overlap["source_image_sha256"]:
        raise ValueError(f"train/test identity overlap: {overlap}")

    created_at = datetime.now(timezone.utc).isoformat()
    out_dir = Path(args.output_dir)
    download37_records = [
        train_records_by_id[value] for value in classic_order + extended_order
    ]
    classic_records = [train_records_by_id[value] for value in classic_order]
    extended_records = [train_records_by_id[value] for value in extended_order]
    test_records = [test_records_by_id[value] for value in test_order]
    outputs = {
        "monuseg_download37_v1": _manifest(
            protocol_id="monuseg_download37_v1",
            role="stainpms_continuity_training_pool",
            source_config=config_path,
            source_config_sha256=config_sha,
            archive=train_archive_identity,
            records=download37_records,
            access_policy="train_annotations_allowed",
            created_at=created_at,
        ),
        "monuseg_challenge30_v1": _manifest(
            protocol_id="monuseg_challenge30_v1",
            role="original_challenge_training_pool",
            source_config=config_path,
            source_config_sha256=config_sha,
            archive=train_archive_identity,
            records=classic_records,
            access_policy="train_annotations_allowed_no_extended7_for_selection",
            created_at=created_at,
        ),
        "monuseg_extended7_v1": _manifest(
            protocol_id="monuseg_extended7_v1",
            role="candidate_development_identity_audit_only",
            source_config=config_path,
            source_config_sha256=config_sha,
            archive=train_archive_identity,
            records=extended_records,
            access_policy="train_annotations_allowed_not_locked_for_model_selection",
            created_at=created_at,
        ),
        "monuseg_test14_identity_v1": _manifest(
            protocol_id="monuseg_test14_identity_v1",
            role="sealed_final_test_identity_only",
            source_config=config_path,
            source_config_sha256=config_sha,
            archive=test_archive_identity,
            records=test_records,
            access_policy="raw_image_identity_only_no_decode_no_annotation_access",
            created_at=created_at,
        ),
    }
    for protocol_id, payload in outputs.items():
        _write_json(out_dir / f"{protocol_id}.json", payload)

    report = {
        "schema_version": 1,
        "phase": "0.5",
        "status": "complete",
        "created_at_utc": created_at,
        "git_branch": _git_value("branch", "--show-current"),
        "git_commit": _git_value("rev-parse", "HEAD"),
        "release_config": str(config_path),
        "release_config_sha256": config_sha,
        "official_page": config["official_page"],
        "train_archive": train_archive_identity,
        "test_archive": test_archive_identity,
        "official_organ_information_file": _auxiliary_identity(args.organ_info),
        "official_xml_converter_file": _auxiliary_identity(args.official_converter),
        "counts": {"classic30": 30, "extended7": 7, "download37": 37, "test14": 14},
        "train_test_isolation": {"status": "isolated", "overlap": overlap},
        "test_access_attestation": {
            "decoded_images": False,
            "opened_annotation_members": False,
            "hashed_raw_image_members": True,
        },
        "generated_manifests": {
            key: str((out_dir / f"{key}.json").resolve()) for key in outputs
        },
    }
    _write_json(out_dir / "monuseg_release_audit_v1.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--release-config",
        default="configs/manifests/monuseg_release_v1.json",
    )
    parser.add_argument("--train-archive", required=True)
    parser.add_argument("--test-archive", required=True)
    parser.add_argument("--prepared-image-root", required=True)
    parser.add_argument("--legacy-label-root", required=True)
    parser.add_argument("--downloaded-at-utc", required=True)
    parser.add_argument("--organ-info", default="")
    parser.add_argument("--official-converter", default="")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> int:
    try:
        report = build_manifests(parse_args())
    except Exception as exc:
        print(json.dumps({"status": "issues_found", "error": str(exc)}))
        return 2
    print(
        json.dumps(
            {
                "status": report["status"],
                "counts": report["counts"],
                "train_test_isolation": report["train_test_isolation"]["status"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
