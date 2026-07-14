"""Immutable-manifest helpers for the ResiMix-PMS data boundary.

The functions in this module intentionally only *read* frozen manifests.  They
never select a new holdout, crop, or image; callers must pass the canonical
paths supplied by the experiment owner.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


REQUIRED_FROZEN_FILES = (
    "monuseg_lite_manifest.json",
    "monuseg_lite_patches.json",
    "SHA256SUMS",
)
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_IMAGE_NAME_FIELDS = ("image_name", "image", "name")
_PATIENT_ID_FIELDS = ("patient_id", "patient", "patient_number")


class ManifestPreflightError(RuntimeError):
    """Raised when a frozen input is missing, malformed, or has changed."""


@dataclass(frozen=True)
class FrozenBundle:
    """The reproducible result of validating a canonical MoNuSeg-Lite bundle."""

    artifact_dir: Path
    file_sha256: Mapping[str, str]
    checksum_file_sha256: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "artifact_dir": str(self.artifact_dir),
            "required_files": list(REQUIRED_FROZEN_FILES),
            "file_sha256": dict(self.file_sha256),
            "checksum_file_sha256": self.checksum_file_sha256,
        }


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a SHA-256 digest without loading the complete file into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_checksum_name(value: str) -> str:
    value = value.strip().lstrip("*").replace("\\", "/")
    pure = PurePosixPath(value)
    if not value or pure.is_absolute() or ".." in pure.parts:
        raise ManifestPreflightError("SHA256SUMS contains an unsafe filename")
    normalized = str(pure)
    if normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def _parse_sha256sums(path: Path) -> Dict[str, str]:
    entries: Dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or not _SHA256_RE.fullmatch(parts[0]):
            raise ManifestPreflightError(
                "Malformed SHA256SUMS entry at line {}".format(line_number)
            )
        filename = _safe_checksum_name(parts[1])
        if filename in entries:
            raise ManifestPreflightError("Duplicate SHA256SUMS entry: {}".format(filename))
        entries[filename] = parts[0].lower()
    return entries


def validate_frozen_bundle(path: str | Path) -> FrozenBundle:
    """Validate the exact frozen MoNuSeg-Lite manifest bundle.

    ``path`` must be the original artifact directory containing exactly the
    required filenames.  A missing checksum entry or a digest mismatch raises
    :class:`ManifestPreflightError`; callers must stop rather than re-pick
    images or patches.
    """

    artifact_dir = Path(path)
    if not artifact_dir.is_dir():
        raise ManifestPreflightError("Frozen artifact directory is unavailable: {}".format(artifact_dir))

    checksum_path = artifact_dir / "SHA256SUMS"
    if not checksum_path.is_file():
        raise ManifestPreflightError("Missing frozen checksum file: {}".format(checksum_path))
    expected = _parse_sha256sums(checksum_path)

    actual: Dict[str, str] = {}
    for filename in REQUIRED_FROZEN_FILES[:-1]:
        file_path = artifact_dir / filename
        if not file_path.is_file():
            raise ManifestPreflightError("Missing frozen manifest: {}".format(file_path))
        if filename not in expected:
            raise ManifestPreflightError("SHA256SUMS has no entry for {}".format(filename))
        digest = sha256_file(file_path)
        if digest != expected[filename]:
            raise ManifestPreflightError(
                "Frozen manifest checksum mismatch for {}: expected {}, got {}".format(
                    filename, expected[filename], digest
                )
            )
        actual[filename] = digest

    return FrozenBundle(
        artifact_dir=artifact_dir.resolve(),
        file_sha256=actual,
        checksum_file_sha256=sha256_file(checksum_path),
    )


def copy_validated_frozen_bundle(source: str | Path, destination: str | Path) -> FrozenBundle:
    """Copy only a validated canonical bundle to a new experiment artifact."""

    bundle = validate_frozen_bundle(source)
    destination_path = Path(destination)
    destination_path.mkdir(parents=True, exist_ok=False)
    for filename in REQUIRED_FROZEN_FILES:
        shutil.copy2(bundle.artifact_dir / filename, destination_path / filename)
    return validate_frozen_bundle(destination_path)


def _read_rows(path: str | Path) -> List[Any]:
    source = Path(path)
    if not source.is_file():
        raise ManifestPreflightError("Manifest is unavailable: {}".format(source))
    suffix = source.suffix.lower()
    if suffix == ".csv":
        with source.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))
    if suffix != ".json":
        raise ManifestPreflightError("Only JSON and CSV manifests are supported: {}".format(source))
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestPreflightError("Invalid JSON manifest: {}".format(source)) from exc
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, Mapping):
        raise ManifestPreflightError("Manifest must contain a JSON list or object: {}".format(source))
    for key in ("crops", "patches", "records", "items", "images", "allowed_images", "data"):
        value = payload.get(key)
        if isinstance(value, list):
            return value
    if any(key in payload for key in _IMAGE_NAME_FIELDS):
        return [payload]
    # A small name -> metadata mapping is a common compact manifest format.
    if payload and all(isinstance(value, Mapping) for value in payload.values()):
        return [dict(value, image_name=name) for name, value in payload.items()]
    raise ManifestPreflightError("Cannot locate records in manifest: {}".format(source))


def _image_name(row: Any) -> str:
    if isinstance(row, str) and row.strip():
        return row.strip()
    if not isinstance(row, Mapping):
        raise ManifestPreflightError("Manifest record is not an image name or object")
    for field in _IMAGE_NAME_FIELDS:
        value = row.get(field)
        if value is not None and str(value).strip():
            return str(value).strip()
    raise ManifestPreflightError("Manifest record lacks image_name/image/name")


def load_allowed_image_names(path: str | Path) -> List[str]:
    """Load canonical image names, accepting ``image_name``, ``image`` or ``name``.

    The result preserves manifest order and rejects duplicates so a caller
    cannot silently amplify a favorable image during training.
    """

    names = [_image_name(row) for row in _read_rows(path)]
    if not names:
        raise ManifestPreflightError("Frozen image manifest is empty")
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ManifestPreflightError("Frozen image manifest contains duplicates: {}".format(duplicates))
    return names


def _patient_id(
    row: Mapping[str, Any], name_to_patient: Optional[Mapping[str, int]] = None
) -> int:
    for field in _PATIENT_ID_FIELDS:
        value = row.get(field)
        if value is None or str(value).strip() == "":
            continue
        text = str(value).strip()
        match = re.fullmatch(r"(?:patient[_ -]?)?(\d+)", text, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
        try:
            numeric = float(text)
        except (TypeError, ValueError) as exc:
            raise ManifestPreflightError("Invalid patient identifier: {}".format(text)) from exc
        if numeric.is_integer():
            return int(numeric)
        raise ManifestPreflightError("Patient identifier must be an integer: {}".format(text))
    if name_to_patient is not None:
        image_name = _image_name(row)
        if image_name not in name_to_patient:
            raise ManifestPreflightError(
                "TNBC image is absent from the approved name_to_patient mapping: {}".format(image_name)
            )
        try:
            return int(name_to_patient[image_name])
        except (TypeError, ValueError) as exc:
            raise ManifestPreflightError(
                "Invalid mapped patient identifier for {}".format(image_name)
            ) from exc
    raise ManifestPreflightError(
        "TNBC manifest record lacks patient_id/patient/patient_number and no name_to_patient mapping was supplied"
    )


def validate_manifest_patient_isolation(
    path: str | Path,
    allowed_patient_ids: Iterable[int],
    forbidden_patient_ids: Iterable[int],
    name_to_patient: Optional[Mapping[str, int]] = None,
) -> List[Dict[str, Any]]:
    """Fail closed before opening any TNBC image, GT, or prediction file.

    This function reads only the supplied manifest, requires an explicit patient
    identifier on every record (or resolves it through the caller-provided,
    approved ``name_to_patient`` mapping), and returns records with a canonical
    integer ``patient_id``.  It rejects records outside ``allowed_patient_ids``
    and all IDs in ``forbidden_patient_ids`` (in particular patients 9--11).
    """

    allowed = {int(patient_id) for patient_id in allowed_patient_ids}
    forbidden = {int(patient_id) for patient_id in forbidden_patient_ids}
    if not allowed:
        raise ManifestPreflightError("At least one allowed TNBC patient ID is required")
    if allowed & forbidden:
        raise ManifestPreflightError("Allowed and forbidden patient ID sets overlap")

    validated: List[Dict[str, Any]] = []
    for index, row in enumerate(_read_rows(path)):
        if not isinstance(row, Mapping):
            raise ManifestPreflightError("TNBC manifest record {} is not an object".format(index))
        patient_id = _patient_id(row, name_to_patient=name_to_patient)
        if patient_id in forbidden or patient_id not in allowed:
            raise ManifestPreflightError(
                "TNBC manifest contains disallowed patient {} at record {}".format(patient_id, index)
            )
        normalized = dict(row)
        normalized["patient_id"] = patient_id
        validated.append(normalized)
    if not validated:
        raise ManifestPreflightError("TNBC manifest is empty")
    return validated


def _as_int(record: Mapping[str, Any], field: str) -> int:
    value = record.get(field)
    if value is None or str(value).strip() == "":
        raise ManifestPreflightError("Crop record lacks {}".format(field))
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ManifestPreflightError("Crop {} is not numeric".format(field)) from exc
    if not numeric.is_integer():
        raise ManifestPreflightError("Crop {} must be an integer".format(field))
    return int(numeric)


def load_crop_records(path: str | Path) -> List[Dict[str, int | str]]:
    """Load frozen crop coordinates in one canonical representation.

    Input records may use ``x,y,width,height`` or ``x1,y1,x2,y2`` and any of
    ``image_name``, ``image`` or ``name``.  Returned records always use
    ``image_name,x,y,width,height``.  Invalid or duplicate crops fail closed.
    """

    normalized: List[Dict[str, int | str]] = []
    seen = set()
    for row in _read_rows(path):
        if not isinstance(row, Mapping):
            raise ManifestPreflightError("Crop record is not an object")
        name = _image_name(row)
        xywh_present = all(field in row and str(row[field]).strip() != "" for field in ("x", "y", "width", "height"))
        xyxy_present = all(field in row and str(row[field]).strip() != "" for field in ("x1", "y1", "x2", "y2"))
        if xywh_present == xyxy_present:
            raise ManifestPreflightError(
                "Crop record must contain exactly one coordinate convention for {}".format(name)
            )
        if xywh_present:
            x, y = _as_int(row, "x"), _as_int(row, "y")
            width, height = _as_int(row, "width"), _as_int(row, "height")
        else:
            x, y = _as_int(row, "x1"), _as_int(row, "y1")
            x2, y2 = _as_int(row, "x2"), _as_int(row, "y2")
            width, height = x2 - x, y2 - y
        if x < 0 or y < 0 or width <= 0 or height <= 0:
            raise ManifestPreflightError("Crop coordinates are out of bounds for {}".format(name))
        result: Dict[str, int | str] = {
            "image_name": name,
            "x": x,
            "y": y,
            "width": width,
            "height": height,
        }
        signature = tuple(result.items())
        if signature in seen:
            raise ManifestPreflightError("Frozen crop manifest contains a duplicate crop")
        seen.add(signature)
        normalized.append(result)
    if not normalized:
        raise ManifestPreflightError("Frozen crop manifest is empty")
    return normalized
