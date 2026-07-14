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
_IMAGE_NAME_FIELDS = ("image_name", "image", "name", "filename")
_PATIENT_ID_FIELDS = ("patient_id", "patient", "patient_number")
FROZEN_CROP_INDEX_SCHEDULE_FORMAT = "resimix_frozen_crop_indices_v1"


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


@dataclass(frozen=True)
class FrozenCropIndexSchedule:
    """An immutable per-epoch selection over the loader's existing crop grid.

    The canonical MoNuSeg-Lite artifact stores four crop *indices* per image
    for each of the ten registered epochs.  It does not store coordinates, so
    callers must apply these indices to the already-fixed crop grid rather
    than inventing a coordinate list or collapsing the epoch schedule.
    """

    path: Path
    crop_size: int
    overlap: int
    load: str
    max_crops_per_image: int
    epoch_indices: Mapping[int, Mapping[str, tuple[int, ...]]]

    @property
    def epochs(self) -> tuple[int, ...]:
        return tuple(sorted(self.epoch_indices))

    @property
    def total_assignments(self) -> int:
        return sum(
            len(indices)
            for image_map in self.epoch_indices.values()
            for indices in image_map.values()
        )

    def indices_for(self, image_name: str, *, epoch: int | None = None, union: bool = False) -> tuple[int, ...]:
        name = _schedule_image_name(image_name)
        if union:
            values = set()
            for image_map in self.epoch_indices.values():
                try:
                    values.update(image_map[name])
                except KeyError as exc:
                    raise ManifestPreflightError(
                        f"frozen crop schedule has no entry for {image_name}"
                    ) from exc
            return tuple(sorted(values))
        if epoch is None:
            raise ManifestPreflightError("frozen crop schedule requires an explicit epoch")
        try:
            return self.epoch_indices[int(epoch)][name]
        except KeyError as exc:
            raise ManifestPreflightError(
                f"frozen crop schedule has no entry for epoch {epoch}, image {image_name}"
            ) from exc

    def select_boxes(
        self,
        image_name: str,
        default_boxes: Sequence[Sequence[int]],
        *,
        epoch: int | None = None,
        union: bool = False,
    ) -> list[list[int]]:
        selected = []
        for index in self.indices_for(image_name, epoch=epoch, union=union):
            if index < 0 or index >= len(default_boxes):
                raise ManifestPreflightError(
                    f"frozen crop index {index} is outside the existing grid for {image_name}"
                )
            box = [int(value) for value in default_boxes[index]]
            if len(box) != 4 or box[2] - box[0] != self.crop_size or box[3] - box[1] != self.crop_size:
                raise ManifestPreflightError(
                    f"frozen crop index {index} does not resolve to a {self.crop_size}x{self.crop_size} crop for {image_name}"
                )
            selected.append(box)
        if not selected:
            raise ManifestPreflightError(f"frozen crop schedule selected no crops for {image_name}")
        return selected


def _schedule_image_name(value: str) -> str:
    raw = str(value).strip().replace("\\", "/")
    pure = PurePosixPath(raw)
    if not raw or pure.is_absolute() or ".." in pure.parts:
        raise ManifestPreflightError(f"unsafe frozen crop schedule image name: {value}")
    return str(pure)


def crop_boxes_for_shape(
    shape: Sequence[int], crop_size: int, overlap: int, load: str
) -> list[list[int]]:
    """Reproduce the fixed loader grid without reading pixels or selecting crops."""

    if len(shape) < 2:
        raise ManifestPreflightError("crop grid requires image height and width")
    height, width = int(shape[0]), int(shape[1])
    if height < crop_size or width < crop_size:
        raise ManifestPreflightError("image is smaller than the frozen crop size")
    if crop_size != 256 or overlap < 0 or overlap >= crop_size:
        raise ManifestPreflightError("frozen crop grid geometry is invalid")
    stride = 256 - overlap

    def start_points(size: int) -> list[int]:
        points = [0]
        counter = 1
        while True:
            point = stride * counter
            if point + crop_size >= size:
                if crop_size != size:
                    points.append(size - crop_size)
                break
            points.append(point)
            counter += 1
        return points

    x_points, y_points = start_points(width), start_points(height)
    boxes: list[list[int]] = []
    if load == "sequence":
        for x in x_points:
            for y in y_points:
                boxes.append([x, y, min(x + crop_size, width), min(y + crop_size, height)])
    elif load == "unsequence":
        forward = True
        for x in x_points:
            for y in y_points if forward else reversed(y_points):
                boxes.append([x, y, min(x + crop_size, width), min(y + crop_size, height)])
            forward = not forward
    elif load in ("clockwise", "unclockwise"):
        top, bottom, left, right = 0, len(y_points) - 1, 0, len(x_points) - 1
        while top <= bottom or left <= right:
            if top <= bottom:
                for y in range(left, right + 1):
                    boxes.append([x_points[top], y_points[y], min(x_points[top] + crop_size, width), min(y_points[y] + crop_size, height)])
                top += 1
            if left <= right:
                for x in range(top, bottom + 1):
                    boxes.append([x_points[x], y_points[right], min(x_points[x] + crop_size, width), min(y_points[right] + crop_size, height)])
                right -= 1
            if top <= bottom:
                for y in reversed(range(left, right + 1)):
                    boxes.append([x_points[bottom], y_points[y], min(x_points[bottom] + crop_size, width), min(y_points[y] + crop_size, height)])
                bottom -= 1
            if left <= right:
                for x in reversed(range(top, bottom + 1)):
                    boxes.append([x_points[x], y_points[left], min(x_points[x] + crop_size, width), min(y_points[left] + crop_size, height)])
                left += 1
        if load == "unclockwise":
            boxes.reverse()
    else:
        raise ManifestPreflightError(f"unsupported frozen crop load order: {load}")
    return boxes


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


def _read_json_mapping(path: str | Path) -> Mapping[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise ManifestPreflightError(f"Manifest is unavailable: {source}")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestPreflightError(f"Invalid JSON manifest: {source}") from exc
    if not isinstance(payload, Mapping):
        raise ManifestPreflightError(f"Frozen crop schedule must be a JSON object: {source}")
    return payload


def load_frozen_crop_index_schedule(
    path: str | Path,
    *,
    allowed_image_names: Iterable[str] | None = None,
    expected_crop_size: int | None = None,
    expected_overlap: int | None = None,
    expected_load: str | None = None,
    expected_epochs: int | None = None,
) -> FrozenCropIndexSchedule | None:
    """Read a derived, immutable MoNuSeg-Lite crop-index schedule if present.

    Returning ``None`` means that ``path`` is an ordinary explicit-coordinate
    manifest and should continue through :func:`load_crop_records`.  A file
    declaring the schedule format is validated completely and never falls back
    to a different crop source.
    """

    payload = _read_json_mapping(path)
    if payload.get("format") != FROZEN_CROP_INDEX_SCHEDULE_FORMAT:
        return None

    source = Path(path)
    try:
        crop_size = int(payload["crop_size"])
        overlap = int(payload["overlap"])
        load = str(payload["load"])
        max_crops = int(payload["max_crops_per_image"])
        epoch_count = int(payload["epochs"])
        raw_epochs = payload["epoch_crop_indices"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ManifestPreflightError(f"Frozen crop schedule is missing required fields: {source}") from exc
    if crop_size <= 0 or overlap < 0 or overlap >= crop_size:
        raise ManifestPreflightError(f"Frozen crop schedule geometry is invalid: {source}")
    if load != "unclockwise":
        raise ManifestPreflightError(f"Frozen crop schedule load order is not fixed unclockwise: {source}")
    if max_crops <= 0 or epoch_count <= 0 or not isinstance(raw_epochs, Mapping):
        raise ManifestPreflightError(f"Frozen crop schedule dimensions are invalid: {source}")
    if expected_crop_size is not None and crop_size != int(expected_crop_size):
        raise ManifestPreflightError(f"Frozen crop schedule crop_size differs from the formal protocol: {source}")
    if expected_overlap is not None and overlap != int(expected_overlap):
        raise ManifestPreflightError(f"Frozen crop schedule overlap differs from the formal protocol: {source}")
    if expected_load is not None and load != str(expected_load):
        raise ManifestPreflightError(f"Frozen crop schedule load order differs from the formal protocol: {source}")
    if expected_epochs is not None and epoch_count != int(expected_epochs):
        raise ManifestPreflightError(f"Frozen crop schedule epoch count differs from the formal protocol: {source}")

    expected_epoch_keys = {str(index) for index in range(epoch_count)}
    if set(raw_epochs) != expected_epoch_keys:
        raise ManifestPreflightError(f"Frozen crop schedule epochs are incomplete or changed: {source}")
    allowed = None
    if allowed_image_names is not None:
        allowed = {_schedule_image_name(name) for name in allowed_image_names}

    normalized: Dict[int, Dict[str, tuple[int, ...]]] = {}
    for epoch in range(epoch_count):
        raw_images = raw_epochs[str(epoch)]
        if not isinstance(raw_images, Mapping) or not raw_images:
            raise ManifestPreflightError(f"Frozen crop schedule epoch {epoch} is malformed: {source}")
        image_names = {_schedule_image_name(name) for name in raw_images}
        if allowed is not None and image_names != allowed:
            raise ManifestPreflightError(
                f"Frozen crop schedule epoch {epoch} does not exactly match the admitted training images"
            )
        normalized_epoch: Dict[str, tuple[int, ...]] = {}
        for raw_name, raw_indices in raw_images.items():
            name = _schedule_image_name(raw_name)
            if not isinstance(raw_indices, list) or len(raw_indices) != max_crops:
                raise ManifestPreflightError(
                    f"Frozen crop schedule must retain exactly {max_crops} indices for {name}, epoch {epoch}"
                )
            indices = []
            for raw_index in raw_indices:
                if isinstance(raw_index, bool):
                    raise ManifestPreflightError(f"Frozen crop index is not an integer for {name}, epoch {epoch}")
                try:
                    numeric = int(raw_index)
                except (TypeError, ValueError) as exc:
                    raise ManifestPreflightError(
                        f"Frozen crop index is not an integer for {name}, epoch {epoch}"
                    ) from exc
                if numeric < 0 or numeric != raw_index:
                    raise ManifestPreflightError(f"Frozen crop index is invalid for {name}, epoch {epoch}")
                indices.append(numeric)
            if len(set(indices)) != len(indices):
                raise ManifestPreflightError(f"Frozen crop schedule duplicates an index for {name}, epoch {epoch}")
            normalized_epoch[name] = tuple(indices)
        normalized[epoch] = normalized_epoch

    return FrozenCropIndexSchedule(
        path=source.resolve(), crop_size=crop_size, overlap=overlap, load=load,
        max_crops_per_image=max_crops, epoch_indices=normalized,
    )


def load_crop_plan(
    path: str | Path,
    **schedule_constraints: Any,
) -> tuple[List[Dict[str, int | str]], FrozenCropIndexSchedule | None]:
    """Load either immutable coordinate records or a frozen index schedule."""

    if Path(path).suffix.lower() != ".json":
        return load_crop_records(path), None
    schedule = load_frozen_crop_index_schedule(path, **schedule_constraints)
    if schedule is not None:
        return [], schedule
    return load_crop_records(path), None
