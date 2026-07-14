"""Immutable static-coverage caches for ResiMix-PMS.

The cache is deliberately a small, model-agnostic boundary.  A caller runs
the frozen StainPMS inference once, writes every allowed training image (or
fixed crop) through :class:`StaticCoverageWriter`, seals the cache, and both
Control and ResiMix subsequently open exactly that sealed cache.  The module
does not discover data, open checkpoints, or run inference.

Two properties are enforced here instead of being left to a training script:

* a cache directory can be generated exactly once; an interrupted generation
  is fail-closed rather than silently resumed or replaced; and
* each cached instance map is shape-checked and file-hashed before a shared
  reader accepts it.

For MoNuSeg-Lite fixed crops, ``write_crop`` writes only the specified local
prediction into its image-sized cache.  Unwritten pixels remain zero and a
conflicting overlapping crop is rejected, so one crop can never overwrite an
unrelated region.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import stat
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = 1
MANIFEST_FILENAME = "coverage_manifest.json"
GENERATION_MARKER_FILENAME = ".coverage_generation.json"
COVERAGE_DIRECTORY_NAME = "coverage"
_KIND = "resimixpms_static_coverage"
_WRITE_BITS = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH


class CoverageError(RuntimeError):
    """Base class for static-coverage cache failures."""


class CoverageGenerationError(CoverageError):
    """Raised when a writer would violate the one-generation contract."""


class CoverageIntegrityError(CoverageError):
    """Raised when a sealed cache is absent, changed, or structurally invalid."""


@dataclass(frozen=True)
class CoverageImageRecord:
    """One sealed image-sized coverage map described by the manifest."""

    image_id: str
    file: str
    shape: tuple[int, int]
    sha256: str
    write_mode: str
    written_pixels: int
    written_fraction: float
    crop_boxes: tuple[tuple[int, int, int, int], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_id": self.image_id,
            "file": self.file,
            "shape": list(self.shape),
            "sha256": self.sha256,
            "write_mode": self.write_mode,
            "written_pixels": self.written_pixels,
            "written_fraction": self.written_fraction,
            "crop_boxes": [list(box) for box in self.crop_boxes],
        }


@dataclass(frozen=True)
class CoverageManifest:
    """Validated immutable description of one static-coverage cache."""

    root: Path
    provenance: Mapping[str, Any]
    images: Mapping[str, CoverageImageRecord]

    @property
    def path(self) -> Path:
        return self.root / MANIFEST_FILENAME

    @property
    def image_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.images))


def _canonical_json(value: Any) -> bytes:
    try:
        text = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        raise CoverageGenerationError("coverage provenance must be JSON serializable") from exc
    return (text + "\n").encode("utf-8")


def _write_json_atomic(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(_canonical_json(payload))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _save_npy_atomic(path: Path, array: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.save(handle, array, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _normalise_image_id(image_id: object) -> str:
    value = str(image_id)
    if not value or "\x00" in value:
        raise CoverageGenerationError("coverage image_id must be a non-empty NUL-free string")
    return value


def _normalise_shape(shape: Sequence[object], *, image_id: str) -> tuple[int, int]:
    if len(shape) != 2:
        raise CoverageGenerationError(f"coverage shape for {image_id!r} must be (height, width)")
    try:
        height, width = int(shape[0]), int(shape[1])
    except (TypeError, ValueError) as exc:
        raise CoverageGenerationError(f"coverage shape for {image_id!r} is not integral") from exc
    if height <= 0 or width <= 0:
        raise CoverageGenerationError(f"coverage shape for {image_id!r} must be positive")
    return height, width


def _normalise_shapes(image_shapes: Mapping[object, Sequence[object]]) -> dict[str, tuple[int, int]]:
    if not isinstance(image_shapes, Mapping) or not image_shapes:
        raise CoverageGenerationError("image_shapes must be a non-empty mapping of image_id to (H, W)")
    result: dict[str, tuple[int, int]] = {}
    for raw_id, raw_shape in image_shapes.items():
        image_id = _normalise_image_id(raw_id)
        if image_id in result:
            raise CoverageGenerationError(f"duplicate coverage image_id: {image_id!r}")
        result[image_id] = _normalise_shape(raw_shape, image_id=image_id)
    return result


def _normalise_prediction(
    prediction: np.ndarray | Sequence[Sequence[object]],
    *,
    shape: tuple[int, int],
    label: str,
) -> np.ndarray:
    array = np.asarray(prediction)
    if array.ndim != 2 or tuple(array.shape) != tuple(shape):
        raise CoverageGenerationError(
            f"{label} has shape {tuple(array.shape)}, expected {tuple(shape)}"
        )
    if not np.issubdtype(array.dtype, np.integer) or array.dtype == np.bool_:
        raise CoverageGenerationError(f"{label} must be a non-boolean integer instance map")
    if array.size and int(array.min()) < 0:
        raise CoverageGenerationError(f"{label} contains a negative instance id")
    if array.size and int(array.max()) > np.iinfo(np.int32).max:
        raise CoverageGenerationError(f"{label} contains an instance id outside int32")
    return np.ascontiguousarray(array, dtype=np.int32)


def _coverage_file_name(image_id: str) -> str:
    # The record retains the human-readable id; the filename is path-safe and
    # collision-resistant even when ids contain dataset-specific separators.
    return sha256(image_id.encode("utf-8")).hexdigest() + ".npy"


def _coverage_relative_path(image_id: str) -> str:
    return f"{COVERAGE_DIRECTORY_NAME}/{_coverage_file_name(image_id)}"


def _freeze_path(path: Path) -> None:
    """Remove writable mode bits without assuming a platform-specific ACL."""
    try:
        os.chmod(path, path.stat().st_mode & ~_WRITE_BITS)
    except OSError as exc:
        raise CoverageGenerationError(f"failed to make static coverage read-only: {path}") from exc


def _require_readonly(path: Path) -> None:
    try:
        mode = path.stat().st_mode
    except OSError as exc:
        raise CoverageIntegrityError(f"cannot stat static coverage path: {path}") from exc
    if mode & _WRITE_BITS:
        raise CoverageIntegrityError(f"static coverage path is writable: {path}")


def _load_manifest_payload(root: Path) -> dict[str, Any]:
    marker = root / GENERATION_MARKER_FILENAME
    if marker.exists():
        raise CoverageIntegrityError(
            f"static coverage generation is incomplete (marker remains): {marker}"
        )
    manifest_path = root / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise CoverageIntegrityError(f"static coverage manifest is missing: {manifest_path}")
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise CoverageIntegrityError(f"cannot parse static coverage manifest: {manifest_path}") from exc
    if not isinstance(payload, dict):
        raise CoverageIntegrityError("static coverage manifest must be a JSON object")
    return payload


def _parse_box(raw_box: Sequence[object], *, shape: tuple[int, int], image_id: str) -> tuple[int, int, int, int]:
    if len(raw_box) != 4:
        raise CoverageIntegrityError(f"crop box for {image_id!r} must contain four coordinates")
    try:
        x1, y1, x2, y2 = (int(item) for item in raw_box)
    except (TypeError, ValueError) as exc:
        raise CoverageIntegrityError(f"crop box for {image_id!r} is not integral") from exc
    height, width = shape
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise CoverageIntegrityError(
            f"crop box {(x1, y1, x2, y2)} is outside coverage shape {shape} for {image_id!r}"
        )
    return x1, y1, x2, y2


def _parse_record(raw: Mapping[str, Any]) -> CoverageImageRecord:
    try:
        image_id = _normalise_image_id(raw["image_id"])
        shape = _normalise_shape(raw["shape"], image_id=image_id)
        relative_file = str(raw["file"])
        digest = str(raw["sha256"]).lower()
        write_mode = str(raw["write_mode"])
        written_pixels = int(raw["written_pixels"])
        written_fraction = float(raw["written_fraction"])
        raw_boxes = raw["crop_boxes"]
    except (KeyError, TypeError, ValueError) as exc:
        raise CoverageIntegrityError("static coverage image record is malformed") from exc
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise CoverageIntegrityError(f"invalid SHA256 for static coverage image {image_id!r}")
    expected_file = _coverage_relative_path(image_id)
    if relative_file != expected_file:
        raise CoverageIntegrityError(
            f"unsafe or unexpected cache filename for {image_id!r}: {relative_file!r}"
        )
    if write_mode not in {"full", "fixed_crops"}:
        raise CoverageIntegrityError(f"unknown write_mode for {image_id!r}: {write_mode!r}")
    if not isinstance(raw_boxes, list):
        raise CoverageIntegrityError(f"crop_boxes for {image_id!r} must be a list")
    boxes = tuple(_parse_box(box, shape=shape, image_id=image_id) for box in raw_boxes)
    area = shape[0] * shape[1]
    if not 0 <= written_pixels <= area:
        raise CoverageIntegrityError(f"written_pixels is invalid for {image_id!r}")
    expected_fraction = written_pixels / area
    if not np.isclose(written_fraction, expected_fraction, rtol=0.0, atol=1.0e-12):
        raise CoverageIntegrityError(f"written_fraction is invalid for {image_id!r}")
    if write_mode == "full":
        if boxes or written_pixels != area:
            raise CoverageIntegrityError(f"full coverage record is inconsistent for {image_id!r}")
    elif not boxes:
        raise CoverageIntegrityError(f"fixed-crop record has no crop boxes for {image_id!r}")
    return CoverageImageRecord(
        image_id=image_id,
        file=relative_file,
        shape=shape,
        sha256=digest,
        write_mode=write_mode,
        written_pixels=written_pixels,
        written_fraction=written_fraction,
        crop_boxes=boxes,
    )


def _validate_crop_footprint(record: CoverageImageRecord, array: np.ndarray) -> None:
    if record.write_mode != "fixed_crops":
        return
    written = np.zeros(record.shape, dtype=bool)
    for x1, y1, x2, y2 in record.crop_boxes:
        written[y1:y2, x1:x2] = True
    if int(written.sum()) != record.written_pixels:
        raise CoverageIntegrityError(
            f"crop footprint does not match written_pixels for {record.image_id!r}"
        )
    # The writer starts from zeros; anything outside the union of fixed crops
    # therefore proves that a local result overwrote another image region.
    if np.any(array[~written] != 0):
        raise CoverageIntegrityError(
            f"coverage outside fixed crops is nonzero for {record.image_id!r}"
        )


def validate_static_coverage_cache(
    cache_dir: str | Path,
    *,
    expected_image_shapes: Mapping[object, Sequence[object]] | None = None,
    expected_provenance: Mapping[str, Any] | None = None,
    require_readonly: bool = True,
) -> CoverageManifest:
    """Fail closed unless a sealed shared cache matches its manifest exactly.

    ``expected_image_shapes`` should come from the already-frozen allowed
    training manifest.  Passing it prevents a cache for a different split from
    being silently reused.  ``expected_provenance`` is compared canonically
    when callers need to bind the cache to checkpoint and input-manifest hashes.
    """
    root = Path(cache_dir)
    if not root.is_dir():
        raise CoverageIntegrityError(f"static coverage cache directory is missing: {root}")
    payload = _load_manifest_payload(root)
    if payload.get("schema_version") != SCHEMA_VERSION or payload.get("kind") != _KIND:
        raise CoverageIntegrityError("static coverage manifest schema/kind mismatch")
    if payload.get("state") != "sealed":
        raise CoverageIntegrityError("static coverage manifest is not sealed")
    provenance = payload.get("provenance", {})
    if not isinstance(provenance, Mapping):
        raise CoverageIntegrityError("static coverage provenance must be an object")
    try:
        canonical_provenance = json.loads(_canonical_json(dict(provenance)).decode("utf-8"))
    except CoverageGenerationError as exc:
        raise CoverageIntegrityError("static coverage provenance is not serializable") from exc
    if expected_provenance is not None:
        try:
            canonical_expected = json.loads(
                _canonical_json(dict(expected_provenance)).decode("utf-8")
            )
        except CoverageGenerationError as exc:
            raise CoverageIntegrityError("expected coverage provenance is not serializable") from exc
        if canonical_provenance != canonical_expected:
            raise CoverageIntegrityError("static coverage provenance does not match the expected frozen inputs")

    raw_images = payload.get("images")
    if not isinstance(raw_images, list) or not raw_images:
        raise CoverageIntegrityError("static coverage manifest has no image records")
    records: dict[str, CoverageImageRecord] = {}
    for raw_record in raw_images:
        if not isinstance(raw_record, Mapping):
            raise CoverageIntegrityError("static coverage image record must be an object")
        record = _parse_record(raw_record)
        if record.image_id in records:
            raise CoverageIntegrityError(f"duplicate static coverage image_id: {record.image_id!r}")
        records[record.image_id] = record

    if expected_image_shapes is not None:
        expected = _normalise_shapes(expected_image_shapes)
        actual_shapes = {image_id: record.shape for image_id, record in records.items()}
        if actual_shapes != expected:
            raise CoverageIntegrityError(
                "static coverage records do not exactly match the allowed image/crop manifest"
            )

    coverage_dir = root / COVERAGE_DIRECTORY_NAME
    if not coverage_dir.is_dir():
        raise CoverageIntegrityError(f"static coverage array directory is missing: {coverage_dir}")
    expected_files = {record.file for record in records.values()}
    actual_files = {
        f"{COVERAGE_DIRECTORY_NAME}/{path.name}"
        for path in coverage_dir.iterdir()
        if path.is_file() or path.is_symlink()
    }
    if actual_files != expected_files:
        raise CoverageIntegrityError("static coverage cache contains missing or unexpected array files")

    if require_readonly:
        _require_readonly(root)
        _require_readonly(coverage_dir)
        _require_readonly(root / MANIFEST_FILENAME)

    for record in records.values():
        path = root / record.file
        if path.is_symlink() or not path.is_file():
            raise CoverageIntegrityError(f"static coverage array is missing or symlinked: {path}")
        if require_readonly:
            _require_readonly(path)
        actual_digest = _sha256_file(path)
        if actual_digest.lower() != record.sha256:
            raise CoverageIntegrityError(
                f"SHA256 mismatch for static coverage image {record.image_id!r}"
            )
        try:
            array = np.load(path, allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise CoverageIntegrityError(f"cannot load static coverage array: {path}") from exc
        if array.ndim != 2 or tuple(array.shape) != record.shape:
            raise CoverageIntegrityError(
                f"shape mismatch for static coverage image {record.image_id!r}"
            )
        if not np.issubdtype(array.dtype, np.integer) or array.dtype == np.bool_:
            raise CoverageIntegrityError(
                f"static coverage array has invalid dtype for {record.image_id!r}"
            )
        if array.size and int(array.min()) < 0:
            raise CoverageIntegrityError(
                f"static coverage array has negative ids for {record.image_id!r}"
            )
        _validate_crop_footprint(record, array)

    return CoverageManifest(root=root, provenance=canonical_provenance, images=records)


class StaticCoverageCache:
    """Read-only view of a previously validated static-coverage cache."""

    def __init__(self, manifest: CoverageManifest):
        self._manifest = manifest

    @property
    def manifest(self) -> CoverageManifest:
        return self._manifest

    @classmethod
    def open(
        cls,
        cache_dir: str | Path,
        *,
        expected_image_shapes: Mapping[object, Sequence[object]] | None = None,
        expected_provenance: Mapping[str, Any] | None = None,
        require_readonly: bool = True,
    ) -> "StaticCoverageCache":
        return cls(
            validate_static_coverage_cache(
                cache_dir,
                expected_image_shapes=expected_image_shapes,
                expected_provenance=expected_provenance,
                require_readonly=require_readonly,
            )
        )

    @property
    def image_ids(self) -> tuple[str, ...]:
        return self._manifest.image_ids

    def load(self, image_id: object, *, verify_sha256: bool = False) -> np.ndarray:
        """Load one array as a non-writeable in-memory instance map.

        The whole cache is verified on ``open``.  ``verify_sha256=True`` is
        available for a defensive hand-off boundary without paying the hashing
        cost on every training crop.
        """
        key = _normalise_image_id(image_id)
        try:
            record = self._manifest.images[key]
        except KeyError as exc:
            raise CoverageIntegrityError(f"static coverage image is not declared: {key!r}") from exc
        path = self._manifest.root / record.file
        if verify_sha256 and _sha256_file(path).lower() != record.sha256:
            raise CoverageIntegrityError(f"SHA256 mismatch while loading static coverage {key!r}")
        try:
            array = np.load(path, allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise CoverageIntegrityError(f"cannot load static coverage array: {path}") from exc
        if array.ndim != 2 or tuple(array.shape) != record.shape:
            raise CoverageIntegrityError(f"shape mismatch while loading static coverage {key!r}")
        array.setflags(write=False)
        return array


class StaticCoverageWriter:
    """One-shot writer for frozen full-image or fixed-crop coverage predictions."""

    def __init__(
        self,
        root: Path,
        image_shapes: Mapping[str, tuple[int, int]],
        provenance: Mapping[str, Any],
    ):
        self._root = root
        self._image_shapes = dict(image_shapes)
        self._provenance = dict(provenance)
        self._arrays: dict[str, np.ndarray] = {}
        self._written: dict[str, np.ndarray] = {}
        self._write_modes: dict[str, str | None] = {
            image_id: None for image_id in self._image_shapes
        }
        self._crop_boxes: dict[str, list[tuple[int, int, int, int]]] = {
            image_id: [] for image_id in self._image_shapes
        }
        self._sealed = False

    @classmethod
    def create(
        cls,
        cache_dir: str | Path,
        image_shapes: Mapping[object, Sequence[object]],
        *,
        provenance: Mapping[str, Any] | None = None,
    ) -> "StaticCoverageWriter":
        """Create a fresh cache directory, refusing any existing path.

        The generation marker is intentionally left behind if the process
        fails before ``seal``.  A later run must choose a new cache directory
        (or have the incomplete result explicitly audited); it cannot overwrite
        or silently regenerate the original static cache.
        """
        root = Path(cache_dir)
        if root.exists():
            raise CoverageGenerationError(
                f"refusing to generate static coverage in existing path: {root}"
            )
        shapes = _normalise_shapes(image_shapes)
        raw_provenance = {} if provenance is None else dict(provenance)
        canonical_provenance = json.loads(_canonical_json(raw_provenance).decode("utf-8"))
        root.mkdir(parents=True, exist_ok=False)
        (root / COVERAGE_DIRECTORY_NAME).mkdir(exist_ok=False)
        _write_json_atomic(
            root / GENERATION_MARKER_FILENAME,
            {
                "schema_version": SCHEMA_VERSION,
                "kind": _KIND,
                "state": "generating",
                "image_shapes": {image_id: list(shape) for image_id, shape in sorted(shapes.items())},
                "provenance": canonical_provenance,
            },
        )
        return cls(root, shapes, canonical_provenance)

    @property
    def image_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._image_shapes))

    def _require_active(self) -> None:
        if self._sealed:
            raise CoverageGenerationError("static coverage writer is already sealed")

    def _require_image(self, image_id: object) -> tuple[str, tuple[int, int]]:
        key = _normalise_image_id(image_id)
        try:
            return key, self._image_shapes[key]
        except KeyError as exc:
            raise CoverageGenerationError(f"image is not in the frozen coverage manifest: {key!r}") from exc

    def write_full(self, image_id: object, prediction: np.ndarray | Sequence[Sequence[object]]) -> None:
        """Record the one full-image frozen prediction for ``image_id``."""
        self._require_active()
        key, shape = self._require_image(image_id)
        if self._write_modes[key] is not None:
            raise CoverageGenerationError(
                f"static coverage for {key!r} was already written; replacement is forbidden"
            )
        self._arrays[key] = _normalise_prediction(
            prediction, shape=shape, label=f"full coverage prediction for {key!r}"
        )
        self._written[key] = np.ones(shape, dtype=bool)
        self._write_modes[key] = "full"

    def write_crop(
        self,
        image_id: object,
        crop_box: Sequence[object],
        prediction: np.ndarray | Sequence[Sequence[object]],
    ) -> None:
        """Insert one fixed-crop prediction without replacing other pixels.

        ``crop_box`` follows ``(x1, y1, x2, y2)`` image coordinates.  A crop
        may overlap a previous crop only where the instance-map values are
        byte-for-byte identical; otherwise the operation fails before mutating
        the cached array.  This makes first-write ownership explicit and avoids
        a hidden last-write-wins policy.
        """
        self._require_active()
        key, full_shape = self._require_image(image_id)
        if self._write_modes[key] == "full":
            raise CoverageGenerationError(
                f"cannot add a crop after full static coverage for {key!r}"
            )
        try:
            x1, y1, x2, y2 = (int(item) for item in crop_box)
        except (TypeError, ValueError) as exc:
            raise CoverageGenerationError(f"crop box for {key!r} is not integral") from exc
        height, width = full_shape
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise CoverageGenerationError(
                f"crop box {(x1, y1, x2, y2)} is outside coverage shape {full_shape} for {key!r}"
            )
        crop_shape = (y2 - y1, x2 - x1)
        crop_prediction = _normalise_prediction(
            prediction, shape=crop_shape, label=f"crop coverage prediction for {key!r}"
        )
        if key not in self._arrays:
            self._arrays[key] = np.zeros(full_shape, dtype=np.int32)
            self._written[key] = np.zeros(full_shape, dtype=bool)
            self._write_modes[key] = "fixed_crops"

        target = self._arrays[key][y1:y2, x1:x2]
        already_written = self._written[key][y1:y2, x1:x2]
        if np.any(already_written & (target != crop_prediction)):
            raise CoverageGenerationError(
                f"conflicting fixed-crop coverage overlaps existing pixels for {key!r}"
            )
        # Assign only cells that have never been owned by a prior crop.  Values
        # in an identical overlap are intentionally left untouched.
        target[~already_written] = crop_prediction[~already_written]
        already_written[...] = True
        self._crop_boxes[key].append((x1, y1, x2, y2))

    def seal(self) -> StaticCoverageCache:
        """Hash, manifest, and make the completed cache read-only."""
        self._require_active()
        missing = [image_id for image_id in self.image_ids if image_id not in self._arrays]
        if missing:
            raise CoverageGenerationError(
                "cannot seal incomplete static coverage; missing predictions for " + repr(missing)
            )

        records: list[CoverageImageRecord] = []
        coverage_dir = self._root / COVERAGE_DIRECTORY_NAME
        for image_id in self.image_ids:
            array = self._arrays[image_id]
            shape = self._image_shapes[image_id]
            mode = self._write_modes[image_id]
            assert mode in {"full", "fixed_crops"}
            relative_file = _coverage_relative_path(image_id)
            output_path = self._root / relative_file
            _save_npy_atomic(output_path, array)
            written_pixels = int(self._written[image_id].sum())
            records.append(
                CoverageImageRecord(
                    image_id=image_id,
                    file=relative_file,
                    shape=shape,
                    sha256=_sha256_file(output_path),
                    write_mode=mode,
                    written_pixels=written_pixels,
                    written_fraction=written_pixels / (shape[0] * shape[1]),
                    crop_boxes=tuple(self._crop_boxes[image_id]),
                )
            )

        manifest_path = self._root / MANIFEST_FILENAME
        _write_json_atomic(
            manifest_path,
            {
                "schema_version": SCHEMA_VERSION,
                "kind": _KIND,
                "state": "sealed",
                "provenance": self._provenance,
                "images": [record.to_dict() for record in records],
            },
        )
        marker = self._root / GENERATION_MARKER_FILENAME
        try:
            marker.unlink()
        except OSError as exc:
            raise CoverageGenerationError(f"failed to remove generation marker: {marker}") from exc

        # Freeze files before their directories.  The cache directory itself is
        # dedicated to coverage, so making it read-only prevents a second path
        # from adding or replacing maps after sealing.
        for record in records:
            _freeze_path(self._root / record.file)
        _freeze_path(manifest_path)
        _freeze_path(coverage_dir)
        _freeze_path(self._root)
        self._sealed = True
        self._arrays.clear()
        self._written.clear()
        return StaticCoverageCache.open(
            self._root,
            expected_image_shapes=self._image_shapes,
            expected_provenance=self._provenance,
            require_readonly=True,
        )


def begin_static_coverage_generation(
    cache_dir: str | Path,
    image_shapes: Mapping[object, Sequence[object]],
    *,
    provenance: Mapping[str, Any] | None = None,
) -> StaticCoverageWriter:
    """Convenience entry point for the only allowed cache-generation path."""
    return StaticCoverageWriter.create(cache_dir, image_shapes, provenance=provenance)
