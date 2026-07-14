"""Training-only frozen-error audits and deterministic ResiMix donor-bank logic.

This module operates on arrays already admitted by the experiment's data
manifest.  It does not load images, checkpoints, development data, or test
data.  Call :func:`audit_training_samples` only after manifest preflight.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


DONOR_SAMPLING_PROPORTIONS: Mapping[str, float] = {
    "Missed": 0.50,
    "IoU-Cliff": 0.30,
    "Low-Quality Matched": 0.20,
}
_TNBC_ALLOWED_PATIENTS = frozenset(range(1, 7))
_FORBIDDEN_SPLIT_TOKENS = frozenset(("dev", "development", "valid", "validation", "test", "eval", "evaluation", "holdout"))
_HE_STAIN_MATRIX = np.asarray(
    ((0.650, 0.072), (0.704, 0.990), (0.286, 0.105)), dtype=np.float64
)
_HE_PINV = np.linalg.pinv(_HE_STAIN_MATRIX)


class DataIsolationError(ValueError):
    """Raised before an audit can use a non-training source."""


class DonorAuditError(ValueError):
    """Raised for malformed image, GT, prediction, or coverage arrays."""


@dataclass(frozen=True)
class TrainingSample:
    """An already-authorized training image and its frozen step-0 inputs."""

    source_id: str
    dataset: str
    split: str
    instance_map: np.ndarray
    prediction_masks: Any
    coverage_map: np.ndarray
    rgb: np.ndarray
    od: Optional[np.ndarray] = None
    patient_id: Optional[int] = None
    source_metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class InstanceAudit:
    """Metrics, source provenance, and preserved transplant payload for one GT."""

    source_id: str
    dataset: str
    patient_id: Optional[int]
    instance_id: int
    best_iou: float
    coverage_fraction: float
    covered: bool
    hardness: float
    area: int
    bbox_xyxy: Tuple[int, int, int, int]
    eccentricity: float
    solidity: float
    touches_image_boundary: bool
    primary_component_fraction: float
    rgb_finite: bool
    od_finite: bool
    nucleus_rgb_stats: Mapping[str, Optional[List[float]]]
    nucleus_od_stats: Mapping[str, Optional[List[float]]]
    nucleus_he_stats: Mapping[str, Optional[List[float]]]
    annulus_he_stats: Mapping[str, Optional[List[float]]]
    annulus_gradient_energy: Optional[float]
    donor_class: Optional[str] = None
    eligible: bool = False
    rejection_reasons: Tuple[str, ...] = ()
    source_metadata: Mapping[str, Any] = field(default_factory=dict)
    # These are deliberately retained rather than reconstructed from a
    # development/test source.  They are excluded from CSV serialization.
    mask: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=bool), repr=False)
    rgb_patch: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 3), dtype=np.uint8), repr=False)
    od_patch: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 3), dtype=np.float32), repr=False)
    annulus_mask: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=bool), repr=False)
    patch_bbox_xyxy: Tuple[int, int, int, int] = (0, 0, 0, 0)


@dataclass
class DonorBank:
    """A frozen, training-only donor bank plus rejected-audit provenance."""

    audits: List[InstanceAudit]
    donors: List[InstanceAudit]
    matched_iou_q25: Optional[float]
    area_q05: Optional[float]
    area_q95: Optional[float]

    def summary(self) -> Dict[str, Any]:
        donor_classes = Counter(record.donor_class for record in self.donors)
        rejection_counts: Counter[str] = Counter()
        for record in self.audits:
            rejection_counts.update(record.rejection_reasons)
        sources = sorted({record.source_id for record in self.audits})
        return {
            "all_gt_instances": len(self.audits),
            "eligible_donors": len(self.donors),
            "matched_iou_q25": self.matched_iou_q25,
            "gt_area_q05": self.area_q05,
            "gt_area_q95": self.area_q95,
            "donor_class_counts": {key: donor_classes.get(key, 0) for key in DONOR_SAMPLING_PROPORTIONS},
            "formal_sampling_proportions": dict(DONOR_SAMPLING_PROPORTIONS),
            "rejection_counts": dict(sorted(rejection_counts.items())),
            "source_ids": sources,
            "training_only": True,
        }


def _normalize_split(value: str) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def _metadata_has_forbidden_split(metadata: Mapping[str, Any]) -> bool:
    for field in ("split", "subset", "partition", "role", "phase"):
        value = metadata.get(field)
        if value is None:
            continue
        tokens = set(_normalize_split(str(value)).split("_"))
        if tokens & _FORBIDDEN_SPLIT_TOKENS:
            return True
    return False


def validate_training_sample(
    sample: TrainingSample,
    allowed_tnbc_patients: Iterable[int] = _TNBC_ALLOWED_PATIENTS,
) -> None:
    """Fail closed for development/test sources before examining their arrays."""

    split = _normalize_split(sample.split)
    if split not in {"train", "training"} or _metadata_has_forbidden_split(sample.source_metadata):
        raise DataIsolationError("ResiMix donor audit accepts training sources only: {}".format(sample.source_id))
    dataset = _normalize_split(sample.dataset)
    if dataset == "tnbc":
        allowed = {int(patient_id) for patient_id in allowed_tnbc_patients}
        if sample.patient_id is None:
            raise DataIsolationError("TNBC training sample has no explicit patient_id: {}".format(sample.source_id))
        patient_id = int(sample.patient_id)
        if patient_id not in allowed or patient_id in {9, 10, 11}:
            raise DataIsolationError(
                "TNBC patient is outside the authorized donor/training range: {}".format(patient_id)
            )


def _validate_sample_arrays(sample: TrainingSample) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[np.ndarray]]:
    instance_map = np.asarray(sample.instance_map)
    coverage = np.asarray(sample.coverage_map)
    rgb = np.asarray(sample.rgb)
    if instance_map.ndim != 2:
        raise DonorAuditError("instance_map must be HxW")
    if coverage.shape != instance_map.shape:
        raise DonorAuditError("coverage_map must have the same HxW shape as instance_map")
    if rgb.ndim != 3 or rgb.shape[:2] != instance_map.shape or rgb.shape[2] != 3:
        raise DonorAuditError("rgb must be HxWx3 and align with instance_map")
    if not np.issubdtype(instance_map.dtype, np.integer):
        raise DonorAuditError("instance_map must use an integer instance ID dtype")
    if sample.od is None:
        od = rgb_to_od(rgb)
    else:
        od = np.asarray(sample.od)
        if od.shape != rgb.shape:
            raise DonorAuditError("od must be HxWx3 and align with rgb")
    prediction_masks = coerce_prediction_masks(sample.prediction_masks, instance_map.shape)
    return instance_map, coverage, rgb, od, prediction_masks


def coerce_prediction_masks(predictions: Any, expected_shape: Tuple[int, int]) -> List[np.ndarray]:
    """Accept a mask sequence, an NxHxW tensor, or a labelled HxW prediction map."""

    if predictions is None:
        return []
    if isinstance(predictions, np.ndarray):
        array = np.asarray(predictions)
        if array.ndim == 2:
            if array.shape != expected_shape:
                raise DonorAuditError("prediction map shape does not match GT")
            if array.dtype == bool:
                return [array.astype(bool, copy=False)]
            return [array == value for value in np.unique(array) if value != 0]
        if array.ndim == 3:
            masks = [array[index] for index in range(array.shape[0])]
        else:
            raise DonorAuditError("prediction masks must be HxW or NxHxW")
    else:
        masks = list(predictions)
    result: List[np.ndarray] = []
    for mask in masks:
        binary = np.asarray(mask).astype(bool, copy=False)
        if binary.shape != expected_shape:
            raise DonorAuditError("a prediction mask does not align with GT")
        result.append(binary)
    return result


def rgb_to_od(rgb: np.ndarray) -> np.ndarray:
    """Convert 8-bit or unit-range RGB to finite optical density values."""

    values = np.asarray(rgb, dtype=np.float64)
    if not np.isfinite(values).all():
        # Keep the invalid value visible to the audit rather than hiding it.
        return values
    if values.size and float(np.max(values)) > 1.0:
        values = values / 255.0
    return -np.log(np.clip(values, 1.0 / 255.0, 1.0))


def binary_iou(first: np.ndarray, second: np.ndarray) -> float:
    first_binary = np.asarray(first, dtype=bool)
    second_binary = np.asarray(second, dtype=bool)
    if first_binary.shape != second_binary.shape:
        raise DonorAuditError("IoU masks must share a shape")
    union = int(np.logical_or(first_binary, second_binary).sum())
    if union == 0:
        return 0.0
    return float(np.logical_and(first_binary, second_binary).sum() / union)


def best_iou_for_gt(gt_mask: np.ndarray, prediction_masks: Sequence[np.ndarray]) -> float:
    """Return the best IoU with any frozen prediction, or zero without overlap."""

    if not prediction_masks:
        return 0.0
    return max(binary_iou(gt_mask, prediction) for prediction in prediction_masks)


def _bbox_xyxy(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        raise DonorAuditError("GT instance is empty")
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def _connected_component_areas(mask: np.ndarray) -> List[int]:
    binary = np.asarray(mask, dtype=bool)
    visited = np.zeros_like(binary, dtype=bool)
    height, width = binary.shape
    areas: List[int] = []
    for start_y, start_x in zip(*np.nonzero(binary)):
        if visited[start_y, start_x]:
            continue
        stack = [(int(start_y), int(start_x))]
        visited[start_y, start_x] = True
        area = 0
        while stack:
            y, x = stack.pop()
            area += 1
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    next_y, next_x = y + dy, x + dx
                    if (
                        0 <= next_y < height
                        and 0 <= next_x < width
                        and binary[next_y, next_x]
                        and not visited[next_y, next_x]
                    ):
                        visited[next_y, next_x] = True
                        stack.append((next_y, next_x))
        areas.append(area)
    return areas


def _convex_hull_area(mask: np.ndarray) -> float:
    """Area of the hull of foreground pixel squares, without scipy/skimage."""

    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return 0.0
    points = sorted(
        {
            (int(x), int(y))
            for y, x in zip(ys, xs)
            for x, y in ((x, y), (x + 1, y), (x, y + 1), (x + 1, y + 1))
        }
    )
    if len(points) < 3:
        return float(len(ys))

    def cross(origin: Tuple[int, int], first: Tuple[int, int], second: Tuple[int, int]) -> int:
        return (first[0] - origin[0]) * (second[1] - origin[1]) - (first[1] - origin[1]) * (second[0] - origin[0])

    lower: List[Tuple[int, int]] = []
    for point in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: List[Tuple[int, int]] = []
    for point in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    hull = lower[:-1] + upper[:-1]
    twice_area = sum(
        hull[index][0] * hull[(index + 1) % len(hull)][1]
        - hull[(index + 1) % len(hull)][0] * hull[index][1]
        for index in range(len(hull))
    )
    return abs(twice_area) / 2.0


def _eccentricity(mask: np.ndarray) -> float:
    ys, xs = np.nonzero(mask)
    if len(ys) < 2:
        return 0.0
    covariance = np.cov(np.stack((ys, xs)), bias=True)
    eigenvalues = np.linalg.eigvalsh(covariance)
    maximum = float(max(eigenvalues[-1], 0.0))
    minimum = float(max(eigenvalues[0], 0.0))
    if maximum <= 0.0:
        return 0.0
    return float(math.sqrt(max(0.0, 1.0 - minimum / maximum)))


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    binary = np.asarray(mask, dtype=bool)
    height, width = binary.shape
    result = np.zeros_like(binary)
    radius_squared = radius * radius
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy > radius_squared:
                continue
            source_y0, source_y1 = max(0, -dy), min(height, height - dy)
            source_x0, source_x1 = max(0, -dx), min(width, width - dx)
            target_y0, target_y1 = max(0, dy), min(height, height + dy)
            target_x0, target_x1 = max(0, dx), min(width, width + dx)
            result[target_y0:target_y1, target_x0:target_x1] |= binary[source_y0:source_y1, source_x0:source_x1]
    return result


def annulus_mask(mask: np.ndarray, radius: int = 8) -> np.ndarray:
    if radius <= 0:
        raise DonorAuditError("annulus radius must be positive")
    binary = np.asarray(mask, dtype=bool)
    return np.logical_and(_dilate(binary, radius), ~binary)


def _channel_stats(values: np.ndarray, selector: np.ndarray) -> Mapping[str, Optional[List[float]]]:
    selected = np.asarray(values)[np.asarray(selector, dtype=bool)]
    if selected.size == 0:
        return {"mean": None, "std": None}
    if selected.ndim == 1:
        selected = selected[:, None]
    return {
        "mean": [float(value) for value in np.mean(selected, axis=0)],
        "std": [float(value) for value in np.std(selected, axis=0)],
    }


def _he_concentrations(od: np.ndarray) -> np.ndarray:
    return np.einsum("...c,kc->...k", np.asarray(od, dtype=np.float64), _HE_PINV)


def _gradient_energy(he: np.ndarray, selector: np.ndarray) -> Optional[float]:
    if not np.any(selector):
        return None
    gradients = []
    for channel in range(he.shape[2]):
        grad_y, grad_x = np.gradient(he[:, :, channel])
        gradients.append(grad_y * grad_y + grad_x * grad_x)
    energy = np.sqrt(np.sum(gradients, axis=0))
    return float(np.mean(energy[np.asarray(selector, dtype=bool)]))


def _patch_payload(
    mask: np.ndarray, rgb: np.ndarray, od: np.ndarray, ring: np.ndarray, bbox: Tuple[int, int, int, int], radius: int = 8
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
    x0, y0, x1, y1 = bbox
    height, width = mask.shape
    patch_x0, patch_y0 = max(0, x0 - radius), max(0, y0 - radius)
    patch_x1, patch_y1 = min(width, x1 + radius), min(height, y1 + radius)
    slices = (slice(patch_y0, patch_y1), slice(patch_x0, patch_x1))
    return (
        mask[slices].copy(),
        rgb[slices].copy(),
        od[slices].copy(),
        ring[slices].copy(),
        (patch_x0, patch_y0, patch_x1, patch_y1),
    )


def _touches_boundary(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any())


def _audit_one_instance(
    sample: TrainingSample,
    instance_id: int,
    instance_map: np.ndarray,
    coverage: np.ndarray,
    rgb: np.ndarray,
    od: np.ndarray,
    prediction_masks: Sequence[np.ndarray],
) -> InstanceAudit:
    mask = instance_map == instance_id
    area = int(mask.sum())
    bbox = _bbox_xyxy(mask)
    components = _connected_component_areas(mask)
    primary_fraction = float(max(components) / area) if components else 0.0
    ring = annulus_mask(mask, radius=8)
    he = _he_concentrations(od)
    rgb_finite = bool(np.isfinite(rgb).all())
    od_finite = bool(np.isfinite(od).all())
    best_iou = best_iou_for_gt(mask, prediction_masks)
    coverage_fraction = float(np.mean(np.asarray(coverage)[mask] > 0))
    covered = coverage_fraction >= 0.5
    hardness = 0.5 * (1.0 - best_iou) + 0.3 * float(best_iou < 0.5) + 0.2 * float(not covered)
    hull_area = _convex_hull_area(mask)
    solidity = float(min(1.0, area / hull_area)) if hull_area > 0 else 0.0
    payload = _patch_payload(mask, rgb, od, ring, bbox)
    return InstanceAudit(
        source_id=sample.source_id,
        dataset=sample.dataset,
        patient_id=None if sample.patient_id is None else int(sample.patient_id),
        instance_id=int(instance_id),
        best_iou=float(best_iou),
        coverage_fraction=coverage_fraction,
        covered=covered,
        hardness=float(hardness),
        area=area,
        bbox_xyxy=bbox,
        eccentricity=_eccentricity(mask),
        solidity=solidity,
        touches_image_boundary=_touches_boundary(mask),
        primary_component_fraction=primary_fraction,
        rgb_finite=rgb_finite,
        od_finite=od_finite,
        nucleus_rgb_stats=_channel_stats(rgb, mask),
        nucleus_od_stats=_channel_stats(od, mask),
        nucleus_he_stats=_channel_stats(he, mask),
        annulus_he_stats=_channel_stats(he, ring),
        annulus_gradient_energy=_gradient_energy(he, ring),
        source_metadata=dict(sample.source_metadata),
        mask=payload[0],
        rgb_patch=payload[1],
        od_patch=payload[2],
        annulus_mask=payload[3],
        patch_bbox_xyxy=payload[4],
    )


def donor_class_for(best_iou: float, matched_iou_q25: Optional[float]) -> Optional[str]:
    """Classify exactly according to the preregistered ResiMix thresholds."""

    if best_iou < 0.1:
        return "Missed"
    if best_iou < 0.5:
        return "IoU-Cliff"
    if matched_iou_q25 is not None and 0.5 <= best_iou <= matched_iou_q25:
        return "Low-Quality Matched"
    return None


def _filter_reasons(record: InstanceAudit, area_q05: float, area_q95: float) -> Tuple[str, ...]:
    reasons: List[str] = []
    x0, y0, x1, y1 = record.bbox_xyxy
    if record.touches_image_boundary:
        reasons.append("touches_image_boundary")
    if record.primary_component_fraction < 0.95:
        reasons.append("primary_component_fraction_lt_0.95")
    if not (area_q05 <= record.area <= area_q95):
        reasons.append("area_outside_training_q05_q95")
    if x1 <= x0 or y1 <= y0:
        reasons.append("invalid_bbox")
    if not record.rgb_finite:
        reasons.append("nonfinite_rgb")
    if not record.od_finite:
        reasons.append("nonfinite_od")
    return tuple(reasons)


def audit_training_samples(
    samples: Sequence[TrainingSample],
    allowed_tnbc_patients: Iterable[int] = _TNBC_ALLOWED_PATIENTS,
) -> DonorBank:
    """Build a training-only donor bank from frozen GT/prediction/coverage arrays.

    The 25th percentile is computed over all matched training GT instances
    (best IoU >= 0.5), while area limits are the training-GT 5th/95th
    percentiles.  No development or test array is inspected.
    """

    if not samples:
        raise DonorAuditError("At least one authorized training sample is required")
    audits: List[InstanceAudit] = []
    for sample in samples:
        validate_training_sample(sample, allowed_tnbc_patients=allowed_tnbc_patients)
        instance_map, coverage, rgb, od, predictions = _validate_sample_arrays(sample)
        for instance_id in sorted(int(value) for value in np.unique(instance_map) if value != 0):
            audits.append(
                _audit_one_instance(sample, instance_id, instance_map, coverage, rgb, od, predictions)
            )
    if not audits:
        raise DonorAuditError("Authorized training samples contain no GT instances")

    matched_ious = [record.best_iou for record in audits if record.best_iou >= 0.5]
    matched_iou_q25 = float(np.quantile(matched_ious, 0.25)) if matched_ious else None
    areas = [record.area for record in audits]
    area_q05 = float(np.quantile(areas, 0.05))
    area_q95 = float(np.quantile(areas, 0.95))

    donors: List[InstanceAudit] = []
    for record in audits:
        record.donor_class = donor_class_for(record.best_iou, matched_iou_q25)
        record.rejection_reasons = _filter_reasons(record, area_q05, area_q95)
        record.eligible = record.donor_class is not None and not record.rejection_reasons
        if record.eligible:
            donors.append(record)
    return DonorBank(
        audits=audits,
        donors=donors,
        matched_iou_q25=matched_iou_q25,
        area_q05=area_q05,
        area_q95=area_q95,
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def donor_manifest_row(record: InstanceAudit) -> Dict[str, str]:
    """Create the compact CSV row; transplant arrays stay in the in-memory bank."""

    return {
        "donor_id": "{}__inst_{:06d}".format(record.source_id, int(record.instance_id)),
        "category": record.donor_class or "",
        "payload_path": "donor_payloads/{}__inst_{:06d}.npz".format(record.source_id, int(record.instance_id)),
        "type_id": "1",
        "source_id": record.source_id,
        "dataset": record.dataset,
        "patient_id": "" if record.patient_id is None else str(record.patient_id),
        "instance_id": str(record.instance_id),
        "donor_class": record.donor_class or "",
        "best_iou": "{:.12g}".format(record.best_iou),
        "coverage_fraction": "{:.12g}".format(record.coverage_fraction),
        "covered": str(record.covered).lower(),
        "hardness": "{:.12g}".format(record.hardness),
        "area": str(record.area),
        "bbox_xyxy": json.dumps(record.bbox_xyxy),
        "eccentricity": "{:.12g}".format(record.eccentricity),
        "solidity": "{:.12g}".format(record.solidity),
        "primary_component_fraction": "{:.12g}".format(record.primary_component_fraction),
        "nucleus_rgb_stats": json.dumps(_json_safe(record.nucleus_rgb_stats), sort_keys=True),
        "nucleus_od_stats": json.dumps(_json_safe(record.nucleus_od_stats), sort_keys=True),
        "nucleus_he_stats": json.dumps(_json_safe(record.nucleus_he_stats), sort_keys=True),
        "annulus_he_stats": json.dumps(_json_safe(record.annulus_he_stats), sort_keys=True),
        "annulus_gradient_energy": "" if record.annulus_gradient_energy is None else "{:.12g}".format(record.annulus_gradient_energy),
        "patch_bbox_xyxy": json.dumps(record.patch_bbox_xyxy),
        "source_metadata": json.dumps(_json_safe(record.source_metadata), sort_keys=True),
    }


def write_donor_bank(bank: DonorBank, output_dir: str | Path) -> Tuple[Path, Path]:
    """Write the donor CSV/summary plus immutable per-donor NPZ payloads.

    Existing files are never overwritten; that prevents an accidental revision
    of an artifact after a formal run.
    """

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    csv_path = destination / "donor_bank_manifest.csv"
    summary_path = destination / "donor_bank_summary.json"
    if csv_path.exists() or summary_path.exists():
        raise FileExistsError("Refusing to overwrite an existing donor-bank artifact")
    payload_dir = destination / "donor_payloads"
    if payload_dir.exists():
        raise FileExistsError("Refusing to overwrite an existing donor payload directory")
    payload_dir.mkdir()

    fieldnames = list(donor_manifest_row(bank.donors[0]).keys()) if bank.donors else [
        "donor_id", "category", "payload_path", "type_id", "source_id", "dataset", "patient_id", "instance_id", "donor_class", "best_iou",
        "coverage_fraction", "covered", "hardness", "area", "bbox_xyxy", "eccentricity",
        "solidity", "primary_component_fraction", "nucleus_rgb_stats", "nucleus_od_stats",
        "nucleus_he_stats", "annulus_he_stats", "annulus_gradient_energy", "patch_bbox_xyxy",
        "source_metadata",
    ]
    with csv_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in bank.donors:
            row = donor_manifest_row(record)
            payload_path = destination / row["payload_path"]
            np.savez_compressed(
                payload_path,
                rgb=np.asarray(record.rgb_patch, dtype=np.uint8),
                mask=np.asarray(record.mask, dtype=np.uint8),
                annulus=np.asarray(record.annulus_mask, dtype=np.uint8),
                od=np.asarray(record.od_patch, dtype=np.float32),
                type_id=np.asarray(1, dtype=np.int16),
            )
            writer.writerow(row)
    with summary_path.open("x", encoding="utf-8") as handle:
        json.dump(_json_safe(bank.summary()), handle, indent=2, sort_keys=True)
        handle.write("\n")
    return csv_path, summary_path
