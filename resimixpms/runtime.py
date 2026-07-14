"""Dataset-facing, deterministic ResiMix augmentation runtime.

The runtime is deliberately model-free.  It consumes a pre-audited donor bank
and the one immutable coverage cache, modifies at most one training crop, and
returns the synthetic instance id so the existing PMS branch can add its
ordinary residual point/mask supervision.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from resimixpms.experiment import sha256_file
from resimixpms.transplant import (
    CONTEXT_FEATURE_NAMES,
    CompositeResult,
    QualityDecision,
    annulus_mask,
    boundary_gradient_energy,
    choose_host_candidate,
    composite_transplant,
    deterministic_donor_choice,
    deterministic_geometry,
    deterministic_host_mode,
    enumerate_legal_hosts,
    od_affine_stain_match,
    quality_reject,
    rgb_to_od,
    stable_rng,
    transform_donor,
)


FORMAL_SEED = 3407
FORMAL_AUGMENTATION_PROBABILITY = 0.5
FORMAL_ACTIVE_START_EPOCH = 2
FORMAL_ACTIVE_END_EPOCH = 9


@dataclass(frozen=True)
class AugmentResult:
    image: np.ndarray
    instance_map: np.ndarray
    type_map: np.ndarray
    coverage_map: np.ndarray
    synthetic_instance_id: int | None
    event_index: int | None


def _as_hwc_rgb_uint8(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError("ResiMix expects an HxWx3 crop before Normalize/ToTensor")
    if not np.isfinite(arr).all():
        raise ValueError("ResiMix host crop contains NaN or Inf")
    if arr.dtype == np.uint8:
        return arr
    values = arr.astype(np.float64)
    if values.min(initial=0.0) >= -1e-8 and values.max(initial=0.0) <= 1.0 + 1e-8:
        values = values * 255.0
    return np.rint(np.clip(values, 0.0, 255.0)).astype(np.uint8)


def _number(payload: Mapping[str, Any], names: tuple[str, ...]) -> float:
    for name in names:
        if name in payload:
            return float(payload[name])
    raise KeyError("missing one of {}".format(", ".join(names)))


class ResiMixAugmentor:
    """Frozen ResiMix plan + donor-bank executor used only by the train dataset."""

    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path).resolve()
        with self.config_path.open("r", encoding="utf-8") as handle:
            self.config = json.load(handle)
        if int(self.config.get("seed", FORMAL_SEED)) != FORMAL_SEED:
            raise ValueError("ResiMix formal seed is fixed at 3407")
        if float(self.config.get("augmentation_probability", FORMAL_AUGMENTATION_PROBABILITY)) != FORMAL_AUGMENTATION_PROBABILITY:
            raise ValueError("ResiMix formal augmentation probability is fixed at 0.5")
        if int(self.config.get("active_start_epoch", FORMAL_ACTIVE_START_EPOCH)) != FORMAL_ACTIVE_START_EPOCH:
            raise ValueError("ResiMix must start at epoch 2")
        if int(self.config.get("active_end_epoch", FORMAL_ACTIVE_END_EPOCH)) != FORMAL_ACTIVE_END_EPOCH:
            raise ValueError("ResiMix formal active period ends at epoch 9")
        self.seed = FORMAL_SEED
        self.payload_root = Path(self.config.get("donor_payload_dir", "")).expanduser().resolve()
        self.manifest_path = Path(self.config.get("donor_bank_manifest", "")).expanduser().resolve()
        self.statistics_path = Path(self.config.get("host_context_statistics", "")).expanduser().resolve()
        if not self.manifest_path.is_file() or not self.statistics_path.is_file():
            raise FileNotFoundError("ResiMix donor manifest and context statistics must both exist")
        if not self.payload_root.is_dir():
            raise FileNotFoundError("ResiMix donor_payload_dir is unavailable: {}".format(self.payload_root))
        self.donors_by_category = self._load_donors()
        self.statistics = self._load_statistics()
        self._validate_configured_hashes()
        self.events: list[dict[str, Any]] = []

    def _validate_configured_hashes(self) -> None:
        """Reject a modified donor/statistics artifact before any augmentation."""
        for label, path, key in (
            ("donor bank manifest", self.manifest_path, "donor_bank_manifest_sha256"),
            ("host context statistics", self.statistics_path, "host_context_statistics_sha256"),
        ):
            expected = str(self.config.get(key, "") or "").lower()
            if not expected:
                raise ValueError(f"ResiMix config lacks frozen {key}")
            actual = sha256_file(path).lower()
            if actual != expected:
                raise ValueError(f"{label} SHA256 mismatch: expected {expected}, got {actual}")

    def _load_donors(self) -> dict[str, list[dict[str, str]]]:
        groups: dict[str, list[dict[str, str]]] = {
            "Missed": [], "IoU-Cliff": [], "Low-Quality Matched": []
        }
        with self.manifest_path.open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError("ResiMix donor bank is empty")
        for row in rows:
            category = str(row.get("category", ""))
            if category not in groups:
                raise ValueError("unknown donor category in manifest: {}".format(category))
            if not row.get("donor_id"):
                raise ValueError("donor manifest row lacks donor_id")
            relative_payload = row.get("payload_path") or "{}.npz".format(row["donor_id"])
            candidate = Path(relative_payload)
            if candidate.is_absolute() or ".." in candidate.parts:
                raise ValueError("donor payload_path must be a safe relative path")
            # The generated donor manifest is self-describing: its paths are
            # relative to the manifest directory (e.g.
            # ``donor_payloads/<id>.npz``).  Older synthetic unit fixtures use
            # bare names, which remain relative to the explicitly configured
            # payload directory.  Never concatenate both prefixes.
            if len(candidate.parts) == 1:
                payload = self.payload_root / candidate
            else:
                payload = self.manifest_path.parent / candidate
            payload = payload.resolve()
            try:
                payload.relative_to(self.payload_root)
            except ValueError as exc:
                raise ValueError("donor payload escapes donor_payload_dir") from exc
            if not payload.is_file():
                raise FileNotFoundError("missing donor payload: {}".format(payload))
            row = dict(row)
            if row.get("dataset") and str(row["dataset"]) != str(self.config.get("dataset", "")):
                raise ValueError("donor manifest dataset does not match ResiMix config")
            if str(self.config.get("dataset", "")) == "tnbc":
                try:
                    patient_id = int(row.get("patient_id", ""))
                except (TypeError, ValueError) as exc:
                    raise ValueError("TNBC donor manifest row lacks a valid patient_id") from exc
                if patient_id not in {1, 2, 3, 4, 5, 6}:
                    raise ValueError("TNBC donor is outside patients 1--6")
            row["_payload_path"] = str(payload)
            groups[category].append(row)
        return groups

    def _load_statistics(self) -> dict[str, Any]:
        with self.statistics_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        try:
            mean = np.asarray(payload.get("context_mean", payload.get("feature_mean")), dtype=np.float64)
            std = np.asarray(payload.get("context_std", payload.get("feature_std")), dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("host context statistics are malformed") from exc
        if mean.shape != (len(CONTEXT_FEATURE_NAMES),) or std.shape != mean.shape:
            raise ValueError("host context statistics must have one value per frozen feature")
        if not np.isfinite(mean).all() or not np.isfinite(std).all():
            raise ValueError("host context statistics contain NaN/Inf")
        payload["context_mean"] = mean
        payload["context_std"] = std
        payload["natural_boundary_gradient_p95"] = _number(
            payload, ("natural_boundary_gradient_p95", "boundary_gradient_p95")
        )
        payload["legal_context_distance_p95"] = _number(
            payload, ("legal_context_distance_p95", "context_distance_p95")
        )
        payload["tissue_total_od_threshold"] = float(payload.get("tissue_total_od_threshold", 0.15))
        return payload

    def _load_payload(self, donor: Mapping[str, str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        with np.load(str(donor["_payload_path"]), allow_pickle=False) as payload:
            try:
                rgb = np.asarray(payload["rgb"])
                mask = np.asarray(payload["mask"], dtype=bool)
                annulus = np.asarray(payload["annulus"], dtype=bool)
            except KeyError as exc:
                raise ValueError("donor payload lacks rgb/mask/annulus") from exc
            type_id = int(payload["type_id"]) if "type_id" in payload else int(donor.get("type_id", 1))
        if rgb.shape[:2] != mask.shape or annulus.shape != mask.shape:
            raise ValueError("donor payload has inconsistent spatial shapes")
        if not mask.any() or not annulus.any() or not np.isfinite(rgb).all():
            raise ValueError("donor payload violates finite/nonempty constraints")
        return _as_hwc_rgb_uint8(rgb), mask, annulus, type_id

    def enabled_for_epoch(self, epoch: int) -> bool:
        return FORMAL_ACTIVE_START_EPOCH <= int(epoch) <= FORMAL_ACTIVE_END_EPOCH

    def _tissue_mask(self, rgb: np.ndarray) -> np.ndarray:
        return rgb_to_od(rgb).sum(axis=-1) >= float(self.statistics["tissue_total_od_threshold"])

    def _event(self, **fields: Any) -> int:
        normalized = {key: value for key, value in fields.items()}
        normalized.setdefault("synthetic_prompt_added", False)
        self.events.append(normalized)
        return len(self.events) - 1

    def mark_prompt_added(self, event_index: int | None) -> None:
        if event_index is not None:
            self.events[event_index]["synthetic_prompt_added"] = True

    def consume_events(self) -> list[dict[str, Any]]:
        result, self.events = self.events, []
        return result

    def validate_formal_bindings(
        self,
        *,
        dataset: str,
        train_manifest: str | Path,
        train_crop_manifest: str | Path | None,
        coverage_manifest: str | Path,
    ) -> None:
        """Bind this donor bank to the exact cache and manifests used by PMS.

        This closes the otherwise subtle Cache-A/Cache-B failure mode where a
        donor audit and a training loader could each be internally valid but
        describe different static coverage maps.
        """
        if str(self.config.get("dataset", "")) != str(dataset):
            raise ValueError("ResiMix config dataset does not match data_identity")
        checks = (
            ("train_manifest", train_manifest, "train_manifest_sha256"),
            ("static_coverage_manifest", coverage_manifest, "static_coverage_manifest_sha256"),
        )
        if train_crop_manifest:
            checks = (*checks, ("train_crop_manifest", train_crop_manifest, "train_crop_manifest_sha256"))
        elif str(self.config.get("train_crop_manifest", "") or ""):
            raise ValueError("ResiMix config requires a frozen training crop manifest")
        for config_path_key, supplied_path, hash_key in checks:
            configured = Path(str(self.config.get(config_path_key, "") or "")).resolve()
            supplied = Path(supplied_path).resolve()
            if configured != supplied:
                raise ValueError(f"ResiMix {config_path_key} differs from the active dataset input")
            expected = str(self.config.get(hash_key, "") or "").lower()
            actual = sha256_file(supplied).lower()
            if not expected or expected != actual:
                raise ValueError(f"ResiMix {hash_key} does not bind the active input")

    def augment(
        self,
        image: np.ndarray,
        instance_map: np.ndarray,
        type_map: np.ndarray,
        coverage_map: np.ndarray,
        *,
        epoch: int,
        sample_key: str,
    ) -> AugmentResult:
        """Maybe transplant one donor; input coverage is returned bit-identical."""
        host = _as_hwc_rgb_uint8(image)
        inst = np.asarray(instance_map).copy()
        types = np.asarray(type_map).copy()
        coverage = np.asarray(coverage_map)
        if inst.shape != host.shape[:2] or types.shape != inst.shape or coverage.shape != inst.shape:
            raise ValueError("image, instance/type maps and coverage must share crop geometry")
        if not self.enabled_for_epoch(epoch):
            return AugmentResult(host, inst, types, coverage, None, None)
        activation_rng = stable_rng(self.seed, "resimix_activation", int(epoch), sample_key)
        if float(activation_rng.random()) >= FORMAL_AUGMENTATION_PROBABILITY:
            self._event(epoch=int(epoch), sample_key=sample_key, status="not_selected")
            return AugmentResult(host, inst, types, coverage, None, None)

        chosen = deterministic_donor_choice(
            self.donors_by_category, self.seed, (int(epoch), sample_key)
        )
        if chosen is None:
            raise RuntimeError("ResiMix enabled with no usable donor category")
        category, donor = chosen
        event = {
            "epoch": int(epoch), "sample_key": sample_key, "status": "rejected",
            "donor_id": str(donor["donor_id"]), "donor_category": category,
            "donor_source_id": str(donor.get("source_id", "")),
            "donor_patient_id": str(donor.get("patient_id", "")),
        }
        try:
            rgb, donor_mask, donor_annulus, donor_type = self._load_payload(donor)
            geometry = deterministic_geometry(self.seed, (int(epoch), sample_key, donor["donor_id"]))
            transformed = transform_donor(rgb, donor_mask, donor_annulus, geometry)
            tissue = self._tissue_mask(host)
            candidates = enumerate_legal_hosts(
                host, transformed, inst, coverage, tissue,
                self.statistics["context_mean"], self.statistics["context_std"],
                seed=self.seed, sample_key=(int(epoch), sample_key), max_candidates=32,
                clearance=3, coverage_threshold=0.5,
            )
            requested_mode = deterministic_host_mode(self.seed, (int(epoch), sample_key))
            selection = choose_host_candidate(
                candidates, requested_mode, self.seed, (int(epoch), sample_key), top_k=5)
            if selection is None:
                event.update({"reason": "no_legal_host", "requested_host_mode": requested_mode})
                event_index = self._event(**event)
                return AugmentResult(host, inst, types, coverage, None, event_index)
        except Exception as exc:
            # Configuration/payload failures must surface, whereas routine host
            # infeasibility is represented by the explicit no_legal_host path.
            raise RuntimeError("ResiMix donor preparation failed for {}".format(donor["donor_id"])) from exc

        # The annulus lives in host coordinates and must be derived from the
        # actual placed mask (not from an unplaced donor patch).
        from resimixpms.transplant import render_placed_mask
        placed_mask = render_placed_mask(transformed.mask, selection.candidate.placement, inst.shape)
        host_annulus = annulus_mask(placed_mask, width=8)
        stain_match = od_affine_stain_match(
            transformed.rgb, transformed.annulus, host, host_annulus, transformed.mask
        )
        composite: CompositeResult = composite_transplant(
            host, stain_match.rgb, transformed.mask, selection.candidate.center_yx, taper_width=2
        )
        seam = boundary_gradient_energy(composite.rgb, composite.placed_mask)
        decision: QualityDecision = quality_reject(
            # Area acceptance is relative to the untransformed donor, not the
            # scaled intermediate mask; otherwise the ±25% guard is vacuous.
            donor_mask,
            composite.placed_mask,
            occupied_instances=inst,
            composited_rgb=composite.rgb,
            stain_match=stain_match,
            seam_gradient=seam,
            natural_boundary_p95=float(self.statistics["natural_boundary_gradient_p95"]),
            context_distance=float(selection.candidate.context_distance),
            legal_context_p95=float(self.statistics["legal_context_distance_p95"]),
        )
        event.update({
            "requested_host_mode": selection.requested_mode,
            "host_mode": selection.used_mode,
            "host_fallback": bool(selection.used_fallback),
            "context_distance": float(selection.candidate.context_distance),
            "nearest_gt_distance": float(selection.candidate.nearest_gt_distance),
            "proposal_ranked_count": int(selection.ranked_count),
            "coverage_overlap_pixels": int(np.count_nonzero(coverage[composite.placed_mask] >= 0.5)),
            **{key: float(value) for key, value in decision.diagnostics.items()},
        })
        if not decision.accepted:
            event.update({"status": "rejected", "reason": "|".join(decision.reasons)})
            event_index = self._event(**event)
            return AugmentResult(host, inst, types, coverage, None, event_index)

        next_id = int(np.max(inst)) + 1
        augmented_inst = inst.copy()
        augmented_types = types.copy()
        augmented_inst[composite.placed_mask] = next_id
        augmented_types[composite.placed_mask] = donor_type
        event.update({
            "status": "accepted", "synthetic_instance_id": next_id,
            "instance_count_before": int(np.count_nonzero(np.unique(inst))),
            "instance_count_after": int(np.count_nonzero(np.unique(augmented_inst))),
        })
        event_index = self._event(**event)
        return AugmentResult(
            np.rint(np.clip(composite.rgb, 0.0, 255.0)).astype(np.uint8),
            augmented_inst,
            augmented_types,
            coverage,
            next_id,
            event_index,
        )
