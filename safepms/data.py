"""Closed-patient manifests and deterministic patient-balanced sampling."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from torch.utils.data import Sampler


TRAIN_PATIENTS = frozenset(range(1, 7))
DEVELOPMENT_PATIENTS = frozenset((7, 8))


def patient_of(image_id: str) -> int:
    try:
        return int(image_id.split("_", 1)[0])
    except (ValueError, IndexError) as error:
        raise ValueError(f"Invalid TNBC image ID: {image_id!r}") from error


def load_cache_manifest_ids(path: Path, *, role: str) -> list[str]:
    """Read exact formal cache image IDs without enumerating train_12."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "nurank_automatic_prompt_cache_v1" or payload.get("role") != role:
        raise ValueError(f"Unexpected formal cache manifest: {path}")
    ids = list(payload.get("image_ids", []))
    allowed = TRAIN_PATIENTS if role == "train" else DEVELOPMENT_PATIENTS
    if not ids or any(patient_of(image_id) not in allowed for image_id in ids):
        raise ValueError(f"Closed-patient guard failed for {role} manifest")
    if role == "development" and len(ids) != 7:
        raise ValueError("SafePMS requires the fixed seven-image development manifest")
    return ids


def manifest_sha256(image_ids: list[str]) -> str:
    return hashlib.sha256("\n".join(image_ids).encode("utf-8")).hexdigest()


class PatientBalancedSampler(Sampler[int]):
    """Round-robin, deterministic sampler with equal per-patient exposure."""

    def __init__(self, image_ids: list[str], *, rounds_per_patient: int, seed: int = 3407):
        self.image_ids = list(image_ids)
        self.seed = int(seed)
        self.rounds_per_patient = int(rounds_per_patient)
        self._patient_indices: dict[int, list[int]] = {}
        for index, image_id in enumerate(self.image_ids):
            self._patient_indices.setdefault(patient_of(image_id), []).append(index)
        if set(self._patient_indices) != TRAIN_PATIENTS:
            raise ValueError("Patient-balanced train sampler requires exactly patients 1-6")
        self.epoch = 0
        self._order: list[int] = []
        self.set_epoch(0)

    def set_epoch(self, epoch: int) -> None:
        import random

        self.epoch = int(epoch)
        groups: dict[int, list[int]] = {}
        for patient, values in self._patient_indices.items():
            shuffled = list(values)
            random.Random(self.seed + 1009 * self.epoch + patient).shuffle(shuffled)
            groups[patient] = shuffled
        self._order = []
        for round_index in range(self.rounds_per_patient):
            for patient in sorted(TRAIN_PATIENTS):
                values = groups[patient]
                self._order.append(values[round_index % len(values)])

    @property
    def image_order(self) -> list[str]:
        return [self.image_ids[index] for index in self._order]

    @property
    def checksum(self) -> str:
        return manifest_sha256(self.image_order)

    def __iter__(self):
        return iter(self._order)

    def __len__(self) -> int:
        return len(self._order)
