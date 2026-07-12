"""Data, provenance, and freeze guards for SemiPMS Phase 0.

These functions are intentionally independent of torch/model code so that the
most important privacy and leakage contracts can be tested without a GPU.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


SEED = 3407
ALLOWED_PATIENTS = frozenset(range(1, 7))
CLOSED_PATIENTS = frozenset({7, 8, 9, 10, 11})


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def patient_from_stem(stem: str) -> int:
    match = re.match(r"^(\d{1,2})_", stem)
    if not match:
        raise ValueError(f"TNBC stem {stem!r} is missing its '<patient>_' prefix.")
    return int(match.group(1))


@dataclass(frozen=True)
class ImageRecord:
    patient: int
    stem: str
    image_path: str
    label_path: str
    image_sha256: str


def list_allowed_images(data_root: Path) -> list[ImageRecord]:
    """List only patients 1--6 and never open a closed-patient image/label."""
    image_root = data_root / "train_12" / "images"
    label_root = data_root / "train_12" / "labels"
    if not image_root.is_dir() or not label_root.is_dir():
        raise FileNotFoundError(
            f"Expected TNBC train_12 images/labels below {data_root}; neither test nor MoNuSeg is allowed."
        )
    records: list[ImageRecord] = []
    for image_path in sorted(path for path in image_root.iterdir() if path.is_file()):
        patient = patient_from_stem(image_path.stem)
        if patient in CLOSED_PATIENTS:
            # Directory metadata is inspected only to enforce the guard; the
            # closed file itself is neither opened nor checksummed.
            continue
        if patient not in ALLOWED_PATIENTS:
            raise ValueError(f"Unexpected TNBC patient {patient} in {image_path.name}.")
        label_path = label_root / f"{image_path.stem}.mat"
        if not label_path.is_file():
            raise FileNotFoundError(f"Missing label for allowed image {image_path.name}.")
        records.append(
            ImageRecord(
                patient=patient,
                stem=image_path.stem,
                image_path=str(image_path.resolve()),
                label_path=str(label_path.resolve()),
                image_sha256=sha256_file(image_path),
            )
        )
    patients = {record.patient for record in records}
    if patients != ALLOWED_PATIENTS:
        raise RuntimeError(f"SemiPMS needs patients 1--6 exactly; found {sorted(patients)}.")
    if len(records) != 30:
        raise RuntimeError(f"SemiPMS preregistration requires 30 images across patients 1--6; found {len(records)}.")
    return records


def deterministic_split(records: Iterable[ImageRecord], seed: int = SEED) -> tuple[list[ImageRecord], list[ImageRecord]]:
    """One checksum-stable labeled image per allowed patient; remaining 24 unlabeled."""
    by_patient: dict[int, list[ImageRecord]] = {patient: [] for patient in ALLOWED_PATIENTS}
    for record in records:
        if record.patient not in ALLOWED_PATIENTS:
            raise PermissionError(f"Closed patient {record.patient} reached deterministic_split.")
        by_patient[record.patient].append(record)
    labeled: list[ImageRecord] = []
    unlabeled: list[ImageRecord] = []
    for patient in sorted(ALLOWED_PATIENTS):
        items = sorted(by_patient[patient], key=lambda item: item.stem)
        if not items:
            raise RuntimeError(f"Patient {patient} has no Phase-0 image.")
        # The hash selection is insensitive to filesystem iteration order and
        # gives every patient exactly one labeled file.
        selected = min(
            items,
            key=lambda item: hashlib.sha256(f"{seed}:{patient}:{item.stem}".encode("utf-8")).hexdigest(),
        )
        labeled.append(selected)
        unlabeled.extend(item for item in items if item != selected)
    if len(labeled) != 6 or len(unlabeled) != 24:
        raise AssertionError(
            f"SemiPMS requires exactly 6 labeled + 24 unlabeled images; got {len(labeled)} + {len(unlabeled)}."
        )
    return labeled, sorted(unlabeled, key=lambda item: item.stem)


class HiddenGTGuard:
    """Explicitly prevents unlabeled label reads before the rule is frozen."""

    def __init__(self) -> None:
        self._frozen = False
        self.hidden_gt_reads = 0

    @property
    def frozen(self) -> bool:
        return self._frozen

    def freeze_acceptance_rule(self) -> None:
        self._frozen = True

    def allow_unlabeled_label_read(self, record: ImageRecord) -> None:
        if record.patient not in ALLOWED_PATIENTS:
            raise PermissionError(f"Closed patient {record.patient} is forbidden.")
        if not self._frozen:
            raise PermissionError(
                "Hidden unlabeled GT cannot be opened before the labeled-only acceptance rule is frozen."
            )
        self.hidden_gt_reads += 1


def validate_clean_checkpoint_name(checkpoint: Path) -> None:
    """Reject known TNBC-derived checkpoints before their payload is opened."""
    name = checkpoint.name.lower()
    forbidden_markers = ("tnbc", "pms", "e147", "e156", "stain", "finetune", "epoch")
    if any(marker in name for marker in forbidden_markers):
        raise PermissionError(f"Initialization checkpoint name is not clean official SAM2 provenance: {checkpoint.name}")
    if not (name.startswith("sam2_hiera_") and checkpoint.suffix.lower() in {".pt", ".pth"}):
        raise PermissionError("Initialization must be an official sam2_hiera_* pretrained weight, not a derived checkpoint.")


def inspect_clean_initialization(checkpoint: Path, payload: Any | None = None) -> dict[str, Any]:
    """Reject fine-tuned TNBC/PMS checkpoints before any model is instantiated.

    Official SAM2 public weights are expected to be named ``sam2_hiera_*.pt``
    and contain only the SAM2 model state. A provenance decision is recorded
    from filename plus checkpoint top-level schema; a checkpoint with point
    head, epoch, optimizer, texture bank, or TNBC/PMS naming is rejected.
    """
    validate_clean_checkpoint_name(checkpoint)
    # Importing torch here keeps the remainder of the guards CPU-testable.
    import torch

    if payload is None:
        payload = torch.load(checkpoint, map_location="cpu")
    if not isinstance(payload, dict) or "model" not in payload:
        raise PermissionError("Official SAM2 checkpoint must contain a top-level 'model' state dictionary.")
    forbidden_keys = {"model1", "optimizer", "epoch", "texture_memory_bank_list", "scheduler"}
    present = sorted(forbidden_keys.intersection(payload))
    if present:
        raise PermissionError(f"Initialization contains derived-training keys: {present}")
    return {
        "path": str(checkpoint.resolve()),
        "sha256": sha256_file(checkpoint),
        "provenance": "official_sam2_pretrained_schema_and_name_verified",
        "top_level_keys": sorted(payload.keys()),
        "tnbc_training_supervision": False,
        "point_head_initialization": "random",
    }


def data_manifest(data_root: Path, labeled: Iterable[ImageRecord], unlabeled: Iterable[ImageRecord]) -> dict[str, Any]:
    return {
        "dataset": "TNBC",
        "data_root": str(data_root.resolve()),
        "allowed_patients": sorted(ALLOWED_PATIENTS),
        "closed_patients": sorted(CLOSED_PATIENTS),
        "monuseg": "forbidden",
        "seed": SEED,
        "labeled": [asdict(record) for record in labeled],
        "unlabeled": [asdict(record) for record in unlabeled],
        "guards": {
            "unlabeled_gt_before_acceptance_freeze": "forbidden",
            "patients_7_to_11": "forbidden",
            "monuseg": "forbidden",
        },
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
