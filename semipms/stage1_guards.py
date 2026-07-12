"""Access and leakage guards specific to SemiPMS Stage 1.

Stage 1 adds TNBC patients 7--8 as a development set, while keeping the
24 train-side unlabeled annotations unavailable until every optimizer update
and pseudo-label decision has completed.  This module does not import torch so
the access contract is cheap to test on a CPU-only machine.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from semipms.guards import (
    ALLOWED_PATIENTS,
    ImageRecord,
    patient_from_stem,
    sha256_file,
)


DEVELOPMENT_PATIENTS = frozenset({7, 8})
CLOSED_STAGE1_PATIENTS = frozenset({9, 10, 11})


def list_stage1_records(data_root: Path) -> tuple[list[ImageRecord], list[ImageRecord]]:
    """Return train (1--6) and development (7--8) records without opening 9--11.

    The caller may checksum/open only the returned files.  In particular,
    filenames for patients 9--11 are inspected solely to enforce the guard;
    their pixels and labels are never read.
    """
    image_root = data_root / "train_12" / "images"
    label_root = data_root / "train_12" / "labels"
    if not image_root.is_dir() or not label_root.is_dir():
        raise FileNotFoundError("Expected TNBC train_12 images/labels; test and MoNuSeg are forbidden.")
    train: list[ImageRecord] = []
    development: list[ImageRecord] = []
    for image_path in sorted(path for path in image_root.iterdir() if path.is_file()):
        patient = patient_from_stem(image_path.stem)
        if patient in CLOSED_STAGE1_PATIENTS:
            continue
        if patient not in ALLOWED_PATIENTS and patient not in DEVELOPMENT_PATIENTS:
            raise PermissionError(f"Unexpected patient {patient} in {image_path.name}.")
        label_path = label_root / f"{image_path.stem}.mat"
        if not label_path.is_file():
            raise FileNotFoundError(f"Missing allowed label {label_path.name}.")
        record = ImageRecord(
            patient=patient,
            stem=image_path.stem,
            image_path=str(image_path.resolve()),
            label_path=str(label_path.resolve()),
            image_sha256=sha256_file(image_path),
        )
        (train if patient in ALLOWED_PATIENTS else development).append(record)
    if {item.patient for item in train} != ALLOWED_PATIENTS or len(train) != 30:
        raise RuntimeError(f"Stage 1 requires 30 train records from patients 1--6; found {len(train)}.")
    if {item.patient for item in development} != DEVELOPMENT_PATIENTS:
        raise RuntimeError("Stage 1 requires development patients 7 and 8 exactly.")
    return train, development


class Stage1AccessGuard:
    """Proves that train-side hidden GT was not used while training decisions ran."""

    def __init__(self) -> None:
        self.config_frozen = False
        self.training_finished = False
        self.hidden_train_label_reads = 0
        self.development_label_reads = 0

    def freeze_training_configuration(self) -> None:
        self.config_frozen = True

    def mark_training_finished(self) -> None:
        if not self.config_frozen:
            raise PermissionError("Training cannot finish before the pseudo-label configuration is frozen.")
        self.training_finished = True

    def allow_development_label_read(self, record: ImageRecord) -> None:
        if record.patient not in DEVELOPMENT_PATIENTS:
            raise PermissionError(f"Patient {record.patient} is not a Stage-1 development record.")
        self.development_label_reads += 1

    def allow_hidden_train_audit_read(self, record: ImageRecord) -> None:
        if record.patient not in ALLOWED_PATIENTS:
            raise PermissionError(f"Patient {record.patient} is not an allowed train-side audit record.")
        if not (self.config_frozen and self.training_finished):
            raise PermissionError(
                "Unlabeled train GT is diagnostic-only and cannot be read before all Stage-1 training finishes."
            )
        self.hidden_train_label_reads += 1

    def manifest(self) -> dict[str, object]:
        return {
            "development_patients": sorted(DEVELOPMENT_PATIENTS),
            "closed_patients": sorted(CLOSED_STAGE1_PATIENTS),
            "pseudo_label_train_gt": "forbidden until all training is finished",
            "development_model_selection": "permitted for patients 7--8 only",
        }


def stage1_data_manifest(
    data_root: Path,
    labeled: Iterable[ImageRecord],
    unlabeled: Iterable[ImageRecord],
    development: Iterable[ImageRecord],
) -> dict[str, object]:
    return {
        "dataset": "TNBC",
        "data_root": str(data_root.resolve()),
        "train_patients": sorted(ALLOWED_PATIENTS),
        "development_patients": sorted(DEVELOPMENT_PATIENTS),
        "closed_patients": sorted(CLOSED_STAGE1_PATIENTS),
        "monuseg": "forbidden",
        "labeled": [asdict(item) for item in labeled],
        "unlabeled_train": [asdict(item) for item in unlabeled],
        "development": [asdict(item) for item in development],
        "guards": Stage1AccessGuard().manifest(),
    }
