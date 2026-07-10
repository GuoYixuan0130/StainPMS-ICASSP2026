"""Create immutable StainRoute train/calibration manifests without GT access.

The script reads only image filenames.  TNBC assignment is patient-level;
MoNuSeg uses a deterministic image-level split because source/site metadata is
not supplied in the current repository.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stainroute.utils import canonical_json_sha256


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def _image_names(image_dir: Path) -> list[str]:
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
    return sorted(
        path.stem
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _with_checksum(payload: dict) -> dict:
    result = dict(payload)
    result["content_sha256"] = canonical_json_sha256(payload)
    return result


def build_monuseg_split_from_names(
    names: list[str], seed: int = 3407, calibration_count: int = 8
) -> dict:
    """Create an image-level 29/8 split from the official 37-image train set."""

    names = sorted(names)
    if len(names) < 2:
        raise ValueError(f"Need at least two MoNuSeg train images, found {len(names)}")
    if calibration_count <= 0 or calibration_count >= len(names):
        raise ValueError("calibration_count must be in [1, num_images - 1]")
    shuffled = list(names)
    random.Random(seed).shuffle(shuffled)
    calibration = sorted(shuffled[:calibration_count])
    router_train = sorted(shuffled[calibration_count:])
    return _with_checksum(
        {
            "schema_version": 1,
            "dataset": "MoNuSeg",
            "source_metadata_available": False,
            "split_method": "deterministic_image_level_shuffle",
            "seed": int(seed),
            "image_root_relative": "train_12/images",
            "router_train": router_train,
            "calibration": calibration,
            "official_test": "sealed; not listed or used by Stage 1",
        }
    )


def build_monuseg_split(image_dir: Path, seed: int = 3407, calibration_count: int = 8) -> dict:
    return build_monuseg_split_from_names(
        _image_names(image_dir), seed=seed, calibration_count=calibration_count
    )


def build_tnbc_split_from_names(train_names: list[str], test_names: list[str]) -> dict:
    """Create the fixed patient-level TNBC split from image name prefixes."""

    grouped: dict[int, list[str]] = {}
    for name in train_names + test_names:
        token = name.split("_", 1)[0]
        if not token.isdigit():
            raise ValueError(f"TNBC image name lacks numeric patient prefix: {name}")
        grouped.setdefault(int(token), []).append(name)

    expected_patients = set(range(1, 12))
    missing = sorted(expected_patients - set(grouped))
    unexpected = sorted(set(grouped) - expected_patients)
    train_patients = {int(name.split("_", 1)[0]) for name in train_names}
    test_patients = {int(name.split("_", 1)[0]) for name in test_names}
    if (
        missing
        or unexpected
        or train_patients != set(range(1, 9))
        or test_patients != set(range(9, 12))
    ):
        raise ValueError(f"Unexpected TNBC patient set; missing={missing}, unexpected={unexpected}")

    def select(patient_ids: range) -> list[str]:
        return sorted(name for patient in patient_ids for name in grouped[patient])

    return _with_checksum(
        {
            "schema_version": 1,
            "dataset": "TNBC",
            "split_method": "fixed_patient_level",
            "train_image_root_relative": "train_12/images",
            "test_image_root_relative": "test/images",
            "router_train_patients": [1, 2, 3, 4, 5, 6],
            "calibration_patients": [7, 8],
            "test_patients": [9, 10, 11],
            "router_train": select(range(1, 7)),
            "calibration": select(range(7, 9)),
            "test": select(range(9, 12)),
        }
    )


def build_tnbc_split(train_image_dir: Path, test_image_dir: Path) -> dict:
    return build_tnbc_split_from_names(
        _image_names(train_image_dir), _image_names(test_image_dir)
    )


def _write(payload: dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {destination} (content_sha256={payload['content_sha256']})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--monuseg-root", required=True, type=Path)
    parser.add_argument("--tnbc-root", required=True, type=Path)
    parser.add_argument("--seed", default=3407, type=int)
    parser.add_argument("--monuseg-calibration-count", default=8, type=int)
    parser.add_argument("--data-splits-dir", default=Path("data/splits"), type=Path)
    parser.add_argument("--config-splits-dir", default=Path("configs/splits"), type=Path)
    args = parser.parse_args()

    monuseg = build_monuseg_split(
        args.monuseg_root / "train_12" / "images",
        seed=args.seed,
        calibration_count=args.monuseg_calibration_count,
    )
    tnbc = build_tnbc_split(
        args.tnbc_root / "train_12" / "images",
        args.tnbc_root / "test" / "images",
    )
    for directory in (args.data_splits_dir, args.config_splits_dir):
        _write(monuseg, directory / "stainroute_monuseg.json")
        _write(tnbc, directory / "stainroute_tnbc.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
