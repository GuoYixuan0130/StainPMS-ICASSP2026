"""Freeze the authorised local MoNuSeg 37-image training tree into a manifest.

This tool deliberately accepts only a train image root and a train label root.
It rejects paths containing a ``test`` component before listing any files, so it
cannot be repurposed to enumerate or inspect the sealed official test set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_train_only(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if any(part.lower() == "test" or "test14" in part.lower() for part in resolved.parts):
        raise ValueError(f"{label} must be a MoNuSeg training path, not a test path: {resolved}")
    if not resolved.is_dir():
        raise FileNotFoundError(f"{label} is not a directory: {resolved}")
    return resolved


def build_manifest(args: argparse.Namespace) -> dict:
    image_root = require_train_only(Path(args.image_root), "--image-root")
    label_root = require_train_only(Path(args.label_root), "--label-root")
    images = sorted(path for path in image_root.iterdir() if path.is_file() and path.suffix.lower() in {".tif", ".tiff"})
    if len(images) != int(args.expected_count):
        raise ValueError(f"expected {args.expected_count} training images, found {len(images)}")
    records = []
    for image_path in images:
        sample_id = image_path.stem
        label_path = (label_root / f"{sample_id}.mat").resolve()
        if not label_path.is_file():
            raise FileNotFoundError(f"missing prepared label for {sample_id}: {label_path}")
        records.append(
            {
                "sample_id": sample_id,
                "case_id": sample_id,
                "image_path": str(image_path.resolve()),
                "image_sha256": sha256_file(image_path),
                "label_path": str(label_path),
                "label_sha256": sha256_file(label_path),
            }
        )
    return {
        "schema_version": 1,
        "dataset": "monuseg",
        "protocol_id": args.protocol_id,
        "status": "train_only_continuity_prepared_labels",
        "role": "phase1_training_set_mechanism_diagnosis",
        "record_count": len(records),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "authorised_local_train37_tree",
        "sealed_test_policy": "test14 is rejected before directory enumeration",
        "prepared_label_policy": "StainPMS historical continuity labels; no XML/GT selection in Phase 1",
        "records": records,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--label-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-count", type=int, default=37)
    parser.add_argument("--protocol-id", default="monuseg_download37_continuity_v1_phase1_trainonly")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_manifest(args)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "output": str(output), "records": manifest["record_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
