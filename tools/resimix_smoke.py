"""Mechanical pre-training smoke for one frozen ResiMix dataset configuration.

The smoke creates real synthetic training crops from the authorized training
split but performs no optimizer/model step.  It fails on label/cache/numerical
errors and deliberately does not apply a scientific quality NO-GO threshold.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _set_seed(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_args(options, *, resimix_enabled):
    return SimpleNamespace(
        crop_size=256,
        overlap=int(options.overlap),
        use_pms=True,
        pms_self_bootstrap=False,
        pms_gt_match_radius=8,
        pms_baseline_prompts=False,
        pms_preserve_max_prompts=0,
        baseline_masks_dir=str(Path(options.coverage_manifest).parent),
        coverage_manifest=str(options.coverage_manifest),
        resimix_enabled=bool(resimix_enabled),
        resimix_config=str(options.resimix_config) if resimix_enabled else "",
        data_identity=options.dataset,
        train_manifest=str(options.train_manifest),
        test_manifest=str(options.test_manifest),
        train_crop_manifest=str(options.train_crop_manifest or ""),
        eval_crop_manifest=str(options.eval_crop_manifest or ""),
        train_image_root=str(options.train_image_root),
        train_label_root=str(options.train_label_root),
        test_image_root=str(options.test_image_root),
        test_label_root=str(options.test_label_root),
        allowed_patient_ids="",
        train_allowed_patient_ids="1,2,3,4,5,6" if options.dataset == "tnbc" else "",
        test_allowed_patient_ids="7,8" if options.dataset == "tnbc" else "",
        forbidden_patient_ids="9,10,11",
    )


def _build_dataset(options, *, resimix_enabled):
    from mmengine.config import Config
    from run.dataset.monuseg import MONUSEG

    args = _build_args(options, resimix_enabled=resimix_enabled)
    model_args = Config.fromfile(str(ROOT / "args.py"))
    model_args.criterion.pms_loss_coef = 0.5
    model_args.criterion.pms_residual_mask_weight = 0.3
    model_args.criterion.pms_preserve_loss_coef = 1.0
    model_args.criterion.pms_object_weight = 1.0
    return MONUSEG(args, model_args, str(options.data_path), options.load, mode="train")


def _as_rgb(tensor):
    array = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    array = array * np.asarray((0.229, 0.224, 0.225)) + np.asarray((0.485, 0.456, 0.406))
    return np.rint(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)


def _check_batch(dataset, batch, events, frames):
    checks = {
        "accepted": 0, "label_errors": 0, "cache_errors": 0,
        "medoid_errors": 0, "numeric_errors": 0,
    }
    for event in events:
        if event.get("status") != "accepted":
            continue
        checks["accepted"] += 1
        if int(event.get("instance_count_after", -1)) != int(event.get("instance_count_before", -2)) + 1:
            checks["label_errors"] += 1
        crop_index = int(str(event["sample_key"]).split(":")[-3])
        synthetic_id = int(event["synthetic_instance_id"])
        image = batch[0][crop_index]
        instance_masks = batch[1][crop_index]
        b_coords = batch[11][crop_index].detach().cpu().numpy()
        b_gt_masks = batch[13][crop_index].detach().cpu().numpy().astype(bool)
        if not np.isfinite(image.detach().cpu().numpy()).all():
            checks["numeric_errors"] += 1
            continue
        if instance_masks.shape[0] < 1:
            checks["label_errors"] += 1
            continue
        # Instance IDs are sorted before stacking, so a new max ID is the
        # final synthetic mask.  It must remain one connected mask and enter
        # exactly one ordinary PMS positive mask/point pair.
        synthetic_mask = instance_masks[-1].detach().cpu().numpy().astype(bool)
        if not synthetic_mask.any():
            checks["label_errors"] += 1
            continue
        # This diagnostic is computed on the post-Albumentations coverage map
        # passed to the runtime, not on an untransformed full-image cache.
        if int(event.get("coverage_overlap_pixels", -1)) != 0:
            checks["cache_errors"] += 1
        matched = [index for index, gt_mask in enumerate(b_gt_masks) if np.array_equal(gt_mask, synthetic_mask)]
        if len(matched) != 1:
            checks["medoid_errors"] += 1
        else:
            point = b_coords[matched[0]].astype(int)
            if not synthetic_mask[point[1], point[0]]:
                checks["medoid_errors"] += 1
        if event.get("synthetic_prompt_added") is not True:
            checks["medoid_errors"] += 1
        if len(frames) < 64:
            frame = _as_rgb(image)
            frame[synthetic_mask] = (255, 45, 45)
            frames.append(frame)
    return checks


def _save_montage(frames, path):
    from PIL import Image
    if len(frames) < 64:
        raise RuntimeError(f"smoke produced only {len(frames)} synthetic crops; 64 are required")
    side = 8
    height, width = frames[0].shape[:2]
    montage = np.zeros((side * height, side * width, 3), dtype=np.uint8)
    for index, frame in enumerate(frames[:64]):
        row, col = divmod(index, side)
        montage[row * height:(row + 1) * height, col * width:(col + 1) * width] = frame
    Image.fromarray(montage).save(path)


def _digest(value, digest):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(np.ascontiguousarray(value).tobytes())
    elif isinstance(value, (list, tuple)):
        digest.update(b"[")
        for item in value:
            _digest(item, digest)
        digest.update(b"]")
    else:
        digest.update(repr(value).encode("utf-8"))


def _baseline_equivalence(options):
    """Compare Static-PMS against the enabled ResiMix warm-up tensor path."""
    _set_seed(3407)
    control_dataset = _build_dataset(options, resimix_enabled=False)
    control_dataset.set_epoch(1)
    control = control_dataset[0]
    _set_seed(3407)
    warmup_dataset = _build_dataset(options, resimix_enabled=True)
    warmup_dataset.set_epoch(1)
    warmup = warmup_dataset[0]
    left, right = hashlib.sha256(), hashlib.sha256()
    _digest(control, left)
    _digest(warmup, right)
    return {
        "setpms_enabled": False,
        "control_resimix_warmup_pixel_exact": left.digest() == right.digest(),
        "control_batch_sha256": left.hexdigest(),
        "resimix_warmup_batch_sha256": right.hexdigest(),
        "warmup_epoch": 1,
    }


def _deterministic_replay(options):
    """Replay one active epoch twice, including real dataset/PMS assembly."""
    def run_once():
        _set_seed(3407)
        dataset = _build_dataset(options, resimix_enabled=True)
        dataset.set_epoch(2)
        digest = hashlib.sha256()
        events = []
        for index in range(len(dataset)):
            _digest(dataset[index], digest)
            events.extend(dataset.consume_resimix_events())
        return digest.hexdigest(), json.dumps(events, sort_keys=True, default=str)
    first_hash, first_events = run_once()
    second_hash, second_events = run_once()
    return {
        "epoch": 2,
        "batch_sha256_first": first_hash,
        "batch_sha256_second": second_hash,
        "events_exact": first_events == second_events,
        "pixel_exact": first_hash == second_hash,
    }


def _argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("tnbc", "monuseg_lite"))
    parser.add_argument("--data-path", required=True, type=Path)
    parser.add_argument("--train-manifest", required=True, type=Path)
    parser.add_argument("--test-manifest", required=True, type=Path)
    parser.add_argument("--coverage-manifest", required=True, type=Path)
    parser.add_argument("--resimix-config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--overlap", required=True, type=int)
    parser.add_argument("--load", default="unclockwise")
    # ``Path(\"\")`` is ``.`` and would incorrectly be treated as a supplied
    # manifest.  TNBC has no crop manifests; retain None until one is passed.
    parser.add_argument("--train-crop-manifest", default=None, type=Path)
    parser.add_argument("--eval-crop-manifest", default=None, type=Path)
    parser.add_argument("--train-image-root", required=True, type=Path)
    parser.add_argument("--train-label-root", required=True, type=Path)
    parser.add_argument("--test-image-root", required=True, type=Path)
    parser.add_argument("--test-label-root", required=True, type=Path)
    return parser


def _parse_options(argv=None):
    return _argument_parser().parse_args(argv)


def main():
    options = _parse_options()
    options.output_dir.mkdir(parents=True, exist_ok=False)
    _set_seed(3407)
    dataset = _build_dataset(options, resimix_enabled=True)
    all_events, frames = [], []
    totals = {"accepted": 0, "label_errors": 0, "cache_errors": 0, "medoid_errors": 0, "numeric_errors": 0}
    for epoch in range(2, 10):
        dataset.set_epoch(epoch)
        for index in range(len(dataset)):
            batch = dataset[index]
            events = dataset.consume_resimix_events()
            all_events.extend(events)
            outcome = _check_batch(dataset, batch, events, frames)
            for key, value in outcome.items():
                totals[key] += value
    fields = sorted({field for event in all_events for field in event} | {"epoch", "sample_key", "status"})
    with (options.output_dir / "synthetic_acceptance.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_events)
    equivalence = _baseline_equivalence(options)
    replay = _deterministic_replay(options)
    (options.output_dir / "baseline_equivalence.json").write_text(json.dumps(equivalence, indent=2) + "\n", encoding="utf-8")
    _save_montage(frames, options.output_dir / "synthetic_montage.png")
    report = {
        "dataset": options.dataset, "seed": 3407, "synthetic_crops": len(frames),
        "label_error_rate": totals["label_errors"] / max(1, totals["accepted"]),
        "checks": totals, "baseline_equivalence": equivalence, "deterministic_replay": replay,
        "passed": all(totals[key] == 0 for key in ("label_errors", "cache_errors", "medoid_errors", "numeric_errors")) and len(frames) >= 64 and equivalence["control_resimix_warmup_pixel_exact"] and replay["pixel_exact"] and replay["events_exact"],
    }
    (options.output_dir / "smoke_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not report["passed"]:
        raise SystemExit("ResiMix smoke failed mechanical checks")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
