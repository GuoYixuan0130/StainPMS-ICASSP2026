"""Authorised one-seed SetPMS Stage 1 dual-development runner.

This runner is intentionally narrow.  It only touches TNBC patients 1--8
through ``train_12`` (1--6 train, 7--8 dev) and MoNuSeg ``train_12``.  It never
constructs a dataset rooted at TNBC patients 9--11 or MoNuSeg official test.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# ``python tools/run_setpms_stage1.py`` otherwise places only ``tools/`` on
# sys.path.  Make the repository root explicit before importing project code.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np
from scipy.io import loadmat
from skimage import io

from run.dataset.monuseg import crop_with_overlap, deterministic_crop_indices


CANONICAL_SHA = "2a1348cb7a1158a6f77aae2f92c168f9552d8068"
SEED = 3407
TNBC_CHECKPOINT_SHA256 = "44a3cb3e93051301d789e44f93769588abfa727d7a174b0270f55305ef023781"
MONUSEG_CHECKPOINT_SHA256 = "6616c24626a162580d79aa7035d0b6c1a4ae6240a6c2f4599960dd4efac95db1"
TNBC_REFERENCE_AJI = 0.750788
TNBC_REFERENCE_PQ = 0.742225
TNBC_REFERENCE_TOLERANCE = 0.005
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json_dump(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=root, text=True, stderr=subprocess.STDOUT
    ).strip()


def _run(root: Path, command: list[str], log_path: Path) -> float:
    """Run one authorised child process and preserve its complete stdout."""

    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("COMMAND\n")
        handle.write(" ".join(command))
        handle.write("\n\nOUTPUT\n")
        completed = subprocess.run(
            command,
            cwd=root,
            text=True,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
            env={**os.environ, "PYTHONHASHSEED": str(SEED)},
        )
    elapsed = time.monotonic() - started
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed ({completed.returncode}); see {log_path}")
    return elapsed


def _image_files(image_dir: Path) -> list[str]:
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Missing authorised image directory: {image_dir}")
    files = sorted(path.name for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if not files:
        raise RuntimeError(f"No images found in {image_dir}")
    return files


def _require_labels(root: Path, filenames: list[str]) -> None:
    for filename in filenames:
        label = root / "labels" / f"{Path(filename).stem}.mat"
        if not label.is_file():
            raise FileNotFoundError(f"Missing label required by manifest: {label}")


def _tnbc_patient(filename: str) -> int:
    match = re.match(r"^(\d{1,2})(?:[_-]|$)", Path(filename).stem)
    if not match:
        raise ValueError(f"Cannot safely infer TNBC patient from filename {filename!r}")
    return int(match.group(1))


def _make_tnbc_manifest(data_root: Path, output_path: Path) -> dict:
    train_root = data_root / "train_12"
    filenames = _image_files(train_root / "images")
    _require_labels(train_root, filenames)
    patient_map = {filename: _tnbc_patient(filename) for filename in filenames}
    prohibited = [name for name, patient in patient_map.items() if patient in {9, 10, 11}]
    if prohibited:
        raise RuntimeError(
            "TNBC closed patients 9--11 appeared in train_12 manifest construction: "
            f"{prohibited}"
        )
    unknown = [
        name for name, patient in patient_map.items() if patient not in set(range(1, 9))
    ]
    if unknown:
        raise RuntimeError(f"TNBC manifest contains unapproved patient IDs: {unknown}")
    train_files = [name for name in filenames if patient_map[name] in set(range(1, 7))]
    eval_files = [name for name in filenames if patient_map[name] in {7, 8}]
    if len(eval_files) != 7:
        raise RuntimeError(
            f"TNBC Full-Dev requires exactly seven patient-7/8 images, found {len(eval_files)}"
        )
    if not train_files:
        raise RuntimeError("TNBC patients 1--6 yielded no training images")
    payload = {
        "dataset": "TNBC",
        "source": str(train_root),
        "seed": SEED,
        "train_patients": [1, 2, 3, 4, 5, 6],
        "development_patients": [7, 8],
        "closed_patients": [9, 10, 11],
        "train_files": train_files,
        "eval_files": eval_files,
        "patient_by_file": patient_map,
        "development_image_count": len(eval_files),
    }
    _json_dump(output_path, payload)
    return payload


def _hash_rank(*tokens: object) -> str:
    return hashlib.sha256("".join(str(token) for token in tokens).encode("utf-8")).hexdigest()


def _three_way_density_groups(rows: list[dict]) -> dict[str, list[dict]]:
    ordered = sorted(rows, key=lambda row: (row["instance_count"], row["filename"]))
    chunks = np.array_split(np.asarray(ordered, dtype=object), 3)
    names = ("low", "mid", "high")
    return {name: list(chunk.tolist()) for name, chunk in zip(names, chunks, strict=True)}


def _candidate_windows(height: int, width: int, patch_size: int = 512) -> list[tuple[int, int]]:
    if height < patch_size or width < patch_size:
        raise RuntimeError(
            f"MoNuSeg-Lite requires 512x512 patches, got image {width}x{height}"
        )
    xs = [int(round(value)) for value in np.linspace(0, width - patch_size, num=3)]
    ys = [int(round(value)) for value in np.linspace(0, height - patch_size, num=3)]
    windows = [(x, y) for y in ys for x in xs]
    if len(windows) != 9:
        raise AssertionError("Expected a fixed 3x3 window grid")
    return windows


def _choose_patch_records(data_root: Path, holdouts: list[str]) -> list[dict]:
    image_root = data_root / "train_12" / "images"
    label_root = data_root / "train_12" / "labels"
    records = []
    for filename in holdouts:
        image = io.imread(image_root / filename)[..., :3]
        inst_map = loadmat(label_root / f"{Path(filename).stem}.mat")["inst_map"]
        candidates = []
        for candidate_index, (x, y) in enumerate(_candidate_windows(*inst_map.shape)):
            patch = inst_map[y : y + 512, x : x + 512]
            density = int(max(0, np.unique(patch).size - 1))
            candidates.append({"index": candidate_index, "x": x, "y": y, "density": density})
        median_density = float(np.median([item["density"] for item in candidates]))
        median_choice = min(
            candidates,
            key=lambda item: (
                abs(item["density"] - median_density),
                _hash_rank(SEED, filename, "median", item["index"]),
            ),
        )
        strict_nonmax = [
            item for item in candidates if item["density"] < max(candidate["density"] for candidate in candidates)
        ]
        high_pool = strict_nonmax or [
            item for item in candidates if item["index"] != median_choice["index"]
        ]
        if not high_pool:
            raise RuntimeError(f"No distinct high-density candidate for {filename}")
        high_density = max(item["density"] for item in high_pool)
        high_choice = min(
            [item for item in high_pool if item["density"] == high_density],
            key=lambda item: _hash_rank(SEED, filename, "high", item["index"]),
        )
        for role, choice in (("median", median_choice), ("high_nonmax", high_choice)):
            x, y = choice["x"], choice["y"]
            image_patch = np.ascontiguousarray(image[y : y + 512, x : x + 512])
            label_patch = np.ascontiguousarray(inst_map[y : y + 512, x : x + 512])
            records.append(
                {
                    "patch_id": f"{Path(filename).stem}__{role}",
                    "filename": filename,
                    "role": role,
                    "x": int(x),
                    "y": int(y),
                    "width": 512,
                    "height": 512,
                    "gt_instance_density": int(choice["density"]),
                    "candidate_densities": [int(item["density"]) for item in candidates],
                    "image_patch_sha256": _sha256_bytes(image_patch.tobytes()),
                    "label_patch_sha256": _sha256_bytes(label_patch.tobytes()),
                }
            )
    if len(records) != 12:
        raise AssertionError(f"Expected 12 frozen MoNuSeg-Lite patches, got {len(records)}")
    return records


def _make_monuseg_lite_manifest(data_root: Path, output_path: Path) -> dict:
    train_root = data_root / "train_12"
    filenames = _image_files(train_root / "images")
    _require_labels(train_root, filenames)
    if len(filenames) != 37:
        raise RuntimeError(f"MoNuSeg-Lite requires train_12=37 images, found {len(filenames)}")
    rows = []
    for filename in filenames:
        inst_map = loadmat(train_root / "labels" / f"{Path(filename).stem}.mat")["inst_map"]
        rows.append(
            {
                "filename": filename,
                "instance_count": int(max(0, np.unique(inst_map).size - 1)),
            }
        )
    groups = _three_way_density_groups(rows)
    holdouts = []
    selected_by_group = {}
    for group_name, group_rows in groups.items():
        if len(group_rows) < 2:
            raise RuntimeError(f"MoNuSeg density group {group_name} has fewer than two images")
        selected = sorted(
            group_rows,
            key=lambda row: _hash_rank(SEED, row["filename"]),
        )[:2]
        selected_names = [row["filename"] for row in selected]
        selected_by_group[group_name] = selected_names
        holdouts.extend(selected_names)
    holdouts = sorted(holdouts)
    if len(holdouts) != 6 or len(set(holdouts)) != 6:
        raise RuntimeError("MoNuSeg-Lite holdout selection was not six unique images")
    train_files = [filename for filename in filenames if filename not in set(holdouts)]
    if len(train_files) != 31:
        raise RuntimeError(f"MoNuSeg-Lite requires 31 continuation images, found {len(train_files)}")

    crop_indices = {}
    image_root = train_root / "images"
    for epoch in range(10):
        epoch_entries = {}
        for filename in train_files:
            image = io.imread(image_root / filename)[..., :3]
            image_chw = np.transpose(image, (2, 0, 1))
            boxes = crop_with_overlap(image_chw, 256, 256, 92, "unclockwise").tolist()
            epoch_entries[filename] = deterministic_crop_indices(
                SEED, epoch, filename, len(boxes), 4
            )
        crop_indices[str(epoch)] = epoch_entries

    payload = {
        "dataset": "MoNuSeg-Lite",
        "source": str(train_root),
        "seed": SEED,
        "selection_note": (
            "The initial StainPMS checkpoint historically saw the complete train split; "
            "this is dataset-response screening, not an independent generalisation result."
        ),
        "density_by_file": {row["filename"]: row["instance_count"] for row in rows},
        "density_groups": {
            name: [row["filename"] for row in group_rows] for name, group_rows in groups.items()
        },
        "holdout_by_density_group": selected_by_group,
        "train_files": train_files,
        "eval_files": holdouts,
        "holdout_files": holdouts,
        "max_train_crops_per_image": 4,
        "crop_indices": crop_indices,
        "patches": _choose_patch_records(data_root, holdouts),
    }
    _json_dump(output_path, payload)
    return payload


def _base_command(
    python: str,
    data_root: Path,
    checkpoint: Path,
    train_manifest: Path,
    eval_manifest: Path,
    run_dir: Path,
    metrics_dir: Path,
    label: str,
    overlap: int,
    save_epochs: str,
    eval_epochs: str,
    *,
    setpms: bool,
    crop_manifest: Path | None = None,
    patch_manifest: Path | None = None,
) -> list[str]:
    command = [
        python,
        "main.py",
        "--dataset", "monuseg",
        "--data_path", str(data_root),
        "--sam_ckpt", str(checkpoint),
        "--sam_config", "sam2_hiera_l",
        "--seed", str(SEED),
        "--epochs", "10",
        "--lr", "1e-5",
        "--lr_min", "1e-6",
        "--weight_decay", "1e-4",
        "--b", "1",
        "--texture",
        "--context",
        "--overlap", str(overlap),
        "--test_nms_thr", "12",
        "--eval_on_train",
        "--train_manifest", str(train_manifest),
        "--eval_manifest", str(eval_manifest),
        "--continuation_save_epochs", save_epochs,
        "--continuation_eval_epochs", eval_epochs,
        "--run_dir", str(run_dir),
        "--metrics_output_dir", str(metrics_dir),
        "--run_label", label,
    ]
    if setpms:
        command.append("--setpms")
    if crop_manifest is not None:
        command.extend(["--max_train_crops_per_image", "4", "--train_crop_manifest", str(crop_manifest)])
    if patch_manifest is not None:
        command.extend(["--eval_patch_manifest", str(patch_manifest)])
    return command


def _read_rows(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Expected child artifact missing: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write empty required artifact {path.name}")
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _child_rows(child: Path, name: str) -> list[dict]:
    return _read_rows(child / "metrics" / name)


def _metric_rows_for_label(rows: list[dict], label: str) -> list[dict]:
    selected = [row for row in rows if row["run_label"] == label]
    if not selected:
        raise RuntimeError(f"No metrics found for {label}")
    for row in selected:
        row["epoch"] = int(row["epoch"])
        for key in ("dice", "dice2", "aji", "aji_plus", "dq", "sq", "pq"):
            row[key] = float(row[key])
    return selected


def _by_epoch(rows: list[dict], epoch: int) -> dict:
    matches = [row for row in rows if int(row["epoch"]) == epoch]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one metric row at epoch {epoch}, found {len(matches)}")
    return matches[0]


def _best_metric_row(rows: list[dict]) -> dict:
    return max(rows, key=lambda row: (row["aji"] + row["pq"], -int(row["epoch"])))


def _metric_delta(left: dict, right: dict) -> dict:
    return {
        key: float(left[key] - right[key])
        for key in ("dice", "dice2", "aji", "aji_plus", "dq", "sq", "pq")
    }


def _enrich_tnbc_patients(rows: list[dict]) -> list[dict]:
    for row in rows:
        image = str(row["image"])
        source = image.split("__", 1)[0]
        row["patient"] = _tnbc_patient(source)
    return rows


def _per_patient(rows: list[dict], label: str, epoch: int) -> dict:
    selected = [row for row in rows if row["run_label"] == label and int(row["epoch"]) == epoch]
    groups = defaultdict(list)
    for row in selected:
        groups[str(row["patient"])].append(row)
    out = {}
    for patient, patient_rows in sorted(groups.items(), key=lambda item: int(item[0])):
        out[patient] = {
            "images": len(patient_rows),
            "dice": float(np.mean([float(row["dice"]) for row in patient_rows])),
            "aji": float(np.mean([float(row["aji"]) for row in patient_rows])),
            "aji_plus": float(np.mean([float(row["aji_plus"]) for row in patient_rows])),
            "dq": float(np.mean([float(row["dq"]) for row in patient_rows])),
            "sq": float(np.mean([float(row["sq"]) for row in patient_rows])),
            "pq": float(np.mean([float(row["pq"]) for row in patient_rows])),
            "tp": int(sum(int(row["tp"]) for row in patient_rows)),
            "fp": int(sum(int(row["fp"]) for row in patient_rows)),
            "fn": int(sum(int(row["fn"]) for row in patient_rows)),
        }
    return out


def _patch_non_decrease(control_rows: list[dict], set_rows: list[dict], control_epoch: int, set_epoch: int) -> dict:
    control = {
        row["image"]: row
        for row in control_rows
        if int(row["epoch"]) == control_epoch
    }
    experiment = {
        row["image"]: row
        for row in set_rows
        if int(row["epoch"]) == set_epoch
    }
    if set(control) != set(experiment) or len(control) != 12:
        raise RuntimeError("MoNuSeg-Lite patch comparison must contain the same 12 frozen patches")
    details = []
    for patch_id in sorted(control):
        c_row = control[patch_id]
        s_row = experiment[patch_id]
        delta_aji = float(s_row["aji"]) - float(c_row["aji"])
        delta_pq = float(s_row["pq"]) - float(c_row["pq"])
        details.append(
            {
                "patch_id": patch_id,
                "delta_aji": delta_aji,
                "delta_pq": delta_pq,
                "nondecrease_both": delta_aji >= 0.0 and delta_pq >= 0.0,
            }
        )
    return {
        "details": details,
        "both_non_decrease_count": sum(item["nondecrease_both"] for item in details),
        "aji_non_decrease_count": sum(item["delta_aji"] >= 0.0 for item in details),
        "pq_non_decrease_count": sum(item["delta_pq"] >= 0.0 for item in details),
    }


def _decision(tnbc_delta: dict, monu_delta: dict, patch_info: dict) -> dict:
    strong_tnbc = (
        (tnbc_delta["aji"] >= 0.020 and tnbc_delta["pq"] >= 0.0)
        or (tnbc_delta["pq"] >= 0.010 and tnbc_delta["aji"] >= 0.0)
        or (tnbc_delta["aji"] >= 0.010 and tnbc_delta["pq"] >= 0.010)
    )
    strong_monu = (
        (monu_delta["aji"] >= 0.010 or monu_delta["pq"] >= 0.010)
        and monu_delta["aji"] >= 0.0
        and monu_delta["pq"] >= 0.0
        and patch_info["both_non_decrease_count"] >= 8
    )
    promising_tnbc = (
        (tnbc_delta["aji"] >= 0.005 and tnbc_delta["pq"] >= 0.0)
        or (tnbc_delta["pq"] >= 0.005 and tnbc_delta["aji"] >= 0.0)
    )
    promising_monu = (
        (monu_delta["aji"] >= 0.005 or monu_delta["pq"] >= 0.005)
        and monu_delta["aji"] >= 0.0
        and monu_delta["pq"] >= 0.0
        and patch_info["both_non_decrease_count"] >= 7
    )
    directionally_consistent = (
        tnbc_delta["aji"] > 0.0
        and tnbc_delta["pq"] >= 0.0
        and monu_delta["aji"] > 0.0
        and monu_delta["pq"] >= 0.0
    ) or (
        tnbc_delta["pq"] > 0.0
        and tnbc_delta["aji"] >= 0.0
        and monu_delta["pq"] > 0.0
        and monu_delta["aji"] >= 0.0
    )
    if strong_tnbc or strong_monu:
        category = "STRONG_GO"
        recommendation = "SetPMS meets a pre-authorised strong-go condition."
    elif promising_tnbc or promising_monu or directionally_consistent:
        category = "PROMISING_FULL_MONUSEG_RECOMMENDED"
        recommendation = "Do not call NO-GO automatically; the owner decides whether to authorise full MoNuSeg."
    elif (
        tnbc_delta["aji"] <= 0.0
        and tnbc_delta["pq"] <= 0.0
        and monu_delta["aji"] <= 0.0
        and monu_delta["pq"] <= 0.0
    ):
        category = "NO_GO_RECOMMENDED"
        recommendation = "Both authorised datasets show no actual aggregate SetPMS improvement over Control."
    else:
        category = "INCONCLUSIVE_OWNER_REVIEW"
        recommendation = "The fixed gate is mixed; preserve artifacts and request owner judgement."
    return {
        "category": category,
        "recommendation": recommendation,
        "strong_tnbc": strong_tnbc,
        "strong_monuseg_lite": strong_monu,
        "promising_tnbc": promising_tnbc,
        "promising_monuseg_lite": promising_monu,
        "directionally_consistent": directionally_consistent,
    }


def _collect_checkpoint_checksums(artifact_root: Path) -> list[dict]:
    rows = []
    for checkpoint in sorted(artifact_root.glob("*/Model/*.pth")):
        rows.append(
            {
                "path": str(checkpoint.relative_to(artifact_root)),
                "bytes": checkpoint.stat().st_size,
                "sha256": _sha256_file(checkpoint),
            }
        )
    if not rows:
        raise RuntimeError("No continuation checkpoints were produced")
    return rows


def _write_environment(root: Path, output_path: Path) -> None:
    lines = [
        f"python={sys.version}",
        f"platform={platform.platform()}",
        f"cwd={root}",
        f"utc={datetime.now(timezone.utc).isoformat()}",
    ]
    for command in ([sys.executable, "-m", "pip", "freeze"], ["nvidia-smi"]):
        try:
            output = subprocess.check_output(command, cwd=root, text=True, stderr=subprocess.STDOUT)
        except (OSError, subprocess.CalledProcessError) as error:
            output = f"unavailable: {error}"
        lines.append("\n$ " + " ".join(command) + "\n" + output)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _write_sha256s(artifact_root: Path) -> None:
    rows = []
    for path in sorted(path for path in artifact_root.rglob("*") if path.is_file()):
        if path.name == "SHA256SUMS":
            continue
        rows.append(f"{_sha256_file(path)}  {path.relative_to(artifact_root).as_posix()}")
    (artifact_root / "SHA256SUMS").write_text("\n".join(rows) + "\n", encoding="utf-8")


def _assert_step0_equivalence(control: list[dict], experiment: list[dict], dataset: str) -> None:
    control_zero = _by_epoch(control, 0)
    experiment_zero = _by_epoch(experiment, 0)
    differences = _metric_delta(experiment_zero, control_zero)
    if any(abs(value) > 1.0e-10 for value in differences.values()):
        raise RuntimeError(
            f"{dataset} Control and SetPMS step-0 are not equivalent: {differences}"
        )


def _assert_required_epochs(rows: list[dict], expected: set[int], label: str) -> None:
    found = {int(row["epoch"]) for row in rows}
    if found != expected:
        raise RuntimeError(f"{label} evaluated epochs {sorted(found)}, expected {sorted(expected)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tnbc-data", default="data/tnbc")
    parser.add_argument("--monuseg-data", default="data/monuseg")
    parser.add_argument("--tnbc-checkpoint", default="deliver_ckpts/tnbc_pms_best_e156.pth")
    parser.add_argument("--monuseg-checkpoint", default="deliver_ckpts/monuseg_pms_best_pq.pth")
    parser.add_argument("--artifact-root", default="")
    parser.add_argument("--skip-unit-tests", action="store_true")
    return parser.parse_args()


def main() -> None:
    options = parse_args()
    root = Path.cwd().resolve()
    if _git(root, "branch", "--show-current") != "research/setpms":
        raise RuntimeError("SetPMS runner must execute from the isolated research/setpms branch")
    if _git(root, "merge-base", CANONICAL_SHA, "HEAD") != CANONICAL_SHA:
        raise RuntimeError("Current branch does not descend from the fixed canonical baseline SHA")
    dirty = _git(root, "status", "--short")
    if dirty:
        raise RuntimeError(f"Refusing formal run with a dirty SetPMS worktree:\n{dirty}")

    head = _git(root, "rev-parse", "HEAD")
    default_artifact = root / "logs" / "setpms" / "stage1_dual_dev" / (
        datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + head[:12]
    )
    artifact_root = Path(options.artifact_root).resolve() if options.artifact_root else default_artifact
    if artifact_root.exists():
        raise FileExistsError(f"Artifact directory already exists: {artifact_root}")
    artifact_root.mkdir(parents=True)

    tnbc_data = (root / options.tnbc_data).resolve()
    monuseg_data = (root / options.monuseg_data).resolve()
    tnbc_checkpoint = (root / options.tnbc_checkpoint).resolve()
    monuseg_checkpoint = (root / options.monuseg_checkpoint).resolve()
    for path, expected, label in (
        (tnbc_checkpoint, TNBC_CHECKPOINT_SHA256, "TNBC"),
        (monuseg_checkpoint, MONUSEG_CHECKPOINT_SHA256, "MoNuSeg"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Fixed {label} initialization checkpoint is absent: {path}")
        actual = _sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"Fixed {label} checkpoint SHA256 mismatch: {actual}")

    tnbc_manifest_path = artifact_root / "tnbc_data_manifest.json"
    monu_manifest_path = artifact_root / "monuseg_lite_manifest.json"
    tnbc_manifest = _make_tnbc_manifest(tnbc_data, tnbc_manifest_path)
    monu_manifest = _make_monuseg_lite_manifest(monuseg_data, monu_manifest_path)
    patch_manifest_path = artifact_root / "monuseg_lite_patches.json"
    _json_dump(patch_manifest_path, {"patches": monu_manifest["patches"]})

    git_manifest = {
        "canonical_baseline_sha": CANONICAL_SHA,
        "head_sha": head,
        "branch": "research/setpms",
        "worktree": str(root),
        "status_short": dirty,
    }
    _json_dump(artifact_root / "git_manifest.json", git_manifest)
    _json_dump(
        artifact_root / "checkpoint_manifest.json",
        {
            "tnbc": {"path": str(tnbc_checkpoint), "sha256": TNBC_CHECKPOINT_SHA256},
            "monuseg": {"path": str(monuseg_checkpoint), "sha256": MONUSEG_CHECKPOINT_SHA256},
            "official_sam2": "/root/autodl-tmp/projects/CA-SAM2-HRC/checkpoints/sam2_hiera_large.pt",
        },
    )
    _json_dump(
        artifact_root / "training_config.json",
        {
            "seed": SEED,
            "tta": False,
            "batch_size": 1,
            "nms": 12,
            "inclusive_iou_threshold": 0.5,
            "texture": True,
            "context": True,
            "epochs": 10,
            "optimizer": "AdamW",
            "lr": 1e-5,
            "lr_min": 1e-6,
            "weight_decay": 1e-4,
            "scheduler": "cosine",
            "tnbc": {"overlap": 32, "save_epochs": [0, 2, 4, 6, 8, 10], "eval_epochs": [0, 2, 4, 6, 8, 10]},
            "monuseg_lite": {"overlap": 92, "save_epochs": [0, 2, 4, 5, 6, 8, 10], "eval_epochs": [0, 5, 10], "max_train_crops_per_image": 4},
            "prohibited": [
                "TNBC patients 9-11",
                "MoNuSeg official test",
                "TTA",
                "coverage refresh",
                "dynamic PMS mining",
                "pseudo prompts",
                "second seed",
                "inference modules",
            ],
        },
    )
    _json_dump(
        artifact_root / "setpms_formula.json",
        {
            "K": "min(64, max(16, 2N)); N=0 keeps top-16",
            "cost": "(1-soft_iou) + 0.25*clamp(point_distance/crop_diagonal,0,1)",
            "uot": {"epsilon": 0.10, "tau": 1.0, "iterations": 20, "transport_detached": True},
            "gate": "sigmoid((IoU-0.5)/0.05)",
            "loss": "0.5*(1-soft_PQ)+0.5*(1-soft_AJI)+0.1*transport_cost+0.1*duplicate",
            "lambda_set": "0.0 at epoch 0; 0.05 at epoch 1; 0.1 from epoch 2",
            "anchor": "1e-3 * mean((theta-theta0)^2)/(mean(theta0^2)+eps)",
        },
    )
    _write_environment(root, artifact_root / "environment.txt")

    unit_output = artifact_root / "unit_tests.txt"
    if options.skip_unit_tests:
        unit_output.write_text("SKIPPED BY EXPLICIT --skip-unit-tests\n", encoding="utf-8")
    else:
        _run(
            root,
            [
                sys.executable,
                "-m",
                "unittest",
                "discover",
                "-s",
                "tests",
                "-p",
                "test_setpms_loss.py",
                "-v",
            ],
            unit_output,
        )

    runtime = {}
    smoke_dir = artifact_root / "smoke"
    smoke_metrics = smoke_dir / "metrics"
    smoke_command = _base_command(
        sys.executable,
        tnbc_data,
        tnbc_checkpoint,
        tnbc_manifest_path,
        tnbc_manifest_path,
        smoke_dir,
        smoke_metrics,
        "tnbc_setpms_smoke",
        32,
        "",
        "",
        setpms=True,
    )
    smoke_command.extend(["--setpms_smoke_batches", "2"])
    runtime["smoke"] = _run(root, smoke_command, artifact_root / "commands" / "smoke.log")
    shutil.copy2(smoke_metrics / "smoke_report.json", artifact_root / "smoke_report.json")
    shutil.copy2(smoke_metrics / "baseline_equivalence.json", artifact_root / "baseline_equivalence.json")

    def child(name: str) -> tuple[Path, Path]:
        child_root = artifact_root / name
        metrics = child_root / "metrics"
        return child_root, metrics

    # TNBC control must prove canonical step-0 before any continuation update.
    tnbc_control_dir, tnbc_control_metrics = child("tnbc_control")
    tnbc_control_label = "tnbc_control"
    tnbc_control_command = _base_command(
        sys.executable, tnbc_data, tnbc_checkpoint, tnbc_manifest_path, tnbc_manifest_path,
        tnbc_control_dir, tnbc_control_metrics, tnbc_control_label, 32, "0,2,4,6,8,10", "0,2,4,6,8,10", setpms=False,
    )
    tnbc_control_command.extend([
        "--baseline_reference_aji", str(TNBC_REFERENCE_AJI),
        "--baseline_reference_pq", str(TNBC_REFERENCE_PQ),
        "--baseline_reference_tolerance", str(TNBC_REFERENCE_TOLERANCE),
    ])
    runtime["tnbc_control"] = _run(root, tnbc_control_command, artifact_root / "commands" / "tnbc_control.log")
    tnbc_control_rows = _metric_rows_for_label(_child_rows(tnbc_control_dir, "metrics.csv"), tnbc_control_label)
    _assert_required_epochs(tnbc_control_rows, {0, 2, 4, 6, 8, 10}, tnbc_control_label)
    tnbc_control_zero = _by_epoch(tnbc_control_rows, 0)

    tnbc_set_dir, tnbc_set_metrics = child("tnbc_setpms")
    tnbc_set_label = "tnbc_setpms"
    tnbc_set_command = _base_command(
        sys.executable, tnbc_data, tnbc_checkpoint, tnbc_manifest_path, tnbc_manifest_path,
        tnbc_set_dir, tnbc_set_metrics, tnbc_set_label, 32, "0,2,4,6,8,10", "0,2,4,6,8,10", setpms=True,
    )
    tnbc_set_command.extend([
        "--baseline_reference_aji", str(tnbc_control_zero["aji"]),
        "--baseline_reference_pq", str(tnbc_control_zero["pq"]),
        "--baseline_reference_tolerance", "1e-10",
    ])
    runtime["tnbc_setpms"] = _run(root, tnbc_set_command, artifact_root / "commands" / "tnbc_setpms.log")
    tnbc_set_rows = _metric_rows_for_label(_child_rows(tnbc_set_dir, "metrics.csv"), tnbc_set_label)
    _assert_required_epochs(tnbc_set_rows, {0, 2, 4, 6, 8, 10}, tnbc_set_label)
    _assert_step0_equivalence(tnbc_control_rows, tnbc_set_rows, "TNBC")

    monu_control_dir, monu_control_metrics = child("monuseg_lite_control")
    monu_control_label = "monuseg_lite_control"
    monu_control_command = _base_command(
        sys.executable, monuseg_data, monuseg_checkpoint, monu_manifest_path, monu_manifest_path,
        monu_control_dir, monu_control_metrics, monu_control_label, 92, "0,2,4,5,6,8,10", "0,5,10", setpms=False,
        crop_manifest=monu_manifest_path, patch_manifest=patch_manifest_path,
    )
    runtime["monuseg_lite_control"] = _run(root, monu_control_command, artifact_root / "commands" / "monuseg_lite_control.log")
    monu_control_rows = _metric_rows_for_label(_child_rows(monu_control_dir, "metrics.csv"), monu_control_label)
    _assert_required_epochs(monu_control_rows, {0, 5, 10}, monu_control_label)
    monu_control_zero = _by_epoch(monu_control_rows, 0)

    monu_set_dir, monu_set_metrics = child("monuseg_lite_setpms")
    monu_set_label = "monuseg_lite_setpms"
    monu_set_command = _base_command(
        sys.executable, monuseg_data, monuseg_checkpoint, monu_manifest_path, monu_manifest_path,
        monu_set_dir, monu_set_metrics, monu_set_label, 92, "0,2,4,5,6,8,10", "0,5,10", setpms=True,
        crop_manifest=monu_manifest_path, patch_manifest=patch_manifest_path,
    )
    monu_set_command.extend([
        "--baseline_reference_aji", str(monu_control_zero["aji"]),
        "--baseline_reference_pq", str(monu_control_zero["pq"]),
        "--baseline_reference_tolerance", "1e-10",
    ])
    runtime["monuseg_lite_setpms"] = _run(root, monu_set_command, artifact_root / "commands" / "monuseg_lite_setpms.log")
    monu_set_rows = _metric_rows_for_label(_child_rows(monu_set_dir, "metrics.csv"), monu_set_label)
    _assert_required_epochs(monu_set_rows, {0, 5, 10}, monu_set_label)
    _assert_step0_equivalence(monu_control_rows, monu_set_rows, "MoNuSeg-Lite")

    # Required aggregate artifacts.
    tnbc_metric_rows = _child_rows(tnbc_control_dir, "metrics.csv") + _child_rows(tnbc_set_dir, "metrics.csv")
    tnbc_image_rows = _enrich_tnbc_patients(
        _child_rows(tnbc_control_dir, "per_image.csv") + _child_rows(tnbc_set_dir, "per_image.csv")
    )
    monu_metric_rows = _child_rows(monu_control_dir, "metrics.csv") + _child_rows(monu_set_dir, "metrics.csv")
    monu_patch_rows = _child_rows(monu_control_dir, "per_image.csv") + _child_rows(monu_set_dir, "per_image.csv")
    curve_rows = (
        _child_rows(tnbc_control_dir, "training_curves.csv")
        + _child_rows(tnbc_set_dir, "training_curves.csv")
        + _child_rows(monu_control_dir, "training_curves.csv")
        + _child_rows(monu_set_dir, "training_curves.csv")
    )
    _write_rows(artifact_root / "tnbc_metrics.csv", tnbc_metric_rows)
    _write_rows(artifact_root / "tnbc_per_image.csv", tnbc_image_rows)
    _write_rows(artifact_root / "monuseg_lite_metrics.csv", monu_metric_rows)
    _write_rows(artifact_root / "monuseg_lite_per_patch.csv", monu_patch_rows)
    _write_rows(artifact_root / "training_curves.csv", curve_rows)

    tnbc_control_best = _best_metric_row(tnbc_control_rows)
    tnbc_set_best = _best_metric_row(tnbc_set_rows)
    monu_control_best = _best_metric_row(monu_control_rows)
    monu_set_best = _best_metric_row(monu_set_rows)
    tnbc_delta_control = _metric_delta(tnbc_set_best, tnbc_control_best)
    monu_delta_control = _metric_delta(monu_set_best, monu_control_best)
    tnbc_delta_step0 = _metric_delta(tnbc_set_best, _by_epoch(tnbc_set_rows, 0))
    monu_delta_step0 = _metric_delta(monu_set_best, _by_epoch(monu_set_rows, 0))
    patch_info = _patch_non_decrease(
        _child_rows(monu_control_dir, "per_image.csv"),
        _child_rows(monu_set_dir, "per_image.csv"),
        int(monu_control_best["epoch"]),
        int(monu_set_best["epoch"]),
    )
    decision = _decision(tnbc_delta_control, monu_delta_control, patch_info)

    report = {
        "questions": {
            "setpms_improves_tnbc": {
                "answer": tnbc_delta_control["aji"] > 0.0 or tnbc_delta_control["pq"] > 0.0,
                "setpms_best_vs_control_best": tnbc_delta_control,
            },
            "monuseg_lite_response_differs_from_tnbc": {
                "answer": (
                    math.copysign(1.0, monu_delta_control["aji"] or 1.0)
                    != math.copysign(1.0, tnbc_delta_control["aji"] or 1.0)
                    or math.copysign(1.0, monu_delta_control["pq"] or 1.0)
                    != math.copysign(1.0, tnbc_delta_control["pq"] or 1.0)
                ),
                "monuseg_lite_best_vs_control_best": monu_delta_control,
            },
            "dq_sq_aji_contribution": {
                "tnbc": {key: tnbc_delta_control[key] for key in ("dq", "sq", "aji", "pq")},
                "monuseg_lite": {key: monu_delta_control[key] for key in ("dq", "sq", "aji", "pq")},
            },
            "full_monuseg_recommendation": decision,
            "holds_vs_control_and_step0": {
                "tnbc_vs_control": tnbc_delta_control,
                "tnbc_vs_step0": tnbc_delta_step0,
                "monuseg_lite_vs_control": monu_delta_control,
                "monuseg_lite_vs_step0": monu_delta_step0,
            },
        },
        "tnbc": {
            "step0": _by_epoch(tnbc_set_rows, 0),
            "control_final": _by_epoch(tnbc_control_rows, 10),
            "setpms_final": _by_epoch(tnbc_set_rows, 10),
            "control_best": tnbc_control_best,
            "setpms_best": tnbc_set_best,
            "per_patient_control_best": _per_patient(
                tnbc_image_rows, tnbc_control_label, int(tnbc_control_best["epoch"])
            ),
            "per_patient_setpms_best": _per_patient(
                tnbc_image_rows, tnbc_set_label, int(tnbc_set_best["epoch"])
            ),
        },
        "monuseg_lite": {
            "screening_disclaimer": monu_manifest["selection_note"],
            "step0": _by_epoch(monu_set_rows, 0),
            "control_final": _by_epoch(monu_control_rows, 10),
            "setpms_final": _by_epoch(monu_set_rows, 10),
            "control_best": monu_control_best,
            "setpms_best": monu_set_best,
            "patch_non_decrease": patch_info,
        },
        "decision": decision,
    }
    _json_dump(artifact_root / "report.json", report)

    checkpoint_checksums = _collect_checkpoint_checksums(artifact_root)
    _json_dump(artifact_root / "checkpoint_checksums.json", checkpoint_checksums)
    checkpoint_manifest = json.loads((artifact_root / "checkpoint_manifest.json").read_text(encoding="utf-8"))
    checkpoint_manifest["produced_checkpoints"] = checkpoint_checksums
    _json_dump(artifact_root / "checkpoint_manifest.json", checkpoint_manifest)
    _json_dump(
        artifact_root / "runtime_memory.json",
        {
            "command_seconds": runtime,
            "smoke": json.loads((artifact_root / "smoke_report.json").read_text(encoding="utf-8")),
            "continuations": {
                name: json.loads((artifact_root / name / "metrics" / "runtime_memory.json").read_text(encoding="utf-8"))
                for name in (
                    "tnbc_control",
                    "tnbc_setpms",
                    "monuseg_lite_control",
                    "monuseg_lite_setpms",
                )
            },
        },
    )
    _write_sha256s(artifact_root)
    print(artifact_root)


if __name__ == "__main__":
    main()
