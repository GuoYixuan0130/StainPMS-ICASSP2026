"""Render development-only C2-AR assembly/FP examples from oracle artifacts."""

from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stainpms.zero_training_oracle import decode_binary_rle


SEEDS = (2027, 1337)
ARMS = ("c0", "c1", "c2_ar")


def parse_assignment(value: str) -> tuple[int, str, Path]:
    try:
        left, raw_path = value.split("=", 1)
        raw_seed, arm = left.split(":", 1)
        seed = int(raw_seed)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--input must be SEED:ARM=/oracle/run/directory") from exc
    if seed not in SEEDS or arm not in ARMS:
        raise argparse.ArgumentTypeError("C2 cases require 2027/1337 and c0/c1/c2_ar")
    return seed, arm, Path(raw_path).resolve()


def read_artifacts(root: Path) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for path in sorted((root / "completed_images").glob("*.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        output[str(payload["artifact"]["sample_id"])] = payload
    if len(output) != 7:
        raise ValueError(f"expected seven p7/p8 artifacts in {root}, got {len(output)}")
    return output


def native_pq(payload: dict[str, Any]) -> float:
    return float(payload["image_record"]["stages"]["native_final"]["pq"])


def errors(payload: dict[str, Any]) -> dict[str, int]:
    record = payload["image_record"]["errors"]
    return {
        "assembly_loss": int(record["counts"]["assembly_loss"]),
        "fp": int(record["counts"]["native_final_false_positive_count"]),
        "merge": int(record["native_final_structural_errors"]["sensitivity"]["overlap_fraction_gt_or_pred_gt_0"]["merge"]),
        "split": int(record["native_final_structural_errors"]["sensitivity"]["overlap_fraction_gt_or_pred_gt_0"]["split"]),
    }


def choose_cases(artifacts: dict[tuple[int, str], dict[str, dict[str, Any]]]) -> dict[str, dict[str, Any] | None]:
    candidates: dict[str, list[tuple[float, dict[str, Any]]]] = {
        "c2_assembly_loss_reduced": [],
        "c2_final_fp_reduced": [],
        "c2_native_pq_improved": [],
        "c2_remaining_merge_or_split": [],
    }
    for seed in SEEDS:
        for sample_id, c2 in artifacts[(seed, "c2_ar")].items():
            c0 = artifacts[(seed, "c0")][sample_id]
            c1 = artifacts[(seed, "c1")][sample_id]
            c2_error, c1_error = errors(c2), errors(c1)
            if c1_error["assembly_loss"] > c2_error["assembly_loss"]:
                candidates["c2_assembly_loss_reduced"].append((
                    c1_error["assembly_loss"] - c2_error["assembly_loss"],
                    {"seed": seed, "sample_id": sample_id, "arm": "c2_ar", "reason": "C2-AR has fewer selected-pool TP losses in native assembly than C1-full"},
                ))
            if c1_error["fp"] > c2_error["fp"]:
                candidates["c2_final_fp_reduced"].append((
                    c1_error["fp"] - c2_error["fp"],
                    {"seed": seed, "sample_id": sample_id, "arm": "c2_ar", "reason": "C2-AR has fewer native-final false positives than C1-full"},
                ))
            if native_pq(c2) > native_pq(c0):
                candidates["c2_native_pq_improved"].append((
                    native_pq(c2) - native_pq(c0),
                    {"seed": seed, "sample_id": sample_id, "arm": "c2_ar", "reason": "C2-AR native-final PQ improves over paired C0"},
                ))
            structural = c2_error["merge"] + c2_error["split"]
            if structural:
                candidates["c2_remaining_merge_or_split"].append((
                    structural,
                    {"seed": seed, "sample_id": sample_id, "arm": "c2_ar", "reason": "remaining frozen-definition merge or split in C2-AR native final"},
                ))
    return {name: max(rows, default=(None, None), key=lambda item: item[0])[1] for name, rows in candidates.items()}


def render_case(name: str, case: dict[str, Any], payload: dict[str, Any], output: Path) -> None:
    artifact = payload["artifact"]
    image = np.asarray(Image.open(Path(payload["_root"]) / artifact["source_image_png"]).convert("RGB"))
    fig, ax = plt.subplots(figsize=(7, 7), dpi=160)
    ax.imshow(image)
    for record in artifact["gt_instances"]:
        ax.contour(decode_binary_rle(record["mask_rle"]), levels=[0.5], colors="#20c978", linewidths=0.35)
    for record in artifact["native_final_instances"]:
        ax.contour(decode_binary_rle(record["mask_rle"]), levels=[0.5], colors="#ff5a5f", linewidths=0.5)
    ax.set_title(f"{name}: seed={case['seed']} sample={case['sample_id']}\n{case['reason']}", fontsize=8)
    ax.set_axis_off()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.15)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, type=parse_assignment)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    assignments = {(seed, arm): path for seed, arm, path in args.input}
    expected = {(seed, arm) for seed in SEEDS for arm in ARMS}
    if set(assignments) != expected:
        raise ValueError("all six two-seed C0/C1/C2-AR oracle directories are required")
    artifacts = {key: read_artifacts(path) for key, path in assignments.items()}
    for key, records in artifacts.items():
        for record in records.values():
            record["_root"] = str(assignments[key])
    cases = choose_cases(artifacts)
    output_dir = Path(args.output_dir).resolve()
    index: dict[str, Any] = {"schema_version": 1, "scope": "TNBC p7/p8 only", "cases": {}}
    for name, case in cases.items():
        if case is None:
            index["cases"][name] = {"status": "unavailable_under_frozen_results"}
            continue
        payload = artifacts[(int(case["seed"]), "c2_ar")][str(case["sample_id"])]
        png = output_dir / "c2_ar_cases" / f"{name}.png"
        render_case(name, case, payload, png)
        index["cases"][name] = {**case, "png": str(png)}
    write_json_atomic(output_dir / "c2_ar_case_index.json", index)
    print(json.dumps({"status": "complete", "case_index": str(output_dir / "c2_ar_case_index.json")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
