#!/usr/bin/env python3
"""Render fixed C3 GT-only score-intervention examples from compact artifacts."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stainpms.c2_component_audit import deserialize_gt, deserialize_selected
from stainpms.c3_score_control_audit import audit_image
from stainpms.zero_training_oracle import decode_binary_rle


SEEDS = (2027, 1337)
OPERATIONS = (
    "fp_demotion_oracle",
    "duplicate_order_oracle",
    "conflict_order_oracle",
    "merge_risk_demotion_oracle",
)


def parse_assignment(value: str) -> tuple[int, Path]:
    raw_seed, raw_path = value.split("=", 1)
    seed = int(raw_seed)
    if seed not in SEEDS:
        raise argparse.ArgumentTypeError("only seeds 2027 and 1337 are allowed")
    return seed, Path(raw_path).resolve()


def read_artifacts(root: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in sorted((root / "completed_images").glob("*.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        payload["_root"] = str(root)
        result[str(payload["artifact"]["sample_id"])] = payload
    if len(result) != 7:
        raise ValueError(f"expected exactly seven p7/p8 compact artifacts: {root}")
    return result


def choose_cases(audit: dict[str, Any]) -> dict[str, dict[str, Any] | None]:
    candidates = {operation: [] for operation in OPERATIONS}
    for seed_record in audit["per_seed"]:
        seed = int(seed_record["seed"])
        for row in seed_record["per_image"]:
            for operation in OPERATIONS:
                delta_pq = float(row["deltas_vs_native"][operation]["pq"])
                if delta_pq > 0.0:
                    candidates[operation].append((delta_pq, seed, str(row["sample_id"]), row))
    output: dict[str, dict[str, Any] | None] = {}
    for operation, values in candidates.items():
        if not values:
            output[operation] = None
            continue
        delta_pq, seed, sample_id, row = max(values, key=lambda value: (value[0], -value[1], value[2]))
        output[operation] = {
            "seed": seed,
            "sample_id": sample_id,
            "patient": int(row["patient"]),
            "delta_pq": delta_pq,
            "native": row["stages"]["native"],
            "intervention": row["stages"][operation],
            "target_counts": row["targets"],
            "retention_count_preserved": bool(row["retention_count_preserved"][operation]),
        }
    return output


def draw_boundaries(axis, image: np.ndarray, gt_map: np.ndarray, pred_map: np.ndarray, title: str) -> None:
    axis.imshow(image)
    axis.contour(gt_map > 0, levels=[0.5], colors="#20c978", linewidths=0.35)
    axis.contour(pred_map > 0, levels=[0.5], colors="#ff5a5f", linewidths=0.45)
    axis.set_title(title, fontsize=8)
    axis.set_axis_off()


def render_case(
    operation: str,
    case: dict[str, Any],
    payload: dict[str, Any],
    output: Path,
) -> None:
    artifact = payload["artifact"]
    image = np.asarray(Image.open(Path(payload["_root"]) / artifact["source_image_png"]).convert("RGB"))
    gt_map = deserialize_gt(artifact)
    result = audit_image(deserialize_selected(artifact), gt_map, nms_iou=0.5, return_maps=True)
    native = result["maps"]["native"]
    changed = result["maps"][operation]
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=160)
    draw_boundaries(
        axes[0],
        image,
        gt_map,
        native,
        f"Native C1\nPQ={result['stages']['native']['pq']:.4f}, TP/FP/FN="
        f"{result['stages']['native']['tp']}/{result['stages']['native']['fp']}/{result['stages']['native']['fn']}",
    )
    draw_boundaries(
        axes[1],
        image,
        gt_map,
        changed,
        f"GT-only {operation}\nPQ={result['stages'][operation]['pq']:.4f}, "
        f"delta={result['deltas_vs_native'][operation]['pq']:+.4f}",
    )
    fig.suptitle(
        f"C3 score-control upper bound — seed {case['seed']}, p{case['patient']}, {case['sample_id']}\n"
        "green: GT; red: native final (left) or GT-only score-intervention final (right)",
        fontsize=9,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.3)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, type=parse_assignment, help="SEED=C1_ORACLE_DIR")
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    roots = dict(args.input)
    if set(roots) != set(SEEDS):
        raise ValueError("requires both C1 source directories: 2027 and 1337")
    audit = json.loads(args.audit.resolve().read_text(encoding="utf-8"))
    if audit.get("status") != "complete":
        raise ValueError("requires a completed C3 audit")
    data = {seed: read_artifacts(root) for seed, root in roots.items()}
    cases = choose_cases(audit)
    output = args.output_dir.resolve()
    if output.exists():
        raise ValueError(f"refusing to overwrite case directory: {output}")
    index: dict[str, Any] = {
        "schema_version": 1,
        "scope": "TNBC p7/p8 only; C1 fixed selected masks; GT-only score upper bounds",
        "cases": {},
    }
    for operation, case in cases.items():
        if case is None:
            index["cases"][operation] = {"status": "no_positive_per_image_gain"}
            continue
        payload = data[int(case["seed"])][str(case["sample_id"])]
        png = output / "c3_score_control_cases" / f"{operation}.png"
        render_case(operation, case, payload, png)
        index["cases"][operation] = {"status": "rendered", **case, "png": str(png)}
    write_json(output / "c3_score_control_case_index.json", index)
    print(json.dumps({"status": "complete", "case_index": str(output / "c3_score_control_case_index.json")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
