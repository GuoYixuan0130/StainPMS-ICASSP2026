"""Render fixed, development-only examples from completed oracle artifacts."""

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
DIAGNOSIS_SEEDS = (2027, 1337)
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stainpms.zero_training_oracle import decode_binary_rle


def parse_assignment(value: str) -> tuple[int, str, Path]:
    try:
        left, raw_path = value.split("=", 1)
        seed_raw, arm = left.split(":", 1)
        seed = int(seed_raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--input must be SEED:ARM=/completed/run/directory") from exc
    if seed not in DIAGNOSIS_SEEDS or arm not in {"c0", "c1"}:
        raise argparse.ArgumentTypeError("invalid fixed seed/arm")
    return seed, arm, Path(raw_path).resolve()


def read_artifacts(root: Path) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for path in sorted((root / "completed_images").glob("*.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        artifact = payload["artifact"]
        output[artifact["sample_id"]] = payload
    if len(output) != 7:
        raise ValueError(f"expected 7 p7/p8 artifacts under {root}, got {len(output)}")
    return output


def best_for_gt(records: list[dict[str, Any]], gt_id: int) -> dict[str, Any] | None:
    eligible = [record for record in records if float(record.get("gt_ious", {}).get(str(gt_id), 0.0)) > 0.5]
    return max(eligible, key=lambda row: (float(row["gt_ious"][str(gt_id)]), -int(row["record_index"]))) if eligible else None


def native_pq(payload: dict[str, Any]) -> float:
    return float(payload["image_record"]["stages"]["native_final"]["pq"])


def find_cases(arms: dict[tuple[int, str], dict[str, dict[str, Any]]]) -> dict[str, dict[str, Any] | None]:
    candidate_improvement: list[tuple[float, dict[str, Any]]] = []
    final_improvement: list[tuple[float, dict[str, Any]]] = []
    for seed in DIAGNOSIS_SEEDS:
        for sample_id, c0 in arms[(seed, "c0")].items():
            c1 = arms[(seed, "c1")][sample_id]
            c0_record = c0["image_record"]
            c1_record = c1["image_record"]
            c0_per_gt = {int(row["gt_instance_id"]): row for row in c0_record["per_gt"]}
            c1_per_gt = {int(row["gt_instance_id"]): row for row in c1_record["per_gt"]}
            if native_pq(c1) <= native_pq(c0):
                for gt_id, c1_gt in c1_per_gt.items():
                    gain = float(c1_gt["all_candidate_best_iou"]) - float(c0_per_gt[gt_id]["all_candidate_best_iou"])
                    if gain > 0.0:
                        candidate_improvement.append((gain, {"seed": seed, "sample_id": sample_id, "arm": "c1", "gt_instance_id": gt_id, "reason": "C1 candidate IoU improves while native image PQ does not", "highlight": best_for_gt(c1["artifact"]["all_native_candidates"], gt_id)}))
            pq_gain = native_pq(c1) - native_pq(c0)
            if pq_gain > 0.0:
                final_improvement.append((pq_gain, {"seed": seed, "sample_id": sample_id, "arm": "c1", "reason": "C1 native final PQ improves over paired C0", "highlight": None}))

    assembly: list[dict[str, Any]] = []
    structural: list[tuple[int, dict[str, Any]]] = []
    for (seed, arm), images in arms.items():
        for sample_id, payload in images.items():
            record = payload["image_record"]
            for gt in record["per_gt"]:
                if gt["error_class"] == "assembly_loss":
                    selected = best_for_gt(payload["artifact"]["native_selected_before_assembly"], int(gt["gt_instance_id"]))
                    assembly.append({"seed": seed, "sample_id": sample_id, "arm": arm, "gt_instance_id": int(gt["gt_instance_id"]), "reason": "selected pool has IoU > 0.5 candidate but native final has no strict match", "highlight": selected})
            structural_error = record["errors"]["native_final_structural_errors"]
            score = int(structural_error["duplicate_unmatched_prediction_count"])
            score += int(structural_error["sensitivity"]["overlap_fraction_gt_or_pred_gt_0"]["split"])
            score += int(structural_error["sensitivity"]["overlap_fraction_gt_or_pred_gt_0"]["merge"])
            if score:
                structural.append((score, {"seed": seed, "sample_id": sample_id, "arm": arm, "reason": "native final duplicate, split, or merge under frozen overlap definitions", "highlight": None}))
    return {
        "c1_candidate_improved_final_not": max(candidate_improvement, default=(None, None), key=lambda item: item[0])[1],
        "selected_tp_lost_by_assembly": assembly[0] if assembly else None,
        "duplicate_merge_or_split": max(structural, default=(None, None), key=lambda item: item[0])[1],
        "c1_true_final_improvement": max(final_improvement, default=(None, None), key=lambda item: item[0])[1],
    }


def render_case(case_name: str, case: dict[str, Any], payload: dict[str, Any], output: Path) -> None:
    artifact = payload["artifact"]
    image = np.asarray(Image.open(Path(payload["_root"]) / artifact["source_image_png"]).convert("RGB"))
    fig, ax = plt.subplots(figsize=(7, 7), dpi=160)
    ax.imshow(image)
    for record in artifact["gt_instances"]:
        mask = decode_binary_rle(record["mask_rle"])
        ax.contour(mask, levels=[0.5], colors="#20c978", linewidths=0.35)
    for record in artifact["native_final_instances"]:
        mask = decode_binary_rle(record["mask_rle"])
        ax.contour(mask, levels=[0.5], colors="#ff5a5f", linewidths=0.45)
    if case.get("highlight") is not None:
        mask = decode_binary_rle(case["highlight"]["mask_rle"])
        ax.contour(mask, levels=[0.5], colors="#28a9ff", linewidths=1.2)
    ax.set_title(f"{case_name}: seed={case['seed']} arm={case['arm']} sample={case['sample_id']}\n{case['reason']}", fontsize=8)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, type=parse_assignment)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    assignments = {(seed, arm): path for seed, arm, path in args.input}
    expected = {(seed, arm) for seed in DIAGNOSIS_SEEDS for arm in ("c0", "c1")}
    if set(assignments) != expected:
        raise ValueError("all four paired seed/arm oracle directories are required")
    arms = {key: read_artifacts(path) for key, path in assignments.items()}
    for key, records in arms.items():
        for payload in records.values():
            payload["_root"] = str(assignments[key])
    cases = find_cases(arms)
    output_dir = Path(args.output_dir).resolve()
    index: dict[str, Any] = {"schema_version": 1, "scope": "TNBC p7/p8 only", "cases": {}}
    for name, case in cases.items():
        if case is None:
            index["cases"][name] = {"status": "unavailable_under_frozen_results"}
            continue
        payload = arms[(int(case["seed"]), str(case["arm"]))][str(case["sample_id"])]
        png = output_dir / "cases" / f"{name}.png"
        render_case(name, case, payload, png)
        serializable = {key: value for key, value in case.items() if key != "highlight"}
        if case.get("highlight") is not None:
            serializable["highlight_record_index"] = int(case["highlight"]["record_index"])
        serializable["png"] = str(png)
        index["cases"][name] = serializable
    write_json_atomic(output_dir / "case_index.json", index)
    print(json.dumps({"status": "complete", "case_index": str(output_dir / "case_index.json")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
