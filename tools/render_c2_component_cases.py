"""Render a few fixed p7/p8 component-attribution examples."""

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
ARMS = ("c1", "c2_eu", "c2_e", "c2_u")


def parse(value: str) -> tuple[int, str, Path]:
    left, path = value.split("=", 1); seed_s, arm = left.split(":", 1); seed = int(seed_s)
    if seed not in SEEDS or arm not in ARMS:
        raise argparse.ArgumentTypeError("inputs require seed 2027/1337 and c1/c2_eu/c2_e/c2_u")
    return seed, arm, Path(path).resolve()


def read(root: Path) -> dict[str, Any]:
    out = {}
    for path in sorted((root / "completed_images").glob("*.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as handle: payload = json.load(handle)
        payload["_root"] = str(root); out[str(payload["artifact"]["sample_id"])] = payload
    if len(out) != 7: raise ValueError(f"expected 7 artifacts: {root}")
    return out


def error(payload: dict[str, Any]) -> dict[str, int]:
    data = payload["image_record"]["errors"]
    return {"assembly": int(data["counts"]["assembly_loss"]), "fp": int(data["counts"]["native_final_false_positive_count"]), "merge": int(data["native_final_structural_errors"]["sensitivity"]["overlap_fraction_gt_or_pred_gt_0"]["merge"]), "split": int(data["native_final_structural_errors"]["sensitivity"]["overlap_fraction_gt_or_pred_gt_0"]["split"])}


def choose(data: dict[tuple[int, str], dict[str, Any]]) -> dict[str, dict[str, Any] | None]:
    candidates = {"c2_e_assembly_improvement": [], "c2_u_fp_improvement": [], "component_native_pq_improvement": [], "remaining_structural_error": []}
    for seed in SEEDS:
        for sample_id, c1 in data[(seed, "c1")].items():
            c1e = error(c1)
            for arm in ("c2_e", "c2_u"):
                current = data[(seed, arm)][sample_id]; cur_error = error(current)
                if arm == "c2_e" and c1e["assembly"] > cur_error["assembly"]:
                    candidates["c2_e_assembly_improvement"].append((c1e["assembly"] - cur_error["assembly"], seed, arm, sample_id, "fewer selected-pool assembly losses than C1"))
                if arm == "c2_u" and c1e["fp"] > cur_error["fp"]:
                    candidates["c2_u_fp_improvement"].append((c1e["fp"] - cur_error["fp"], seed, arm, sample_id, "fewer native-final false positives than C1"))
                pq_delta = float(current["image_record"]["stages"]["native_final"]["pq"]) - float(c1["image_record"]["stages"]["native_final"]["pq"])
                if pq_delta > 0:
                    candidates["component_native_pq_improvement"].append((pq_delta, seed, arm, sample_id, "native-final PQ improves over C1"))
                structural = cur_error["merge"] + cur_error["split"]
                if structural:
                    candidates["remaining_structural_error"].append((structural, seed, arm, sample_id, "remaining frozen-definition merge or split"))
    return {key: (None if not values else {"seed": value[1], "arm": value[2], "sample_id": value[3], "reason": value[4]}) for key, values in candidates.items() for value in [max(values, key=lambda item: item[0]) if values else None]}


def render(name: str, case: dict[str, Any], payload: dict[str, Any], output: Path) -> None:
    artifact = payload["artifact"]
    image = np.asarray(Image.open(Path(payload["_root"]) / artifact["source_image_png"]).convert("RGB"))
    fig, axis = plt.subplots(figsize=(7, 7), dpi=160); axis.imshow(image)
    for row in artifact["gt_instances"]: axis.contour(decode_binary_rle(row["mask_rle"]), levels=[0.5], colors="#20c978", linewidths=.35)
    for row in artifact["native_final_instances"]: axis.contour(decode_binary_rle(row["mask_rle"]), levels=[0.5], colors="#ff5a5f", linewidths=.5)
    axis.set_title(f"{name}: seed={case['seed']} {case['arm']} {case['sample_id']}\n{case['reason']}", fontsize=8); axis.set_axis_off()
    output.parent.mkdir(parents=True, exist_ok=True); fig.tight_layout(pad=.15); fig.savefig(output, bbox_inches="tight"); plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--input", action="append", required=True, type=parse); parser.add_argument("--output-dir", required=True); args = parser.parse_args()
    assignments = {(seed, arm): root for seed, arm, root in args.input}; expected = {(seed, arm) for seed in SEEDS for arm in ARMS}
    if set(assignments) != expected: raise ValueError("requires 8 two-seed C1/C2-EU/C2-E/C2-U oracle directories")
    data = {key: read(root) for key, root in assignments.items()}; cases = choose(data); out = Path(args.output_dir).resolve(); index = {"schema_version": 1, "scope": "TNBC p7/p8 development only", "cases": {}}
    for name, case in cases.items():
        if case is None: index["cases"][name] = {"status": "unavailable_under_frozen_results"}; continue
        payload = data[(int(case["seed"]), str(case["arm"]))][str(case["sample_id"])]; png = out / "c2_component_cases" / f"{name}.png"; render(name, case, payload, png); index["cases"][name] = {**case, "png": str(png)}
    target = out / "c2_component_case_index.json"; target.parent.mkdir(parents=True, exist_ok=True); temp = target.with_name(f".{target.name}.{os.getpid()}.tmp"); temp.write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"); os.replace(temp, target); print(json.dumps({"status": "complete", "case_index": str(target)}, ensure_ascii=False)); return 0


if __name__ == "__main__": raise SystemExit(main())
