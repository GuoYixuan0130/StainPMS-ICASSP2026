#!/usr/bin/env python3
"""Render two representative C4 prediction-only conflict-ranking cases."""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from PIL import Image

from stainpms.c2_component_audit import deserialize_gt, deserialize_selected
from stainpms.c4_csr import normalize_graph, prediction_conflict_graph, prediction_only_ranked_assembly, residual_rank_keys
from tools.run_c4_csr import load_ranker, parse_assignment, read_json


SEEDS = (2027, 1337)


def read_gzip(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def colorize(labels: np.ndarray) -> np.ndarray:
    palette = np.asarray([[0, 0, 0], [230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48], [145, 30, 180], [70, 240, 240], [240, 50, 230], [210, 245, 60]], dtype=np.uint8)
    result = np.zeros((*labels.shape, 3), dtype=np.uint8)
    for value in np.unique(labels):
        if value:
            result[labels == value] = palette[int(value) % len(palette)]
    return result


def panel(image: np.ndarray, labels: np.ndarray) -> np.ndarray:
    color = colorize(labels)
    return np.clip(0.55 * image.astype(np.float32) + 0.45 * color.astype(np.float32), 0, 255).astype(np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", required=True, type=Path)
    parser.add_argument("--prepared-dir", required=True, type=Path)
    parser.add_argument("--c1-source", action="append", required=True, type=parse_assignment)
    parser.add_argument("--ranker-weights", action="append", required=True, type=parse_assignment)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--gpu-device", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    evaluation = read_json(args.evaluation)
    sources, weights = dict(args.c1_source), dict(args.ranker_weights)
    if set(sources) != set(SEEDS) or set(weights) != set(SEEDS):
        raise ValueError("requires C1 sources and ranker weights for seeds 2027 and 1337")
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite C4 cases directory: {output}")
    device = torch.device("cuda", args.gpu_device)
    torch.cuda.set_device(device)
    schema = read_json(args.prepared_dir.resolve() / "c4_csr_feature_schema.json")
    output.mkdir(parents=True, exist_ok=False)
    index = {"schema_version": 1, "protocol": "tnbc_c4_csr_v1", "cases": []}
    for seed_row in evaluation["per_seed"]:
        seed = int(seed_row["seed"])
        selected = max(seed_row["per_image"], key=lambda row: float(row["c4"]["pq"] - row["c1"]["pq"]))
        source_payloads = [read_gzip(path) for path in sorted((sources[seed] / "completed_images").glob("*.json.gz"))]
        payload = next(row for row in source_payloads if str(row["artifact"]["sample_id"]) == selected["sample_id"])
        artifact = payload["artifact"]
        gt = deserialize_gt(artifact); records = deserialize_selected(artifact)
        graph = normalize_graph(prediction_conflict_graph(records, gt.shape, instance_nms_iou=0.5), schema["normalization"]["by_seed"][str(seed)]["normalizer"])
        ranker, _, _ = load_ranker(weights[seed], device)
        ranked = prediction_only_ranked_assembly(records, graph, residual_rank_keys(ranker, graph, device=device), gt.shape, instance_nms_iou=0.5)
        source_path = sources[seed] / artifact["source_image_png"]
        image = np.asarray(Image.open(source_path).convert("RGB"), dtype=np.uint8)
        strips = [image, panel(image, gt), panel(image, ranked["native_final_map"]), panel(image, ranked["final_map"])]
        composite = np.concatenate(strips, axis=1)
        target = output / f"seed{seed}_{artifact['sample_id']}_c4_csr.png"
        Image.fromarray(composite).save(target)
        index["cases"].append({"seed": seed, "sample_id": artifact["sample_id"], "patient": int(artifact["patient"]), "selection": "largest per-image C4-C1 PQ delta", "c4_minus_c1_pq": float(selected["c4"]["pq"] - selected["c1"]["pq"]), "panel_order": ["source", "GT", "C1 native", "C4 CSR"], "image": target.name})
    (output / "c4_csr_case_index.json").write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "case_index": str(output / "c4_csr_case_index.json")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
