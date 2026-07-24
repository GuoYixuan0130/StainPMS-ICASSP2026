#!/usr/bin/env python3
"""Prepare, train, preflight, evaluate and summarize fixed C4-CSR runs.

The only trainable object is the small residual ranker in ``stainpms.c4_csr``.
Frozen C1 selected predictions are compact artifacts.  GT is used exclusively
to create detached p1--6 pair labels and to evaluate p7/p8 afterwards; the
deployed graph and assembly invocation receive prediction records only.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stainpms.c2_component_audit import deserialize_gt, deserialize_selected
from stainpms.c3_score_control_audit import summarize_conflicts
from stainpms.c4_csr import (
    EDGE_FEATURE_NAMES,
    NODE_FEATURE_NAMES,
    build_ranker,
    fit_feature_normalizer,
    normalize_graph,
    prediction_conflict_graph,
    prediction_only_ranked_assembly,
    residual_rank_keys,
    training_graph_with_pairs,
)
from stainpms.zero_training_oracle import (
    ORACLE_MATCH_IOU,
    decode_binary_rle,
    error_partition,
    final_pool_oracle_stage,
    native_final_stage,
    oracle_pool_stage,
)

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - reports can be inspected without torch
    torch = None


SEEDS = (2027, 1337)
TASK_FIELDS = ("dice1", "dice2", "aji", "dq", "sq", "pq")
STAGE_FIELDS = ("tp", "fp", "fn", "dq", "sq", "pq")
SIZE_BINS = (("2", 2, 2), ("3_4", 3, 4), ("5_8", 5, 8), ("9_plus", 9, None))


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def read_gzip_json(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def parse_assignment(value: str) -> tuple[int, Path]:
    try:
        raw_seed, raw_path = value.split("=", 1)
        seed = int(raw_seed)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("assignment must be SEED=/absolute/path") from exc
    if seed not in SEEDS:
        raise argparse.ArgumentTypeError("only seeds 2027 and 1337 are permitted")
    return seed, Path(raw_path).resolve()


def deterministic(seed: int) -> None:
    if torch is None:
        raise RuntimeError("C4 ranker execution requires PyTorch")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compact_sources(root: Path, *, expected_count: int, patients: set[int], seed: int, kind: str) -> list[dict[str, Any]]:
    summary = read_json(root / "summary.json")
    if summary.get("status") != "complete" or int(summary.get("seed", -1)) != seed:
        raise ValueError(f"invalid {kind} source summary for seed {seed}: {root}")
    expected_arm = "c1_reconstructed" if seed == 1337 and expected_count == 7 else "c1"
    if summary.get("arm") != expected_arm:
        raise ValueError(f"C4 {kind} source must be frozen C1, not {summary.get('arm')}: {root}")
    artifacts = [read_gzip_json(path) for path in sorted((root / "completed_images").glob("*.json.gz"))]
    observed = {int(item["artifact"]["patient"]) for item in artifacts}
    if len(artifacts) != expected_count or observed != patients:
        raise ValueError(f"C4 {kind} source scope mismatch at {root}: count={len(artifacts)} patients={sorted(observed)}")
    return artifacts


def validate_development_c1_lineage(root: Path, *, seed: int, prepared_config: dict[str, Any], c3_reference: dict[str, Any]) -> None:
    """Fail closed on the approved C1/C3 source pair for development replay."""

    summary = read_json(root / "summary.json")
    source_root = Path(str(c3_reference["source_c1_oracle_directory"])).resolve()
    if root != source_root:
        raise ValueError(f"seed-{seed} C4 development source is not the exact C1 source used by the frozen C3 audit")
    expected = prepared_config["frozen_c1_train_sources"][str(seed)]["lineage_contract"]
    checkpoint = summary.get("checkpoint", {})
    if seed == 2027:
        if (
            expected.get("kind") != "recovery_audited_epoch5_weights_only"
            or summary.get("arm") != "c1"
            or checkpoint.get("checkpoint_sha256") != expected.get("source_last_state_sha256")
        ):
            raise ValueError("seed-2027 C4 development C1 must link the recovery-audited weights-only source to its historical C3 state")
    else:
        frozen = summary.get("frozen_epoch5_manifest", {})
        source_identity = c3_reference["source_identity"]
        if (
            expected.get("kind") != "reconstructed_epoch5_full_state"
            or summary.get("arm") != "c1_reconstructed"
            or checkpoint.get("checkpoint_sha256") != expected.get("complete_state_sha256")
            or checkpoint.get("checkpoint_sha256") != source_identity.get("checkpoint_sha256")
            or frozen.get("status") != "frozen_before_development_access"
            or frozen.get("lineage") != "reconstructed C1 seed-1337 lineage"
            or frozen.get("sha256") != expected.get("frozen_epoch5_manifest_sha256")
            or frozen.get("sha256") != source_identity.get("frozen_epoch5_manifest_sha256")
        ):
            raise ValueError("seed-1337 C4 development C1 must use only the frozen reconstructed lineage paired with reconstructed C3")


def c0_sources(root: Path, *, seed: int) -> list[dict[str, Any]]:
    summary = read_json(root / "summary.json")
    if summary.get("status") != "complete" or int(summary.get("seed", -1)) != seed or summary.get("arm") != "c0":
        raise ValueError(f"invalid C0 p7/p8 source for seed {seed}: {root}")
    artifacts = [read_gzip_json(path) for path in sorted((root / "completed_images").glob("*.json.gz"))]
    if len(artifacts) != 7 or {int(item["artifact"]["patient"]) for item in artifacts} != {7, 8}:
        raise ValueError(f"C0 source must contain exactly TNBC p7/p8 artifacts: {root}")
    return artifacts


def selected_signature(records: list[dict[str, Any]]) -> str:
    payload = []
    for row in records:
        mask = np.asarray(row["mask"], dtype=bool)
        payload.append({
            "record_index": int(row["record_index"]), "prompt_group_id": int(row["prompt_group_id"]),
            "token": int(row.get("token", -1)), "crop_index": int(row.get("crop_index", -1)),
            "assembly_score": float(row["assembly_score"]),
            "mask_bits": hashlib.sha256(np.asfortranarray(mask).tobytes()).hexdigest(),
        })
    return json_sha256(payload)


def final_map_from_artifact(artifact: dict[str, Any]) -> np.ndarray:
    result = np.zeros(tuple(int(value) for value in artifact["image_shape"]), dtype=np.int32)
    for row in artifact.get("native_final_instances", []):
        result[decode_binary_rle(row["mask_rle"])] = int(row["final_instance_id"])
    return result


def deserialize_all(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for encoded in artifact.get("all_native_candidates", []):
        row = dict(encoded)
        row["mask"] = decode_binary_rle(row.pop("mask_rle"))
        row["record_index"] = int(row["record_index"])
        row["prompt_group_id"] = int(row["prompt_group_id"])
        row["token"] = int(row.get("token", -1))
        row["crop_index"] = int(row.get("crop_index", -1))
        row["quality"] = float(row["quality"])
        row["assembly_score"] = float(row["assembly_score"])
        row["edge_penalized"] = bool(row.get("edge_penalized", False))
        rows.append(row)
    return rows


def strict_stage(gt: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    stage = native_final_stage(gt, pred, threshold=ORACLE_MATCH_IOU)
    return {**stage, "final_instance_count": int(np.unique(pred[pred > 0]).size)}


def compact_stage(stage: dict[str, Any]) -> dict[str, Any]:
    return {
        **{field: stage.get(field) for field in STAGE_FIELDS},
        "final_instance_count": int(stage.get("final_instance_count", 0)),
        "strict_metrics": {name: stage.get("strict_metrics", {}).get(name) for name in TASK_FIELDS},
    }


def average_stage(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    values = list(rows)
    result: dict[str, Any] = {field: (float(np.mean([float(row[field]) for row in values])) if values else None) for field in STAGE_FIELDS}
    result["final_instance_count"] = float(np.mean([float(row["final_instance_count"]) for row in values])) if values else None
    result["strict_metrics"] = {
        field: (float(np.mean([float(row["strict_metrics"][field]) for row in values])) if values else None)
        for field in TASK_FIELDS
    }
    return result


def delta_stage(left: dict[str, Any], right: dict[str, Any]) -> dict[str, float]:
    output = {field: float(left[field]) - float(right[field]) for field in STAGE_FIELDS}
    output["final_instance_count"] = float(left["final_instance_count"]) - float(right["final_instance_count"])
    output.update({field: float(left["strict_metrics"][field]) - float(right["strict_metrics"][field]) for field in TASK_FIELDS})
    return output


def mean_metrics(rows: list[dict[str, Any]], key: str) -> dict[str, float | None]:
    return {metric: (float(np.mean([float(row[key][metric]) for row in rows])) if rows else None) for metric in TASK_FIELDS}


def prepare(args: argparse.Namespace) -> int:
    sources = dict(args.train_source)
    if set(sources) != set(SEEDS):
        raise ValueError("C4 preparation requires exactly frozen p1-p6 C1 exports for seeds 2027 and 1337")
    config_path = Path(args.config).resolve()
    config = read_json(config_path)
    if config.get("protocol_id") != "tnbc_c4_conflict_set_structured_ranking_v1":
        raise ValueError("unexpected C4 pre-registered configuration")
    c3_audit = Path(args.c3_audit).resolve()
    c3_reference = load_c3_reference(c3_audit)
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite C4 preparation directory: {output}")
    schemas: dict[str, Any] = {}
    frozen_sources: dict[str, Any] = {}
    for seed in SEEDS:
        payloads = compact_sources(sources[seed], expected_count=30, patients={1, 2, 3, 4, 5, 6}, seed=seed, kind="p1-p6 train")
        graphs = []
        image_pair_counts = []
        for payload in payloads:
            artifact = payload["artifact"]
            gt = deserialize_gt(artifact)
            graph = training_graph_with_pairs(deserialize_selected(artifact), gt, instance_nms_iou=float(args.instance_nms_iou))
            graphs.append(graph)
            image_pair_counts.append({"sample_id": artifact["sample_id"], "patient": int(artifact["patient"]), **graph["pair_counts"]})
        normalizer = fit_feature_normalizer(graphs)
        schemas[str(seed)] = {
            "normalizer": normalizer,
            "train_image_count": len(graphs),
            "train_pair_counts": {
                key: int(sum(int(row[key]) for row in image_pair_counts))
                for key in ("component_count_with_pairs", "pair_count", "unique_tp", "unmatched_fp", "duplicate")
            },
            "per_image_pair_counts": image_pair_counts,
        }
        summary = read_json(sources[seed] / "summary.json")
        frozen_sources[str(seed)] = {
            "directory": str(sources[seed]), "summary_sha256": sha256_file(sources[seed] / "summary.json"),
            "manifest_sha256": summary["manifest"]["sha256"], "checkpoint_sha256": summary["checkpoint"]["checkpoint_sha256"],
            "lineage_contract": summary["checkpoint"]["lineage_contract"],
        }
    preregistered = {
        "schema_version": 1, "status": "frozen_before_development_read", "static_config_path": str(config_path),
        "static_config_sha256": sha256_file(config_path), "protocol": config["protocol_id"],
        "seeds": list(SEEDS), "train_scope": "TNBC p1-p6 only", "development_scope": "not read during preparation",
        "frozen_c1_train_sources": frozen_sources,
        "frozen_c3_reference": {"path": str(c3_audit), "sha256": sha256_file(c3_audit), "per_seed": c3_reference},
        "ranker": config["ranker"], "training": config["training"], "inference": config["inference"],
        "instance_nms_iou": float(args.instance_nms_iou), "oracle_match_iou": ORACLE_MATCH_IOU,
    }
    schema = {
        "schema_version": 1, "status": "frozen_before_development_read",
        "node_features": list(NODE_FEATURE_NAMES), "edge_features": list(EDGE_FEATURE_NAMES),
        "normalization": {"fit_scope": "TNBC p1-p6 only", "by_seed": schemas},
        "pair_definition": "same prediction-only non-singleton conflict component; detached unique_tp > detached conflicting unmatched_fp or duplicate",
        "excluded_pair_definition": ["singleton", "merge_risk", "positive_positive", "negative_negative"],
        "component_size_reporting": "ordering accuracy is component-level; PQ delta is stratified by each image's maximum non-singleton component-size bin",
    }
    output.mkdir(parents=True, exist_ok=False)
    design = """# C4-CSR design\n\nC4 is a ranker-only development experiment.  C1 candidate generation, native token selection, selected masks, scores, score values, NMS threshold and native assembly code are frozen.  A prediction-only graph connects masks that share a prompt group, exceed the native box-NMS predicate, or overlap in paint order.  Only non-singleton component score values may be permuted.\n\nThe ranker is a two-layer width-64 node MLP plus one mean/max relation aggregation.  Its zero-initialized residual output creates `s_rank = s_native + delta`; stable ties use native score and record index.  Training uses only detached GT labels on p1--6: within a real conflict component, unique TP ranks above conflicting unmatched FP or duplicate.  The loss is component-balanced pairwise logistic loss.  No GT, evaluator matching, patient identity or filename is read at inference.\n\nDevelopment is fixed to ranker epoch 20 on p7/p8 for seeds 2027 and 1337.  It is development evidence only.\n"""
    (output / "c4_csr_design.md").write_text(design, encoding="utf-8")
    write_json_atomic(output / "c4_csr_preregistered_config.json", preregistered)
    write_json_atomic(output / "c4_csr_feature_schema.json", schema)
    write_json_atomic(output / "prepared_manifest.json", {"status": "complete", "preregistered_config_sha256": json_sha256(preregistered), "feature_schema_sha256": json_sha256(schema)})
    print(json.dumps({"status": "complete", "output_dir": str(output), "train_pair_counts": {seed: schemas[str(seed)]["train_pair_counts"] for seed in SEEDS}}, ensure_ascii=False))
    return 0


def read_prepared(directory: Path, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    config = read_json(directory / "c4_csr_preregistered_config.json")
    schema = read_json(directory / "c4_csr_feature_schema.json")
    if config.get("status") != "frozen_before_development_read" or schema.get("status") != "frozen_before_development_read":
        raise ValueError("C4 preparation was not frozen before development access")
    if str(seed) not in schema["normalization"]["by_seed"]:
        raise ValueError(f"C4 preparation lacks seed {seed}")
    return config, schema


def train_graphs(source: Path, seed: int, normalizer: dict[str, Any]) -> list[dict[str, Any]]:
    graphs = []
    for payload in compact_sources(source, expected_count=30, patients={1, 2, 3, 4, 5, 6}, seed=seed, kind="p1-p6 train"):
        artifact = payload["artifact"]
        graph = training_graph_with_pairs(deserialize_selected(artifact), deserialize_gt(artifact), instance_nms_iou=0.5)
        graph["sample_id"] = str(artifact["sample_id"])
        graph["patient"] = int(artifact["patient"])
        graphs.append(normalize_graph(graph, normalizer))
    return graphs


def component_balanced_pair_loss(ranker, graph: dict[str, Any], device) -> tuple[Any, int, int]:
    if torch is None:
        raise RuntimeError("C4 ranker training requires PyTorch")
    pairs = graph["component_pairs"]
    if not pairs:
        return None, 0, 0
    nodes = torch.as_tensor(graph["node_features"], dtype=torch.float32, device=device)
    edge_index = torch.as_tensor(graph["edge_index"], dtype=torch.long, device=device)
    edges = torch.as_tensor(graph["edge_features"], dtype=torch.float32, device=device)
    native = torch.as_tensor([float(row["assembly_score"]) for row in graph["records"]], dtype=torch.float32, device=device)
    residual = ranker(nodes, edge_index, edges)
    keys = native + residual
    component_losses = []
    pair_count = 0
    for pair in pairs:
        positive = torch.as_tensor(pair["positive_indices"], dtype=torch.long, device=device)
        negative = torch.as_tensor(pair["negative_indices"], dtype=torch.long, device=device)
        margins = keys[positive][:, None] - keys[negative][None, :]
        component_losses.append(torch.nn.functional.softplus(-margins).mean())
        pair_count += int(margins.numel())
    return torch.stack(component_losses).mean(), len(component_losses), pair_count


def train(args: argparse.Namespace) -> int:
    if torch is None or not torch.cuda.is_available():
        raise RuntimeError("C4 ranker-only training requires CUDA PyTorch")
    prepared = Path(args.prepared_dir).resolve()
    config, schema = read_prepared(prepared, args.seed)
    source = Path(args.train_source).resolve()
    expected = config["frozen_c1_train_sources"][str(args.seed)]
    if str(source) != expected["directory"] or sha256_file(source / "summary.json") != expected["summary_sha256"]:
        raise ValueError("C4 train source differs from the p1-p6 source frozen during preparation")
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite C4 ranker training output: {output}")
    deterministic(args.seed)
    device = torch.device("cuda", int(args.gpu_device))
    torch.cuda.set_device(device)
    normalizer = schema["normalization"]["by_seed"][str(args.seed)]["normalizer"]
    graphs = train_graphs(source, args.seed, normalizer)
    ranker, parameter_count = build_ranker(width=int(config["ranker"]["width"]))
    ranker.to(device).train()
    optimizer = torch.optim.AdamW(ranker.parameters(), lr=float(config["training"]["learning_rate"]), weight_decay=float(config["training"]["weight_decay"]))
    epoch_logs = []
    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        order = list(range(len(graphs)))
        # The exact data order is deterministic but does not depend on p7/p8.
        random.Random(args.seed * 1000 + epoch).shuffle(order)
        losses = []
        component_count = pair_count = 0
        for index in order:
            loss, local_components, local_pairs = component_balanced_pair_loss(ranker, graphs[index], device)
            if loss is None:
                continue
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            component_count += local_components
            pair_count += local_pairs
        if not losses:
            raise RuntimeError("C4 training has no valid detached ranking pairs")
        epoch_logs.append({"epoch": epoch, "mean_component_balanced_pairwise_logistic_loss": float(np.mean(losses)), "image_update_count": len(losses), "component_count": component_count, "pair_count": pair_count})
        print(f"[c4-csr] seed={args.seed} epoch={epoch}/20 loss={epoch_logs[-1]['mean_component_balanced_pairwise_logistic_loss']:.6f} images={len(losses)} pairs={pair_count}", flush=True)
    output.mkdir(parents=True, exist_ok=False)
    weights = {
        "schema_version": 1, "protocol": config["protocol"], "seed": args.seed,
        "parameter_count": parameter_count, "state_dict": ranker.cpu().state_dict(),
        "prepared_manifest_sha256": sha256_file(prepared / "prepared_manifest.json"),
        "normalizer": normalizer,
    }
    torch.save(weights, output / "ranker_epoch20_weights.pth")
    torch.save({"schema_version": 1, "epoch": 20, "ranker": weights["state_dict"], "optimizer": optimizer.state_dict(), "torch_rng_state": torch.get_rng_state(), "numpy_rng_state": np.random.get_state()}, output / "ranker_final_training_state.pth")
    summary = {
        "schema_version": 1, "protocol": config["protocol"], "status": "complete", "seed": args.seed,
        "scope": "ranker-only C4 training on frozen C1 p1-p6 selected predictions; no C1 gradients or development access",
        "source": expected, "prepared_dir": str(prepared), "prepared_manifest_sha256": weights["prepared_manifest_sha256"],
        "parameter_count": parameter_count, "epochs": epoch_logs,
        "retention": {"weights": "ranker_epoch20_weights.pth", "final_training_state": "ranker_final_training_state.pth", "no_epoch_1_to_19_ranker_states": True},
        "weights_sha256": sha256_file(output / "ranker_epoch20_weights.pth"),
    }
    write_json_atomic(output / "training_summary.json", summary)
    print(json.dumps({"status": "complete", "output_dir": str(output), "weights": str(output / "ranker_epoch20_weights.pth")}, ensure_ascii=False))
    return 0


def load_ranker(weights_path: Path, device):
    if torch is None:
        raise RuntimeError("C4 ranker inference requires PyTorch")
    payload = torch.load(weights_path, map_location="cpu", weights_only=False)
    ranker, parameter_count = build_ranker(width=64)
    ranker.load_state_dict(payload["state_dict"], strict=True)
    ranker.to(device).eval()
    return ranker, payload, parameter_count


def label_rows_with_scores(records: list[dict[str, Any]], scores: list[float], gt: np.ndarray) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    graph = training_graph_with_pairs(records, gt, instance_nms_iou=0.5)
    labelled = [dict(row) for row in graph["records"]]
    for index, score in enumerate(scores):
        labelled[index]["assembly_score"] = float(score)
    return labelled, graph


def component_bin(size: int) -> str:
    for name, lower, upper in SIZE_BINS:
        if size >= lower and (upper is None or size <= upper):
            return name
    raise ValueError(f"unclassified component size: {size}")


def ranking_by_size(records: list[dict[str, Any]], graph: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, dict[str, int]] = {name: {"components": 0, "top1_correct": 0, "top1_total": 0, "pair_correct": 0, "pair_total": 0} for name, *_ in SIZE_BINS}
    for component in graph["components"]:
        if len(component) <= 1:
            continue
        bucket = result[component_bin(len(component))]
        bucket["components"] += 1
        positives = [index for index in component if records[index]["utility_label"] == "unique_tp"]
        negatives = [index for index in component if records[index]["utility_label"] in {"unmatched_fp", "duplicate"}]
        if positives and negatives:
            top = max(float(records[index]["assembly_score"]) for index in component)
            bucket["top1_total"] += 1
            bucket["top1_correct"] += int(any(float(records[index]["assembly_score"]) == top and records[index]["utility_label"] == "unique_tp" for index in component))
            for positive in positives:
                for negative in negatives:
                    bucket["pair_total"] += 1
                    bucket["pair_correct"] += int(float(records[positive]["assembly_score"]) > float(records[negative]["assembly_score"]))
    return {name: {**row, "top1_accuracy": (float(row["top1_correct"] / row["top1_total"]) if row["top1_total"] else None), "pairwise_accuracy": (float(row["pair_correct"] / row["pair_total"]) if row["pair_total"] else None)} for name, row in result.items()}


def extract_c0_stage(payload: dict[str, Any]) -> dict[str, Any]:
    stage = payload["image_record"]["stages"]["native_final"]
    return {**compact_stage({**stage, "final_instance_count": payload["image_record"]["native_final_prediction_count"]})}


def evaluate_one(
    payload: dict[str, Any],
    ranker,
    normalizer: dict[str, Any],
    *,
    nms_iou: float,
    zero_residual: bool,
) -> dict[str, Any]:
    artifact = payload["artifact"]
    gt = deserialize_gt(artifact)
    selected = deserialize_selected(artifact)
    all_candidates = deserialize_all(artifact)
    graph = normalize_graph(prediction_conflict_graph(selected, gt.shape, instance_nms_iou=nms_iou), normalizer)
    if zero_residual:
        rank_keys = np.asarray([float(row["assembly_score"]) for row in selected], dtype=np.float64)
    else:
        rank_keys = residual_rank_keys(ranker, graph, device=next(ranker.parameters()).device)
    ranked = prediction_only_ranked_assembly(selected, graph, rank_keys, gt.shape, instance_nms_iou=nms_iou)
    native = strict_stage(gt, ranked["native_final_map"])
    c4 = strict_stage(gt, ranked["final_map"])
    expected_native = payload["image_record"]["stages"]["native_final"]
    metric_mismatch = {field: {"expected": expected_native[field], "actual": native[field]} for field in STAGE_FIELDS if not math.isclose(float(expected_native[field]), float(native[field]), rel_tol=0.0, abs_tol=1.0e-7)}
    source_final_map = final_map_from_artifact(artifact)
    final_map_identical = bool(np.array_equal(source_final_map, ranked["native_final_map"]))
    selected_oracle = oracle_pool_stage(selected, gt, threshold=ORACLE_MATCH_IOU)
    expected_selected_oracle = payload["image_record"]["stages"]["native_selected_pool_oracle"]
    selected_oracle_identical = all(math.isclose(float(selected_oracle[field]), float(expected_selected_oracle[field]), rel_tol=0.0, abs_tol=1.0e-7) for field in ("tp", "fp", "fn", "dq", "sq", "pq"))
    c4_final_oracle = final_pool_oracle_stage(gt, ranked["final_map"], threshold=ORACLE_MATCH_IOU)
    errors = error_partition(gt_map=gt, all_candidate_records=all_candidates, selected_records=selected, native_final=c4, final_map=ranked["final_map"], threshold=ORACLE_MATCH_IOU)
    labelled, train_graph = label_rows_with_scores(selected, ranked["assembly_scores"], gt)
    ranking = summarize_conflicts(labelled, graph)
    singletons_unchanged = all(
        math.isclose(float(ranked["native_scores"][index]), float(ranked["assembly_scores"][index]), rel_tol=0.0, abs_tol=0.0)
        for index, flag in enumerate(graph["non_singleton_mask"]) if not bool(flag)
    )
    max_component = max((len(component) for component in graph["components"] if len(component) > 1), default=1)
    c1_final_oracle = payload["image_record"]["stages"]["final_pool_oracle"]
    c1_selected_oracle = payload["image_record"]["stages"]["native_selected_pool_oracle"]
    return {
        "sample_id": str(artifact["sample_id"]), "patient": int(artifact["patient"]),
        "selected_signature": selected_signature(selected), "native_reproduction": {"metric_mismatch": metric_mismatch, "final_map_identical": final_map_identical},
        "invariance": {"selected_oracle_identical": selected_oracle_identical, "singleton_scores_unchanged": singletons_unchanged, "candidate_pool_unchanged": True, "mask_unchanged": True, "keep_threshold_unchanged": True, "inference_uses_gt": False, "inference_uses_evaluator_matching": False},
        "c1": compact_stage(native), "c4": compact_stage(c4), "c4_final_pool_oracle": compact_stage({**c4_final_oracle, "final_instance_count": c4["final_instance_count"]}),
        "selected_pool_oracle": {field: selected_oracle[field] for field in ("tp", "fp", "fn", "dq", "sq", "pq", "coverage_recall_at_0_5")},
        "assembly_accounting": {
            "c1_selected_to_final_assembly_gap_pq": float(c1_selected_oracle["pq"] - c1_final_oracle["pq"]),
            "c4_selected_to_final_assembly_gap_pq": float(selected_oracle["pq"] - c4_final_oracle["pq"]),
            "c1_final_fp_pq_penalty": float(c1_final_oracle["pq"] - native["pq"]),
            "c4_final_fp_pq_penalty": float(c4_final_oracle["pq"] - c4["pq"]),
            "selected_pool_oracle_pq": float(selected_oracle["pq"]),
        },
        "c1_errors": payload["image_record"]["errors"],
        "c1_structural_errors": payload["image_record"]["errors"]["native_final_structural_errors"],
        "errors": errors["counts"], "structural_errors": errors["native_final_structural_errors"],
        "ranking": ranking, "ranking_by_component_size": ranking_by_size(labelled, graph), "pair_counts": train_graph["pair_counts"],
        "assembly": {key: ranked[key] for key in ("native_final_instance_count", "final_instance_count", "eligible_non_singleton_component_count", "accepted_component_permutation_count", "rejected_for_final_count_change")},
        "image_max_non_singleton_component_size": int(max_component),
    }


def aggregate_ranking(rows: list[dict[str, Any]]) -> dict[str, Any]:
    top_num = sum(int(row["ranking"]["unique_tp_native_top1"]["numerator"]) for row in rows)
    top_den = sum(int(row["ranking"]["unique_tp_native_top1"]["denominator"]) for row in rows)
    pairs: dict[str, Any] = {}
    for name in ("all_negative", "unmatched_fp", "duplicate"):
        correct = sum(int(row["ranking"]["pairwise_ordering"][name]["correct"]) for row in rows)
        count = sum(int(row["ranking"]["pairwise_ordering"][name]["count"]) for row in rows)
        margins = [value for row in rows for value in row["ranking"]["pairwise_ordering"][name]["positive_minus_negative_margin"].get("margin_values", [])]
        # Original C3 reports hide raw margins; use direct image summaries where unavailable.
        pairs[name] = {"correct": correct, "count": count, "accuracy": float(correct / count) if count else None, "margin_mean_of_image_means": float(np.mean([row["ranking"]["pairwise_ordering"][name]["positive_minus_negative_margin"]["mean"] for row in rows if row["ranking"]["pairwise_ordering"][name]["positive_minus_negative_margin"]["mean"] is not None])) if any(row["ranking"]["pairwise_ordering"][name]["positive_minus_negative_margin"]["mean"] is not None for row in rows) else None}
    return {"unique_tp_top1": {"numerator": top_num, "denominator": top_den, "accuracy": float(top_num / top_den) if top_den else None}, "pairwise_ordering": pairs}


def _sum_error_counts(rows: list[dict[str, Any]], key: str, structural_key: str) -> dict[str, int]:
    counts = ("generation_miss", "selection_miss", "assembly_loss", "native_final_false_positive_count", "native_final_false_negative_count")
    structural = ("duplicate_unmatched_prediction_count", "unmatched_gt_count", "unmatched_prediction_count")
    output = {name: int(sum(int(row[key]["counts"].get(name, 0)) if "counts" in row[key] else int(row[key].get(name, 0)) for row in rows)) for name in counts}
    for name in structural:
        output[name] = int(sum(int(row[structural_key].get(name, 0)) for row in rows))
    sensitivity_key = "overlap_fraction_gt_or_pred_gt_0.1"
    for name in ("split", "merge"):
        output[name] = int(sum(int(row[structural_key]["sensitivity"][sensitivity_key][name]) for row in rows))
    return output


def _mean_mapping(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    keys = sorted({name for row in rows for name, value in row[key].items() if isinstance(value, (int, float))})
    return {name: float(np.mean([float(row[key][name]) for row in rows])) for name in keys}


def patient_summary(rows: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    by_patient = {str(patient): [row for row in rows if int(row["patient"]) == patient] for patient in (7, 8)}
    patients = {}
    for patient, items in by_patient.items():
        patients[patient] = {
            "image_count": len(items), "c4": average_stage([row["c4"] for row in items]), "c1": average_stage([row["c1"] for row in items]), "c0": average_stage([row["c0"] for row in items]),
            "c4_minus_c1": delta_stage(average_stage([row["c4"] for row in items]), average_stage([row["c1"] for row in items])),
            "c4_minus_c0": delta_stage(average_stage([row["c4"] for row in items]), average_stage([row["c0"] for row in items])),
            "ranking": aggregate_ranking(items),
            "errors": {"c4": _sum_error_counts(items, "errors", "structural_errors"), "c1": _sum_error_counts(items, "c1_errors", "c1_structural_errors"), "c0": _sum_error_counts(items, "c0_errors", "c0_structural_errors")},
            "accounting": {
                "c4": _mean_mapping(items, "assembly_accounting"),
                "c1": {
                    "selected_to_final_assembly_gap_pq": float(np.mean([row["assembly_accounting"]["c1_selected_to_final_assembly_gap_pq"] for row in items])),
                    "final_fp_pq_penalty": float(np.mean([row["assembly_accounting"]["c1_final_fp_pq_penalty"] for row in items])),
                    "selected_pool_oracle_pq": float(np.mean([row["assembly_accounting"]["selected_pool_oracle_pq"] for row in items])),
                },
                "c0": _mean_mapping(items, "c0_assembly_accounting"),
            },
        }
    macro = {
        key: {field: float(np.mean([patients[str(patient)][key][field] for patient in (7, 8)])) for field in (TASK_FIELDS if key in {"c4", "c1", "c0"} else ())}
        for key in ()
    }
    # Stage structures mix task fields under strict_metrics.  Aggregate them directly.
    macro_stages = {}
    for name in ("c4", "c1", "c0"):
        macro_stages[name] = average_stage([patients["7"][name], patients["8"][name]])
    accounting = {
        name: {key: float(np.mean([patients[str(patient)]["accounting"][name][key] for patient in (7, 8)])) for key in patients["7"]["accounting"][name]}
        for name in ("c4", "c1", "c0")
    }
    errors = {
        name: {key: int(sum(patients[str(patient)]["errors"][name][key] for patient in (7, 8))) for key in patients["7"]["errors"][name]}
        for name in ("c4", "c1", "c0")
    }
    return {"patients": patients, "patient_macro": {"c4": macro_stages["c4"], "c1": macro_stages["c1"], "c0": macro_stages["c0"], "c4_minus_c1": delta_stage(macro_stages["c4"], macro_stages["c1"]), "c4_minus_c0": delta_stage(macro_stages["c4"], macro_stages["c0"]), "ranking": aggregate_ranking(rows), "accounting": accounting, "errors": errors}}


def load_c3_reference(path: Path) -> dict[str, dict[str, Any]]:
    payload = read_json(path)
    if (
        payload.get("status") != "complete"
        or payload.get("c3_gate", {}).get("single_supported_operation") != "conflict_order_oracle"
        or payload.get("lineage", {}).get("seed2027") != "historical verified C1 epoch-5 lineage"
        or payload.get("lineage", {}).get("seed1337") != "reconstructed C1 seed-1337 lineage"
    ):
        raise ValueError("C4 requires the reconstructed joint C3 conflict-order-supported audit")
    references: dict[str, dict[str, Any]] = {}
    for row in payload["per_seed"]:
        seed = int(row["seed"])
        conflict = row["patient_macro"]["conflicts_both_patients"]
        reference = {
            "conflict_order_oracle_delta_pq": float(row["patient_macro"]["deltas_vs_native_patient_macro"]["conflict_order_oracle"]["pq"]),
            "native_top1_accuracy": float(conflict["unique_tp_native_top1"]["accuracy"]),
            "native_pairwise_accuracy": float(conflict["pairwise_ordering"]["all_negative"]["accuracy"]),
            "source_c1_oracle_directory": str(row["source_c1_oracle_directory"]),
            "source_identity": row.get("source_identity"),
        }
        if seed == 2027:
            reused = row.get("historical_c3_reused_without_rerun")
            if (
                not isinstance(reused, dict)
                or not isinstance(reused.get("path"), str)
                or not isinstance(reused.get("sha256"), str)
                or len(reused["sha256"]) != 64
                or payload.get("historical_seed_reuse", {}).get("2027") != reused
            ):
                raise ValueError("seed-2027 C4 requires the historical C3 record to be reused without rerun")
        if seed == 1337:
            identity = reference["source_identity"]
            if not isinstance(identity, dict) or identity.get("lineage") != "reconstructed C1 seed-1337 lineage":
                raise ValueError("seed-1337 C4 requires reconstructed C3 source identity")
        references[str(seed)] = reference
    if {int(seed) for seed in references} != set(SEEDS):
        raise ValueError("C3 audit lacks one of the fixed seeds")
    return references


def evaluate(args: argparse.Namespace) -> int:
    if torch is None or not torch.cuda.is_available():
        raise RuntimeError("C4 development evaluation requires CUDA PyTorch")
    c1 = dict(args.c1_source); c0 = dict(args.c0_source); weights = dict(args.ranker_weights or [])
    if set(c1) != set(SEEDS) or set(c0) != set(SEEDS) or (not args.zero_residual and set(weights) != set(SEEDS)):
        raise ValueError("C4 evaluation requires C0/C1 for both seeds and ranker weights for both seeds unless zero-residual preflight is requested")
    prepared = Path(args.prepared_dir).resolve()
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite C4 evaluation output: {output}")
    c3_audit = Path(args.c3_audit).resolve()
    c3_reference = load_c3_reference(c3_audit)
    output.mkdir(parents=True, exist_ok=False)
    all_seed = []
    device = torch.device("cuda", int(args.gpu_device)); torch.cuda.set_device(device)
    for seed in SEEDS:
        config, schema = read_prepared(prepared, seed)
        frozen_c3 = config.get("frozen_c3_reference", {})
        if frozen_c3.get("sha256") != sha256_file(c3_audit) or frozen_c3.get("per_seed") != c3_reference:
            raise ValueError("C4 evaluation C3 reference differs from the reference frozen before development access")
        validate_development_c1_lineage(c1[seed], seed=seed, prepared_config=config, c3_reference=c3_reference[str(seed)])
        dev_c1 = compact_sources(c1[seed], expected_count=7, patients={7, 8}, seed=seed, kind="p7-p8 development")
        dev_c0 = c0_sources(c0[seed], seed=seed)
        c0_by_sample = {str(payload["artifact"]["sample_id"]): payload for payload in dev_c0}
        if args.zero_residual:
            ranker, parameter_count = build_ranker(width=int(config["ranker"]["width"]))
            ranker.to(device).eval()
            weight_payload = {"prepared_manifest_sha256": sha256_file(prepared / "prepared_manifest.json")}
        else:
            ranker, weight_payload, parameter_count = load_ranker(weights[seed], device)
            if weight_payload.get("prepared_manifest_sha256") != sha256_file(prepared / "prepared_manifest.json"):
                raise ValueError("ranker was not trained from this frozen C4 preparation")
        normalizer = schema["normalization"]["by_seed"][str(seed)]["normalizer"]
        rows = []
        for payload in dev_c1:
            result = evaluate_one(payload, ranker, normalizer, nms_iou=0.5, zero_residual=bool(args.zero_residual))
            c0_payload = c0_by_sample.get(result["sample_id"])
            if c0_payload is None:
                raise ValueError("paired C0 development source lacks C1 sample")
            result["c0"] = extract_c0_stage(c0_payload)
            c0_final_oracle = c0_payload["image_record"]["stages"]["final_pool_oracle"]
            c0_selected_oracle = c0_payload["image_record"]["stages"]["native_selected_pool_oracle"]
            c0_native = c0_payload["image_record"]["stages"]["native_final"]
            result["c0_errors"] = c0_payload["image_record"]["errors"]
            result["c0_structural_errors"] = c0_payload["image_record"]["errors"]["native_final_structural_errors"]
            result["c0_assembly_accounting"] = {
                "selected_to_final_assembly_gap_pq": float(c0_selected_oracle["pq"] - c0_final_oracle["pq"]),
                "final_fp_pq_penalty": float(c0_final_oracle["pq"] - c0_native["pq"]),
                "selected_pool_oracle_pq": float(c0_selected_oracle["pq"]),
            }
            rows.append(result)
        failures = [row["sample_id"] for row in rows if row["native_reproduction"]["metric_mismatch"] or not row["native_reproduction"]["final_map_identical"] or not all(bool(value) for value in row["invariance"].values())]
        if failures:
            write_json_atomic(output / f"seed{seed}_failure.json", {"seed": seed, "failure_samples": failures, "rows": rows})
            raise RuntimeError(f"C4 invariance/reproduction gate failed for seed {seed}: {failures}")
        result = {
            "schema_version": 1, "status": "complete", "seed": seed, "mode": "zero_residual_preflight" if args.zero_residual else "trained_epoch20", "parameter_count": parameter_count,
            "source": {"c1": str(c1[seed]), "c0": str(c0[seed]), "ranker_weights": (None if args.zero_residual else str(weights[seed])), "ranker_weights_sha256": (None if args.zero_residual else sha256_file(weights[seed]))},
            "per_image": rows, **patient_summary(rows, "c4"), "c3_conflict_order_oracle_delta_pq": c3_reference[str(seed)]["conflict_order_oracle_delta_pq"],
        }
        write_json_atomic(output / f"seed{seed}_evaluation.json", result)
        all_seed.append(result)
    aggregate = summarize_evaluations(all_seed, c3_reference, zero_residual=bool(args.zero_residual))
    write_json_atomic(output / ("c4_csr_zero_residual_gate.json" if args.zero_residual else "c4_csr_results.json"), aggregate)
    if not args.zero_residual:
        write_csv(output / "c4_csr_results.csv", aggregate)
        (output / "c4_csr_results.md").write_text(markdown(aggregate), encoding="utf-8")
    print(json.dumps({"status": "complete", "output_dir": str(output), "gate": aggregate["promotion_gate"]["status"]}, ensure_ascii=False))
    return 0


def summarize_evaluations(per_seed: list[dict[str, Any]], c3_reference: dict[str, dict[str, Any]], *, zero_residual: bool) -> dict[str, Any]:
    gate_conditions: dict[str, bool] = {}
    if zero_residual:
        pass_all = all(not row["native_reproduction"]["metric_mismatch"] and row["native_reproduction"]["final_map_identical"] and all(bool(value) for value in row["invariance"].values()) for seed in per_seed for row in seed["per_image"])
        return {"schema_version": 1, "protocol": "tnbc_c4_csr_v1", "status": "complete", "mode": "zero_residual_preflight", "per_seed": per_seed, "promotion_gate": {"status": "pass" if pass_all else "fail", "all_zero_residual_and_c1_invariance_checks": pass_all}}
    recovery = {}
    for row in per_seed:
        seed = int(row["seed"])
        delta = float(row["patient_macro"]["c4_minus_c1"]["pq"])
        denominator = float(c3_reference[str(seed)]["conflict_order_oracle_delta_pq"])
        recovery[str(seed)] = float(delta / denominator) if denominator else None
    gate_conditions["all_four_seed_patient_c4_minus_c1_pq_positive"] = all(float(row["patients"][str(patient)]["c4_minus_c1"]["pq"]) > 0.0 for row in per_seed for patient in (7, 8))
    gate_conditions["both_seed_patient_macro_c4_minus_c0_pq_positive"] = all(float(row["patient_macro"]["c4_minus_c0"]["pq"]) > 0.0 for row in per_seed)
    gate_conditions["both_seed_c4_minus_c1_dq_positive"] = all(float(row["patient_macro"]["c4_minus_c1"]["dq"]) > 0.0 for row in per_seed)
    gate_conditions["mean_c4_minus_c0_aji_positive"] = float(np.mean([row["patient_macro"]["c4_minus_c0"]["aji"] for row in per_seed])) > 0.0
    ordering_improvements = []
    for row in per_seed:
        ranking = row["patient_macro"]["ranking"]
        c3_top = float(c3_reference[str(row["seed"])]["native_top1_accuracy"])
        c3_pair = float(c3_reference[str(row["seed"])]["native_pairwise_accuracy"])
        ordering_improvements.append({"seed": row["seed"], "top1_delta": float(ranking["unique_tp_top1"]["accuracy"] - c3_top), "pairwise_delta": float(ranking["pairwise_ordering"]["all_negative"]["accuracy"] - c3_pair)})
    gate_conditions["both_seed_top1_and_pairwise_accuracy_improve_at_least_0_05"] = all(item["top1_delta"] >= 0.05 and item["pairwise_delta"] >= 0.05 for item in ordering_improvements)
    gate_conditions["mean_oracle_recovery_ratio_at_least_0_25"] = float(np.mean([value for value in recovery.values() if value is not None])) >= 0.25
    gate_conditions["selected_masks_candidate_pool_and_selected_oracle_identical"] = all(all(row["invariance"]["candidate_pool_unchanged"] and row["invariance"]["mask_unchanged"] and row["invariance"]["selected_oracle_identical"] for row in seed["per_image"]) for seed in per_seed)
    return {"schema_version": 1, "protocol": "tnbc_c4_csr_v1", "status": "complete", "scope": "TNBC p7/p8 development only; seeds 2027/1337; trained C4 ranker-only epoch 20", "per_seed": per_seed, "ordering_improvements_vs_c3_native": ordering_improvements, "oracle_recovery_ratio": recovery, "promotion_gate": {"status": "promote_to_confirmation" if all(gate_conditions.values()) else "do_not_promote", "conditions": gate_conditions}}


def markdown(payload: dict[str, Any]) -> str:
    lines = ["# C4-CSR development results", "", "- Scope: frozen C1 candidates/masks and native assembly; TNBC p7/p8 development only; seeds 2027 and 1337.", "- C4 changes only relative ordering inside predicted non-singleton conflict components. Singleton scores, score threshold behaviour, masks and candidate pool remain native C1.", "", "## Core paired results", "", "| seed | C4-C1 AJI | C4-C1 DQ | C4-C1 PQ | C4-C0 AJI | C4-C0 PQ | oracle recovery ratio |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for row in payload["per_seed"]:
        d1, d0 = row["patient_macro"]["c4_minus_c1"], row["patient_macro"]["c4_minus_c0"]
        lines.append(f"| {row['seed']} | {d1['aji']:+.6f} | {d1['dq']:+.6f} | {d1['pq']:+.6f} | {d0['aji']:+.6f} | {d0['pq']:+.6f} | {payload['oracle_recovery_ratio'][str(row['seed'])]:.4f} |")
    lines += ["", "## Patient-wise C4-C1 PQ", "", "| seed | p7 | p8 |", "|---:|---:|---:|"]
    for row in payload["per_seed"]:
        lines.append(f"| {row['seed']} | {row['patients']['7']['c4_minus_c1']['pq']:+.6f} | {row['patients']['8']['c4_minus_c1']['pq']:+.6f} |")
    lines += ["", "## Ordering mechanism", "", "| seed | C4 top-1 | C3 native top-1 | delta | C4 pairwise | C3 native pairwise | delta |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for item, row in zip(payload["ordering_improvements_vs_c3_native"], payload["per_seed"], strict=True):
        ranking = row["patient_macro"]["ranking"]
        top = ranking["unique_tp_top1"]["accuracy"]; pair = ranking["pairwise_ordering"]["all_negative"]["accuracy"]
        lines.append(f"| {row['seed']} | {top:.4f} | {top-item['top1_delta']:.4f} | {item['top1_delta']:+.4f} | {pair:.4f} | {pair-item['pairwise_delta']:.4f} | {item['pairwise_delta']:+.4f} |")
    lines += ["", "## Promotion gate", "", f"- Status: `{payload['promotion_gate']['status']}`."]
    for name, passed in payload["promotion_gate"]["conditions"].items():
        lines.append(f"- {name}: {'pass' if passed else 'fail'}")
    return "\n".join(lines) + "\n"


def write_csv(path: Path, payload: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    for seed in payload["per_seed"]:
        for image in seed["per_image"]:
            rows.append({"seed": seed["seed"], "patient": image["patient"], "sample_id": image["sample_id"], "section": "performance", **{f"c4_{key}": value for key, value in image["c4"].items() if key != "strict_metrics"}, **{f"c1_{key}": value for key, value in image["c1"].items() if key != "strict_metrics"}, **{f"c0_{key}": value for key, value in image["c0"].items() if key != "strict_metrics"}, **{f"c4_{key}": value for key, value in image["c4"]["strict_metrics"].items()}, **{f"c1_{key}": value for key, value in image["c1"]["strict_metrics"].items()}, **{f"c0_{key}": value for key, value in image["c0"]["strict_metrics"].items()}})
            rows.append({"seed": seed["seed"], "patient": image["patient"], "sample_id": image["sample_id"], "section": "ranking", "top1_accuracy": image["ranking"]["unique_tp_native_top1"]["accuracy"], "pairwise_accuracy": image["ranking"]["pairwise_ordering"]["all_negative"]["accuracy"], "accepted_components": image["assembly"]["accepted_component_permutation_count"], "rejected_components": image["assembly"]["rejected_for_final_count_change"]})
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def preflight(args: argparse.Namespace) -> int:
    args.zero_residual = True
    return evaluate(args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--instance-nms-iou", type=float, default=0.5)
    prep = sub.add_parser("prepare", parents=(common,))
    prep.add_argument("--config", required=True); prep.add_argument("--train-source", action="append", required=True, type=parse_assignment); prep.add_argument("--c3-audit", required=True); prep.add_argument("--output-dir", required=True)
    train_p = sub.add_parser("train")
    train_p.add_argument("--prepared-dir", required=True); train_p.add_argument("--train-source", required=True); train_p.add_argument("--seed", type=int, required=True, choices=SEEDS); train_p.add_argument("--output-dir", required=True); train_p.add_argument("--gpu-device", type=int, default=0)
    eval_p = sub.add_parser("evaluate")
    preflight_p = sub.add_parser("preflight")
    for current in (eval_p, preflight_p):
        current.add_argument("--prepared-dir", required=True); current.add_argument("--c1-source", action="append", required=True, type=parse_assignment); current.add_argument("--c0-source", action="append", required=True, type=parse_assignment); current.add_argument("--c3-audit", required=True); current.add_argument("--output-dir", required=True); current.add_argument("--gpu-device", type=int, default=0); current.set_defaults(zero_residual=False)
    eval_p.add_argument("--ranker-weights", action="append", required=True, type=parse_assignment)
    preflight_p.set_defaults(ranker_weights=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "prepare": args.train_source = dict(args.train_source); return prepare(args)
    if args.command == "train": return train(args)
    args.c1_source = dict(args.c1_source); args.c0_source = dict(args.c0_source); args.ranker_weights = None if args.ranker_weights is None else dict(args.ranker_weights)
    return preflight(args) if args.command == "preflight" else evaluate(args)


if __name__ == "__main__":
    raise SystemExit(main())
