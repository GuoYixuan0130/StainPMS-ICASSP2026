"""Project-lead-approved MoNuSeg runtime and futility gates.

This module is deliberately separate from the formal Stage 1 oracle runner.
It operates only on the precommitted MoNuSeg router-train split and never
decodes actions during the Gate 1 candidate-opportunity audit.
"""

from __future__ import annotations

import copy
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from skimage.io import imread
from torch.utils.data import DataLoader

from stainroute.actions import (
    ActionCandidate,
    ActionType,
    AddCandidateConfig,
    SplitAssemblyConfig,
    SplitCandidateConfig,
    apply_add_action,
    generate_add_candidates,
    generate_split_candidates,
)
from stainroute.metrics import PQEvaluation, evaluate_pq
from stainroute.oracle_actions import compute_action_utility, exact_joint_oracle
from stainroute.utils import canonical_json_sha256, sha256_file

from .stage1_runner import (
    DecodedAction,
    _assign_tiles,
    _base_prediction_with_cache,
    _bootstrap,
    _candidate_error_diagnostics,
    _decode_grouped_actions,
    _find_image,
    _json_config,
    _require_frozen_baseline,
    _safe_name,
    _subset_loader,
    _write_csv,
)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Futility config must be JSON-compatible YAML: {path}") from exc


def _require_monuseg_router_train(cfgs: Any) -> None:
    if Path(cfgs.data_path).name.lower() != "monuseg":
        raise ValueError("MoNuSeg futility gates require --data_path ./data/monuseg")
    if cfgs.stainroute_split != "router_train":
        raise ValueError("MoNuSeg futility gates are restricted to --stainroute_split router_train")
    if not cfgs.stainroute_split_manifest:
        raise ValueError("--stainroute_split_manifest is required")


def _filtered_loader(loader: DataLoader, cfgs: Any, image_ids: list[str]) -> DataLoader:
    wanted = set(image_ids)
    dataset = copy.copy(loader.dataset)
    dataset.paths = [path for path in dataset.paths if Path(path).stem in wanted]
    found = {Path(path).stem for path in dataset.paths}
    missing = sorted(wanted - found)
    if missing:
        raise FileNotFoundError(f"Futility image IDs absent from router_train: {missing}")
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=cfgs.num_workers, pin_memory=True)


def _cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _cuda_peak(device: torch.device) -> int | None:
    return int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _split_assembly_config(action_config: dict[str, Any]) -> SplitAssemblyConfig:
    return SplitAssemblyConfig(
        **{
            key: value
            for key, value in action_config["assembly"].items()
            if key in {"min_child_area", "min_parent_coverage", "max_raw_child_iou"}
        }
    )


def _decode_signature(action: DecodedAction, prediction: np.ndarray, split_config: SplitAssemblyConfig, min_added_area: int, gt: np.ndarray) -> dict[str, Any]:
    assembly = action.apply(prediction, split_config, min_added_area)
    utility = compute_action_utility(gt, prediction, assembly.prediction)
    return {
        "add_mask": action.add_mask,
        "add_logits": action.add_logits,
        "child_first": action.child_first,
        "child_second": action.child_second,
        "child_first_logits": action.child_first_logits,
        "child_second_logits": action.child_second_logits,
        "decoded_features": dict(action.candidate.decoded_features),
        "assembly_prediction": assembly.prediction,
        "assembly_applied": assembly.applied,
        "assembly_reason": assembly.reason,
        "utility": utility.as_dict(),
    }


def _max_array_error(first: np.ndarray | None, second: np.ndarray | None) -> float:
    if first is None and second is None:
        return 0.0
    if first is None or second is None or first.shape != second.shape:
        return float("inf")
    return float(np.max(np.abs(np.asarray(first, dtype=np.float32) - np.asarray(second, dtype=np.float32))))


def _decode_equivalence(
    *,
    actions: list[ActionCandidate],
    caches: dict,
    base: np.ndarray,
    gt: np.ndarray,
    ori_shape: Any,
    cfgs: Any,
    net: Any,
    device: torch.device,
    split_config: SplitAssemblyConfig,
    min_added_area: int,
) -> dict[str, Any]:
    """Compare existing tile prompt microbatching against single-action calls."""

    _cuda_sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    batch_profile: dict[str, float] = {}
    started = time.perf_counter()
    batch, _ = _decode_grouped_actions(
        actions, caches, ori_shape=ori_shape, cfgs=cfgs, net=net, device=device, profile=batch_profile
    )
    _cuda_sync(device)
    batch_seconds = time.perf_counter() - started
    batch_peak = _cuda_peak(device)

    _cuda_sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    single_profile: dict[str, float] = {}
    started = time.perf_counter()
    single: dict[str, DecodedAction] = {}
    for action in actions:
        item, _ = _decode_grouped_actions(
            [action], caches, ori_shape=ori_shape, cfgs=cfgs, net=net, device=device, profile=single_profile
        )
        single.update(item)
    _cuda_sync(device)
    single_seconds = time.perf_counter() - started
    single_peak = _cuda_peak(device)

    comparisons = []
    for action in actions:
        first = _decode_signature(batch[action.action_id], base, split_config, min_added_area, gt)
        second = _decode_signature(single[action.action_id], base, split_config, min_added_area, gt)
        logits_error = max(
            _max_array_error(first["add_logits"], second["add_logits"]),
            _max_array_error(first["child_first_logits"], second["child_first_logits"]),
            _max_array_error(first["child_second_logits"], second["child_second_logits"]),
        )
        iou_error = max(
            abs(float(first["decoded_features"].get("decoded_predicted_iou", 0.0)) - float(second["decoded_features"].get("decoded_predicted_iou", 0.0))),
            abs(float(first["decoded_features"].get("child_first_predicted_iou", 0.0)) - float(second["decoded_features"].get("child_first_predicted_iou", 0.0))),
            abs(float(first["decoded_features"].get("child_second_predicted_iou", 0.0)) - float(second["decoded_features"].get("child_second_predicted_iou", 0.0))),
        )
        comparisons.append(
            {
                "action_id": action.action_id,
                "action_type": action.action_type.value,
                "max_abs_logit_error": logits_error,
                "max_abs_predicted_iou_error": iou_error,
                "assembly_equal": bool(np.array_equal(first["assembly_prediction"], second["assembly_prediction"])),
                "assembly_reason_equal": first["assembly_reason"] == second["assembly_reason"],
                "utility_equal": first["utility"] == second["utility"],
            }
        )
    return {
        "actions": comparisons,
        "all_equivalent": bool(
            all(
                item["max_abs_logit_error"] <= 1.0e-6
                and item["max_abs_predicted_iou_error"] <= 1.0e-6
                and item["assembly_equal"]
                and item["assembly_reason_equal"]
                and item["utility_equal"]
                for item in comparisons
            )
        ),
        "microbatch": {
            "seconds": batch_seconds,
            "peak_memory_bytes": batch_peak,
            "actions_per_second": float(len(actions) / batch_seconds) if batch_seconds else None,
            **batch_profile,
        },
        "single": {
            "seconds": single_seconds,
            "peak_memory_bytes": single_peak,
            "actions_per_second": float(len(actions) / single_seconds) if single_seconds else None,
            **single_profile,
        },
    }


@dataclass(frozen=True)
class OptimisticOpportunity:
    action_id: str
    action_type: str
    cost: int
    source_action_id: str
    target_gt_ids: tuple[int, ...]
    parent_pred_id: int | None
    support_box: tuple[int, int, int, int] | None


def _merge_child_ids(gt: np.ndarray, prediction: np.ndarray, parent_id: int) -> tuple[int, int] | None:
    parent = prediction == parent_id
    ids, counts = np.unique(gt[parent], return_counts=True)
    ranked = sorted(
        ((int(count), int(gt_id)) for gt_id, count in zip(ids, counts) if int(gt_id) != 0),
        reverse=True,
    )
    return (ranked[0][1], ranked[1][1]) if len(ranked) >= 2 else None


def _optimistic_opportunities(gt: np.ndarray, base: np.ndarray, actions: list[ActionCandidate], diagnostics: dict[str, Any]) -> list[OptimisticOpportunity]:
    evaluation = evaluate_pq(gt, base)
    missed = set(diagnostics["_missed_gt_ids"])
    near = {gt_id for gt_id, _, iou in evaluation.matched_pairs if 0.5 <= iou < 0.6}
    eligible_add = missed | near
    output: list[OptimisticOpportunity] = []
    seen_add: set[int] = set()
    for action in sorted((item for item in actions if item.action_type is ActionType.ADD), key=lambda item: item.action_id):
        point = action.positive_points[0]
        gt_id = int(gt[point.y, point.x])
        if gt_id in eligible_add and gt_id not in seen_add:
            seen_add.add(gt_id)
            output.append(
                OptimisticOpportunity(
                    action_id=f"OPT:ADD:{gt_id}", action_type="ADD", cost=1,
                    source_action_id=action.action_id, target_gt_ids=(gt_id,), parent_pred_id=None,
                    support_box=action.support_box,
                )
            )
    merge_parents = set(diagnostics["_merge_parent_ids"])
    seen_parent: set[int] = set()
    for action in sorted((item for item in actions if item.action_type is ActionType.SPLIT), key=lambda item: item.action_id):
        parent_id = action.affected_instance_ids[0]
        child_ids = _merge_child_ids(gt, base, parent_id)
        if parent_id in merge_parents and child_ids is not None and parent_id not in seen_parent:
            seen_parent.add(parent_id)
            output.append(
                OptimisticOpportunity(
                    action_id=f"OPT:SPLIT:{parent_id}", action_type="SPLIT", cost=2,
                    source_action_id=action.action_id, target_gt_ids=child_ids, parent_pred_id=parent_id,
                    support_box=action.support_box,
                )
            )
    return output


def _apply_optimistic(base: np.ndarray, gt: np.ndarray, opportunity: OptimisticOpportunity) -> np.ndarray:
    if opportunity.action_type == "ADD":
        ideal = gt == opportunity.target_gt_ids[0]
        return apply_add_action(base, ideal, min_added_area=1).prediction
    assert opportunity.parent_pred_id is not None
    parent = base == opportunity.parent_pred_id
    first = parent & (gt == opportunity.target_gt_ids[0])
    second = parent & (gt == opportunity.target_gt_ids[1])
    # Ideal screening assumption: a decoder returns two valid child masks. It
    # is intentionally more permissive than the real SPLIT assembly checks.
    if not np.any(first) or not np.any(second):
        return base.copy()
    output = base.copy()
    output[parent] = 0
    next_id = int(output.max()) + 1
    output[first] = next_id
    output[second] = next_id + 1
    return output


def _opportunity_conflicts(opportunities: list[OptimisticOpportunity]) -> dict[str, set[str]]:
    graph: dict[str, set[str]] = {item.action_id: set() for item in opportunities}
    for index, first in enumerate(opportunities):
        for second in opportunities[index + 1 :]:
            same_parent = first.parent_pred_id is not None and first.parent_pred_id == second.parent_pred_id
            same_target = bool(set(first.target_gt_ids) & set(second.target_gt_ids))
            if same_parent or same_target:
                graph[first.action_id].add(second.action_id)
                graph[second.action_id].add(first.action_id)
    return graph


def _optimistic_ceiling(
    gt: np.ndarray,
    base: np.ndarray,
    opportunities: list[OptimisticOpportunity],
    *,
    budget: int,
    exact_max_opportunities: int,
) -> dict[str, Any]:
    """Full-PQ candidate-aware ideal-mask screening ceiling at one budget."""

    base_eval = evaluate_pq(gt, base)
    by_id = {item.action_id: item for item in opportunities}
    graph = _opportunity_conflicts(opportunities)
    cache: dict[tuple[str, ...], PQEvaluation] = {}

    def evaluate(ids: tuple[str, ...]) -> PQEvaluation:
        if ids not in cache:
            prediction = base.copy()
            for action_id in ids:
                prediction = _apply_optimistic(prediction, gt, by_id[action_id])
            cache[ids] = evaluate_pq(gt, prediction)
        return cache[ids]

    included = opportunities
    strategy = "exact"
    if len(opportunities) > exact_max_opportunities:
        # This is a labelled screening approximation. The prefilter uses only
        # GT-side optimistic single-action PQ, never a candidate-generator
        # feature, and is reported rather than presented as a decoder oracle.
        scored = []
        for opportunity in opportunities:
            value = evaluate((opportunity.action_id,)).pq - base_eval.pq
            scored.append((value, opportunity.action_id, opportunity))
        included = [item[2] for item in sorted(scored, key=lambda item: (-item[0], item[1]))[:exact_max_opportunities]]
        graph = _opportunity_conflicts(included)
        by_id = {item.action_id: item for item in included}
        strategy = "single_utility_screened_exact"

    # Reuse the exact subset implementation with schema-free local actions.
    # Its required attributes are action_id/action_cost only.
    @dataclass(frozen=True)
    class _SearchAction:
        action_id: str
        action_cost: int

    result = exact_joint_oracle(
        [_SearchAction(item.action_id, item.cost) for item in included],  # type: ignore[arg-type]
        budget=budget,
        conflict_graph=graph,
        evaluate_subset=evaluate,
    )
    return {
        "strategy": strategy,
        "candidate_opportunity_count": len(opportunities),
        "searched_opportunity_count": len(included),
        "selected_action_ids": list(result.action_ids),
        "selected_cost": result.cost,
        "base_pq": base_eval.pq,
        "ceiling_pq": result.evaluation.pq,
        "delta_pq": result.evaluation.pq - base_eval.pq,
        "delta_dq": result.evaluation.dq - base_eval.dq,
        "delta_sq": result.evaluation.sq - base_eval.sq,
    }


def _gate1_status(rows: list[dict[str, Any]], gate_config: dict[str, Any]) -> dict[str, Any]:
    audit = gate_config["candidate_audit"]
    missed_total = int(sum(row["missed_gt_count"] for row in rows))
    merge_total = int(sum(row["merge_parent_count"] for row in rows))
    add_hits = int(sum(row["add_candidate_hit_missed_gt"] for row in rows))
    split_hits = int(sum(row["split_candidate_hit_merge_parent"] for row in rows))
    add_recall = float(add_hits / missed_total) if missed_total else None
    split_recall = float(split_hits / merge_total) if merge_total else None
    joint = [float(row["optimistic_joint_delta_pq"]) for row in rows]
    add = [float(row["optimistic_add_delta_pq"]) for row in rows]
    opportunities = [int(row["optimistic_opportunity_count"]) for row in rows]
    total_opportunities = sum(opportunities)
    largest_fraction = float(max(opportunities) / total_opportunities) if total_opportunities else None
    reasons = []
    if float(np.mean(joint)) < 0.005:
        reasons.append("optimistic_joint_mean_delta_pq_below_0.005")
    if missed_total == 0 or (add_recall is not None and add_recall < 0.10):
        reasons.append("add_missed_gt_coverage_below_0.10_or_no_missed_gt")
    if merge_total == 0:
        reasons.append("no_merge_parent_opportunity")
    elif split_recall is not None and split_recall < 0.10 and float(np.mean(add)) < 0.005:
        reasons.append("split_merge_coverage_below_0.10_and_add_ceiling_insufficient")
    if largest_fraction is not None and largest_fraction >= float(audit["opportunity_concentration_threshold"]):
        reasons.append("correctable_opportunities_concentrated_in_one_image")
    return {
        "status": "NO_GO" if reasons else "PASS_TO_GATE_2",
        "no_go_reasons": reasons,
        "images": len(rows),
        "optimistic_joint_mean_delta_pq_b4": float(np.mean(joint)) if joint else None,
        "optimistic_add_mean_delta_pq_b4": float(np.mean(add)) if add else None,
        "add_missed_gt_coverage": add_recall,
        "split_merge_parent_coverage": split_recall,
        "largest_image_opportunity_fraction": largest_fraction,
        "total_opportunities": total_opportunities,
    }


def _run_runtime_profile(
    *, cfgs: Any, args: Any, loader: DataLoader, image_root: Path, gate_config: dict[str, Any],
    action_config: dict[str, Any], net: Any, point_net: Any, point_encoder: Any,
    texture_memory_bank_list: list, device: torch.device, out_dir: Path,
) -> None:
    image_id = gate_config["runtime_profile"]["image_id"]
    loader = _filtered_loader(loader, cfgs, [image_id])
    batch = next(iter(loader))
    image_tensor, inst_maps, _, _, _, _, ori_shape, _, name = batch
    if _safe_name(name) != image_id:
        raise RuntimeError(f"Fixed Gate 0 image mismatch: {_safe_name(name)} != {image_id}")
    base_started = time.perf_counter()
    base, caches, _, equivalence, counters = _base_prediction_with_cache(
        image_tensor=image_tensor, ori_shape=ori_shape, cfgs=cfgs, args=args, net=net,
        point_net=point_net, point_encoder=point_encoder,
        texture_memory_bank_list=texture_memory_bank_list, device=device, synchronize_for_timing=True,
    )
    _cuda_sync(device)
    base_seconds = time.perf_counter() - base_started
    candidate_started = time.perf_counter()
    raw = imread(_find_image(image_root, image_id))[..., :3]
    additions = generate_add_candidates(raw, base, image_id=image_id, config=AddCandidateConfig(**action_config["add"]))
    splits = generate_split_candidates(raw, base, image_id=image_id, config=SplitCandidateConfig(**action_config["split"]))
    candidate_seconds = time.perf_counter() - candidate_started
    limit = int(gate_config["runtime_profile"]["microbatch_actions_per_family"])
    selected = _assign_tiles(additions[:limit] + splits[:limit], caches)
    gt = np.asarray(inst_maps.numpy()[0]).astype(np.int32)
    split_config = _split_assembly_config(action_config)
    decode_result = _decode_equivalence(
        actions=selected, caches=caches, base=base, gt=gt, ori_shape=ori_shape, cfgs=cfgs, net=net,
        device=device, split_config=split_config, min_added_area=int(action_config["assembly"]["min_added_area"]),
    )
    assembly_started = time.perf_counter()
    batched, _ = _decode_grouped_actions(selected, caches, ori_shape=ori_shape, cfgs=cfgs, net=net, device=device)
    for item in batched.values():
        item.apply(base, split_config, int(action_config["assembly"]["min_added_area"]))
    assembly_seconds = time.perf_counter() - assembly_started

    oracle_started = time.perf_counter()
    base_eval = evaluate_pq(gt, base)
    by_id = batched

    def evaluate(ids: tuple[str, ...]) -> PQEvaluation:
        prediction = base.copy()
        for action_id in ids:
            prediction = by_id[action_id].apply(prediction, split_config, int(action_config["assembly"]["min_added_area"])).prediction
        return evaluate_pq(gt, prediction)

    @dataclass(frozen=True)
    class _SearchAction:
        action_id: str
        action_cost: int

    exact_result = exact_joint_oracle(
        [_SearchAction(item.candidate.action_id, item.candidate.action_cost) for item in batched.values()],  # type: ignore[arg-type]
        budget=4,
        conflict_graph={action_id: set() for action_id in batched},
        evaluate_subset=evaluate,
    )
    oracle_seconds = time.perf_counter() - oracle_started
    io_started = time.perf_counter()
    payload = {
        "protocol_config_sha256": sha256_file(cfgs.stainroute_futility_config),
        "image_id": image_id,
        "base_cached_decode_equivalence": equivalence,
        "image_encoder_calls": counters["encoder_calls"],
        "action_image_encoder_calls": 0,
        "base_decoder_actions": counters["base_decoder_actions"],
        "candidate_counts": {"ADD": len(additions), "SPLIT": len(splits), "microbatch_selected": len(selected)},
        "runtime_seconds": {"first_pass_total": base_seconds, "candidate_generation": candidate_seconds, "assembly": assembly_seconds, "exact_oracle": oracle_seconds, **counters},
        "microbatch_equivalence": decode_result,
        "exact_oracle_b4": {"base_pq": base_eval.pq, "pq": exact_result.evaluation.pq, "delta_pq": exact_result.evaluation.pq - base_eval.pq},
    }
    _write_json(out_dir / "runtime_profile.json", payload)
    payload["runtime_seconds"]["io"] = time.perf_counter() - io_started
    _write_json(out_dir / "runtime_profile.json", payload)
    print(json.dumps({"out_dir": str(out_dir), "image": image_id, "microbatch_equivalent": decode_result["all_equivalent"]}, indent=2))


def _run_candidate_audit(
    *, cfgs: Any, args: Any, loader: DataLoader, image_root: Path, gate_config: dict[str, Any],
    action_config: dict[str, Any], net: Any, point_net: Any, point_encoder: Any,
    texture_memory_bank_list: list, device: torch.device, out_dir: Path,
) -> None:
    feature_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    optimistic_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    cache_rows: list[dict[str, Any]] = []
    split_config = _split_assembly_config(action_config)
    audit_config = gate_config["candidate_audit"]
    for index, batch in enumerate(loader, start=1):
        image_tensor, inst_maps, _, _, _, _, ori_shape, _, name = batch
        image_id = _safe_name(name)
        print(f"[futility-audit] image={index}/{len(loader)} id={image_id}", flush=True)
        started = time.perf_counter()
        base, _, _, equivalence, counters = _base_prediction_with_cache(
            image_tensor=image_tensor, ori_shape=ori_shape, cfgs=cfgs, args=args, net=net,
            point_net=point_net, point_encoder=point_encoder,
            texture_memory_bank_list=texture_memory_bank_list, device=device,
        )
        candidate_started = time.perf_counter()
        raw = imread(_find_image(image_root, image_id))[..., :3]
        actions = generate_add_candidates(raw, base, image_id=image_id, config=AddCandidateConfig(**action_config["add"]))
        actions += generate_split_candidates(raw, base, image_id=image_id, config=SplitCandidateConfig(**action_config["split"]))
        candidate_seconds = time.perf_counter() - candidate_started
        # First GT access: generation is complete and no action decoder is called.
        gt = np.asarray(inst_maps.numpy()[0]).astype(np.int32)
        diagnostics = _candidate_error_diagnostics(gt, base, actions)
        opportunities = _optimistic_opportunities(gt, base, actions, diagnostics)
        add_opportunities = [item for item in opportunities if item.action_type == "ADD"]
        split_opportunities = [item for item in opportunities if item.action_type == "SPLIT"]
        joint = _optimistic_ceiling(gt, base, opportunities, budget=int(audit_config["optimistic_budget"]), exact_max_opportunities=int(audit_config["exact_max_opportunities"]))
        add_ceiling = _optimistic_ceiling(gt, base, add_opportunities, budget=int(audit_config["optimistic_budget"]), exact_max_opportunities=int(audit_config["exact_max_opportunities"]))
        split_ceiling = _optimistic_ceiling(gt, base, split_opportunities, budget=int(audit_config["optimistic_budget"]), exact_max_opportunities=int(audit_config["exact_max_opportunities"]))
        for action in actions:
            feature_rows.append(
                {
                    "image": image_id,
                    "action_id": action.action_id,
                    "action_type": action.action_type.value,
                    "action_cost": action.action_cost,
                    "positive_points": json.dumps([point.as_dict() for point in action.positive_points]),
                    "generation_features": json.dumps(action.generation_features, sort_keys=True),
                    "support_box": json.dumps(action.support_box),
                    "generator_version": action.generator_version,
                    "config_hash": action.config_hash,
                }
            )
        public_diagnostics = {key: value for key, value in diagnostics.items() if not key.startswith("_")}
        label_rows.append({"image": image_id, **public_diagnostics})
        optimistic_rows.append(
            {
                "image": image_id,
                "base_pq": evaluate_pq(gt, base).pq,
                "add_candidates": sum(item.action_type is ActionType.ADD for item in actions),
                "split_candidates": sum(item.action_type is ActionType.SPLIT for item in actions),
                "optimistic_opportunity_count": len(opportunities),
                "optimistic_add_opportunity_count": len(add_opportunities),
                "optimistic_split_opportunity_count": len(split_opportunities),
                "optimistic_add_delta_pq": add_ceiling["delta_pq"],
                "optimistic_split_delta_pq": split_ceiling["delta_pq"],
                "optimistic_joint_delta_pq": joint["delta_pq"],
                "optimistic_joint_strategy": joint["strategy"],
                **public_diagnostics,
            }
        )
        runtime_rows.append({"image": image_id, "candidate_generation_seconds": candidate_seconds, "elapsed_seconds": time.perf_counter() - started, **counters})
        cache_rows.append({"image": image_id, **equivalence})

    status = _gate1_status(optimistic_rows, gate_config)
    _write_csv(out_dir / "candidate_audit_features.csv", feature_rows)
    _write_csv(out_dir / "candidate_audit_labels.csv", label_rows)
    _write_csv(out_dir / "optimistic_ceiling_per_image.csv", optimistic_rows)
    _write_csv(out_dir / "runtime_summary.csv", runtime_rows)
    _write_json(out_dir / "cached_decode_equivalence.json", cache_rows)
    _write_json(
        out_dir / "candidate_audit_summary.json",
        {
            "protocol_config_sha256": sha256_file(cfgs.stainroute_futility_config),
            "split": "router_train",
            "images": len(optimistic_rows),
            "status": status,
            "optimistic_screening_definition": "Ideal GT ADD/SPLIT masks after GT-free candidate generation; not a decoder oracle.",
            "mean_optimistic_joint_delta_pq_b4": float(np.mean([row["optimistic_joint_delta_pq"] for row in optimistic_rows])) if optimistic_rows else None,
            "bootstrap_joint_delta_pq_b4": _bootstrap([float(row["optimistic_joint_delta_pq"]) for row in optimistic_rows], 2000, 3407),
        },
    )
    print(json.dumps({"out_dir": str(out_dir), "images": len(optimistic_rows), **status}, indent=2))


def run_monuseg_futility_gate(
    *, cfgs: Any, args: Any, test_dataset: Any, net: Any, point_net: Any, point_encoder: Any,
    texture_memory_bank_list: list, device: torch.device,
) -> None:
    """Dispatch the committed Gate 0 or Gate 1 implementation."""

    _require_monuseg_router_train(cfgs)
    if cfgs.stage1_monuseg_futility not in {"runtime_profile", "candidate_audit"}:
        raise ValueError("ADD pilot is locked until a committed Gate 1 pilot manifest exists")
    gate_config = _load_json(Path(cfgs.stainroute_futility_config))
    action_config = _json_config(Path(cfgs.stainroute_action_config))
    _require_frozen_baseline(cfgs, Path(cfgs.stainroute_split_manifest))
    loader, image_root, _ = _subset_loader(cfgs, test_dataset, Path(cfgs.stainroute_split_manifest), "router_train")
    out_dir = Path(cfgs.stainroute_out_dir or f"logs/stainroute/futility/{cfgs.stage1_monuseg_futility}")
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        out_dir / "manifest.json",
        {
            "stage": f"MoNuSeg futility {cfgs.stage1_monuseg_futility}",
            "git_sha": __import__("subprocess").check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "command": __import__("sys").argv,
            "split_manifest": cfgs.stainroute_split_manifest,
            "split_sha256": sha256_file(cfgs.stainroute_split_manifest),
            "futility_config": cfgs.stainroute_futility_config,
            "futility_config_sha256": sha256_file(cfgs.stainroute_futility_config),
            "action_config": cfgs.stainroute_action_config,
            "action_config_sha256": sha256_file(cfgs.stainroute_action_config),
            "gt_policy": "GT only after GT-free candidate generation; no action decoder in candidate_audit",
        },
    )
    net.eval()
    point_net.eval()
    point_encoder.eval()
    if cfgs.stage1_monuseg_futility == "runtime_profile":
        _run_runtime_profile(
            cfgs=cfgs, args=args, loader=loader, image_root=image_root, gate_config=gate_config,
            action_config=action_config, net=net, point_net=point_net, point_encoder=point_encoder,
            texture_memory_bank_list=texture_memory_bank_list, device=device, out_dir=out_dir,
        )
    else:
        _run_candidate_audit(
            cfgs=cfgs, args=args, loader=loader, image_root=image_root, gate_config=gate_config,
            action_config=action_config, net=net, point_net=point_net, point_encoder=point_encoder,
            texture_memory_bank_list=texture_memory_bank_list, device=device, out_dir=out_dir,
        )
