"""Prediction-only conflict graphs and the C4-CSR residual ranker.

C4 deliberately leaves C1 candidate generation, selected masks, native score
and the native assembler untouched.  The ranker emits only a *within-conflict
component* residual rank key.  At inference, the native score multiset is
permuted inside an eligible component according to that key; singleton nodes
and all score values outside the component remain native.

GT is accepted only by :func:`training_graph_with_pairs`, which creates
detached train-only pair labels.  Graph construction and score permutation do
not inspect GT, evaluator matching, patient metadata, or filenames.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

import numpy as np

from stainpms.c2_component_audit import selected_utility_labels
from stainpms.c3_score_control_audit import conflict_components
from stainpms.phase1_metrics import instance_ids
from stainpms.zero_training_oracle import ORACLE_MATCH_IOU

try:  # Local report-only environments need not have PyTorch installed.
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover - exercised on CPU report hosts
    torch = None
    nn = None


NODE_FEATURE_NAMES = (
    "native_assembly_score",
    "predicted_iou_quality",
    "log_mask_area",
    "edge_penalized",
    "conflict_degree",
    "native_relative_rank",
    "max_conflict_mask_iou",
    "mean_conflict_mask_iou",
)
EDGE_FEATURE_NAMES = (
    "pairwise_mask_iou",
    "intersection_over_min_area",
    "log_area_ratio_source_over_target",
    "normalized_centroid_distance",
)


def _mask_iou(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    first = np.asarray(left, dtype=bool)
    second = np.asarray(right, dtype=bool)
    intersection = int((first & second).sum())
    if not intersection:
        return 0.0, 0.0
    area_left, area_right = int(first.sum()), int(second.sum())
    union = area_left + area_right - intersection
    return float(intersection / union) if union else 0.0, float(intersection / min(area_left, area_right)) if min(area_left, area_right) else 0.0


def _centroid_from_bbox(record: dict[str, Any]) -> tuple[float, float]:
    x1, y1, x2, y2 = (float(value) for value in record["bbox_xyxy"])
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


def prediction_conflict_graph(
    records: list[dict[str, Any]],
    image_shape: tuple[int, int],
    *,
    instance_nms_iou: float,
) -> dict[str, Any]:
    """Build the deployable C4 graph from C1 prediction records only."""

    if not records:
        return {
            "node_features_raw": np.zeros((0, len(NODE_FEATURE_NAMES)), dtype=np.float32),
            "edge_index": np.zeros((2, 0), dtype=np.int64),
            "edge_features_raw": np.zeros((0, len(EDGE_FEATURE_NAMES)), dtype=np.float32),
            "components": [],
            "component_for_index": {},
            "edge_reason_counts": {},
            "edge_count": 0,
            "non_singleton_mask": np.zeros((0,), dtype=bool),
            "records": [],
        }
    graph = conflict_components(records, nms_iou=float(instance_nms_iou))
    count = len(records)
    degrees = np.zeros(count, dtype=np.float32)
    neighbor_ious: dict[int, list[float]] = defaultdict(list)
    directed_edges: list[tuple[int, int]] = []
    edge_features: list[list[float]] = []
    height, width = (int(image_shape[0]), int(image_shape[1]))
    diagonal = max(float(np.hypot(height, width)), 1.0)
    for edge in graph["edges"]:
        left, right = int(edge["left"]), int(edge["right"])
        mask_iou, overlap_min = _mask_iou(records[left]["mask"], records[right]["mask"])
        area_left = max(int(np.asarray(records[left]["mask"], dtype=bool).sum()), 1)
        area_right = max(int(np.asarray(records[right]["mask"], dtype=bool).sum()), 1)
        left_center, right_center = _centroid_from_bbox(records[left]), _centroid_from_bbox(records[right])
        distance = float(np.hypot(left_center[0] - right_center[0], left_center[1] - right_center[1]) / diagonal)
        degrees[left] += 1.0
        degrees[right] += 1.0
        neighbor_ious[left].append(mask_iou)
        neighbor_ious[right].append(mask_iou)
        directed_edges.extend(((left, right), (right, left)))
        edge_features.extend(
            (
                [mask_iou, overlap_min, float(np.log(area_left / area_right)), distance],
                [mask_iou, overlap_min, float(np.log(area_right / area_left)), distance],
            )
        )
    relative_rank = np.zeros(count, dtype=np.float32)
    non_singleton = np.zeros(count, dtype=bool)
    for component in graph["components"]:
        if len(component) <= 1:
            continue
        non_singleton[np.asarray(component, dtype=np.int64)] = True
        ordered = sorted(
            component,
            key=lambda index: (-float(records[index]["assembly_score"]), int(records[index]["record_index"])),
        )
        denominator = max(len(ordered) - 1, 1)
        for rank, index in enumerate(ordered):
            relative_rank[index] = float(rank / denominator)
    node_features: list[list[float]] = []
    for index, record in enumerate(records):
        area = max(int(np.asarray(record["mask"], dtype=bool).sum()), 1)
        overlaps = neighbor_ious.get(index, [])
        node_features.append(
            [
                float(record["assembly_score"]),
                float(record["quality"]),
                float(np.log1p(area)),
                float(bool(record.get("edge_penalized", False))),
                float(degrees[index]),
                float(relative_rank[index]),
                float(max(overlaps)) if overlaps else 0.0,
                float(np.mean(overlaps)) if overlaps else 0.0,
            ]
        )
    return {
        # A shallow copy is intentional: the deployable graph needs only the
        # same prediction records that native assembly already consumes.  It
        # contains no evaluator matching or GT-derived labels.
        "records": [dict(row) for row in records],
        "node_features_raw": np.asarray(node_features, dtype=np.float32),
        "edge_index": np.asarray(directed_edges, dtype=np.int64).T if directed_edges else np.zeros((2, 0), dtype=np.int64),
        "edge_features_raw": np.asarray(edge_features, dtype=np.float32) if edge_features else np.zeros((0, len(EDGE_FEATURE_NAMES)), dtype=np.float32),
        "components": [list(component) for component in graph["components"]],
        "component_for_index": dict(graph["component_for_index"]),
        "edge_reason_counts": dict(graph["edge_reason_counts"]),
        "edge_count": int(graph["edge_count"]),
        "non_singleton_mask": non_singleton,
    }


def training_graph_with_pairs(
    records: list[dict[str, Any]],
    gt_map: np.ndarray,
    *,
    instance_nms_iou: float,
    merge_risk_overlap_fraction: float = 0.1,
) -> dict[str, Any]:
    """Attach detached C3 labels and C4 component-balanced pair sets."""

    labelled = selected_utility_labels(
        records,
        gt_map,
        match_iou=ORACLE_MATCH_IOU,
        merge_risk_overlap_fraction=merge_risk_overlap_fraction,
    )
    graph = prediction_conflict_graph(labelled, tuple(np.asarray(gt_map).shape), instance_nms_iou=instance_nms_iou)
    component_pairs: list[dict[str, Any]] = []
    for component_id, component in enumerate(graph["components"]):
        if len(component) <= 1:
            continue
        positives = [index for index in component if labelled[index]["utility_label"] == "unique_tp"]
        negatives = [
            index
            for index in component
            if labelled[index]["utility_label"] in {"unmatched_fp", "duplicate"}
        ]
        if positives and negatives:
            component_pairs.append(
                {
                    "component_id": int(component_id),
                    "positive_indices": positives,
                    "negative_indices": negatives,
                    "pair_count": int(len(positives) * len(negatives)),
                }
            )
    graph.update(
        {
            "records": labelled,
            "component_pairs": component_pairs,
            "pair_counts": {
                "component_count_with_pairs": len(component_pairs),
                "pair_count": int(sum(item["pair_count"] for item in component_pairs)),
                "unique_tp": int(sum(row["utility_label"] == "unique_tp" for row in labelled)),
                "unmatched_fp": int(sum(row["utility_label"] == "unmatched_fp" for row in labelled)),
                "duplicate": int(sum(row["utility_label"] == "duplicate" for row in labelled)),
            },
        }
    )
    return graph


def fit_feature_normalizer(graphs: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Fit all feature normalization from p1--6 graphs only."""

    graph_list = list(graphs)
    nodes = [graph["node_features_raw"] for graph in graph_list if len(graph["node_features_raw"])]
    edges = [graph["edge_features_raw"] for graph in graph_list if len(graph["edge_features_raw"])]
    if not nodes:
        raise ValueError("C4 feature normalization requires at least one prediction node")

    def stats(array: np.ndarray, names: tuple[str, ...]) -> dict[str, Any]:
        mean = array.mean(axis=0)
        std = array.std(axis=0)
        std = np.where(std < 1.0e-6, 1.0, std)
        return {
            "names": list(names),
            "mean": [float(value) for value in mean],
            "std": [float(value) for value in std],
            "sample_count": int(array.shape[0]),
        }

    return {
        "method": "z_score_population_std_with_std_floor_1",
        "fit_scope": "TNBC p1-p6 C1 frozen selected predictions only",
        "node": stats(np.concatenate(nodes, axis=0), NODE_FEATURE_NAMES),
        "edge": stats(np.concatenate(edges, axis=0) if edges else np.zeros((1, len(EDGE_FEATURE_NAMES)), dtype=np.float32), EDGE_FEATURE_NAMES),
    }


def normalize_graph(graph: dict[str, Any], normalizer: dict[str, Any]) -> dict[str, Any]:
    node_stats, edge_stats = normalizer["node"], normalizer["edge"]
    node = (graph["node_features_raw"] - np.asarray(node_stats["mean"], dtype=np.float32)) / np.asarray(node_stats["std"], dtype=np.float32)
    edge = (graph["edge_features_raw"] - np.asarray(edge_stats["mean"], dtype=np.float32)) / np.asarray(edge_stats["std"], dtype=np.float32)
    return {**graph, "node_features": node.astype(np.float32), "edge_features": edge.astype(np.float32)}


if nn is not None:

    class ConflictSetResidualRanker(nn.Module):
        """Two-layer width-64 node encoder with one mean/max relation pass."""

        def __init__(self, node_dim: int = len(NODE_FEATURE_NAMES), edge_dim: int = len(EDGE_FEATURE_NAMES), width: int = 64):
            super().__init__()
            self.node_mlp = nn.Sequential(nn.Linear(node_dim, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU())
            self.edge_mlp = nn.Sequential(nn.Linear(edge_dim, width), nn.ReLU())
            self.output = nn.Sequential(nn.Linear(width * 3, width), nn.ReLU(), nn.Linear(width, 1))
            nn.init.zeros_(self.output[-1].weight)
            nn.init.zeros_(self.output[-1].bias)

        def forward(self, node_features, edge_index, edge_features):
            hidden = self.node_mlp(node_features)
            count = int(hidden.shape[0])
            if int(edge_index.shape[1]) == 0:
                mean = torch.zeros_like(hidden)
                maximum = torch.zeros_like(hidden)
            else:
                source, destination = edge_index[0].long(), edge_index[1].long()
                messages = hidden[source] + self.edge_mlp(edge_features)
                mean = torch.zeros_like(hidden)
                mean.index_add_(0, destination, messages)
                degrees = torch.zeros((count, 1), dtype=hidden.dtype, device=hidden.device)
                degrees.index_add_(0, destination, torch.ones((messages.shape[0], 1), dtype=hidden.dtype, device=hidden.device))
                mean = mean / degrees.clamp_min(1.0)
                maximum = torch.zeros_like(hidden)
                for node in range(count):
                    selected = messages[destination == node]
                    if int(selected.shape[0]):
                        maximum[node] = selected.max(dim=0).values
            return self.output(torch.cat((hidden, mean, maximum), dim=1)).squeeze(1)


def build_ranker(*, width: int = 64):
    if torch is None or nn is None:
        raise RuntimeError("C4 ranker construction requires PyTorch")
    ranker = ConflictSetResidualRanker(width=width)
    parameter_count = int(sum(parameter.numel() for parameter in ranker.parameters()))
    if parameter_count > 100_000:
        raise ValueError(f"C4 ranker exceeds 100k parameters: {parameter_count}")
    return ranker, parameter_count


def residual_rank_keys(ranker, graph: dict[str, Any], *, device) -> np.ndarray:
    """Return deployment-only native-plus-residual rank keys, with singleton delta zero."""

    if torch is None:
        raise RuntimeError("C4 inference requires PyTorch")
    ranker.eval()
    with torch.no_grad():
        nodes = torch.as_tensor(graph["node_features"], dtype=torch.float32, device=device)
        edge_index = torch.as_tensor(graph["edge_index"], dtype=torch.long, device=device)
        edges = torch.as_tensor(graph["edge_features"], dtype=torch.float32, device=device)
        delta = ranker(nodes, edge_index, edges).detach().float().cpu().numpy()
    delta = np.asarray(delta, dtype=np.float64)
    delta[~np.asarray(graph["non_singleton_mask"], dtype=bool)] = 0.0
    native = np.asarray([float(row["assembly_score"]) for row in graph["records"]], dtype=np.float64)
    return native + delta


def _assemble_prediction_map(records: list[dict[str, Any]], scores: list[float], image_shape: tuple[int, int], *, instance_nms_iou: float) -> np.ndarray:
    from run.run_on_epoch import _assemble_instance_map

    return _assemble_instance_map(
        [row["bbox_xyxy"] for row in records],
        scores,
        [row["mask"] for row in records],
        [int(row["prompt_group_id"]) for row in records],
        image_shape,
        float(instance_nms_iou),
    )


def _component_score_permutation(
    scores: list[float],
    rank_keys: np.ndarray,
    component: list[int],
    records: list[dict[str, Any]],
) -> list[float]:
    """Permute only a component's native score values using a stable rank key."""

    output = [float(value) for value in scores]
    native_values = sorted(float(scores[index]) for index in component)
    ranked = sorted(
        component,
        key=lambda index: (
            float(rank_keys[index]),
            float(scores[index]),
            int(records[index]["record_index"]),
        ),
    )
    for value, index in zip(native_values, ranked, strict=True):
        output[index] = value
    return output


def prediction_only_ranked_assembly(
    records: list[dict[str, Any]],
    graph: dict[str, Any],
    rank_keys: np.ndarray,
    image_shape: tuple[int, int],
    *,
    instance_nms_iou: float,
) -> dict[str, Any]:
    """Run C4's deployable score-only conflict ordering.

    Each proposed component permutation is accepted only when the unchanged
    native assembler emits the same *predicted* final-instance count.  This
    uses no GT and is the prediction-only counterpart of C3's frozen
    conflict-order semantics.
    """

    native_scores = [float(row["assembly_score"]) for row in records]
    native_map = _assemble_prediction_map(records, native_scores, image_shape, instance_nms_iou=instance_nms_iou)
    native_count = len(instance_ids(native_map))
    current = list(native_scores)
    eligible = accepted = rejected = 0
    for component in graph["components"]:
        if len(component) <= 1:
            continue
        eligible += 1
        candidate = _component_score_permutation(current, rank_keys, component, records)
        if candidate == current:
            accepted += 1
            continue
        candidate_map = _assemble_prediction_map(records, candidate, image_shape, instance_nms_iou=instance_nms_iou)
        if len(instance_ids(candidate_map)) == native_count:
            current = candidate
            accepted += 1
        else:
            rejected += 1
    final_map = _assemble_prediction_map(records, current, image_shape, instance_nms_iou=instance_nms_iou)
    return {
        "native_scores": native_scores,
        "rank_keys": [float(value) for value in rank_keys],
        "assembly_scores": current,
        "native_final_map": native_map,
        "final_map": final_map,
        "native_final_instance_count": int(native_count),
        "final_instance_count": int(len(instance_ids(final_map))),
        "eligible_non_singleton_component_count": int(eligible),
        "accepted_component_permutation_count": int(accepted),
        "rejected_for_final_count_change": int(rejected),
    }
