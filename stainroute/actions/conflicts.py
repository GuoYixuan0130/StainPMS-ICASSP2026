"""Deterministic action conflict graph construction."""

from __future__ import annotations

from collections import defaultdict

from .schema import ActionCandidate


def _box_iou(first: tuple[int, int, int, int] | None, second: tuple[int, int, int, int] | None) -> float:
    if first is None or second is None:
        return 0.0
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = (first[2] - first[0]) * (first[3] - first[1]) + (second[2] - second[0]) * (second[3] - second[1]) - intersection
    return float(intersection / union) if union > 0 else 0.0


def build_conflict_graph(actions: list[ActionCandidate], support_iou_threshold: float = 0.5) -> dict[str, set[str]]:
    """Build symmetric conflicts from parent identity and support-box overlap."""

    graph: dict[str, set[str]] = defaultdict(set)
    ordered = sorted(actions, key=lambda action: action.action_id)
    for index, first in enumerate(ordered):
        graph.setdefault(first.action_id, set())
        for second in ordered[index + 1 :]:
            same_parent = bool(set(first.affected_instance_ids) & set(second.affected_instance_ids))
            support_conflict = _box_iou(first.support_box, second.support_box) >= support_iou_threshold
            declared = second.action_id in first.conflict_ids or first.action_id in second.conflict_ids
            if same_parent or support_conflict or declared:
                graph[first.action_id].add(second.action_id)
                graph[second.action_id].add(first.action_id)
    return {key: set(value) for key, value in graph.items()}
