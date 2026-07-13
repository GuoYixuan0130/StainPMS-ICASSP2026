"""CPU-testable static-cache selection rules for Anchored SemiPMS Stage 1B."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Mapping, Sequence

import numpy as np

from semipms.residual import frozen_accept


PROPOSAL_BUDGETS = (8, 16, 32, 64)
VIEW_IOU_GRID = tuple(round(value, 2) for value in np.arange(0.35, 0.91, 0.05))
TARGET_CALIBRATION_PRECISION = 0.90
VIEW_MATCH_IOU = 0.50


def _mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    union = int(np.logical_or(left, right).sum())
    return float(np.logical_and(left, right).sum() / union) if union else 0.0


def one_to_one_cross_view(rows: Sequence[Mapping[str, Any]], rule: Mapping[str, float], component_ids: np.ndarray) -> tuple[list[dict[str, Any]], Counter]:
    """Accept at most one original/stain/geometry mask per matched view set."""
    accepted_views: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    component_seen: set[int] = set()
    stats: Counter = Counter()
    out: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: (-float(item["evidence"]), int(item["candidate_index"]))):
        item = dict(row)
        y = min(max(int(round(float(item["y"]))), 0), component_ids.shape[0] - 1)
        x = min(max(int(round(float(item["x"]))), 0), component_ids.shape[1] - 1)
        component = int(component_ids[y, x]); item["h_component"] = component
        item["cross_view_accepted"] = bool(frozen_accept(item["features"], dict(rule)))
        if not item["cross_view_accepted"]:
            item["status"] = "cross_view_rejected"; stats[item["status"]] += 1
        elif component and component in component_seen:
            item["status"] = "same_h_component_duplicate"; stats[item["status"]] += 1
        else:
            triple = (np.asarray(item["mask"], bool), np.asarray(item["stain_mask"], bool), np.asarray(item["geometry_mask"], bool))
            duplicate = any(all(_mask_iou(left, right) >= VIEW_MATCH_IOU for left, right in zip(triple, existing)) for existing in accepted_views)
            if duplicate:
                item["status"] = "cross_view_not_one_to_one"; stats[item["status"]] += 1
            else:
                accepted_views.append(triple)
                if component:
                    component_seen.add(component)
                item["status"] = "cross_view_matched"; stats[item["status"]] += 1
        out.append(item)
    return out, stats


def select_rule_lopo(rows: Sequence[Mapping[str, Any]], base_rule: Mapping[str, float]) -> tuple[dict[str, float], int, list[dict[str, Any]]]:
    """LOPO: meet ~90% candidate precision first, then maximise recall."""
    folds = []
    for held_out in range(1, 7):
        train = [row for row in rows if int(row["patient"]) != held_out]
        trials = []
        for threshold in VIEW_IOU_GRID:
            rule = dict(base_rule, min_view_iou=float(threshold))
            for budget in PROPOSAL_BUDGETS:
                subset = []
                for image in sorted({str(row["image"]) for row in train}):
                    image_rows = [row for row in train if row["image"] == image and frozen_accept(row["features"], rule)]
                    subset.extend(sorted(image_rows, key=lambda row: -float(row["evidence"]))[:budget])
                positives = sum(bool(row["is_true"]) for row in subset)
                precision = positives / len(subset) if subset else 0.0
                recall = positives / max(1, sum(bool(row["is_true"]) for row in train))
                met = precision >= TARGET_CALIBRATION_PRECISION
                trials.append((met, recall if met else precision, precision if met else recall, -threshold, -budget, threshold, budget, len(subset)))
        if not trials:
            raise AssertionError("LOPO calibration had no candidate trials.")
        met, first, second, _, _, threshold, budget, count = max(trials)
        precision, recall = (second, first) if met else (first, second)
        folds.append({"held_out_patient": held_out, "target_precision_met": bool(met), "train_precision": precision, "train_recall": recall, "min_view_iou": threshold, "proposal_budget": budget, "selected_count": count})
    rule = dict(base_rule, min_view_iou=float(np.median([row["min_view_iou"] for row in folds])))
    budget = int(np.median([row["proposal_budget"] for row in folds]))
    return rule, budget, folds
