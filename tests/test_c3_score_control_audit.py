from __future__ import annotations

import unittest

import numpy as np

from stainpms.c3_score_control_audit import (
    _demote_scores,
    _permute_component_scores,
    conflict_components,
    duplicate_competition_pairs,
    summarize_conflicts,
)


def row(index, group, mask, bbox, score, label, *, matched=None, eligible=None, merge=False):
    return {
        "record_index": index,
        "prompt_group_id": group,
        "mask": np.asarray(mask, dtype=bool),
        "bbox_xyxy": list(bbox),
        "assembly_score": float(score),
        "utility_label": label,
        "utility_target": 1.0 if label == "unique_tp" else 0.0,
        "matched_gt_instance_id": matched,
        "gt_ious": {str(value): 0.75 for value in (eligible or [])},
        "merge_risk": bool(merge),
    }


class C3ScoreControlAuditTests(unittest.TestCase):
    def setUp(self):
        self.left = np.zeros((12, 12), dtype=bool)
        self.left[1:5, 1:5] = True
        self.right = np.zeros((12, 12), dtype=bool)
        self.right[1:5, 5:9] = True
        self.overlap = self.left.copy()
        self.overlap[3:6, 4:7] = True

    def test_conflict_components_follow_native_group_box_and_mask_paths(self):
        records = [
            row(0, 10, self.left, [1, 1, 5, 5], 0.9, "unique_tp", matched=1, eligible=[1]),
            row(1, 10, self.right, [5, 1, 9, 5], 0.8, "unmatched_fp"),  # prompt-group edge
            row(2, 12, self.overlap, [3, 3, 7, 7], 0.7, "unmatched_fp"),  # mask edge to 0
            row(3, 13, self.right, [5, 1, 9, 5], 0.6, "unmatched_fp"),  # NMS-box edge to 1
        ]
        graph = conflict_components(records, nms_iou=0.5)
        self.assertEqual(graph["components"], [[0, 1, 2, 3]])
        self.assertEqual(graph["edge_reason_counts"]["prompt_group"], 1)
        self.assertGreaterEqual(graph["edge_reason_counts"]["paint_mask_overlap"], 1)
        self.assertGreaterEqual(graph["edge_reason_counts"]["nms_box_iou"], 1)

    def test_duplicate_pairs_require_same_gt_and_true_conflict_component(self):
        records = [
            row(0, 10, self.left, [1, 1, 5, 5], 0.7, "unique_tp", matched=1, eligible=[1]),
            row(1, 11, self.left, [1, 1, 5, 5], 0.8, "duplicate", eligible=[1]),
            row(2, 12, self.right, [5, 1, 9, 5], 0.6, "duplicate", eligible=[1]),
        ]
        graph = conflict_components(records, nms_iou=0.5)
        self.assertEqual(duplicate_competition_pairs(records, graph["component_for_index"]), [(0, 1)])

    def test_score_demotion_keeps_non_targets_unchanged_and_targets_below_them(self):
        native = [0.2, 0.8, 0.4, 0.7]
        changed = _demote_scores(native, [1, 3])
        self.assertEqual(changed[0], native[0])
        self.assertEqual(changed[2], native[2])
        self.assertLess(max(changed[1], changed[3]), min(native[0], native[2]))

    def test_component_permutation_preserves_score_multiset_and_prioritizes_unique_tp(self):
        native = [0.2, 0.8, 0.4]
        changed = _permute_component_scores(
            native,
            [[0, 1, 2]],
            priority=lambda index: {0: 2, 1: 0, 2: 1}[index],
            apply_component=lambda _: True,
        )
        self.assertEqual(sorted(changed), sorted(native))
        self.assertGreater(changed[0], changed[1])

    def test_conflict_summary_reports_ordering_and_score_margins(self):
        records = [
            row(0, 10, self.left, [1, 1, 5, 5], 0.9, "unique_tp", matched=1, eligible=[1]),
            row(1, 11, self.left, [1, 1, 5, 5], 0.8, "duplicate", eligible=[1]),
            row(2, 12, self.right, [5, 1, 9, 5], 0.1, "unmatched_fp"),
        ]
        graph = conflict_components(records, nms_iou=0.5)
        summary = summarize_conflicts(records, graph)
        self.assertEqual(summary["duplicate_count"], 1)
        self.assertEqual(summary["unique_tp_native_top1"]["accuracy"], 1.0)
        self.assertEqual(summary["pairwise_ordering"]["duplicate"]["accuracy"], 1.0)
        self.assertGreater(summary["pairwise_ordering"]["duplicate"]["positive_minus_negative_margin"]["mean"], 0.0)


if __name__ == "__main__":
    unittest.main()
