from __future__ import annotations

import unittest

import numpy as np

from stainpms.c4_csr import (
    EDGE_FEATURE_NAMES,
    NODE_FEATURE_NAMES,
    _component_score_permutation,
    prediction_conflict_graph,
    training_graph_with_pairs,
)
from stainpms.zero_training_oracle import annotate_pool_ious


def record(index, group, mask, score):
    ys, xs = np.where(mask)
    return {
        "record_index": index,
        "prompt_group_id": group,
        "token": 0,
        "crop_index": 0,
        "mask": np.asarray(mask, dtype=bool),
        "bbox_xyxy": [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)],
        "quality": float(score),
        "assembly_score": float(score),
        "edge_penalized": False,
    }


class C4CSRGraphTests(unittest.TestCase):
    def setUp(self):
        self.gt = np.zeros((12, 12), dtype=np.int32)
        self.gt[1:5, 1:5] = 1
        self.gt[1:5, 6:10] = 2
        self.first = self.gt == 1
        self.duplicate = self.gt == 1
        self.second = self.gt == 2

    def test_prediction_graph_uses_no_gt_and_has_only_deployment_features(self):
        records = [record(0, 10, self.first, 0.7), record(1, 11, self.duplicate, 0.8), record(2, 12, self.second, 0.5)]
        graph = prediction_conflict_graph(records, self.gt.shape, instance_nms_iou=0.5)
        self.assertEqual(graph["node_features_raw"].shape, (3, len(NODE_FEATURE_NAMES)))
        self.assertEqual(graph["edge_features_raw"].shape[1], len(EDGE_FEATURE_NAMES))
        self.assertTrue(graph["non_singleton_mask"][0])
        self.assertTrue(graph["non_singleton_mask"][1])
        self.assertFalse(graph["non_singleton_mask"][2])
        self.assertNotIn("gt", repr(graph).lower())

    def test_training_pairs_are_only_unique_tp_over_conflicting_duplicate_or_fp(self):
        records = annotate_pool_ious(
            [record(0, 10, self.first, 0.7), record(1, 11, self.duplicate, 0.8), record(2, 12, self.second, 0.5)],
            self.gt,
        )
        graph = training_graph_with_pairs(records, self.gt, instance_nms_iou=0.5)
        self.assertEqual(graph["pair_counts"]["component_count_with_pairs"], 1)
        pair = graph["component_pairs"][0]
        self.assertEqual(pair["positive_indices"], [0])
        self.assertEqual(pair["negative_indices"], [1])

    def test_zero_residual_rank_key_preserves_every_component_score(self):
        records = [record(0, 10, self.first, 0.7), record(1, 11, self.duplicate, 0.8)]
        scores = [0.7, 0.8]
        permuted = _component_score_permutation(scores, np.asarray(scores), [0, 1], records)
        self.assertEqual(permuted, scores)


if __name__ == "__main__":
    unittest.main()
