from __future__ import annotations

import unittest

import numpy as np

from promptcredit.matching import collision_groups, hungarian_assignment, nearest_assignment


class PromptCreditAssignmentTest(unittest.TestCase):
    def test_nearest_collision_toy_case(self) -> None:
        proposals = np.asarray([[0.0, 0.0], [10.0, 0.0]])
        gt = np.asarray([[0.3, 0.0], [0.7, 0.0]])
        nearest = nearest_assignment(proposals, gt)
        self.assertEqual(nearest.source_for_gt.tolist(), [0, 0])
        self.assertEqual(collision_groups(nearest.source_for_gt), {0: [0, 1]})

    def test_hungarian_is_one_to_one(self) -> None:
        proposals = np.asarray([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
        gt = np.asarray([[0.3, 0.0], [0.7, 0.0]])
        result = hungarian_assignment(proposals, gt, foreground_probability=np.asarray([0.9, 0.9, 0.1]))
        self.assertEqual(len(set(result.source_for_gt.tolist())), 2)
        self.assertEqual(set(result.source_for_gt.tolist()), {0, 1})

    def test_gt_permutation_preserves_match_semantics(self) -> None:
        proposals = np.asarray([[0.0, 0.0], [10.0, 0.0], [50.0, 0.0]])
        gt = np.asarray([[0.4, 0.0], [9.6, 0.0]])
        probability = np.asarray([0.8, 0.7, 0.1])
        original = hungarian_assignment(proposals, gt, probability)
        permutation = np.asarray([1, 0])
        permuted = hungarian_assignment(proposals, gt[permutation], probability)
        original_by_centroid = {tuple(gt[index]): int(source) for index, source in enumerate(original.source_for_gt)}
        permuted_by_centroid = {
            tuple(gt[permutation[index]]): int(source) for index, source in enumerate(permuted.source_for_gt)
        }
        self.assertEqual(original_by_centroid, permuted_by_centroid)


if __name__ == "__main__":
    unittest.main()

