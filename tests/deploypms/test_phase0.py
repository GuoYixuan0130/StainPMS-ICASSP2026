import unittest

import numpy as np
import torch

from deploypms.phase0 import (
    GateResult,
    assembly_gate,
    assess_associations,
    availability_gate,
    conditioning_gate,
    final_verdict,
    find_nearest_points_with_indices,
    point_nms_indices,
)


class DeployPMSPhase0Test(unittest.TestCase):
    def test_nearest_prompt_matches_training_rule(self):
        predicted = torch.tensor([[2.0, 2.0], [8.0, 8.0], [4.0, 5.0]])
        selected = torch.tensor([[3.0, 4.0], [9.0, 9.0]])
        points, indices = find_nearest_points_with_indices(predicted, selected)
        self.assertEqual(indices.tolist(), [2, 1])
        self.assertEqual(points.tolist(), [[4.0, 5.0], [8.0, 8.0]])

    def test_point_nms_keeps_high_score_and_preserves_distant_point(self):
        points = np.asarray([[1.0, 1.0], [2.0, 1.0], [20.0, 20.0]])
        kept = point_nms_indices(points, np.asarray([0.7, 0.9, 0.8]), 2)
        self.assertEqual(kept.tolist(), [1, 2])

    def test_association_has_no_mask_iou_selection_and_keeps_duplicates(self):
        labels = np.zeros((8, 8), dtype=np.int32)
        labels[1:4, 1:4] = 1
        labels[4:7, 4:7] = 2
        teachers = {
            1: {"spatial_gt_id": 1, "point": [2, 2]},
            2: {"spatial_gt_id": 0, "point": [0, 0]},
        }
        deployment = [
            {"point_id": 5, "point": [2, 2], "point_score": 0.8},
            {"point_id": 6, "point": [2, 3], "point_score": 0.9},
            {"point_id": 7, "point": [5, 5], "point_score": 0.7},
            {"point_id": 8, "point": [0, 0], "point_score": 0.6},
        ]
        rows, extras = assess_associations(labels, teachers, deployment)
        self.assertEqual([row["category"] for row in rows], ["both-covered", "deployment-only"])
        self.assertEqual(rows[0]["deployment_primary"]["point_id"], 6)
        self.assertEqual(rows[0]["duplicate_count"], 1)
        self.assertEqual({row["association"] for row in extras}, {"duplicate", "background_unmatched"})

    def test_preregistered_gates_and_verdict(self):
        instances = []
        for image in range(7):
            instances.extend([
                {"image": str(image), "teacher_present": True, "deployment_present": True, "category": "both-covered"},
                {"image": str(image), "teacher_present": True, "deployment_present": False, "category": "teacher-only"},
                {"image": str(image), "teacher_present": False, "deployment_present": False, "category": "neither"},
            ])
        availability = availability_gate(instances)
        self.assertTrue(availability.passed)
        both = [
            {"image": str(index % 7), "hard_iou_gap": 0.02}
            for index in range(70)
        ]
        conditioning = conditioning_gate(both)
        self.assertTrue(conditioning.passed)
        standard = [{"dice": .7, "dice2": .7, "aji": .5, "aji_plus": .5, "dq": .6, "sq": .7, "pq": .42, "tp": 10, "fp": 2, "fn": 2} for _ in range(7)]
        swap = [{**row, "aji": .5, "pq": .425} for row in standard]
        assembly = assembly_gate(standard, swap)
        self.assertTrue(assembly.passed)
        self.assertEqual(final_verdict(availability, conditioning, assembly), "STRONG GO")


if __name__ == "__main__":
    unittest.main()
