import unittest

import numpy as np

from semipms.stage1b_protocol import one_to_one_cross_view, select_rule_lopo


RULE = {
    "min_view_iou": 0.35,
    "max_centroid_displacement": 6.0,
    "min_area_stability": 0.45,
    "min_h_occupancy": 0.1,
    "min_boundary_stability": 0.25,
    "max_pseudo_conflict": 0.35,
}


def _features(view_iou=0.9):
    return {
        "stain_inverse_iou": view_iou,
        "geometric_inverse_iou": view_iou,
        "centroid_displacement": 0.0,
        "area_stability": 1.0,
        "h_occupancy": 1.0,
        "boundary_stability": 1.0,
        "pseudo_conflict": 0.0,
    }


class StaticProtocolTest(unittest.TestCase):
    def test_one_to_one_view_matching_keeps_one_mask(self):
        mask = np.zeros((8, 8), dtype=bool); mask[2:5, 2:5] = True
        rows = [
            {"candidate_index": 0, "x": 3, "y": 3, "evidence": 1.0, "mask": mask, "stain_mask": mask, "geometry_mask": mask, "features": _features()},
            {"candidate_index": 1, "x": 4, "y": 3, "evidence": 0.9, "mask": mask, "stain_mask": mask, "geometry_mask": mask, "features": _features()},
        ]
        components = np.ones((8, 8), dtype=np.int32)
        resolved, stats = one_to_one_cross_view(rows, RULE, components)
        self.assertEqual(sum(row["status"] == "cross_view_matched" for row in resolved), 1)
        self.assertEqual(stats["same_h_component_duplicate"], 1)

    def test_lopo_prefers_high_precision_before_recall(self):
        rows = []
        for patient in range(1, 7):
            for index in range(3):
                rows.append({
                    "patient": patient, "image": f"{patient:02d}_1", "evidence": 1.0 - index * .1,
                    "is_true": index == 0,
                    "features": _features(0.90 if index == 0 else 0.35),
                })
        first = select_rule_lopo(rows, RULE)
        second = select_rule_lopo(rows, RULE)
        self.assertEqual(first, second)
        rule, budget, folds = first
        # 0.40 is the first threshold that excludes the two 0.35 false
        # candidates while retaining the 0.90 true candidate in every fold.
        self.assertGreaterEqual(rule["min_view_iou"], 0.40)
        self.assertIn(budget, (8, 16, 32, 64))
        self.assertEqual(len(folds), 6)


if __name__ == "__main__":
    unittest.main()
