from __future__ import annotations

import inspect
import unittest

import numpy as np

from nupart.core import connected_components, distinct_gt_conflicts, gt_ownership_oracle, logit_wta, nearest_prompt_wta


class NuPartCoreTest(unittest.TestCase):
    def _stage_source(self) -> str:
        from nupart import stage0
        return inspect.getsource(stage0)

    def test_closed_patient_guard(self) -> None:
        from nupart.stage0 import TRAIN_PATIENTS, _patient

        self.assertEqual(_patient("09_1"), 9)
        self.assertNotIn(_patient("09_1"), TRAIN_PATIENTS)
        self.assertIn("cache crosses the authorized patient boundary", self._stage_source())

    def test_token_zero_only_guard(self) -> None:
        source = self._stage_source()
        self.assertIn("TOKEN_INDEX = 0", source)
        self.assertIn("logits[:, TOKEN_INDEX]", source)
        self.assertIn('"low_res_logits" in group', source)
        self.assertIn("cached_mask_logits_max_abs_error", source)

    def test_development_seven_image_guard(self) -> None:
        from nupart.stage0 import REQUIRED_DEVELOPMENT_IMAGES

        self.assertEqual(REQUIRED_DEVELOPMENT_IMAGES, 7)
        self.assertIn("development cache must contain the preregistered seven images", self._stage_source())

    def test_baseline_equivalence_contract(self) -> None:
        source = self._stage_source()
        self.assertIn("final_instance_map_identical", source)
        self.assertIn("metric_max_abs_error", source)
        self.assertIn("<= 1e-7", source)

    def test_distinct_same_gt_and_unmatched_are_separated(self) -> None:
        masks = np.zeros((4, 3, 3), dtype=bool)
        masks[:, 1, 1] = True
        edges = distinct_gt_conflicts(masks, np.asarray((1, 1, 2, 0)))
        self.assertEqual([(edge.left, edge.right) for edge in edges], [(0, 2), (1, 2)])

    def test_one_pixel_overlap_and_components(self) -> None:
        masks = np.zeros((3, 3, 3), dtype=bool)
        masks[0, 1, 1] = masks[1, 1, 1] = True
        masks[1, 1, 2] = masks[2, 1, 2] = True
        edges = distinct_gt_conflicts(masks, np.asarray((1, 2, 3)))
        self.assertEqual([edge.overlap_pixels for edge in edges], [1, 1])
        self.assertEqual(connected_components(3, edges), [[0, 1, 2]])

    def test_resolvers_only_change_overlap_and_ties_use_standard(self) -> None:
        masks = np.zeros((2, 3, 3), dtype=bool); masks[0, 1, 1] = masks[1, 1, 1] = True; masks[0, 0, 0] = True
        logits = np.zeros_like(masks, dtype=np.float32)
        owners = np.full((3, 3), -1, dtype=np.int64); owners[1, 1] = 1
        result, changed = logit_wta(masks, logits, owners)
        self.assertTrue(result[1, 1, 1]); self.assertFalse(result[0, 1, 1]); self.assertTrue(result[0, 0, 0]); self.assertTrue(changed[1, 1])
        nearest, _ = nearest_prompt_wta(masks, np.asarray(((0., 0.), (2., 2.))), owners)
        self.assertTrue(nearest[1, 1, 1])

    def test_oracle_only_changes_authorized_distinct_gt_pixels(self) -> None:
        masks = np.zeros((3, 3, 3), dtype=bool); masks[0, 1, 1] = masks[1, 1, 1] = True; masks[0, 0, 0] = masks[2, 0, 0] = True
        gt = np.zeros((3, 3), dtype=np.int64); gt[1, 1] = 2; gt[0, 0] = 1
        owners = np.full((3, 3), -1, dtype=np.int64); owners[1, 1] = 0; owners[0, 0] = 0
        result, changed, authorized = gt_ownership_oracle(masks, np.asarray((1, 2, 1)), gt, owners)
        self.assertTrue(result[1, 1, 1]); self.assertFalse(result[0, 1, 1]); self.assertTrue(authorized[1, 1])
        self.assertTrue(np.array_equal(result[:, 0, 0], masks[:, 0, 0]))
        self.assertFalse(changed[0, 0])

    def test_non_conflict_masks_remain_unchanged(self) -> None:
        masks = np.zeros((3, 4, 4), dtype=bool)
        masks[0, 1, 1] = masks[1, 1, 1] = True
        masks[2, 3, 3] = True
        gt = np.zeros((4, 4), dtype=np.int64); gt[1, 1] = 2
        owners = np.full((4, 4), -1, dtype=np.int64); owners[1, 1] = 0
        result, _, _ = gt_ownership_oracle(masks, np.asarray((1, 2, 3)), gt, owners)
        self.assertTrue(np.array_equal(result[2], masks[2]))

    def test_background_logit_is_fixed_zero(self) -> None:
        self.assertIn("torch.zeros((len(detached), 1)", self._stage_source())

    def test_local_softmax_normalization_contract(self) -> None:
        self.assertIn("functional.cross_entropy", self._stage_source())
        self.assertIn("local_softmax_normalized\": True", self._stage_source())

    def test_boundary_band_is_fixed_to_two_pixels(self) -> None:
        self.assertIn("distance_transform_edt(mask) <= 2", self._stage_source())

    def test_ownership_label_conservation_contract(self) -> None:
        source = self._stage_source()
        self.assertIn("target[index] =", source)
        self.assertIn("else 0", source)
        self.assertIn("wrong_winner_requires_defined_owner_and_competitor", source)

    def test_no_point_or_mask_deletion_contract(self) -> None:
        self.assertIn("resolver deleted a point or mask", self._stage_source())

    def test_inclusive_iou_half_contract(self) -> None:
        self.assertIn("match_iou=0.5", self._stage_source())

    def test_image_first_aggregation_contract(self) -> None:
        source = self._stage_source()
        self.assertIn("np.mean([row[metric] for row in selected])", source)

    def test_frozen_checksum_contract(self) -> None:
        source = self._stage_source()
        self.assertIn("frozen_checksums", source)
        self.assertIn("checks.get(\"before\")", source)

    def test_no_optimizer_or_model_backward_in_stage_source(self) -> None:
        source = self._stage_source()
        self.assertNotIn("optimizer.step(", source)
        self.assertNotIn("model.load_state_dict", source)
        self.assertNotIn("model.backward(", source)
        self.assertIn("TOKEN_INDEX = 0", source)
        self.assertIn("match_iou=0.5", source)

    def test_deterministic_rerun_of_resolver(self) -> None:
        masks = np.zeros((2, 3, 3), dtype=bool); masks[:, 1, 1] = True
        logits = np.asarray([[[0.0] * 3] * 3, [[0.0] * 3] * 3], dtype=np.float32)
        owners = np.full((3, 3), -1, dtype=np.int64); owners[1, 1] = 1
        first = logit_wta(masks, logits, owners)
        second = logit_wta(masks, logits, owners)
        self.assertTrue(np.array_equal(first[0], second[0]))



if __name__ == "__main__":
    unittest.main()
