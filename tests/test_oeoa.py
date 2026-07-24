from __future__ import annotations

import math
import unittest

import numpy as np

from stainpms.oeoa import (
    ACTION_CLASSES,
    action_mask,
    actions_for_mask,
    all_action_masks,
    apply_component_oracle,
    build_overlap_components,
    candidate_pool_ceiling,
    compact_metrics,
    localize_final_fns,
    map_metrics,
    pairwise_interactions,
    relabel_contiguously,
    shapley_contributions,
)
from tools.run_phase3a_oeoa import audit_code_is_read_only


def synthetic_all_components() -> tuple[np.ndarray, np.ndarray]:
    """One non-overlapping map containing every preregistered category."""

    gt = np.zeros((32, 32), dtype=np.int32)
    pred = np.zeros_like(gt)
    # tp_boundary and subthreshold_1to1
    gt[1:4, 1:4] = 1
    pred[1:4, 1:4] = 1
    gt[1:4, 6:10] = 2
    pred[1:2, 6:7] = 2
    # merge (one native prediction overlaps two GT instances)
    gt[6:9, 1:4] = 3
    gt[6:9, 5:8] = 4
    pred[6:9, 1:8] = 3
    # split_or_duplicate (two native predictions overlap one GT instance)
    gt[12:16, 1:5] = 5
    pred[12:16, 1:3] = 4
    pred[12:16, 3:5] = 5
    # complex_topology (two predictions each touch both GT instances)
    gt[12:16, 8:12] = 6
    gt[12:16, 13:17] = 7
    pred[12:14, 8:17] = 6
    pred[14:16, 8:17] = 7
    # pure_fn and pure_fp
    gt[20:23, 1:4] = 8
    pred[20:23, 8:11] = 8
    return gt, pred


def record(index: int, group: int, mask: np.ndarray) -> dict[str, object]:
    return {"record_index": index, "prompt_group_id": group, "token": 0, "crop_index": 0, "mask": mask}


class OEOATests(unittest.TestCase):
    def test_component_taxonomy_is_mutually_exclusive_and_complete(self):
        gt, pred = synthetic_all_components()
        components, graph = build_overlap_components(gt, pred, sample_id="synthetic")
        self.assertEqual({component["category"] for component in components}, set(ACTION_CLASSES))
        self.assertEqual(sum(graph["category_counts"].values()), graph["component_count"])
        self.assertEqual(len(components), len(ACTION_CLASSES))

    def test_zero_and_all_actions_have_required_exactness(self):
        gt, pred = synthetic_all_components()
        components, _ = build_overlap_components(gt, pred)
        zero = apply_component_oracle(gt, pred, components, ())
        all_actions = apply_component_oracle(gt, pred, components, ACTION_CLASSES)
        self.assertTrue(np.array_equal(zero, pred))
        self.assertTrue(np.array_equal(all_actions, gt))
        self.assertEqual(compact_metrics(map_metrics(gt, zero)), compact_metrics(map_metrics(gt, pred)))

    def test_all_128_subsets_are_order_and_relabel_invariant(self):
        gt, pred = synthetic_all_components()
        components, _ = build_overlap_components(gt, pred)
        pq_values: dict[int, float] = {}
        for mask in all_action_masks():
            actions = actions_for_mask(mask)
            forward = apply_component_oracle(gt, pred, components, actions)
            reverse = apply_component_oracle(gt, pred, components, tuple(reversed(actions)))
            self.assertTrue(np.array_equal(forward, reverse))
            native_metrics = compact_metrics(map_metrics(gt, forward))
            relabelled_metrics = compact_metrics(map_metrics(gt, relabel_contiguously(forward)))
            for name, value in native_metrics.items():
                self.assertAlmostEqual(value, relabelled_metrics[name], places=7)
            pq_values[mask] = native_metrics["pq"]
        contributions = shapley_contributions(pq_values)
        self.assertTrue(math.isclose(sum(contributions.values()), pq_values[action_mask(ACTION_CLASSES)] - pq_values[0], abs_tol=1.0e-12))
        self.assertEqual(len(pairwise_interactions(pq_values)), 21)

    def test_fn_candidate_localization_priority_and_ceiling(self):
        gt = np.zeros((16, 20), dtype=np.int32)
        for identifier, left in enumerate((1, 6, 11, 16), start=1):
            gt[2:5, left:left + 3] = identifier
        final = np.zeros_like(gt)
        selected = [record(0, 10, gt == 1)]
        all_candidates = [record(0, 10, gt == 1), record(1, 11, gt == 2), record(2, 12, gt == 3)]
        # Replace the third all-pool candidate with a positive but subthreshold mask.
        all_candidates[2]["mask"] = np.pad(np.ones((1, 1), dtype=bool), ((2, 13), (11, 8)))
        rows = localize_final_fns(gt_map=gt, final_map=final, selected_records=selected, all_records=all_candidates)
        by_gt = {row["gt_instance_id"]: row["fn_localization"] for row in rows}
        self.assertEqual(by_gt, {1: "assembly_or_keep_miss", 2: "selection_miss", 3: "candidate_mask_near_miss", 4: "generation_miss"})
        ceiling = candidate_pool_ceiling(all_candidates, gt)
        self.assertEqual(ceiling["maximum_attainable_tp"], 2)
        self.assertEqual(ceiling["remaining_fn"], 2)

    def test_audit_code_has_no_model_or_training_api(self):
        self.assertEqual(audit_code_is_read_only()["status"], "pass")


if __name__ == "__main__":
    unittest.main()
