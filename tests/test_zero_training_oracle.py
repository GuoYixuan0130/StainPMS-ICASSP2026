from __future__ import annotations

import unittest

import numpy as np

from stainpms.zero_training_oracle import (
    annotate_pool_ious,
    decode_binary_rle,
    encode_binary_rle,
    error_partition,
    maximum_cardinality_max_iou_matching,
    native_final_stage,
    oracle_pool_stage,
)


def record(index: int, group: int, mask: np.ndarray, token: int = 0):
    return {
        "record_index": index,
        "prompt_group_id": group,
        "token": token,
        "crop_index": 0,
        "mask": np.asarray(mask, dtype=bool),
    }


class ZeroTrainingOracleTests(unittest.TestCase):
    def test_rle_round_trip_preserves_fortran_order(self):
        mask = np.array([[0, 1, 0], [1, 1, 0], [0, 0, 1]], dtype=bool)
        self.assertTrue(np.array_equal(mask, decode_binary_rle(encode_binary_rle(mask))))

    def test_matching_prioritises_cardinality_before_iou_sum(self):
        # Group 1 could take GT 1 at .99, but then group 2 has no eligible
        # partner.  The required lexicographic objective instead keeps two
        # matches: group 1 -> GT 2 and group 2 -> GT 1.
        records = [
            {"record_index": 0, "prompt_group_id": 1, "token": 0, "crop_index": 0, "gt_ious": {"1": 0.99, "2": 0.60}},
            {"record_index": 1, "prompt_group_id": 2, "token": 0, "crop_index": 0, "gt_ious": {"1": 0.70}},
        ]
        result = maximum_cardinality_max_iou_matching(records, [1, 2])
        self.assertEqual(result["tp"], 2)
        self.assertEqual(
            {(row["prompt_group_id"], row["gt_instance_id"]) for row in result["matched"]},
            {(1, 2), (2, 1)},
        )

    def test_oracle_pool_removes_unmatched_groups_only_after_matching(self):
        gt = np.zeros((8, 8), dtype=np.int32)
        gt[1:3, 1:3] = 1
        gt[5:7, 5:7] = 2
        records = annotate_pool_ious(
            [
                record(0, 10, gt == 1),
                record(1, 11, gt == 2),
                record(2, 12, np.zeros_like(gt, dtype=bool)),
            ],
            gt,
        )
        result = oracle_pool_stage(records, gt)
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (2, 0, 0))
        self.assertEqual(result["raw_prediction_group_count"], 3)
        self.assertEqual(result["raw_unmatched_group_count"], 1)
        self.assertAlmostEqual(result["pq"], 1.0)

    def test_error_partition_separates_generation_selection_and_assembly(self):
        gt = np.zeros((12, 12), dtype=np.int32)
        gt[1:3, 1:3] = 1  # selected and final TP
        gt[1:3, 5:7] = 2  # all pool reaches it, selected pool does not
        gt[5:7, 1:3] = 3  # selected reaches it, final assembly drops it
        gt[5:7, 5:7] = 4  # all candidates miss
        all_records = annotate_pool_ious(
            [record(0, 1, gt == 1), record(1, 2, gt == 2), record(2, 3, gt == 3)], gt
        )
        selected_records = annotate_pool_ious([record(0, 1, gt == 1), record(1, 3, gt == 3)], gt)
        final = np.zeros_like(gt)
        final[gt == 1] = 1
        native = native_final_stage(gt, final)
        result = error_partition(
            gt_map=gt,
            all_candidate_records=all_records,
            selected_records=selected_records,
            native_final=native,
            final_map=final,
        )
        self.assertEqual(result["counts"]["generation_miss"], 1)
        self.assertEqual(result["counts"]["selection_miss"], 1)
        self.assertEqual(result["counts"]["assembly_loss"], 1)
        self.assertEqual(result["counts"]["native_final_tp"], 1)


if __name__ == "__main__":
    unittest.main()
