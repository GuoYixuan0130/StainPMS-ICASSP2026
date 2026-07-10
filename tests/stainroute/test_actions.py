import inspect
import unittest

import numpy as np

from stainroute.actions import (
    ActionCandidate,
    ActionType,
    Point,
    SplitAssemblyConfig,
    apply_add_action,
    apply_split_action,
    assert_no_gt_leakage,
    build_conflict_graph,
    generate_add_candidates,
    generate_split_candidates,
)
from stainroute.inference.coordinates import crop_box_around_point, crop_to_global, global_to_crop


class ActionSchemaAndCandidateTest(unittest.TestCase):
    def test_schema_round_trip_and_leakage_guard(self) -> None:
        candidate = ActionCandidate(
            action_id="sample:ADD:000",
            image_id="sample",
            action_type=ActionType.ADD,
            affected_instance_ids=(),
            positive_points=(Point(10, 11),),
            negative_points=(),
            action_cost=1,
            generation_features={"h_evidence": 0.8, "candidate_rank": 0},
            decoded_features={"decoded_predicted_iou": 0.7},
            utility_fields={"delta_pq": 0.1},
        )
        self.assertEqual(ActionCandidate.from_json(candidate.to_json()), candidate)
        with self.assertRaisesRegex(ValueError, "Forbidden"):
            assert_no_gt_leakage({"target_gt_id": 3})

    def test_add_generation_has_no_gt_argument_and_is_deterministic(self) -> None:
        signature = inspect.signature(generate_add_candidates)
        self.assertNotIn("gt", signature.parameters)
        image = np.full((48, 48, 3), 220, dtype=np.uint8)
        image[16:24, 16:24, 2] = 20
        prediction = np.zeros((48, 48), dtype=np.int32)
        first = generate_add_candidates(image, prediction, image_id="image")
        second = generate_add_candidates(image, prediction, image_id="image")
        self.assertEqual([item.to_json() for item in first], [item.to_json() for item in second])

    def test_replacing_or_randomizing_gt_cannot_change_candidate_geometry(self) -> None:
        """The candidate API deliberately has no GT input or hidden state."""

        image = np.full((64, 64, 3), 220, dtype=np.uint8)
        image[12:22, 12:22, 2] = 10
        image[38:48, 38:48, 2] = 15
        prediction = np.zeros((64, 64), dtype=np.int32)
        gt_empty = np.zeros((64, 64), dtype=np.int32)
        gt_random = np.random.default_rng(3407).integers(0, 9, size=(64, 64), dtype=np.int32)
        # Neither label map is passed: if this ever changes, the signature
        # assertions above and this strict geometry comparison must be updated
        # as a leakage review rather than silently accepting the change.
        _ = gt_empty
        first = [item.to_json() for item in generate_add_candidates(image, prediction, image_id="image")]
        _ = gt_random
        second = [item.to_json() for item in generate_add_candidates(image, prediction, image_id="image")]
        self.assertEqual(first, second)
        for candidate in generate_split_candidates(image, prediction, image_id="image"):
            self.assertNotIn("gt", str(candidate.generation_features).lower())

    def test_coordinate_round_trip(self) -> None:
        point = Point(3, 60)
        box = crop_box_around_point(point, image_width=80, image_height=100, crop_size=32)
        local = global_to_crop(point, box)
        self.assertEqual(crop_to_global(local, box), point)

    def test_coordinate_round_trip_at_image_boundaries(self) -> None:
        for point in (Point(0, 0), Point(79, 0), Point(0, 99), Point(79, 99)):
            box = crop_box_around_point(point, image_width=80, image_height=100, crop_size=32)
            self.assertEqual(crop_to_global(global_to_crop(point, box), box), point)


class AssemblyAndConflictTest(unittest.TestCase):
    def test_add_assembly_only_inserts_background(self) -> None:
        prediction = np.zeros((8, 8), dtype=np.int32)
        prediction[0:2, 0:2] = 1
        decoded = np.zeros((8, 8), dtype=bool)
        decoded[0:2, 0:2] = True
        decoded[4:7, 4:7] = True
        result = apply_add_action(prediction, decoded, min_added_area=4)
        self.assertTrue(result.applied)
        self.assertEqual(result.details["added_area"], 9)
        self.assertEqual(set(np.unique(result.prediction)), {0, 1, 2})

    def test_split_replacement_increases_instance_count_by_one(self) -> None:
        prediction = np.zeros((12, 12), dtype=np.int32)
        prediction[2:10, 2:10] = 4
        first = np.zeros_like(prediction, dtype=bool)
        second = np.zeros_like(prediction, dtype=bool)
        first[2:10, 2:6] = True
        second[2:10, 6:10] = True
        result = apply_split_action(
            prediction,
            parent_id=4,
            child_first=first,
            child_second=second,
            first_point=Point(3, 5),
            second_point=Point(8, 5),
            config=SplitAssemblyConfig(min_child_area=8),
        )
        self.assertTrue(result.applied)
        self.assertEqual(result.details["instance_count_after"], result.details["instance_count_before"] + 1)

    def test_split_order_swap_preserves_partition_semantics(self) -> None:
        prediction = np.zeros((12, 12), dtype=np.int32)
        prediction[2:10, 2:10] = 1
        first = np.zeros_like(prediction, dtype=bool)
        second = np.zeros_like(prediction, dtype=bool)
        first[2:10, 2:7] = True
        second[2:10, 5:10] = True
        normal = apply_split_action(
            prediction, parent_id=1, child_first=first, child_second=second,
            first_point=Point(3, 5), second_point=Point(8, 5),
        )
        swapped = apply_split_action(
            prediction, parent_id=1, child_first=second, child_second=first,
            first_point=Point(8, 5), second_point=Point(3, 5),
        )
        self.assertTrue(normal.applied and swapped.applied)
        self.assertTrue(np.array_equal(normal.prediction > 0, swapped.prediction > 0))
        self.assertEqual(
            sorted(int((normal.prediction == value).sum()) for value in np.unique(normal.prediction) if value),
            sorted(int((swapped.prediction == value).sum()) for value in np.unique(swapped.prediction) if value),
        )

    def test_split_candidate_has_no_gt_argument(self) -> None:
        self.assertNotIn("gt", inspect.signature(generate_split_candidates).parameters)

    def test_single_peak_does_not_propose_split_but_two_peaks_do(self) -> None:
        prediction = np.zeros((64, 64), dtype=np.int32)
        prediction[12:52, 10:54] = 1
        image = np.full((64, 64, 3), 220, dtype=np.uint8)
        image[20:27, 18:25] = [40, 40, 120]
        one_peak = generate_split_candidates(image, prediction, image_id="single")
        image[20:27, 38:45] = [40, 40, 120]
        two_peaks = generate_split_candidates(image, prediction, image_id="double")
        self.assertEqual(one_peak, [])
        self.assertGreaterEqual(len(two_peaks), 1)

    def test_conflict_graph_marks_shared_parent(self) -> None:
        first = ActionCandidate("i:SPLIT:1:0", "i", ActionType.SPLIT, (1,), (Point(1, 1), Point(3, 1)), (Point(3, 1), Point(1, 1)), 2)
        second = ActionCandidate("i:SPLIT:1:1", "i", ActionType.SPLIT, (1,), (Point(1, 2), Point(3, 2)), (Point(3, 2), Point(1, 2)), 2)
        graph = build_conflict_graph([first, second])
        self.assertEqual(graph[first.action_id], {second.action_id})
