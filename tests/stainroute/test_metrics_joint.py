import unittest

import numpy as np

from stainroute.actions import ActionCandidate, ActionType, Point
from stainroute.metrics import PQEvaluation, evaluate_pq
from stainroute.oracle_actions import beam_joint_oracle, compute_action_utility, exact_joint_oracle


def _add(action_id: str) -> ActionCandidate:
    return ActionCandidate(action_id, "image", ActionType.ADD, (), (Point(0, 0),), (), 1)


def _split(action_id: str) -> ActionCandidate:
    return ActionCandidate(
        action_id,
        "image",
        ActionType.SPLIT,
        (1,),
        (Point(0, 0), Point(1, 0)),
        (Point(1, 0), Point(0, 0)),
        2,
    )


class InclusiveMetricAndJointOracleTest(unittest.TestCase):
    def test_inclusive_half_iou_and_full_utility(self) -> None:
        gt = np.array([[1, 1, 1, 1, 0]], dtype=np.int32)
        base = np.array([[1, 1, 0, 0, 0]], dtype=np.int32)
        improved = gt.copy()
        evaluation = evaluate_pq(gt, base)
        self.assertEqual(evaluation.tp, 1)
        self.assertAlmostEqual(evaluation.pq, 0.5)
        utility = compute_action_utility(gt, base, improved)
        self.assertAlmostEqual(utility.delta_pq, 0.5)
        self.assertTrue(utility.positive_utility_label)

    def test_beam_equals_exact_on_small_conflict_case(self) -> None:
        actions = [_add("a"), _add("b"), _split("c")]
        values = {(): 0.5, ("a",): 0.6, ("b",): 0.7, ("c",): 0.75, ("a", "b"): 0.9}

        def evaluate(ids):
            return PQEvaluation(0.0, 0, 0, 0, 0.0, 0.0, values.get(tuple(ids), 0.0), ())

        conflicts = {"a": {"c"}, "b": set(), "c": {"a"}}
        exact = exact_joint_oracle(actions, budget=2, conflict_graph=conflicts, evaluate_subset=evaluate)
        beam = beam_joint_oracle(actions, budget=2, conflict_graph=conflicts, evaluate_subset=evaluate, beam_width=8)
        self.assertEqual(exact, beam)
        self.assertEqual(exact.action_ids, ("a", "b"))
