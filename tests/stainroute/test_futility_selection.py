import unittest

from tools.stainroute_select_monuseg_pilot import select_pilot_images


class FutilityPilotSelectionTest(unittest.TestCase):
    def test_selection_is_deterministic_and_uses_two_per_quartile(self) -> None:
        rows = [{"image": f"image_{index:02d}", "add_candidates": str(index // 2)} for index in range(12)]
        first = select_pilot_images(rows, seed=3407)
        second = select_pilot_images(list(reversed(rows)), seed=3407)
        self.assertEqual(first, second)
        self.assertEqual(len(first["pilot_batch_1"]), 4)
        self.assertEqual(len(first["pilot_batch_2"]), 4)
        self.assertEqual(set(first["pilot_batch_1"]) & set(first["pilot_batch_2"]), set())
        self.assertEqual(first["selection_inputs"], ["image", "add_candidates"])
