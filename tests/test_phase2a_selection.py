import unittest

from stainpms.phase2a_selection import choose_tnbc_checkpoint, tnbc_patient_macro_score


def image(sample_id, aji, pq):
    metrics = {name: 0.5 for name in ("dice1", "dice2", "aji_p", "dq", "sq")}
    metrics.update({"aji": aji, "pq": pq})
    return {"sample_id": sample_id, "included_in_macro": True, "metrics": metrics}


class Phase2ASelectionTests(unittest.TestCase):
    def test_patient_macro_precedes_selection_average(self):
        result = tnbc_patient_macro_score(
            [image("07_1", 0.8, 0.6), image("07_2", 0.6, 0.4), image("08_1", 0.2, 0.8)],
            {"07_1": 7, "07_2": 7, "08_1": 8},
        )
        self.assertAlmostEqual(result["macro_patient_aji"], 0.45)
        self.assertAlmostEqual(result["macro_patient_pq"], 0.65)
        self.assertAlmostEqual(result["selection_score"], 0.55)

    def test_tie_within_point_001_keeps_earlier(self):
        result = choose_tnbc_checkpoint(
            [
                {"optimizer_updates": 100, "selection_score": 0.5},
                {"optimizer_updates": 200, "selection_score": 0.5009},
                {"optimizer_updates": 300, "selection_score": 0.501},
            ]
        )
        self.assertEqual(result["optimizer_updates"], 300)


if __name__ == "__main__":
    unittest.main()
