import json
import unittest
from pathlib import Path

from tools.stainroute_make_splits import (
    build_monuseg_split_from_names,
    build_tnbc_split_from_names,
)


class BaselineFreezeTest(unittest.TestCase):
    def test_monuseg_split_is_deterministic_and_disjoint(self) -> None:
        names = [f"image_{index:02d}" for index in range(37)]
        first = build_monuseg_split_from_names(names, seed=3407, calibration_count=8)
        second = build_monuseg_split_from_names(names, seed=3407, calibration_count=8)

        self.assertEqual(first, second)
        self.assertEqual(len(first["router_train"]), 29)
        self.assertEqual(len(first["calibration"]), 8)
        self.assertFalse(set(first["router_train"]) & set(first["calibration"]))
        self.assertEqual(set(first["router_train"]) | set(first["calibration"]), set(names))

    def test_tnbc_split_is_patient_level_and_sealed(self) -> None:
        train_names = [f"{patient}_{index}" for patient in range(1, 9) for index in range(2)]
        test_names = [f"{patient}_{index}" for patient in range(9, 12) for index in range(2)]
        split = build_tnbc_split_from_names(train_names, test_names)

        self.assertEqual({name.split("_", 1)[0] for name in split["router_train"]}, set(map(str, range(1, 7))))
        self.assertEqual({name.split("_", 1)[0] for name in split["calibration"]}, {"7", "8"})
        self.assertEqual({name.split("_", 1)[0] for name in split["test"]}, {"9", "10", "11"})
        self.assertFalse(set(split["router_train"]) & set(split["calibration"]))
        self.assertFalse(set(split["router_train"]) & set(split["test"]))

    def test_baseline_config_is_json_compatible_yaml(self) -> None:
        config_path = Path("configs/stainroute/baseline_v1.yaml")
        payload = json.loads(config_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["baseline_name"], "StainRoute Development Baseline v1")
        self.assertEqual(payload["evaluation"]["test_nms_thr"], 12)
        self.assertTrue(payload["evaluation"]["matching"]["inclusive"])
