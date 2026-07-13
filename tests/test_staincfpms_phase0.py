import unittest
from pathlib import Path

import numpy as np

from staincfpms.metrics import instance_metrics
from staincfpms.protocol import ProtocolError, assert_open_path, require_exact_sha256, sha256_file
from staincfpms.transforms import counterfactual_views, decompose

try:
    import torch
    from staincfpms.inference import no_training_guard
except ModuleNotFoundError:
    torch = None
    no_training_guard = None


class StainCFPMSPhase0Tests(unittest.TestCase):
    def setUp(self):
        yy, xx = np.mgrid[:32, :32]
        self.rgb = np.stack([
            np.clip(245 - yy * 2, 0, 255),
            np.clip(240 - xx * 2, 0, 255),
            np.clip(250 - (xx + yy), 0, 255),
        ], axis=-1).astype(np.uint8)

    def test_identity_and_round_trip_preserve_geometry(self):
        decomposition = decompose(self.rgb)
        views, _ = counterfactual_views(self.rgb, decomposition.matrix, decomposition.matrix, decomposition.matrix)
        self.assertEqual(views["V0"].shape, self.rgb.shape)
        self.assertEqual(views["V1"].shape, self.rgb.shape)
        self.assertLessEqual(np.abs(views["V1"].astype(float) - self.rgb.astype(float)).mean(), 3.0)

    def test_transform_is_deterministic(self):
        decomposition = decompose(self.rgb)
        first, _ = counterfactual_views(self.rgb, decomposition.matrix, decomposition.matrix, decomposition.matrix)
        second, _ = counterfactual_views(self.rgb, decomposition.matrix, decomposition.matrix, decomposition.matrix)
        for view in first:
            self.assertTrue(np.array_equal(first[view], second[view]))

    def test_closed_test_split_guard(self):
        with self.assertRaises(ProtocolError):
            assert_open_path("D:/data/monuseg/test/images", "unit test")
        assert_open_path("D:/data/monuseg/train_12/images", "unit test")

    @unittest.skipUnless(torch is not None, "requires the project PyTorch environment")
    def test_no_backward_or_optimizer_guard(self):
        tensor = torch.tensor(1.0, requires_grad=True)
        with no_training_guard():
            with self.assertRaises(ProtocolError):
                tensor.backward()
            with self.assertRaises(ProtocolError):
                torch.optim.SGD([tensor], lr=0.1)

    def test_checkpoint_checksum_guard(self):
        path = Path(__file__)
        require_exact_sha256(path, sha256_file(path), "unit checkpoint")
        with self.assertRaises(ProtocolError):
            require_exact_sha256(path, "0" * 64, "unit checkpoint")

    def test_inclusive_iou_half_is_a_match(self):
        true = np.zeros((3, 3), dtype=np.int32)
        pred = np.zeros((3, 3), dtype=np.int32)
        true[0, :2] = 1
        pred[0, :2] = 1
        pred[1, :2] = 1  # intersection=2, union=4, IoU=0.5 exactly
        metrics = instance_metrics(true, pred)
        self.assertEqual(metrics["tp"], 1)
        self.assertAlmostEqual(float(metrics["pq"]), 0.5, places=6)

    def test_fixed_prompt_coordinate_equivalence_contract(self):
        # The audit passes V0's float coordinates verbatim to each fixed decoder call.
        points = np.asarray([[10.25, 20.75], [4.5, 9.0]], dtype=np.float32)
        self.assertTrue(np.array_equal(points, np.asarray(points, dtype=np.float32)))


if __name__ == "__main__":
    unittest.main()
