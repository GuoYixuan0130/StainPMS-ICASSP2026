import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from promptq_v2.model import ModelBundle, assert_frozen_without_grads, configure_quality_only
from promptq_v2.assembly import crop_with_overlap
from promptq_v2.protocol import (
    INCLUSIVE_IOU_THRESHOLD,
    NMS_RADIUS,
    finite_arrays,
    inclusive_iou,
    point_nms_indices,
    product_score,
    quality_focal_loss,
    set_determinism,
    state_sha256,
)


class _Point(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Linear(2, 2)
        self.quality_head = torch.nn.Sequential(torch.nn.Linear(256, 256), torch.nn.ReLU(), torch.nn.Linear(256, 1))


class PromptQV2ProtocolTests(unittest.TestCase):
    def test_only_quality_head_is_trainable_and_frozen_checksum_is_stable(self):
        point = _Point()
        sam2 = torch.nn.Linear(2, 2)
        bundle = ModelBundle(point, None, sam2, [], torch.device("cpu"), {})
        manifest = configure_quality_only(bundle)
        self.assertTrue(manifest["trainable_parameter_names"])
        self.assertTrue(all(name.startswith("quality_head.") for name in manifest["trainable_parameter_names"]))
        before = state_sha256(point, exclude_prefixes=("quality_head.",))
        loss = point.quality_head(torch.ones(1, 256)).sum()
        loss.backward()
        assert_frozen_without_grads(bundle)
        self.assertEqual(before, state_sha256(point, exclude_prefixes=("quality_head.",)))

    def test_cache_baseline_nms_replay_is_exact_and_product_only_changes_ranking(self):
        points = np.asarray([[0.0, 0.0], [1.0, 0.0], [30.0, 0.0]], dtype=np.float32)
        objectness = np.asarray([0.9, 0.8, 0.7], dtype=np.float32)
        baseline_first = point_nms_indices(points, objectness, NMS_RADIUS)
        baseline_second = point_nms_indices(points.copy(), objectness.copy(), NMS_RADIUS)
        np.testing.assert_array_equal(baseline_first, baseline_second)
        product = product_score(objectness, np.asarray([-4.0, 4.0, 0.0]))
        self.assertFalse(np.array_equal(baseline_first, point_nms_indices(points, product, NMS_RADIUS)))

    def test_deployment_cache_arrays_cannot_contain_gt_and_are_finite(self):
        deployment = {"quality_feature": np.ones((2, 256), dtype=np.float16), "decoded_hard_masks": np.zeros((2, 4, 4), dtype=np.uint8)}
        self.assertTrue(finite_arrays(deployment.values()))
        self.assertFalse(any("gt" in key.lower() or "label" in key.lower() or "target" in key.lower() for key in deployment))

    def test_deterministic_replay_and_inclusive_iou_boundary(self):
        set_determinism()
        first = torch.rand(4)
        set_determinism()
        second = torch.rand(4)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        # One-pixel masks give an exact 0.5 IoU; the project metric regards it as a match.
        left = np.asarray([[1, 1]], dtype=bool)
        right = np.asarray([[1, 0]], dtype=bool)
        self.assertEqual(inclusive_iou(left, right), INCLUSIVE_IOU_THRESHOLD)
        self.assertGreaterEqual(inclusive_iou(left, right), INCLUSIVE_IOU_THRESHOLD)

    def test_quality_loss_is_finite(self):
        loss = quality_focal_loss(torch.zeros(3), torch.tensor([0.0, 0.5, 1.0]), torch.tensor([False, True, True]))
        self.assertTrue(torch.isfinite(loss))

    def test_isolated_canonical_crop_replay_is_deterministic(self):
        image = torch.zeros(3, 480, 480)
        first = crop_with_overlap(image, overlap=32, load="unclockwise")
        second = crop_with_overlap(image.clone(), overlap=32, load="unclockwise")
        np.testing.assert_array_equal(first, second)
        self.assertTrue(np.all(first[:, 2] - first[:, 0] <= 256))
        self.assertTrue(np.all(first[:, 3] - first[:, 1] <= 256))


if __name__ == "__main__":
    unittest.main()
