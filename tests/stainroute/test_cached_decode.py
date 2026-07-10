import unittest

import numpy as np

try:
    import torch
except ModuleNotFoundError:  # local CPU-only checkout; run fully on AutoDL
    torch = None


@unittest.skipIf(torch is None, "PyTorch is available in the AutoDL experiment environment")
class CachedDecodePrimitiveTest(unittest.TestCase):
    def test_component_retention_and_logit_equivalence_primitive(self) -> None:
        from stainroute.inference.cached_decode import component_containing_point, max_abs_logit_error

        mask = np.zeros((8, 8), dtype=bool)
        mask[1:3, 1:3] = True
        mask[5:7, 5:7] = True
        retained = component_containing_point(mask, 1, 1)
        self.assertEqual(int(retained.sum()), 4)
        self.assertFalse(retained[5, 5])
        logits = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
        self.assertEqual(max_abs_logit_error(logits, logits.clone()), 0.0)
        with self.assertRaises(ValueError):
            max_abs_logit_error(logits, torch.zeros((1, 3)))
