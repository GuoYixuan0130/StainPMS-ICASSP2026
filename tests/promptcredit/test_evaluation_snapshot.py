from __future__ import annotations

import random
import unittest

try:
    import numpy as np
    import torch
except ModuleNotFoundError:
    np = None
    torch = None


@unittest.skipUnless(torch is not None and np is not None, "requires the project PyTorch environment")
class EvaluationSnapshotTest(unittest.TestCase):
    def test_eval_snapshot_is_deterministic_restores_rng_and_train_modes(self) -> None:
        from promptcredit.smoke.evaluation import (
            capture_rng_snapshot,
            evaluation_snapshot,
            rng_snapshots_equal,
        )

        model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Dropout(p=0.9))
        model.train()
        model[0].eval()  # Preserve a deliberately heterogeneous starting state too.
        value = torch.ones(2, 4)
        random.seed(3407)
        np.random.seed(3407)
        torch.manual_seed(3407)
        before = capture_rng_snapshot()
        with evaluation_snapshot(model):
            self.assertFalse(model.training)
            self.assertFalse(model[1].training)
            first = model(value)
        after = capture_rng_snapshot()
        self.assertTrue(rng_snapshots_equal(before, after))
        self.assertTrue(model.training)
        self.assertFalse(model[0].training)
        self.assertTrue(model[1].training)
        with evaluation_snapshot(model):
            second = model(value)
        torch.testing.assert_close(first, second)


if __name__ == "__main__":
    unittest.main()
