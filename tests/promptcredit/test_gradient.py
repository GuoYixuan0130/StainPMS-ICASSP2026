from __future__ import annotations

import unittest

try:
    import torch
except ModuleNotFoundError:  # Local static-check Python need not carry the AutoDL runtime.
    torch = None


@unittest.skipUnless(torch is not None, "requires the project PyTorch environment")
class PromptCreditGradientTest(unittest.TestCase):
    def test_frozen_parameter_isolation_and_coordinate_gradient(self) -> None:
        from promptcredit.audit.gradient import coordinate_gradient_probe, freeze_parameters_and_clear_gradients

        class ToyPromptDecoder(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor(1.0))

            def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
                return (coordinates[:, 0, 0] * self.scale).view(-1, 1, 1).expand(-1, 2, 2)

        decoder = ToyPromptDecoder()
        parameters = freeze_parameters_and_clear_gradients([decoder])
        result = coordinate_gradient_probe(
            decoder,
            coordinates=torch.zeros(2, 1, 2),
            target_masks=torch.ones(2, 2, 2),
            width=16,
            height=16,
            frozen_parameters=parameters,
        )
        self.assertTrue(result["frozen_parameter_grads_none"])
        self.assertTrue(torch.isfinite(result["coordinate_gradient"]).all())
        self.assertTrue(result["nonzero"].all())
        self.assertTrue((result["moved_loss"] < result["original_loss"]).all())


if __name__ == "__main__":
    unittest.main()
