from __future__ import annotations

import inspect
import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipUnless(torch is not None, "requires the project PyTorch environment")
class NuSetMultiMaskAuditTest(unittest.TestCase):
    class _PromptEncoder:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, *, points, boxes, masks, batch_size):
            self.calls += 1
            coordinates, _ = points
            return torch.zeros(len(coordinates), 1, 2), torch.zeros(1, 2, 1, 1)

        def get_dense_pe(self):
            return torch.zeros(1, 2, 1, 1)

    class _Decoder:
        num_multimask_outputs = 3
        num_mask_tokens = 4

        def __init__(self) -> None:
            self.calls = 0

        def predict_masks(self, **kwargs):
            self.calls += 1
            prompts = kwargs["sparse_prompt_embeddings"].shape[0]
            masks = torch.arange(prompts * 4 * 2 * 2, dtype=torch.float32).reshape(prompts, 4, 2, 2)
            iou = torch.arange(prompts * 4, dtype=torch.float32).reshape(prompts, 4)
            return masks, iou, torch.zeros(prompts, 4, 2), torch.zeros(prompts, 1)

    def test_all_token_extraction_uses_one_prompt_and_decoder_call(self) -> None:
        from nuset.audit.decoder import extract_all_tokens_once

        prompt_encoder, decoder = self._PromptEncoder(), self._Decoder()
        result = extract_all_tokens_once(
            mask_decoder=decoder, prompt_encoder=prompt_encoder,
            image_embeddings=torch.zeros(1, 2, 1, 1), high_res_features=[],
            coordinates=torch.tensor([[[1.0, 1.0]], [[2.0, 2.0]]]), out_size=4,
        )
        self.assertEqual(prompt_encoder.calls, 1)
        self.assertEqual(decoder.calls, 1)
        self.assertEqual(tuple(result.low_res_logits.shape), (2, 4, 2, 2))
        self.assertEqual(tuple(result.upsampled_logits.shape), (2, 4, 4, 4))
        self.assertTrue(torch.equal(result.predicted_iou[1], torch.tensor([4.0, 5.0, 6.0, 7.0])))

    def test_token0_selector_and_predicted_oracle_selection_order(self) -> None:
        from nuset.audit.decoder import AllTokenMasks, token0_view
        from nuset.audit.metrics import selector_indices

        logits = torch.arange(16, dtype=torch.float32).reshape(1, 4, 2, 2)
        masks = AllTokenMasks(logits, logits, torch.tensor([[0.1, 0.5, 0.9, 0.2]]), torch.zeros(1, 4, 1), torch.zeros(1, 1), 0.0, 0.0)
        selected, scores, indices = token0_view(masks)
        self.assertTrue(torch.equal(selected, logits[:, 0]))
        self.assertTrue(torch.equal(scores, torch.tensor([0.1])))
        self.assertTrue(torch.equal(indices, torch.tensor([0])))
        choices = selector_indices(masks.predicted_iou, torch.tensor([[0.2, 0.8, 0.3, 0.7]]))
        self.assertEqual(int(choices["multi_pred"][0]), 2)
        self.assertEqual(int(choices["all_pred"][0]), 2)
        self.assertEqual(int(choices["multi_oracle"][0]), 1)
        self.assertEqual(int(choices["all_oracle"][0]), 1)

    def test_deterministic_token_extraction_rerun(self) -> None:
        from nuset.audit.decoder import extract_all_tokens_once

        coordinates = torch.tensor([[[1.0, 1.0]]])
        first = extract_all_tokens_once(
            mask_decoder=self._Decoder(), prompt_encoder=self._PromptEncoder(), image_embeddings=torch.zeros(1, 2, 1, 1),
            high_res_features=[], coordinates=coordinates, out_size=4,
        )
        second = extract_all_tokens_once(
            mask_decoder=self._Decoder(), prompt_encoder=self._PromptEncoder(), image_embeddings=torch.zeros(1, 2, 1, 1),
            high_res_features=[], coordinates=coordinates, out_size=4,
        )
        torch.testing.assert_close(first.low_res_logits, second.low_res_logits)
        torch.testing.assert_close(first.predicted_iou, second.predicted_iou)

    def test_inclusive_iou_half_is_a_true_positive(self) -> None:
        from nuset.audit.metrics import assembly_metrics

        truth = torch.tensor([[1, 1, 0], [0, 0, 0]]).numpy()
        prediction = torch.tensor([[1, 1, 1], [1, 0, 0]]).numpy()
        metrics = assembly_metrics(truth, prediction)
        self.assertEqual(metrics["tp"], 1)
        self.assertEqual(metrics["fp"], 0)
        self.assertEqual(metrics["fn"], 0)

    def test_unmatched_oracle_uses_predicted_token_not_gt(self) -> None:
        from nuset.audit.decoder import AllTokenMasks
        from nuset.audit.runner import _selector_for_automatic

        logits = torch.zeros(2, 4, 2, 2)
        tokens = AllTokenMasks(logits, logits, torch.tensor([[0.1, 0.9, 0.2, 0.3], [0.3, 0.1, 0.2, 0.8]]), torch.zeros(2, 4, 1), torch.zeros(2, 1), 0.0, 0.0)
        choices = _selector_for_automatic(tokens, None, torch.empty(0, dtype=torch.long).cpu().numpy())
        self.assertEqual(choices["oracle_all"].tolist(), [1, 3])

    def test_fixed_selection_rejects_closed_patient(self) -> None:
        from nuset.audit.data import load_fixed_selection

        self.assertIn("range(1, 7)", inspect.getsource(load_fixed_selection))

    def test_no_model_parameter_updates_in_stage0_source(self) -> None:
        from nuset.audit import runner

        source = inspect.getsource(runner.run_stage0)
        self.assertIn("before != after", source)
        self.assertNotIn("optimizer.step", source)

    def test_baseline_selector_matches_maskdecoder_forward_contract(self) -> None:
        from sam2_train.modeling.sam.mask_decoder import MaskDecoder

        source = inspect.getsource(MaskDecoder.forward)
        self.assertIn("masks = masks[:, 0:1", source)
        self.assertIn("iou_pred = iou_pred[:, 0:1]", source)

    def test_single_decoder_and_encoder_contract_is_explicit(self) -> None:
        from nuset.audit import runner

        source = inspect.getsource(runner._decode_all_tokens)
        self.assertEqual(source.count("extract_all_tokens_once("), 1)
        self.assertIn("counts.sam_image_encoder += 1", source)
        self.assertIn("counts.sam_prompt_encoder += 1", source)
        self.assertIn("counts.sam_mask_decoder += 1", source)


if __name__ == "__main__":
    unittest.main()
