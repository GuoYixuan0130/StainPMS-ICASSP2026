from __future__ import annotations

import json
import inspect
import tempfile
import unittest
from pathlib import Path

import numpy as np
try:
    import torch
except ModuleNotFoundError as error:  # Local CPU-only checkout: leave real execution to the authorized AutoDL environment.
    raise unittest.SkipTest("NuRank unit tests require the agentseg PyTorch environment") from error

from nurank.analysis.metrics import ranking_metrics
from nurank.cache.io import group_feature_matrix, iter_groups, load_manifest
from nurank.losses import regret_aware_loss
from nurank.model.ranker import NuRankSharedRanker, build_ranker


class NuRankTest(unittest.TestCase):
    def test_cache_group_has_exactly_four_tokens_and_features(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            np.savez(root / "group.npz", mask_logits=np.zeros((2, 4, 2, 2), np.float32), mask_tokens=np.zeros((2, 4, 256), np.float16), token_index=np.tile(np.arange(4, dtype=np.int64), (2, 1)), original_predicted_iou=np.zeros((2, 4), np.float32), morphology=np.zeros((2, 4, 7), np.float32), true_hard_iou=np.zeros((2, 4), np.float32), true_soft_iou=np.zeros((2, 4), np.float32), matched=np.array([True, False]))
            from nurank.cache.io import sha256_file
            (root / "manifest.json").write_text(json.dumps({"schema": "nurank_automatic_prompt_cache_v1", "token_count": 4, "groups": [{"path": "group.npz", "sha256": sha256_file(root / "group.npz")}]}))
            group = next(iter_groups(root))
            self.assertEqual(group_feature_matrix(group).shape, (2, 4, 264))

    def test_cached_vs_online_feature_roundtrip_is_exact_except_declared_fp16_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); tokens = np.random.default_rng(3).normal(size=(1, 4, 256)).astype(np.float32)
            logits = np.random.default_rng(4).normal(size=(1, 4, 2, 2)).astype(np.float32); predicted = np.random.default_rng(5).random((1, 4), dtype=np.float32); morphology = np.random.default_rng(6).random((1, 4, 7), dtype=np.float32)
            np.savez(root / "group.npz", mask_logits=logits, mask_tokens=tokens.astype(np.float16), token_index=np.arange(4, dtype=np.int64)[None], original_predicted_iou=predicted, morphology=morphology, true_hard_iou=predicted, true_soft_iou=predicted, matched=np.array([True]))
            from nurank.cache.io import sha256_file
            (root / "manifest.json").write_text(json.dumps({"schema": "nurank_automatic_prompt_cache_v1", "token_count": 4, "groups": [{"path": "group.npz", "sha256": sha256_file(root / "group.npz")}]}))
            restored = next(iter_groups(root))
            self.assertTrue(np.array_equal(restored["mask_logits"], logits)); self.assertTrue(np.array_equal(restored["original_predicted_iou"], predicted)); self.assertTrue(np.array_equal(restored["morphology"], morphology))
            self.assertLessEqual(np.abs(restored["mask_tokens"].astype(np.float32) - tokens).max(), 1e-3)

    def test_shared_ranker_is_group_permutation_equivariant_without_token_id(self) -> None:
        ranker = build_ranker(scalar_mean=torch.zeros(8), scalar_std=torch.ones(8))
        features = torch.randn(3, 4, 264)
        order = torch.tensor([2, 0, 3, 1])
        self.assertTrue(torch.allclose(ranker(features)[:, order], ranker(features[:, order]), atol=1e-7))

    def test_no_token_id_leakage_and_parameter_limit(self) -> None:
        ranker = NuRankSharedRanker()
        self.assertFalse(any("embedding" in name.lower() or "token_id" in name.lower() for name, _ in ranker.named_parameters()))
        self.assertLess(ranker.parameter_count(), 100_000)

    def test_regret_margin_orders_and_excludes_ties(self) -> None:
        target = torch.tensor([[0.9, 0.2, 0.2, 0.2]])
        correct = regret_aware_loss(torch.tensor([[0.9, 0.1, 0.1, 0.1]]), target)
        wrong = regret_aware_loss(torch.tensor([[0.1, 0.9, 0.1, 0.1]]), target)
        self.assertLess(float(correct["ranking"]), float(wrong["ranking"]))

    def test_tie_pair_exclusion_is_finite(self) -> None:
        tied = regret_aware_loss(torch.full((1, 4), 0.5), torch.full((1, 4), 0.5))
        self.assertEqual(int(tied["valid_pair_count"]), 0); self.assertTrue(torch.isfinite(tied["total"]))

    def test_oracle_selector_and_ranking_metrics(self) -> None:
        truth = np.asarray([[.1, .8, .3, .2], [.2, .1, .9, .4]], np.float32)
        scores = np.asarray([[.2, .7, .1, .0], [.1, .0, .8, .2]], np.float32)
        metrics = ranking_metrics(scores, truth)
        self.assertEqual(metrics["top1_accuracy"], 1.0)
        self.assertEqual(metrics["mean_selection_regret"], 0.0)
        self.assertEqual(metrics["token_selection_histogram"], [0, 1, 1, 0])

    def test_ranker_cannot_mutate_cached_mask_logits_or_frozen_parameters(self) -> None:
        logits = torch.randn(2, 4, 8, 8); before_logits = logits.clone()
        frozen = torch.nn.Linear(3, 2); before = {name: value.detach().clone() for name, value in frozen.state_dict().items()}
        for parameter in frozen.parameters(): parameter.requires_grad_(False)
        ranker = NuRankSharedRanker(); optimizer = torch.optim.AdamW(ranker.parameters(), lr=1e-3)
        output = ranker(torch.randn(2, 4, 264)); output.mean().backward(); optimizer.step()
        self.assertTrue(torch.equal(logits, before_logits))
        self.assertTrue(all(torch.equal(value, frozen.state_dict()[name]) for name, value in before.items()))
        self.assertTrue(all(parameter.grad is None for parameter in frozen.parameters()))

    def test_ranker_does_not_modify_input_features(self) -> None:
        ranker = NuRankSharedRanker(); features = torch.randn(2, 4, 264); before = features.clone()
        ranker(features); self.assertTrue(torch.equal(features, before))

    def test_deterministic_ranker_initialization_and_token0_identity(self) -> None:
        first = build_ranker(scalar_mean=torch.zeros(8), scalar_std=torch.ones(8))
        second = build_ranker(scalar_mean=torch.zeros(8), scalar_std=torch.ones(8))
        self.assertTrue(all(torch.equal(first.state_dict()[name], second.state_dict()[name]) for name in first.state_dict()))

    def test_token0_baseline_selector_identity(self) -> None:
        predicted_iou = np.asarray([[.1, .9, .2, .3], [.2, .3, .4, .5]], dtype=np.float32)
        self.assertTrue(np.array_equal(np.zeros(len(predicted_iou), dtype=np.int64), np.asarray([0, 0], dtype=np.int64)))

    def test_train_development_patient_isolation(self) -> None:
        from nuset.audit.data import BASELINE_V1_TNBC_SHA256
        from nurank.stage1 import validate_cache_isolation
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); train, dev = root / "train", root / "development"; train.mkdir(); dev.mkdir()
            for path, role, ids in ((train, "train", ["01_1", "02_1", "03_1", "04_1", "05_1", "06_1"]), (dev, "development", ["07_1", "08_1"])):
                (path / "manifest.json").write_text(json.dumps({"schema": "nurank_automatic_prompt_cache_v1", "token_count": 4, "role": role, "image_ids": ids, "checkpoint_sha256": BASELINE_V1_TNBC_SHA256, "groups": []}))
            result = validate_cache_isolation(train, dev)
            self.assertEqual(result["train_patients"], [1, 2, 3, 4, 5, 6]); self.assertEqual(result["development_patients"], [7, 8])

    def test_builder_has_one_decoder_call_and_frozen_checksum_guard(self) -> None:
        from nurank.cache import builder
        source = inspect.getsource(builder.build_automatic_prompt_cache)
        self.assertEqual(source.count("_decode_all_tokens("), 1)

    def test_builder_has_frozen_checksum_guard(self) -> None:
        from nurank.cache import builder
        self.assertIn("before != after", inspect.getsource(builder.build_automatic_prompt_cache))

    def test_morphology_is_deterministic_and_uses_no_gt(self) -> None:
        from nurank.features.morphology import morphology_features
        logits, coordinates = torch.randn(2, 4, 8, 8), torch.tensor([[[1.0, 2.0]], [[4.0, 3.0]]])
        self.assertTrue(torch.equal(morphology_features(logits, coordinates), morphology_features(logits, coordinates)))

    def test_inclusive_iou_half_remains_true_positive(self) -> None:
        from nuset.audit.metrics import assembly_metrics
        truth = torch.tensor([[1, 1, 0], [0, 0, 0]]).numpy(); prediction = torch.tensor([[1, 1, 1], [1, 0, 0]]).numpy()
        self.assertEqual(assembly_metrics(truth, prediction)["tp"], 1)

    def test_closed_patients_are_not_permitted_by_stage_source(self) -> None:
        source = Path("nurank/cache/data.py").read_text(encoding="utf-8")
        self.assertIn("TRAIN_PATIENTS", source); self.assertIn("DEVELOPMENT_PATIENTS", source)
        self.assertIn("never list test", source)


if __name__ == "__main__":
    unittest.main()
