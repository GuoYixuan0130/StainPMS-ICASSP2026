import unittest

from stainpms.warmstart_equivalence import compare_c0_reference, summarize_c1_scale


def smoke(arm, *, candidate=False):
    runtime = {
        "optimizer_steps": 1,
        "crop_batches_seen": 1,
        "native_candidate_decoder_calls": 2,
        "native_candidate_prompt_count": 7,
        "native_mask_token_count": 4,
        "original_supervised_mask_token": 0,
        "gradient_audit": {
            "group_l2_mean": {
                "point_head": 1.0,
                "mask_decoder": 2.0,
                "quality_head": 0.5,
            },
            "key_gradients": {
                "mask_token_embedding": {
                    "name": "mask_tokens.weight",
                    "shape": [2],
                    "values": [0.25, -0.5],
                }
            },
        },
    }
    if candidate:
        runtime["candidate_loss_audit"] = {
            "means": {"stainpms_loss": 1.0, "weighted_extra": 0.25},
            "groups": {},
        }
    return {
        "status": "complete",
        "training_configuration": {
            "arm": arm,
            "optimizer": {"type": "AdamW", "learning_rate": 1e-5},
            "scheduler": {"type": "MultiStepLR"},
            "data_order": {"seed": 3407, "shuffle": False},
        },
        "data": {
            "protocol_id": "train",
            "manifest_sha256": "manifest",
            "coverage": {"sha256": "coverage"},
        },
        "initialization": {"checkpoint_sha256": "checkpoint"},
        "determinism": {"seed": 3407},
        "losses": {"loss": 1.0, "loss_mask": 0.25},
        "runtime": runtime,
    }


class WarmStartEquivalenceTests(unittest.TestCase):
    def test_equal_reference_passes(self):
        legacy = smoke("legacy")
        c0 = smoke("c0")
        result = compare_c0_reference(legacy, c0)
        self.assertEqual(result["status"], "pass")

    def test_loss_difference_fails(self):
        legacy = smoke("legacy")
        c0 = smoke("c0")
        c0["losses"]["loss"] = 1.1
        result = compare_c0_reference(legacy, c0)
        self.assertEqual(result["status"], "fail")

    def test_c1_reports_gradient_ratios_and_forward_identity(self):
        c0 = smoke("c0")
        c1 = smoke("c1", candidate=True)
        c1["runtime"]["gradient_audit"]["group_l2_mean"]["mask_decoder"] = 3.0
        result = summarize_c1_scale(c0, c1)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["gradient_norm_ratio_C1_over_C0"]["mask_decoder"], 1.5)


if __name__ == "__main__":
    unittest.main()
