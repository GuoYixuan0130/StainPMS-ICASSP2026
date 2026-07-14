"""Synthetic-array tests for the isolated ResiMix transplantation primitives."""

from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from resimixpms.transplant import (  # noqa: E402
    CONTEXT_FEATURE_NAMES,
    DonorGeometry,
    TransplantStats,
    annulus_mask,
    choose_host_candidate,
    composite_transplant,
    deterministic_candidate_centers,
    deterministic_donor_choice,
    deterministic_geometry,
    enumerate_legal_hosts,
    mask_medoid,
    normalized_donor_ratios,
    od_affine_stain_match,
    placement_for_mask,
    quality_reject,
    render_placed_mask,
    transform_donor,
)


def disk(shape, center, radius):
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    return (yy - center[0]) ** 2 + (xx - center[1]) ** 2 <= radius**2


class ResiMixTransplantTest(unittest.TestCase):
    def setUp(self):
        self.donor_mask = disk((17, 17), (8, 8), 4)
        self.donor_ring = annulus_mask(self.donor_mask, width=3)
        self.donor_rgb = np.full((17, 17, 3), (205, 180, 165), dtype=np.float64)
        self.donor_rgb[self.donor_mask] = (55, 72, 115)

    def test_geometry_is_restricted_and_reproducible(self):
        geometry = DonorGeometry(rotation_deg=90, flip="horizontal", scale=1.0)
        transformed = transform_donor(
            self.donor_rgb, self.donor_mask, self.donor_ring, geometry
        )
        expected_rgb = np.flip(np.rot90(self.donor_rgb, 1), axis=1)
        expected_mask = np.flip(np.rot90(self.donor_mask, 1), axis=1)
        self.assertTrue(np.array_equal(transformed.mask, expected_mask))
        self.assertTrue(np.allclose(transformed.rgb, expected_rgb))
        self.assertFalse(np.any(transformed.mask & transformed.annulus))
        self.assertEqual(int(transformed.mask.sum()), int(self.donor_mask.sum()))

        scaled = transform_donor(
            self.donor_rgb,
            self.donor_mask,
            self.donor_ring,
            DonorGeometry(rotation_deg=180, flip="vertical", scale=1.1),
        )
        self.assertGreater(scaled.mask.sum(), 0)
        self.assertFalse(np.any(scaled.mask & scaled.annulus))

        self.assertEqual(deterministic_geometry(3407, "crop-001"), deterministic_geometry(3407, "crop-001"))
        with self.assertRaises(ValueError):
            DonorGeometry(rotation_deg=45).validate()
        with self.assertRaises(ValueError):
            DonorGeometry(scale=1.11).validate()

    def test_od_matching_and_two_pixel_alpha_composite(self):
        host = np.full((65, 65, 3), (160, 135, 125), dtype=np.float64)
        host_instance = disk((65, 65), (32, 32), 5)
        host_ring = annulus_mask(host_instance, width=3)
        matched = od_affine_stain_match(
            self.donor_rgb,
            self.donor_ring,
            host,
            host_ring,
            transplant_mask=self.donor_mask,
        )
        self.assertTrue(np.all(matched.channel_scale >= 0.75))
        self.assertTrue(np.all(matched.channel_scale <= 1.33))
        self.assertTrue(
            np.allclose(
                matched.od[self.donor_ring].mean(axis=0),
                matched.host_ring_mean,
                atol=1e-10,
            )
        )
        composite = composite_transplant(host, matched.rgb, self.donor_mask, (32, 32))
        self.assertEqual(int(composite.placed_mask.sum()), int(self.donor_mask.sum()))
        self.assertTrue(np.array_equal(composite.rgb[~composite.placed_mask], host[~composite.placed_mask]))
        self.assertEqual(float(composite.alpha[32, 32]), 1.0)
        edge_alpha = composite.alpha[composite.placed_mask]
        self.assertGreater(edge_alpha.min(), 0.0)
        self.assertLess(edge_alpha.min(), 1.0)

    def test_host_constraints_context_ranking_and_deterministic_selection(self):
        host = np.full((96, 96, 3), (170, 150, 135), dtype=np.float64)
        # A small deterministic gradient makes context distances non-degenerate.
        host[..., 0] += np.arange(96, dtype=np.float64)[None, :] / 16.0
        donor = transform_donor(
            self.donor_rgb,
            self.donor_mask,
            self.donor_ring,
            DonorGeometry(),
        )
        instance_map = np.zeros((96, 96), dtype=np.int32)
        instance_map[disk((96, 96), (48, 48), 4)] = 1
        tissue = np.ones((96, 96), dtype=bool)
        coverage = np.zeros((96, 96), dtype=np.uint8)
        blocked = placement_for_mask(donor.mask, (16, 16), coverage.shape)
        coverage[render_placed_mask(donor.mask, blocked, coverage.shape)] = 1
        mean = np.zeros(len(CONTEXT_FEATURE_NAMES), dtype=np.float64)
        std = np.ones(len(CONTEXT_FEATURE_NAMES), dtype=np.float64)
        candidates = enumerate_legal_hosts(
            host,
            donor,
            instance_map,
            coverage,
            tissue,
            mean,
            std,
            centers=((16, 16), (20, 40), (48, 62), (90, 90)),
            seed=3407,
            sample_key="synthetic-crop",
        )
        centers = {candidate.center_yx for candidate in candidates}
        self.assertNotIn((16, 16), centers, "static coverage must block a placement")
        self.assertIn((20, 40), centers)
        self.assertIn((48, 62), centers)
        adjacent = choose_host_candidate(candidates, "adjacent", 3407, "synthetic-crop")
        self.assertIsNotNone(adjacent)
        self.assertEqual(adjacent.used_mode, "adjacent")
        self.assertEqual(
            adjacent,
            choose_host_candidate(candidates, "adjacent", 3407, "synthetic-crop"),
        )
        generated_a = deterministic_candidate_centers(tissue, 3407, "synthetic-crop")
        generated_b = deterministic_candidate_centers(tissue, 3407, "synthetic-crop")
        self.assertEqual(generated_a, generated_b)
        self.assertLessEqual(len(generated_a), 32)

    def test_quality_medoid_donor_shortage_and_statistics(self):
        source = disk((17, 17), (8, 8), 4)
        occupied = np.zeros_like(source, dtype=np.int32)
        occupied[8, 8] = 1
        rejected = quality_reject(
            source,
            source,
            occupied_instances=occupied,
            seam_gradient=11.0,
            natural_boundary_p95=10.0,
            context_distance=4.0,
            legal_context_p95=3.0,
        )
        self.assertFalse(rejected.accepted)
        self.assertIn("instance_overlap", rejected.reasons)
        self.assertIn("seam_gradient", rejected.reasons)
        self.assertIn("context_distance", rejected.reasons)
        accepted = quality_reject(source, source, composited_rgb=self.donor_rgb)
        self.assertTrue(accepted.accepted)
        medoid = mask_medoid(source)
        self.assertTrue(source[medoid])

        banks = {
            "Missed": ({"donor_id": "m-1"},),
            "IoU-Cliff": (),
            "Low-Quality Matched": ({"donor_id": "l-1"},),
        }
        ratios = normalized_donor_ratios(banks)
        self.assertAlmostEqual(sum(ratios.values()), 1.0)
        self.assertNotIn("IoU-Cliff", ratios)
        self.assertEqual(
            deterministic_donor_choice(banks, 3407, "crop-1"),
            deterministic_donor_choice(banks, 3407, "crop-1"),
        )

        stats = TransplantStats()
        stats.record(rejected, donor_category="Missed", host_mode="adjacent")
        stats.record(
            accepted,
            donor_category="IoU-Cliff",
            host_mode="isolated",
            synthetic_prompt_added=True,
        )
        summary = stats.as_dict()
        self.assertEqual(summary["accepted_transplants"], 1)
        self.assertEqual(summary["synthetic_prompts_added"], 1)
        self.assertEqual(summary["rejection_reasons"]["instance_overlap"], 1)


if __name__ == "__main__":
    unittest.main()
