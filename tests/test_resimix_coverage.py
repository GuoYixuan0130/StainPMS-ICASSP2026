"""Synthetic-array tests for the ResiMix static-coverage cache boundary."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import stat
import tempfile
import unittest
from uuid import uuid4

import numpy as np

from resimixpms.coverage import (
    COVERAGE_DIRECTORY_NAME,
    MANIFEST_FILENAME,
    CoverageGenerationError,
    CoverageIntegrityError,
    StaticCoverageCache,
    StaticCoverageWriter,
    validate_static_coverage_cache,
)


def _make_tree_writable(root: Path) -> None:
    """Allow TemporaryDirectory cleanup after the cache intentionally freezes it."""
    if not root.exists():
        return
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts)):
        try:
            os.chmod(path, stat.S_IREAD | stat.S_IWRITE | stat.S_IEXEC)
        except OSError:
            pass
    try:
        os.chmod(root, stat.S_IREAD | stat.S_IWRITE | stat.S_IEXEC)
    except OSError:
        pass


class StaticCoverageCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        # Use the sandbox's designated temporary area.  The writer itself
        # creates the final cache directory, which must not already exist.
        self._temp_root = Path(tempfile.mkdtemp(prefix=f"resimix_coverage_test_{uuid4().hex}_"))
        self.cache_dir = self._temp_root / "cache"
        self.shapes = {"tnbc_patient_1_img_a": (4, 5), "tnbc_patient_2_img_b": (3, 4)}
        self.provenance = {
            "checkpoint_sha256": "a" * 64,
            "training_manifest_sha256": "b" * 64,
            "coverage_mode": "single_static_pass",
        }

    def tearDown(self) -> None:
        _make_tree_writable(self._temp_root)
        shutil.rmtree(self._temp_root, ignore_errors=True)

    def test_seal_records_shape_hash_and_readonly_shared_cache(self) -> None:
        writer = StaticCoverageWriter.create(
            self.cache_dir, self.shapes, provenance=self.provenance
        )
        full_prediction = np.array(
            [[0, 1, 1, 0, 0], [0, 1, 0, 2, 2], [0, 0, 0, 2, 2], [3, 3, 0, 0, 0]],
            dtype=np.int32,
        )
        writer.write_full("tnbc_patient_1_img_a", full_prediction)
        writer.write_full("tnbc_patient_2_img_b", np.zeros((3, 4), dtype=np.int32))
        cache = writer.seal()

        self.assertEqual(cache.image_ids, tuple(sorted(self.shapes)))
        loaded = cache.load("tnbc_patient_1_img_a", verify_sha256=True)
        np.testing.assert_array_equal(loaded, full_prediction)
        self.assertFalse(loaded.flags.writeable)
        with self.assertRaises(ValueError):
            loaded[0, 0] = 7

        manifest = validate_static_coverage_cache(
            self.cache_dir,
            expected_image_shapes=self.shapes,
            expected_provenance=self.provenance,
        )
        record = manifest.images["tnbc_patient_1_img_a"]
        self.assertEqual(record.shape, (4, 5))
        self.assertEqual(len(record.sha256), 64)
        self.assertEqual(record.write_mode, "full")
        self.assertEqual(record.written_pixels, 20)
        self.assertFalse(os.stat(self.cache_dir / MANIFEST_FILENAME).st_mode & stat.S_IWUSR)
        self.assertFalse(
            os.stat(self.cache_dir / COVERAGE_DIRECTORY_NAME / Path(record.file).name).st_mode
            & stat.S_IWUSR
        )

        with self.assertRaises(CoverageGenerationError):
            StaticCoverageWriter.create(self.cache_dir, self.shapes, provenance=self.provenance)

    def test_fixed_crop_writes_preserve_unrelated_regions(self) -> None:
        shapes = {"monuseg_holdout_01": (6, 8)}
        writer = StaticCoverageWriter.create(self.cache_dir, shapes, provenance=self.provenance)
        first_crop = np.array([[7, 7, 0], [7, 0, 0]], dtype=np.int32)
        second_crop = np.array([[9, 9], [0, 9]], dtype=np.int32)
        writer.write_crop("monuseg_holdout_01", (2, 1, 5, 3), first_crop)
        writer.write_crop("monuseg_holdout_01", (0, 4, 2, 6), second_crop)
        cache = writer.seal()

        expected = np.zeros((6, 8), dtype=np.int32)
        expected[1:3, 2:5] = first_crop
        expected[4:6, 0:2] = second_crop
        loaded = cache.load("monuseg_holdout_01")
        np.testing.assert_array_equal(loaded, expected)

        record = cache.manifest.images["monuseg_holdout_01"]
        self.assertEqual(record.write_mode, "fixed_crops")
        self.assertEqual(record.written_pixels, first_crop.size + second_crop.size)
        self.assertEqual(record.crop_boxes, ((2, 1, 5, 3), (0, 4, 2, 6)))

    def test_conflicting_crop_overlap_is_rejected_without_last_write_wins(self) -> None:
        writer = StaticCoverageWriter.create(
            self.cache_dir, {"img": (5, 5)}, provenance=self.provenance
        )
        writer.write_crop("img", (1, 1, 4, 4), np.ones((3, 3), dtype=np.int32))
        with self.assertRaises(CoverageGenerationError):
            writer.write_crop("img", (2, 2, 5, 5), np.full((3, 3), 2, dtype=np.int32))
        with self.assertRaises(CoverageGenerationError):
            writer.write_full("img", np.zeros((5, 5), dtype=np.int32))

    def test_integrity_validation_detects_tamper_and_wrong_expected_manifest(self) -> None:
        writer = StaticCoverageWriter.create(
            self.cache_dir, {"img": (3, 3)}, provenance=self.provenance
        )
        writer.write_full("img", np.eye(3, dtype=np.int32))
        cache = writer.seal()
        record = cache.manifest.images["img"]
        data_path = self.cache_dir / record.file

        with self.assertRaises(CoverageIntegrityError):
            StaticCoverageCache.open(
                self.cache_dir,
                expected_image_shapes={"other_allowed_image": (3, 3)},
                expected_provenance=self.provenance,
            )
        with self.assertRaises(CoverageIntegrityError):
            StaticCoverageCache.open(
                self.cache_dir,
                expected_image_shapes={"img": (3, 3)},
                expected_provenance={"checkpoint_sha256": "c" * 64},
            )

        # Simulate an external mutation; the hash check, rather than trust in
        # filesystem permissions alone, must reject it.
        os.chmod(data_path, stat.S_IREAD | stat.S_IWRITE)
        np.save(data_path, np.zeros((3, 3), dtype=np.int32), allow_pickle=False)
        with self.assertRaises(CoverageIntegrityError):
            validate_static_coverage_cache(self.cache_dir, require_readonly=False)

    def test_incomplete_generation_cannot_seal_or_be_restarted(self) -> None:
        writer = StaticCoverageWriter.create(
            self.cache_dir,
            {"first": (2, 2), "second": (2, 2)},
            provenance=self.provenance,
        )
        writer.write_full("first", np.zeros((2, 2), dtype=np.int32))
        with self.assertRaises(CoverageGenerationError):
            writer.seal()
        with self.assertRaises(CoverageGenerationError):
            StaticCoverageWriter.create(
                self.cache_dir,
                {"first": (2, 2), "second": (2, 2)},
                provenance=self.provenance,
            )
        with self.assertRaises(CoverageIntegrityError):
            validate_static_coverage_cache(self.cache_dir, require_readonly=False)


if __name__ == "__main__":
    unittest.main()
