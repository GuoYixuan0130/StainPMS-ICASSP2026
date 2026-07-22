import json
import tempfile
import unittest
from pathlib import Path

from stainpms.warmstart_protocol import (
    MONUSEG_TRAIN37_SAMPLE_IDS,
    build_coverage_manifest,
    new_timing_runtime_stats,
    timing_audit_isolation,
    validate_train_manifest_identity,
    verify_coverage_manifest,
)


class WarmStartProtocolTests(unittest.TestCase):
    def write_manifest(self, root, dataset, records, **extra):
        path = root / "train.json"
        payload = {
            "dataset": dataset,
            "record_count": len(records),
            "records": records,
            **extra,
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_timing_runtime_disables_and_attests_smoke_only_audits(self):
        runtime_stats = new_timing_runtime_stats()
        self.assertEqual(timing_audit_isolation(runtime_stats)["status"], "pass")
        runtime_stats["gradient_audit"] = {"step_count": 1}
        result = timing_audit_isolation(runtime_stats)
        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["unexpected_audit_records"], ["gradient_audit"])

    def test_tnbc_rejects_sealed_identity_before_file_access(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = [
                {
                    "sample_id": f"{patient:02d}_{index}",
                    "patient": patient,
                    "image_path": f"missing/{patient:02d}_{index}.png",
                    "label_path": f"missing/{patient:02d}_{index}.mat",
                }
                for index, patient in enumerate([1] * 29 + [9])
            ]
            path = self.write_manifest(root, "tnbc", records)
            with self.assertRaisesRegex(ValueError, "rejects non-training patient"):
                validate_train_manifest_identity(path, "tnbc")

    def test_monuseg_rejects_test_path_before_file_access(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = [
                {
                    "sample_id": sample_id,
                    "image_path": f"train/images/{index}.tif",
                    "label_path": f"train/labels/{index}.mat",
                }
                for index, sample_id in enumerate(sorted(MONUSEG_TRAIN37_SAMPLE_IDS))
            ]
            records[5]["image_path"] = "test/images/forbidden.tif"
            path = self.write_manifest(root, "monuseg", records)
            with self.assertRaisesRegex(ValueError, "test identity rejected"):
                validate_train_manifest_identity(path, "monuseg")

    def test_monuseg_rejects_nonfrozen_37_identity_before_file_access(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample_ids = sorted(MONUSEG_TRAIN37_SAMPLE_IDS)
            sample_ids[0] = "unknown-train-case"
            records = [
                {
                    "sample_id": sample_id,
                    "image_path": f"missing/train/images/{index}.tif",
                    "label_path": f"missing/train/labels/{index}.mat",
                }
                for index, sample_id in enumerate(sample_ids)
            ]
            path = self.write_manifest(root, "monuseg", records)
            with self.assertRaisesRegex(ValueError, "differs from frozen train37"):
                validate_train_manifest_identity(path, "monuseg")

    def test_shared_coverage_manifest_is_hash_locked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = [
                {
                    "sample_id": sample_id,
                    "image_path": f"train/images/{index}.tif",
                    "label_path": f"train/labels/{index}.mat",
                }
                for index, sample_id in enumerate(sorted(MONUSEG_TRAIN37_SAMPLE_IDS))
            ]
            train_path = self.write_manifest(root, "monuseg", records)
            identity = validate_train_manifest_identity(train_path, "monuseg")
            cache = root / "coverage"
            cache.mkdir()
            for sample_id in identity["sample_ids"]:
                (cache / f"{sample_id}.npy").write_bytes(sample_id.encode("utf-8"))
            checkpoint = root / "checkpoint.pth"
            checkpoint.write_bytes(b"checkpoint")
            checkpoint_sha = (
                "47320987f9a49d5b00119b960f247a956773f57543982b8bfcb6da5bb3afd9ef"
            )
            payload = build_coverage_manifest(
                cache_dir=cache,
                train_manifest_identity=identity,
                dataset="monuseg",
                checkpoint_path=checkpoint,
                checkpoint_sha256=checkpoint_sha,
                wall_seconds=1.25,
                repository={"branch": "branch", "commit": "commit"},
                command=["main.py"],
            )
            coverage_path = root / "coverage.json"
            coverage_path.write_text(json.dumps(payload), encoding="utf-8")
            verified = verify_coverage_manifest(
                coverage_path,
                train_manifest_identity=identity,
                checkpoint_sha256=checkpoint_sha,
                dataset="monuseg",
            )
            self.assertEqual(verified["record_count"], 37)
            changed = identity["sample_ids"][0]
            (cache / f"{changed}.npy").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                verify_coverage_manifest(
                    coverage_path,
                    train_manifest_identity=identity,
                    checkpoint_sha256=checkpoint_sha,
                    dataset="monuseg",
                )


if __name__ == "__main__":
    unittest.main()
