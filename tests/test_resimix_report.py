"""Regression test for mandatory ResiMix Stage-1 report cardinality/gates."""
from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
_SUMMARY_PATH = ROOT / "tools" / "summarize_resimix_stage1.py"
_SPEC = importlib.util.spec_from_file_location("resimix_stage_summary", _SUMMARY_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("cannot import ResiMix summary script")
_SUMMARY = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_SUMMARY)


class ReportGateTest(unittest.TestCase):
    def _write_run(self, root: Path, dataset: str, method: str, count: int, epochs: tuple[int, ...], delta: float) -> None:
        run = root / dataset / method
        run.mkdir(parents=True)
        names = [f"{dataset}_{index:02d}" for index in range(count)]
        for epoch in epochs:
            value = 0.20 + (delta if epoch == epochs[-1] else 0.0)
            payload = {"completed_epochs": epoch, **{metric: value for metric in _SUMMARY.METRICS}}
            (run / f"evaluation_epoch_{epoch:02d}.json").write_text(json.dumps(payload), encoding="utf-8")
            with (run / f"per_image_epoch_{epoch:02d}.csv").open("w", newline="", encoding="utf-8") as handle:
                fields = ["image", *_SUMMARY.METRICS, "tp", "fp", "fn"]
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for name in names:
                    writer.writerow({"image": name, **{metric: value for metric in _SUMMARY.METRICS}, "tp": 2, "fp": 1, "fn": 1})

    def test_report_requires_all_items_and_writes_patient_and_gate_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            development = [{"image_name": f"tnbc_{index:02d}.png", "patient_id": 7 + index % 2} for index in range(7)]
            (root / "tnbc_data_manifest.json").write_text(json.dumps({"development_records": development}), encoding="utf-8")
            for dataset, count in (("tnbc", 7), ("monuseg_lite", 12)):
                for method, delta in (("static_control", 0.0), ("resimix", 0.03)):
                    self._write_run(root, dataset, method, count, _SUMMARY.SCHEDULES[dataset], delta)
            saved_argv = sys.argv
            try:
                sys.argv = ["summarize_resimix_stage1.py", "--artifact-dir", str(root)]
                _SUMMARY.main()
            finally:
                sys.argv = saved_argv
            report = json.loads((root / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["gate"]["verdict"], "STRONG_GO")
            self.assertTrue((root / "tnbc_per_patient.csv").is_file())
            self.assertTrue((root / "monuseg_lite_per_patch.csv").is_file())
            self.assertEqual(report["tnbc"]["step0_equivalence"]["checked_items"], 7)

    def test_report_can_use_an_explicit_recovery_control_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            development = [{"image_name": f"tnbc_{index:02d}.png", "patient_id": 7 + index % 2} for index in range(7)]
            (root / "tnbc_data_manifest.json").write_text(json.dumps({"development_records": development}), encoding="utf-8")
            for method, delta in (("static_control", 0.0), ("resimix", 0.03)):
                self._write_run(root, "tnbc", method, 7, _SUMMARY.SCHEDULES["tnbc"], delta)
            self._write_run(root, "monuseg_lite", "static_control", 12, _SUMMARY.SCHEDULES["monuseg_lite"], 0.0)
            self._write_run(root, "monuseg_lite", "resimix", 12, _SUMMARY.SCHEDULES["monuseg_lite"], 0.03)
            recovery = root / "monuseg_lite" / "static_control_recovery"
            self._write_run(root, "monuseg_lite", "static_control_recovery", 12, _SUMMARY.SCHEDULES["monuseg_lite"], 0.02)
            saved_argv = sys.argv
            try:
                sys.argv = [
                    "summarize_resimix_stage1.py", "--artifact-dir", str(root),
                    "--monuseg-control-dir", str(recovery),
                ]
                _SUMMARY.main()
            finally:
                sys.argv = saved_argv
            report = json.loads((root / "report.json").read_text(encoding="utf-8"))
            self.assertAlmostEqual(report["monuseg_lite"]["control_best"]["aji"], 0.22)


if __name__ == "__main__":
    unittest.main()
