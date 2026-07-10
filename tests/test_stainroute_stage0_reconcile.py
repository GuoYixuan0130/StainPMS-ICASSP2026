import json
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.stainroute_stage0_reconcile import _load_main_metrics


class Stage0ReconcileTest(unittest.TestCase):
    def test_prefers_unrounded_main_metric_artifact(self) -> None:
        artifact_dir = Path("synthetic-artifacts")
        exact_metrics = {
            "dice1": 1.0,
            "dice2": 1.0,
            "aji": 1.0,
            "aji_p": 1.0,
            "dq": 1.0,
            "sq": 0.999999000001,
            "pq": 0.999999000001,
        }
        with (
            patch.object(Path, "is_file", return_value=True),
            patch.object(Path, "read_text", return_value=json.dumps({"metrics": exact_metrics})),
        ):
            metrics, source = _load_main_metrics(artifact_dir, None)

        self.assertEqual(source, str(artifact_dir / "main_eval_metrics.json"))
        self.assertEqual(metrics, exact_metrics)
