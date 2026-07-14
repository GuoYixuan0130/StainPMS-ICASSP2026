"""Independent ResiMix metric/schedule regressions (no retired-route code)."""
from __future__ import annotations

from pathlib import Path
import importlib.util
import sys
import types
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from resimixpms.experiment import EvaluationSchedule, parse_epoch_schedule  # noqa: E402

_STATS_PATH = ROOT / "sam2_train" / "modeling" / "stats_utils.py"
try:
    import cv2  # noqa: F401
except ModuleNotFoundError:
    # get_fast_pq itself has no OpenCV dependency; this minimal import stub
    # keeps its direct inclusive-threshold regression runnable in lightweight
    # local CI without exercising any cv2-backed utility.
    sys.modules.setdefault("cv2", types.ModuleType("cv2"))
_SPEC = importlib.util.spec_from_file_location("resimix_stats_utils", _STATS_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"cannot load metric module: {_STATS_PATH}")
_STATS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_STATS)
get_fast_pq = _STATS.get_fast_pq


class ResiMixMetricTest(unittest.TestCase):
    def test_iou_exactly_half_is_an_inclusive_match(self):
        gt = np.asarray([[1, 1, 1, 0]], dtype=np.int32)
        pred = np.asarray([[0, 1, 1, 1]], dtype=np.int32)
        (dq, sq, pq), pairing = get_fast_pq(gt, pred, match_iou=0.5)
        self.assertEqual(pairing[0], [1])
        self.assertEqual(pairing[1], [1])
        self.assertEqual(pairing[2], [])
        self.assertEqual(pairing[3], [])
        self.assertAlmostEqual(dq, 1.0)
        self.assertAlmostEqual(sq, 0.5, places=5)
        self.assertAlmostEqual(pq, 0.5, places=5)

    def test_registered_schedule_has_frozen_semantics(self):
        self.assertEqual(parse_epoch_schedule("10,0,2,2", 10), (0, 2, 10))
        schedule = EvaluationSchedule.from_cli(10, "0,2,4,6,8,10")
        self.assertTrue(schedule.should_evaluate(0))
        self.assertTrue(schedule.should_evaluate(10))
        self.assertFalse(schedule.should_evaluate(3))
        with self.assertRaises(ValueError):
            parse_epoch_schedule("11", 10)


if __name__ == "__main__":
    unittest.main()
