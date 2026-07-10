"""StainRoute utilities.

The package is intentionally independent from :mod:`stainpqr`.  It contains
only Stage 0 metric code until baseline reconciliation has passed.
"""

from .oracle import matched_iou_sum, pq_factorized

__all__ = ["matched_iou_sum", "pq_factorized"]
