"""Training-only metric-aligned set supervision for StainPMS.

This package deliberately contains no inference helpers.  SetPMS is attached
only to the continuation training loop and the canonical inference path stays
unchanged.
"""

from .anchor import L2SPAnchor
from .loss import (
    SetPMSResult,
    compute_setpms_loss,
    foreground_probability,
    select_set_queries,
    unbalanced_sinkhorn_log,
)

__all__ = [
    "L2SPAnchor",
    "SetPMSResult",
    "compute_setpms_loss",
    "foreground_probability",
    "select_set_queries",
    "unbalanced_sinkhorn_log",
]
