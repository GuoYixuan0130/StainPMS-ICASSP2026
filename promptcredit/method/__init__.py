"""PromptCredit v1 method primitives, isolated from default StainPMS behavior."""

from .checkpoint import load_point_checkpoint_compat
from .credit import directional_credit, gather_nearest_coordinates, legacy_nearest_indices
from .freeze import (
    configure_promptcredit_v1_trainable,
    configure_promptq_trainable,
    frozen_parameters_have_no_grad,
    module_state_sha256,
    module_state_sha256_excluding,
    optimizer_excludes_frozen,
)
from .quality import (
    build_quality_targets,
    prompt_ranking_scores,
    quality_focal_loss,
    quality_focal_loss_with_audit,
    utility_target_from_hard_iou,
)

__all__ = [
    "build_quality_targets",
    "configure_promptcredit_v1_trainable",
    "configure_promptq_trainable",
    "directional_credit",
    "gather_nearest_coordinates",
    "frozen_parameters_have_no_grad",
    "legacy_nearest_indices",
    "load_point_checkpoint_compat",
    "module_state_sha256",
    "module_state_sha256_excluding",
    "optimizer_excludes_frozen",
    "prompt_ranking_scores",
    "quality_focal_loss",
    "quality_focal_loss_with_audit",
    "utility_target_from_hard_iou",
]
