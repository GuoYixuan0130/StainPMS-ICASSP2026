"""Assignment diagnostics that reproduce the current training semantics."""

from .assignments import (
    AssignmentResult,
    collision_groups,
    hungarian_assignment,
    nearest_assignment,
    point_inside_mask,
)

__all__ = [
    "AssignmentResult",
    "collision_groups",
    "hungarian_assignment",
    "nearest_assignment",
    "point_inside_mask",
]

