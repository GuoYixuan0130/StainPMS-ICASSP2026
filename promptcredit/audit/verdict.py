"""Pre-registered PromptCredit Stage 0 decision rule."""

from __future__ import annotations


def stage0_verdict(
    *,
    assignment_gap: bool,
    quality_gap: bool,
    actionable_gradient: bool,
    acceptable_cost: bool,
    single_gap_evidence_weak: bool = False,
) -> str:
    """Apply the project-lead-approved Stage 0 truth table.

    GO requires a usable coordinate gradient, acceptable cost, and at least one
    of the assignment or quality gaps.  ``CONDITIONAL GO`` is reserved for an
    explicitly marked weak single-gap result; it is never inferred from merely
    having one gap.
    """
    if not actionable_gradient or not acceptable_cost or not (assignment_gap or quality_gap):
        return "NO-GO"
    if single_gap_evidence_weak and assignment_gap != quality_gap:
        return "CONDITIONAL GO"
    return "GO"

