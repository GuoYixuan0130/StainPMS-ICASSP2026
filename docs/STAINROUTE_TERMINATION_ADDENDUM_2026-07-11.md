# StainRoute Termination Addendum — 2026-07-11

## Decision

StainRoute is formally terminated as of 2026-07-11.  This is a project-scope
decision, not a deletion or invalidation of the work completed to date.  All
StainRoute code, configurations, logs, artifacts, and historical results are
to be retained unchanged for provenance and possible future review.

No further StainRoute oracle, router, training, inference, or MoNuSeg
experiment may be launched.  In particular, the prepared MoNuSeg Futility
Gate 0 and Gate 1 commands must not be run.

## Evidence considered

The official TNBC router-train results showed limited remaining oracle
headroom:

| Evaluation | Result |
| --- | ---: |
| Joint Oracle@4 Delta PQ | +0.00119 |
| Calibration Delta PQ | +0.00322 |
| SPLIT Oracle@4 Delta PQ | 0.00000 |

MoNuSeg completed smoke validation only; no formal MoNuSeg oracle result was
completed.

## Rationale

The TNBC oracle headroom is too small to justify continued routing work, and
the zero gain from every SPLIT Oracle@4 outcome further weakens the case for a
router-based intervention.  Completing formal MoNuSeg validation was expected
to require approximately 25–50 GPU hours.  Given the weak observed upside,
that validation cost is not justified.

## Preservation and transition

This addendum deliberately makes no change to existing StainRoute source,
reports, experiment records, or artifacts.  It records the stop decision only.
Subsequent exploratory work proceeds separately under PromptCredit and must
not revive or extend StainRoute experiments without a new written
authorization from the project lead.
