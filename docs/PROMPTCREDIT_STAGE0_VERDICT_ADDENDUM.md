# PromptCredit Stage 0 Verdict Addendum

Date: 2026-07-11

## Corrected verdict

The project lead's manual verdict for PromptCredit Stage 0 is **GO**.  This
authorizes implementation and the two-crop mechanism smoke test only.  It does
not authorize a final-method claim, full TNBC training, MoNuSeg, calibration,
or test access.

The pre-registered rule is:

```text
GO = actionable gradient C AND (assignment gap A OR quality gap B) AND acceptable cost
```

The frozen Stage 0 measurements are A=false, B=true, C=true, and
acceptable_cost=true.  Therefore the correct result is GO.

## Why the original automatic verdict was wrong

The original implementation returned `GO` only when both A and B were true,
and returned `CONDITIONAL GO` whenever exactly one was true.  That was stricter
than the pre-registered rule, which requires at least one of A or B.  The code
now implements the truth table directly.  `CONDITIONAL GO` remains available
only when a caller explicitly records weak single-gap evidence; it is not
inferred from the number of positive gap categories.

## Preserved original evidence

No raw Stage 0 artifact was modified and no model was re-run.  In particular,
the original `temple/report.json` remains in place with SHA256:

```text
6f99fbf15efb85b8b24faef9c682b39175b619473b4598c8fc1d8bce1893be1e
```

The original run used code Git SHA
`af0fc4a7a5c64e9456c93b7ddf035a50313c72b9`.  Its numeric measurements,
including assignment collision excess 0.253%, nearest/Hungarian disagreement
8.23%, ECE 0.1703, and 100% finite nonzero coordinate gradients, are unchanged.

## Scope interpretation

Assignment gap was not established, so PromptCredit v1 must not replace the
nearest assignment with Hungarian matching or claim severe duplicate
supervision.  The supported mechanisms are instead point-score/mask-utility
misalignment and stable frozen-decoder coordinate gradients.
