# SafePMS Stage 0 closure: NO-GO

## Decision

SafePMS is closed.  The project lead accepted the preregistered Stage 0
`NO-GO` decision on 2026-07-12.  Do not enter Stage 1, retune coefficients,
replace coverage maps, rerun another seed, or alter the inference path.

## Executed run

- Code branch: `research/safepms`
- SafePMS code commits: `6fa6db9` and `02969c0`
- Starting canonical baseline commit: `2a1348cb7a1158a6f77aae2f92c168f9552d8068`
- Checkpoint: `tnbc_pms_best_e156.pth`
- Checkpoint SHA256: `44a3cb3e93051301d789e44f93769588abfa727d7a174b0270f55305ef023781`
- Stage 0 artifact: `logs/safepms/stage0/safepms_20260712_155408/`
- Stage 1: not run

The continuation JSON was reconstructed from the documented preserve-v1
configuration because the complete historical e156 invocation was not
available in retained logs/checkpoint metadata.  This provenance limitation is
recorded; it is not a reason to retry the closed path.

## Stage 0 result

`report.json` recorded `"verdict": "NO-GO"` after 36 valid,
patient-balanced batches.  The implementation and integrity checks for finite
gradients, frozen modules, coverage artifacts, and zero Stage-0 optimizer
steps passed.

The preregistered gates that failed were:

- `conflict_median_cosine_le_neg_005 = false`.
- `projection_dot_ge_neg_1e7 = false`.
- `anchor_safety_contract = false`.

`projection_validation.json` reported a maximum negative anchor margin of
`-3.0517578125e-05`, a float32 projection residual beyond the strict
`-1e-7` reporting threshold.  Independently, the conflict-median-cosine gate
failed, so correcting numerical reporting would not authorize Stage 1.

## Custody

Keep the complete Stage 0 directory, including `report.json`, gradient CSVs,
`projection_validation.json`, `tests.txt`, and `SHA256SUMS`, immutable.  It is
evidence for the closed SafePMS path rather than a checkpoint for continued
method development.
