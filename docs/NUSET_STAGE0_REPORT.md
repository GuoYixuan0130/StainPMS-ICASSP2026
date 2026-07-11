# REPORT FOR PROJECT LEAD — NUSET STAGE 0

Date: 2026-07-11

## 1. Verdict

**STRONG GO** under the pre-registered NuSet Stage 0 criteria.  This is
evidence for designing a future multimask-training method only; it is not
authorization to train, run TNBC development/test, run MoNuSeg, alter a loss,
or resume StainRoute, PromptCredit-v1, or PromptQ.

## 2. Git SHA and environment

The audited code SHA was `5df6ed4816ec9ab059bb8e565541cdc5dabeb0de` on
`research/nuset`.  The AutoDL environment is recorded without alteration in
the immutable artifact `environment.txt`.

## 3. Fixed six-image manifest

The audit restored the existing Stage-0 fixed list without reselection:

```text
02_1, 05_1, 04_5, 05_4, 03_3, 04_7
```

It used only TNBC patients 1-6 and the frozen baseline-v1 checkpoint SHA256
`44a3cb3e93051301d789e44f93769588abfa727d7a174b0270f55305ef023781`.

## 4. Token-0 baseline equivalence

From one `predict_masks()` call, token 0 exactly reproduced the single-token
selector:

- low-resolution selector error: `0`
- upsampled-logit selector error: `0`
- predicted-IoU selector error: `0`
- hard masks, bbox/assembly, final instance map, and metrics: identical

No second single-mask decoder forward was issued.

## 5. GT-associated prompt headroom

Across 790 GT-associated prompts:

- All-Oracle mean ΔIoU: `+0.01906`; median: `+0.01352`
- All-Pred mean ΔIoU: `+0.00109`
- All-Oracle non-token-0 best: `80.51%`
- All-Oracle ΔIoU >= `.02`: `36.20%`
- token-0 below `.5` while another token reaches `.5`: `0.63%`

## 6. Automatic-prompt headroom

Across 838 automatic prompts inside a GT instance:

- All-Oracle mean ΔIoU: `+0.01817`; median: `+0.01297`
- All-Pred mean ΔIoU: `+0.00092`
- All-Oracle ΔIoU >= `.01/.02/.05`: `56.32% / 35.80% / 5.73%`
- All-Oracle non-token-0 best: `82.46%`
- token collapse: `0%`

There were 22 unmatched automatic prompts.  All-Pred selected a non-token-0
mask for `40.91%`; its selected mask was larger than token 0 for `22.73%`.
They were not removed or GT-corrected.

## 7. Full assembly

| Path | Dice | AJI | DQ | SQ | PQ | ΔPQ vs Single |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline-Single | .88915 | .77117 | .93795 | .79879 | .74999 | — |
| Deployable-All-Pred | .88956 | .77252 | .94144 | .79962 | .75365 | +.00366 |
| Oracle-All | .89318 | .77918 | .94113 | .80491 | .75845 | +.00846 |

Oracle-All PQ was non-decreasing on 5/6 images.  Its largest positive-image
contribution was `34.72%`, below the 60% limit.  Deployable-All-Pred recovered
`43.32%` of aggregate Oracle-All ΔPQ headroom.

## 8. IoU-head ranking

On automatic matched prompts, predicted-IoU versus true-IoU Spearman was
`.62184` and Pearson `.65385`, but top-1 token-selection accuracy was only
`.19093`; mean oracle regret was `.01725` and MRR `.47613`.  ECE was `.23534`.
Thus masks are diverse and useful, while the existing predicted-IoU head does
not reliably rank the best available token.

## 9. Token diversity and collapse

For automatic matched prompts, mean pairwise hard-mask IoU ranged from `.89962`
(tokens 1/2) to `.96298` (tokens 0/3).  Pairwise boundary disagreement was
nonzero for all pairs.  Collapse was `0%`; diversity is sufficient for the
pre-registered criterion.

## 10. Runtime, memory, and calls

- wall time: `19.58 s` (under one-hour cap)
- peak GPU memory: `3,480,961,536 B`
- maximum all-token output tensor memory: `67,961,808 B`
- image encoder calls: 87
- prompt encoder calls: 87
- mask decoder calls: 87
- all-token/single-path call counts: identical
- measured exposure overhead ratio: `1.000086` (under 5%)

## 11. Per-image statistics

Oracle-All ΔPQ by fixed image: `02_1 +.01143`, `03_3 +.00684`,
`04_5 -.03105`, `04_7 +.02840`, `05_1 +.02129`, `05_4 +.01384`.
The full rows are preserved in `per_image_metrics.csv`.

## 12. Tests

All 9 NuSet tests passed in the AutoDL PyTorch environment, including
all-token extraction, one prompt/decoder call, token-0 contract, deterministic
rerun, inclusive IoU `.5`, unmatched-GT isolation, frozen-state guard, and
closed-patient guard.

## 13. Artifact paths and checksums

Formal immutable artifact directory:

```text
logs/nuset/stage0/20260711_5df6ed4/
```

Selected SHA256 entries:

```text
report.json                    ad00d66a82f7a35600841fca5429588c3e9b6596b04783ca19749a92f2c9204e
assembly_summary.json          702fde5ebe74125843c6442598f3f38303015c348066f94e07834c38388c382b
headroom_summary.json          48dd46ba607b1aa1d9f5f5fc223cfaf8ebc9f9e999f7fb2aa1b3c3284f74b903
iou_head_ranking.json          d890b14b056140621b10c98d651cdcdf6e0c815af3f5ac4bfd4fd75542b78e60
per_image_metrics.csv          6fbdff65e985ec3297921de10f11acd9081a5e277d6bd350964dd0135ce82ab8
runtime_summary.json           f8f2eb6e6488f81fac5ac648be030344c91b32344c1cbf0bf0430bb34fc7ba40
```

The local `temple/` handoff copies were read-only and were neither modified
nor staged.

## 14. STRONG GO / CONDITIONAL / NO-GO

All eight STRONG GO conditions passed.  No NO-GO condition fired.

## 15. Recommendation

It is worth designing a narrowly scoped multimask-training method, with the
ranking/calibration gap as the primary target and without adding decoder calls.
Do not implement or train it until a further project-lead authorization.
