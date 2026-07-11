# StainRoute MoNuSeg Futility Protocol

Status: **approved protocol amendment** on 2026-07-11, before inspecting any
MoNuSeg action utility. It supersedes the instruction to launch a complete
MoNuSeg Stage 1 oracle immediately. The existing TNBC formal results remain
unchanged. This amendment is resource-aware and does not revise the research
question, candidate generator, evaluation metric, data split, or paper claim.

## Scope and frozen constraints

- Dataset: **MoNuSeg `router_train` only** from
  `configs/splits/stainroute_monuseg.json`.
- Explicitly sealed: MoNuSeg calibration and official test.
- Frozen baseline: StainRoute Development Baseline v1, seed 3407, NMS 12,
  disabled TTA, texture/context enabled, batch size 1, inclusive one-to-one
  IoU >= 0.5 matching.
- Candidate generation remains GT-free. GT is available only after candidate
  generation for opportunity analysis, optimistic screening, and oracle labels.
- No router training, model fine-tuning, calibration evaluation, or test
  evaluation is authorized under this protocol.
- Do not modify generator thresholds, budget definitions, action families, or
  the fixed pilot list after any MoNuSeg action utility is observed.

If an old full-MoNuSeg job was already launched, finish its active
image/action microbatch, preserve any artifacts already written, and stop
before the next image. Do not delete an artifact or use its utility to choose
pilot images.

## Gate 0 — runtime diagnosis

Select one router-train image using the precommitted seed-3407 ID-only rule.
Profile separately:

- first-pass image encoding;
- candidate generation;
- ADD decoding;
- SPLIT decoding;
- assembly;
- exact/beam oracle evaluation;
- I/O.

Record encoder calls and verify each tile embedding is encoded once. The
current action decoder uses prompt microbatches per tile. Where available, run
one batch-vs-single action-decode check with mask/logit maximum absolute error,
predicted-IoU maximum absolute error, assembly equality, action-utility
equality, peak GPU memory, and throughput. This check is an implementation
equivalence test, not a utility pilot.

## Gate 1 — zero/low-decode opportunity audit

Run all MoNuSeg router-train images, but **do not action-decode** candidates.
For every image record GT-side analysis only:

- missed-GT count and merge-parent count;
- ADD candidate coverage of missed GT;
- SPLIT candidate coverage of merge parents;
- candidate count per image and concentration across images.

Compute a clearly-labelled `optimistic_screening_ceiling`, not a decoder
oracle. A candidate that hits an eligible missed/near-threshold GT is assumed
to yield an ideal ADD mask; a split candidate for a GT-defined merge parent is
assumed to yield ideal child masks. Use ADD cost 1, SPLIT cost 2, joint budget
4, conflict-aware subset assembly, and the complete global PQ evaluator.

Terminate before decoding if any condition holds:

1. optimistic joint ceiling ΔPQ < 0.005;
2. ADD missed-GT coverage < 10%;
3. SPLIT merge-parent coverage < 10% and ADD ceiling is also insufficient;
4. nearly all correctable opportunities are concentrated in one image;
5. MoNuSeg has essentially no residual ADD/SPLIT-like error.

Passing this gate only means the method is not yet eliminated.

## Gate 2 — precommitted four-image ADD pilot

Before utility is inspected, generate and commit a pilot manifest from the
Gate 1 candidate-audit CSV. The selection input may contain only image ID,
ADD candidate count, and seed 3407. Stratify by ADD-count quartiles, select
one seed-resolved image per quartile for batch 1, and precommit one additional
seed-resolved image per quartile for batch 2. Calibration remains sealed.

Run **ADD only** for batch 1 at budgets 1, 2, and 4. Report per-image and
aggregate Oracle ΔPQ, DQ/SQ decomposition, positive action/image counts,
largest-image contribution, and actual GPU time.

Decision rules:

- **Terminate current StainRoute**: no positive ADD action; mean ADD Oracle@4
  ΔPQ < 0.001; or only one positive image with >70% of total gain.
- **Additional 4-image pilot**: mean ΔPQ in [0.001, 0.003), or one positive
  image with contribution <=70%. Batch 2 was precommitted with batch 1.
- **Consider complete MoNuSeg ADD study**: batch-1 mean ΔPQ >= 0.003, at
  least two positive images, largest-image contribution <=50%, and adequate
  positive/negative actions.

Across both pilot batches, continuation requires ADD Oracle@4 ΔPQ >= 0.003,
at least 3/8 positive images, and largest-image contribution <=50%.

## Gate 3 — SPLIT pilot

Only after Gate 2 passes and Gate 1 finds enough merge opportunity, decode
SPLIT on the same precommitted pilot images. Retire SPLIT if it has no positive
utility, SPLIT Oracle@4 < 0.001, or any apparent gain comes from one anomalous
parent. ADD may continue independently; no artificial two-family claim is
permitted.

## Gate 4 — stop and report

After the applicable pilot, stop. Do not launch a full MoNuSeg run without a
new project-lead decision. The report title and required sections are:

`REPORT FOR PROJECT LEAD — MONUSEG FUTILITY PILOT`

1. protocol-amendment git SHA;
2. runtime breakdown and projected full runtime;
3. candidate coverage audit;
4. optimistic candidate-aware ceiling;
5. fixed pilot image list and selection checksum;
6. ADD Oracle@1/2/4 per image;
7. positive action/image statistics;
8. largest-image contribution;
9. SPLIT pilot, only if authorized by the preceding gates;
10. estimated value and cost of a full run;
11. one recommendation: `TERMINATE CURRENT STAINROUTE`, `ADDITIONAL 4-IMAGE
    PILOT`, `FULL ADD-ONLY STAGE 1`, or `FULL ADD+SPLIT STAGE 1`.
