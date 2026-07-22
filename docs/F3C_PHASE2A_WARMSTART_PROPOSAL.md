# Phase 2A warm-start feasibility proposal

Status: owner approval is required before candidate-loss implementation,
timing, or fine-tuning.

## Corrected interpretation of the scratch budget

The measured scratch reference used batch size one, so every crop batch was
one optimizer update:

- TNBC: 30 images x 9 crops x 200 epochs = 54,000 updates.
- MoNuSeg: 37 images x 36 crops x 200 epochs = 266,400 updates.

The resulting 58.2-hour MoNuSeg estimate came from applying that hypothetical
200-epoch scratch schedule to 37 images of size 1000x1000. It is not the
currently requested warm-start fine-tuning budget.

More importantly, the public repository does not identify 200-epoch MoNuSeg
scratch as its reproduction path. At upstream commit
`51c2eac340cb92274d1fef4ff71b27ceae34da5f`, `README.md` specifies:

- MoNuSeg: CA-SAM2 warm-start, 10 epochs, learning rate 1e-5, overlap 92,
  batch size one, PMS active from epoch zero, refresh interval 20.
- The 200-epoch command: generic SAM2 initialization, deferred PMS at epoch
  50, learning rate 1e-4, overlap 32, and a TNBC scratch experiment name.

The imported repository commit `e560b47b3e520cc63fa02b272e207b850ceca237`
contains the same distinction. No reviewed historical log currently proves
that the delivered MoNuSeg checkpoint came from a 200-epoch scratch run.
Therefore the old 69.393-hour estimate is retained only as a scratch-reference
artifact. The frozen file that produced it is not rewritten because its SHA256
is already embedded in the returned reports.

## Checkpoint evidence status

Byte identities are verified:

| dataset | checkpoint | SHA256 | current evidence class |
|---|---|---|---|
| TNBC | `tnbc_pms_best_e156.pth` | `44a3cb3e93051301d789e44f93769588abfa727d7a174b0270f55305ef023781` | warm-start exploratory; manifest and selection history pending |
| MoNuSeg | `monuseg_pms_best_pq.pth` | `6616c24626a162580d79aa7035d0b6c1a4ae6240a6c2f4599960dd4efac95db1` | warm-start exploratory; manifest and test-selection history pending |

Run `tools/audit_warmstart_checkpoints.py` on AutoDL to record embedded epoch,
top-level metadata, training-state availability, and names of possible command
or config evidence. It does not load a dataset, run inference, or read the
contents of historical result logs.

The tracked local repository, its imported history, and the clean public
release contain training examples but no original run log tied byte-for-byte
to either delivered checkpoint. The local Phase 1 declarations therefore
remain authoritative for the present evidence boundary: training manifests,
initialization sources, and whether selection used TNBC p7-p11 or MoNuSeg
test14 are unknown. The `e156` filename is not treated as proof of embedded
epoch until the checkpoint metadata audit confirms it. The AutoDL preflight
only inventories candidate evidence filenames under the historical log root;
any later content review must be targeted and must not read sealed prediction
or metric artifacts.

## Equal-budget arms

Both arms start from the exact same dataset-specific StainPMS checkpoint.
If no optimizer state is embedded, both reset AdamW identically. If complete
optimizer/scheduler/RNG state unexpectedly exists, resume versus weight-only
warm-start becomes an explicit owner decision; the two arms will use the same
choice. Both use seed 3407, manifest order with no shuffle, the same
augmentations, learning rate 1e-5, batch size one, initial train-only
self-coverage refresh, crop budget, update count, and per-epoch checkpoint
positions. The public MultiStepLR milestones are 80/140/200, so the learning
rate remains constant throughout either five- or ten-epoch stage.

- C0: unchanged StainPMS continued-training objective.
- C1: C0 plus the proposed deterministic candidate-set coverage and native
  quality-head auxiliary loss.

Screening is a fixed-final-update paired comparison, not a comparison against
the pre-fine-tuning checkpoint and not a best-epoch search. Per-epoch
trajectories are supporting diagnostics. A ten-epoch run must restart both
arms independently from the original checkpoint rather than resume the
five-epoch screening run.

| dataset | updates/epoch | five epochs | ten epochs |
|---|---:|---:|---:|
| TNBC | 270 | 1,350 | 2,700 |
| MoNuSeg | 1,332 | 6,660 | 13,320 |

The previous synchronized PMS-active measurements give only the following C0
planning proxies (training plus one initial train-only coverage refresh):

| dataset | C0 five epochs | C0 ten epochs | C1 |
|---|---:|---:|---|
| TNBC | about 0.331 GPU h | about 0.648 GPU h | pending dedicated timing |
| MoNuSeg | about 1.896 GPU h | about 3.568 GPU h | pending dedicated timing |

These are not formal warm-start timings: they were measured before the C0/C1
runner existed and used generic-initialization PMS-active timing profiles. The
preflight estimator records the limitation; the required per-arm 10+100
measurements replace these proxies before a formal run.

TNBC uses p1-p6 for training and reports p7 and p8 separately plus the
patient-macro result. Patients p9-p11 remain sealed. MoNuSeg constructs only a
train37 loader, reports training-set mechanism observations, and fixes the
final update; test14 is not constructed or accessed.

## Proposed C1 loss

The first feasibility comparison is deliberately narrower than full F3C. It
tests whether the historical checkpoint can task its native four-mask set
without adding restaining, point stability, LoRA, external ranking, or assembly
changes.

For each eligible positive prompt and assigned GT mask, expose native tokens
0-3 through the decoder's existing `predict_masks` result. Keep every original
StainPMS loss unchanged. For candidate `k`, define the normalized segmentation
loss

```text
ell_k = (20 * existing_DiceLoss(M_k, G) + existing_FocalLoss(M_k, G)) / 21
```

and add

```text
L_coverage = -tau * log(mean_k(exp(-ell_k / tau))), tau = 0.1
IoU_k      = IoU(stopgrad(1[sigmoid(M_k) >= frozen_threshold]), G)
L_quality  = mean_k((q_k - IoU_k)^2)
L_C1       = L_StainPMS + 1.0 * L_coverage + 1.0 * L_quality
```

Eligible prompts are the ordinary GT-assigned positive prompts and PMS
residual/preservation positive prompts that have an assigned GT. Negative and
unmatched prompts are excluded from candidate-mask supervision. The existing
object/negative losses remain unchanged.

This objective asks the set to contain at least one strong candidate; it does
not force all four masks to copy the GT. Candidate utilization and pairwise IoU
are audited, but utilization balancing remains disabled unless actual collapse
is demonstrated. The standard single-mask inference path and native quality
selection remain unchanged.

This is a best-of-K feasibility baseline, not yet the full counterfactual
worst-view claim. Controlled restaining and worst-view aggregation require a
separate owner decision after this isolated test.

## Timing status and required gate

The earlier PMS-active timings can provide only a C0 proxy. C1 cost is unknown
until the proposed loss is approved and implemented. Before either formal
five-epoch run, each dataset and each arm must run 10 warm-up plus 100
CUDA-synchronized timed updates. These arm-specific measurements replace the
200-epoch scratch gate; they do not inherit its 12/24-hour decision.

No candidate loss or warm-start training is implemented by this proposal.
