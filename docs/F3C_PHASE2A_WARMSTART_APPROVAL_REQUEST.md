# Phase 2A warm-start feasibility: approval request

## 1. Conclusion summary

The warm-start preflight is complete. It did not open a dataset, run
inference, or update a model. Both delivered checkpoints match their declared
SHA256 and load with PyTorch's restricted `weights_only` mode. The three
preflight tests passed.

Neither checkpoint is a resumable training checkpoint: both omit optimizer,
scheduler, RNG, manifest, command, and config state. They can support the
project manager's approved exploratory weight-warm-start question, but they
cannot currently support a clean provenance claim or a full-state resume.

The requested C0/C1 experiment is feasible in principle. No C1 loss has been
implemented and no fine-tuning has started. Approval is requested for the
specific isolated candidate-set objective in Section 7.

## 2. Repository state

- Branch: `research/f3c-stainpms`
- Preflight-code commit: `fec21ee79a3ba5f2f7c8239a925e644614c22fc5`
- Preflight proposal SHA256:
  `fd3ca0684b34934f5bf5770011d4b7249e1dbce7e9264d177eaed38345eb51a7`
- The returned audit files remain ignored local evidence and are represented
  by hashes in `configs/phase2a/warmstart_preflight_result_v1.json`.

## 3. Historical command and log reconciliation

The public repository at commit
`51c2eac340cb92274d1fef4ff71b27ceae34da5f` distinguishes two recipes:

| path | initialization | epochs | LR | crop/overlap | batch | PMS start/refresh |
|---|---|---:|---:|---|---:|---|
| public MoNuSeg warm-start | CA-SAM2 MoNuSeg checkpoint | 10 | 1e-5 | 256/92 | 1 | 0/20 epochs |
| public TNBC scratch example | generic SAM2 Hiera-L | 200 | 1e-4 | 256/32 | 1 | 50/20 epochs |

Consequently:

- TNBC scratch accounting was 30 images x 9 crops x 200 epochs = 54,000
  optimizer updates.
- The prior MoNuSeg scratch accounting was 37 images x 36 crops x 200 epochs
  = 266,400 optimizer updates.
- Batch size was one, so every crop batch was one optimizer update.
- The resulting 58.2 GPU-hour MoNuSeg estimate was a hypothetical application
  of the 200-epoch scratch recipe to 37 1000x1000 images. It was not the
  public MoNuSeg warm-start budget.

No checkpoint-bound command, manifest, or original training log was found in
the reviewed repository/history or the inventoried AutoDL historical log
directory. The public recipe describes a code path; it does not prove the
origin of either delivered checkpoint.

## 4. Checkpoint evidence

| dataset | SHA256 | embedded epoch index | interpreted completed epoch | saved state | evidence class |
|---|---|---:|---:|---|---|
| TNBC | `44a3cb3e...23781` | 156 | 157 if zero-based | model, point head, 64-bank, epoch | exploratory only |
| MoNuSeg | `6616c246...c95db1` | 4 | 5 if zero-based | model, point head, 64-bank, epoch | exploratory only |

Both contain 3,184 SAM2 tensors and 389 point-head tensors. Neither contains
optimizer, scheduler, RNG, manifest, command, or configuration evidence.

There is an important provenance risk. The checkpoint payload structure
matches the legacy saver in `main.py`, which evaluates `test_loader` and saves
`base_pq_epoch.pth`/`base_aji_epoch.pth` when those metrics improve. The
MoNuSeg delivered filename also says `best_pq`. This is evidence of elevated
test-metric-selection risk, not proof that a particular historical run used
test14 or TNBC p9-p11. Exposure therefore remains `unknown`, and neither
checkpoint may be used as clean or final-performance evidence. This is
compatible with the manager's explicit permission to use unclear historical
checkpoints for exploratory warm-start feasibility.

## 5. Proposed weight-warm-start contract

For both C0 and C1:

- load only `model` and `model1` from the same dataset checkpoint;
- do not load the embedded 64-item texture bank because its provenance is
  unknown and it is unnecessary for the train-only comparison;
- create identical fresh AdamW optimizers because no optimizer state exists;
- create identical fresh MultiStepLR schedulers; milestones 80/140/200 leave
  LR constant at 1e-5 during five or ten epochs;
- generate a fresh initial coverage cache from the approved training manifest;
- start the per-epoch training texture bank empty, matching the existing path;
- use seed 3407, deterministic mode, manifest order, `shuffle=false`, identical
  augmentations, crop counts, updates, and checkpoint positions.

This must be called exploratory **weight warm-start**, not continued full-state
resume.

## 6. Exact equal budgets and timing

| dataset | updates/epoch | five-epoch updates | ten-epoch updates | C0 5-epoch proxy | C0 10-epoch proxy |
|---|---:|---:|---:|---:|---:|
| TNBC | 270 | 1,350 | 2,700 | 0.331 GPU h | 0.648 GPU h |
| MoNuSeg | 1,332 | 6,660 | 13,320 | 1.896 GPU h | 3.568 GPU h |

Each five- or ten-epoch run has one epoch-zero train-only coverage refresh;
the next interval would be epoch 20. Checkpoints are written at every epoch in
both arms. C0 and C1 are compared at the identical fixed final update. A
ten-epoch experiment restarts independently from the original checkpoint and
does not continue the five-epoch run.

The listed times are C0 planning proxies from earlier PMS-active measurements,
not formal arm timings. C1 cost remains unknown. After implementation, every
dataset/arm must run 10 warm-up plus 100 CUDA-synchronized timed updates before
any five-epoch run.

## 7. Proposed C1 candidate coverage/quality loss

This is the minimum mechanism-isolating experiment, equivalent to the required
ordinary best-of-K candidate baseline. It is not yet full counterfactual F3C.

For each eligible positive prompt with an assigned GT and native candidates
`k=0..3`, keep all original StainPMS losses and define:

```text
ell_k = (20 * existing_DiceLoss(M_k, G)
         + existing_BinaryFocalLoss(M_k, G)) / 21

L_coverage = -tau * log(mean_k(exp(-ell_k / tau)))
tau = 0.1

IoU_k = IoU(stopgrad(1[upsample(M_k) > 0.0]), G)
L_quality = mean_k((q_k - IoU_k)^2)

L_C1 = L_StainPMS + 1.0 * L_coverage + 1.0 * L_quality
```

Here `M_k` denotes the bilinearly upsampled candidate logits at the existing
training output size. The soft minimum will be implemented with `logsumexp`
for numerical stability.

The `0.0` logit threshold is the frozen Phase 1/inference binarization rule
(equivalent to sigmoid probability 0.5); it is not tuned during this study.
`q_k` is the native raw quality-head output, matching the existing quality
loss convention.

Eligible prompts:

- ordinary GT-assigned positive prompts;
- PMS residual positive prompts with assigned GT;
- PMS preservation positive prompts with assigned GT.

Coverage and quality losses are averaged separately within the ordinary,
residual, and preservation groups; an empty group contributes zero. The
ordinary group has weight 1.0. The residual group inherits
`pms_loss_coef * pms_residual_mask_weight`; the preservation group inherits
`pms_preserve_loss_coef`. Thus the number of mined prompts does not silently
change the relative branch scale, and the original PMS branch weighting is
preserved.

Negative and unmatched prompts receive no candidate-mask auxiliary supervision;
their existing object/negative losses remain unchanged. C1 introduces no new
parameter and no random draw. Standard single-mask inference, NMS, assembly,
and all thresholds remain unchanged.

This objective asks for at least one good candidate. It does not force all four
candidates to match the same GT. Token utilization, candidate count at
IoU>=0.5/0.7, and pairwise IoU will be logged. No utilization balancing is
added unless later evidence demonstrates collapse.

Excluded from this first test are restaining/worst-view aggregation, point
stability, LoRA, external ranking, and assembly changes. This isolates whether
tasking the existing unused candidate tokens has value before adding the main
counterfactual component.

## 8. Risks and scientific interpretation

- The experiment can answer whether the candidate objective adds value beyond
  ordinary continued training under the same historical initialization.
- It cannot establish clean generalization because checkpoint selection
  exposure remains unknown.
- TNBC p7-p8 can compare C1-C0 descriptively; p9-p11 remain sealed.
- MoNuSeg train37 can provide training/mechanism observations only; test14 is
  never constructed or accessed.
- A gain in C1 over C0 would justify continued method development, but would
  not by itself establish that counterfactual stain robustness is the cause.
- A failure of this narrow best-of-K objective would argue against adding the
  more complex worst-view objective before reconsidering loss scale or token
  trainability.

## 9. Decisions requested from the project manager

1. Approve or revise the exact C1 definition: `tau=0.1`,
   `lambda_coverage=1.0`, `lambda_quality=1.0`, and the eligible prompt scope.
2. Confirm that both delivered checkpoints may be used only as exploratory
   weight warm-starts despite unresolved historical test-selection exposure.
3. Approve discarding their embedded texture banks and rebuilding all coverage
   from the approved training manifests.
4. If the loss is approved, authorize implementation, unit tests, 1-2 batch
   smoke, and the four required 10+100 timing runs only. The formal five-epoch
   C0/C1 runs will not start until the measured budget is reported and accepted.
