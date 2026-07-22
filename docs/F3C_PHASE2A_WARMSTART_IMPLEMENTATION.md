# Phase 2A warm-start C0/C1 implementation

Status: implementation, unit tests, 1--2 update smoke, and 10+100 update
timing are approved. Formal 5-epoch fine-tuning is not approved.

The frozen runtime contract is
`configs/phase2a/warmstart_runtime_v1.json`. The earlier
`warmstart_feasibility_v1.json` and `warmstart_preflight_result_v1.json` are
retained as historical proposal/preflight records rather than rewritten.

## Arms and decoder mapping

- `legacy` is a one-update equivalence reference. It calls the existing mask
  decoder with `multimask_output=False`.
- In training mode that existing call first computes native mask tokens 0--3
  and then returns token 0. Dynamic stability fallback is inactive in training.
- `C0` and `C1` both call `sam_mask_decoder.predict_masks` exactly once for
  each prompt group, retain tokens 0--3, and give token 0 to every original
  StainPMS loss and downstream training operation.
- `C1` alone computes the approved auxiliary losses from the retained four
  candidates. It adds no parameter and changes no inference or assembly path.

The common forward and original token mapping are implemented in
`run/run_on_epoch.py::_decode_training_candidates`.

## Frozen C1 objective

`stainpms/candidate_coverage.py` implements, in FP32,

```text
ell_i,k = (20 * DiceLoss(M_i,k, G_i) + FocalLoss(M_i,k, G_i)) / 21
Lcoverage_i = -tau * (logsumexp(-ell_i,k / tau) - log(K))
tau = 0.1, K = 4
```

The quality target is the detached IoU between each bilinearly upsampled,
`logit > 0` candidate and its assigned GT. Quality loss is mean squared error
over all four native quality predictions.

Ordinary, PMS residual, and PMS preservation positives are averaged within
their own groups and then combined with weights `1.0`,
`pms_loss_coef * pms_residual_mask_weight`, and
`pms_preserve_loss_coef`. An empty group contributes an exact zero. Negative,
unmatched, and unassigned prompts remain governed only by the original
StainPMS objective.

The scientific name of C1 is **native best-of-K coverage plus quality
calibration baseline**. A selected-candidate improvement without a
best-candidate improvement is quality/ranking evidence, not candidate
generation evidence.

## Safety and equality gates

Before constructing a dataset, the warm-start entry point validates the
train-only manifest identity and full checkpoint SHA256. TNBC accepts only the
30 p1--6 records; MoNuSeg accepts only the frozen train37 identity. No
evaluation loader is constructed.

One train-only coverage cache is generated from `model` and `model1` with the
checkpoint texture bank discarded. Its record order and every NPY SHA256 are
frozen in a coverage manifest. Independent C0/C1 processes must verify and use
that same cache.

`tools/compare_warmstart_smokes.py` compares the legacy and C0 original loss
components, gradient group norms, and fixed key-parameter gradient vectors. It
also verifies C0/C1 checkpoint, manifest, coverage, seed, optimizer,
scheduler, crop count, decoder call count, prompt count, and token mapping.
Full gradient norms, key-gradient snapshots, and softmin diagnostic reductions
are collected in smoke only. They are disabled during the timed interval to
avoid contaminating seconds-per-update with CPU synchronization; this does not
change the C0 or C1 training loss.

## Execution order

1. Run `tools/run_phase2a_warmstart_smokes.sh` on AutoDL. It runs PyTorch unit
   tests, prepares shared train-only coverage, runs one update for
   legacy/C0/C1, and writes a gate per dataset.
2. Inspect both smoke gates and loss/gradient scale. Stop on any non-finite
   value, skipped update, C0 mismatch, decoder mismatch, or abnormal C1 scale.
3. Only after both smoke gates pass, run
   `tools/run_phase2a_warmstart_timing.sh`. It performs 10 warm-up and 100
   CUDA-synchronized measured updates independently for C0 and C1.
4. `tools/estimate_warmstart_budget.py` projects the approved 5/10-epoch
   update counts, but does not authorize either training run.
