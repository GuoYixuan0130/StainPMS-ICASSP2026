# PromptCredit corrected paired smoke protocol

Authorization is limited to one corrected two-crop, 100-step control versus
PromptCredit run.  It does not authorize a full TNBC experiment, patients
7–11, calibration, test, MoNuSeg, or any StainRoute work.

The fixed crops remain the first two nucleus-containing crops from image
`02_1`, selected before model construction and recorded with crop and GT
checksums.  The frozen baseline-v1 checkpoint, seed 3407, AdamW, lr `1e-4`,
weight decay `1e-4`, crop order, 100 steps, alpha warm-up to `0.10`, Quality
Focal Loss gamma `2`, and quality-loss coefficient `1.0` are unchanged.

## Corrections fixed before the run

- `evaluation_snapshot` saves Python, NumPy, torch CPU, and torch CUDA RNG
  states; sets the complete point model and SAM2 to `eval()` under
  `torch.no_grad()`; and restores every module state plus every RNG stream.
- Control and PromptCredit are separately rebuilt from the original checkpoint
  and trained from independent copies of the same RNG state.  Evaluation cannot
  perturb either training sequence.
- The quality-head final layer now has zero weight and bias `logit(0.01)`.
  Consequently every initial proposal has the same quality score and
  `objectness × quality` has exactly the `objectness` ranking at step 0.  This
  is a pre-performance numerical-stability and paired-fairness correction, not
  a tuned prior.  Both runs retain the same initialized quality head; the
  control only sets its quality-loss coefficient to zero.
- Before training, the runner records proposal/positive/negative counts,
  positive and negative QFL sums, the matched-positive normalization
  denominator, and per-loss gradient norms on the shared fusion convolution.
  It does not alter gamma, utility-target formula, alpha, or the loss
  coefficient.

## Fail-closed step 0

The runner stops before training unless all legacy-checkpoint common parameter
checksums and quality-head checksums are equal, coordinate/logit/selected
prompt/decoded-logit errors are zero, objectness and product-score rankings
are equal, crop-local point-NMS action IDs are equal, and common-score mask
IoU/loss are equal.  Baseline legacy-versus-disabled-PromptCredit equivalence
must remain zero-error.

At steps 0 and 100, and every ten training steps, metrics come only from the
snapshot helper.  Reports include both views:

- **Common-score:** both arms use `objectness`, isolating directional credit.
- **Deployment-score:** Control uses `objectness`; PromptCredit uses
  `objectness_x_quality`, measuring the complete scalar-plus-directional path.

Crop-local NMS is an evaluation proxy only.  It uses the unchanged NMS radius
and semantic filter and does not change the formal model's decoder-call path.
Each evaluation view decodes the already matched prompts in one decoder call
per crop, then applies the current ranking/NMS action to those decoded records;
it never adds an image-encoder or mask-decoder call to the model itself.

## One authorized AutoDL command

After pulling the corrected commit on `research/promptcredit`, run the unit
tests and then exactly one fresh output directory.  Substitute the printed
commit SHA for `<CORRECTED_SHA>`; do not reuse the invalid directory.

```bash
python -m unittest discover -s tests/promptcredit -p "test_*.py" -v

python tools/smoke_prompt_credit.py \
  --data-root ./data/tnbc \
  --checkpoint ../CA-SAM2-HRC/deliver_ckpts/tnbc_pms_best_e156.pth \
  --out-dir logs/promptcredit/stage1_smoke_corrected/<CORRECTED_SHA>
```

Return every file from that new directory, including `report.json`, the two
curves, `run_manifest.json`, `baseline_equivalence.json`,
`step0_strict_equivalence.json`, `quality_loss_scale_audit.json`, and
`smoke_crop_selection.json`.  Copy them into `temple/` only as a staging copy;
leave the formal output in `logs/promptcredit/stage1_smoke_corrected/`.

The report's automatic outcome is PASS, CONDITIONAL, or FAIL only for this
mechanism smoke.  A CONDITIONAL outcome is mandatory when the common-score
mask-loss and IoU comparisons conflict, or when scalar quality fits but no
relative common-score coordinate/mask gain is demonstrated.  No outcome may
be described as a generalization result.
