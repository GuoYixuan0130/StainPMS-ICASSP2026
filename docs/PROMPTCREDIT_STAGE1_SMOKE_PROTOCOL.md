# PromptCredit Stage 1 Smoke Protocol

Authorization: implementation and a two-crop mechanism smoke test only.  This
is not authorization for full TNBC training, MoNuSeg, calibration, test, a
final-method claim, or a return to StainRoute.

## v1 scope

Stage 0 did **not** establish an assignment gap (collision excess 0.253%;
nearest/Hungarian disagreement 8.23%).  PromptCredit v1 therefore preserves
the current nearest assignment and does not implement Hungarian reassignment
or any new assignment algorithm.  Assignment is retained as a negative
mechanism result.

The v1 mechanisms are:

- Directional credit: frozen-decoder focal+dice loss reaches selected point
  coordinates only through `stop_gradient(p) + alpha * (p - stop_gradient(p))`.
- Scalar credit: a lightweight ROI-feature `quality_head` receives detached
  threshold-aware decoded-mask utility targets and only changes point-NMS
  ranking when explicitly enabled.

Only `conv`, `deform_layer`, `reg_head`, `cls_head`, and `quality_head` are
trainable.  SAM2 image/prompt/mask/memory modules, the point backbone, and the
auxiliary semantic mask head are frozen and excluded from the optimizer.

Default flags preserve StainPMS behavior.  PromptCredit options are explicit:

```text
--prompt_credit_enabled
--prompt_credit_grad_scale {0..0.10}
--prompt_credit_quality_loss_coef {>=0}
--prompt_score_mode {objectness,objectness_x_quality,quality}
```

`objectness_x_quality` changes only point-NMS ranking.  It does not change the
class decision, semantic filtering, SAM IoU, decoder-call count, crop order,
NMS radius, or instance assembly.

## GPU-only AutoDL smoke command

Run once on AutoDL 4090 after pulling `research/promptcredit`.  Do not run it
locally and do not rerun it with changed parameters.

```bash
python -m unittest discover -s tests/promptcredit -p "test_*.py" -v

python tools/smoke_prompt_credit.py \
  --data-root ./data/tnbc \
  --checkpoint ../CA-SAM2-HRC/deliver_ckpts/tnbc_pms_best_e156.pth \
  --out-dir logs/promptcredit/stage1_smoke_tnbc_02_1_two_crops
```

Before model construction, the runner freezes the first two
nucleus-containing crops from the first Stage 0 image (`02_1`) and writes the
image and GT crop checksums to `smoke_crop_selection.json`.  It then initializes
two identical 100-step AdamW runs (seed 3407, lr 1e-4, weight decay 1e-4):

- control: alpha=0, quality coefficient=0, objectness ranking;
- PromptCredit: alpha linearly warms to 0.10 in steps 1--20, quality coefficient
  1.0, objectness-times-quality ranking.

The runner records every ten steps and compares step 0/100 only on those same
two training crops.  PASS additionally fixes “no clear localization damage” to
an end-vs-start mean localization-error ratio of at most 1.10, and requires
PromptCredit mean step time at most 1.30x control.  It refuses existing output
directories.  Do not commit or push generated logs, CSVs, figures, checkpoints,
or caches.

## Required handoff

Return these files after the one authorized run:

```text
report.json
run_manifest.json
smoke_crop_selection.json
baseline_equivalence.json
control_metrics.csv
promptcredit_metrics.csv
```

The report must be titled `REPORT FOR PROJECT LEAD — PROMPTCREDIT STAGE 1
SMOKE` and concludes only PASS/FAIL for code correctness and numerical
stability.  It must not claim generalization from the two training crops.
