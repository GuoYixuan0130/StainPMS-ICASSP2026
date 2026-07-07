# StainPMS

Code for **StainPMS: Self-Bootstrapped Prompt-Mask Supervision for Nuclei Instance Segmentation**.

StainPMS is a training-time framework for point-prompted SAM2-style nuclei instance segmentation. It builds on the CA-SAM2 auto-point pipeline and adds self-bootstrapped stain-guided prompt-mask supervision. During inference, the standard CA-SAM2 path is used, so no extra learnable parameters or extra inference branch are required for the reported main results.

## Highlights

- Self-bootstrapped online coverage maps generated from the current model on the training split.
- Hematoxylin residual mining outside the accumulated coverage map.
- Positive, negative, and coverage-preservation prompt supervision for the shared SAM2 mask decoder.
- Same inference path and cost as the CA-SAM2 baseline for the paper's main setting.

## ICASSP 2026 Direction: StainPQR

This repository is now also being used as the starting point for **StainPQR**:
Panoptic-Quality Risk-Calibrated Selective Refinement for prompted nuclei
instance segmentation.

The working hypothesis is that StainPMS has already reduced the broad coverage
deficit, while the remaining PQ loss is concentrated in a smaller set of
high-risk actions. StainPQR therefore keeps the StainPMS first-pass model
frozen, estimates the expected PQ utility of local corrective actions, and only
spends extra decoder calls on actions that are likely to improve PQ.

Current AutoDL Stage 0/1 findings:

| Dataset | Method | AJI | DQ | SQ | PQ |
| --- | --- | ---: | ---: | ---: | ---: |
| MoNuSeg | CA-SAM2 | 0.6436 | 0.8263 | 0.7494 | 0.6199 |
| MoNuSeg | StainPMS | 0.6667 | 0.8525 | 0.7710 | 0.6577 |
| TNBC | CA-SAM2 | 0.6220 | 0.8315 | 0.7971 | 0.6634 |
| TNBC | StainPMS | 0.6471 | 0.8306 | 0.8039 | 0.6681 |

Coverage-action oracle results show why selection is necessary. Executing all
residual H-peak actions is harmful on average, but actions that truly target
missed FNs are highly useful:

| Dataset | All coverage positive rate | Missed-FN positive rate | Main takeaway |
| --- | ---: | ---: | --- |
| MoNuSeg | 25.7% | 86.7% | Coverage actions help only when they identify missed FNs. |
| TNBC | 20.4% | 70.8% | Small budgets are essential; blind correction hurts PQ. |

Stage 2A selector checks sharpened the target. On MoNuSeg, the learned
`selector_prob` ranks useful coverage actions well, but the first
`selector_expected_utility` score is weakened by the small utility-regression
head. On TNBC and the combined split, the first selector run exposed the need
for stricter NaN/empty-value handling. Strong rule baselines remain
`missed_like_proxy` and `added_area`; the revised selector script therefore
reports both pure learned scores and probability-rule hybrids:
`selector_prob_added_area`, `selector_prob_missed_like`, and
`selector_prob_iou_area`.

After the NaN-safe selector update, group-CV on the current test-oracle action
sets supports the selective-refinement hypothesis:

| Selector audit | Strongest stable observation |
| --- | --- |
| MoNuSeg | `selector_prob_iou_area` improves budget-2/4 selected Delta PQ over `missed_like_proxy` and `added_area`. |
| TNBC | `selector_prob_added_area` gives the best budget-1 gain; budget 2 remains close to `missed_like_proxy`. |
| Combined | learned probability hybrids beat the hand-built rules at budgets 1/2, while larger budgets start to over-correct. |

The current working choice is to treat `selector_prob_iou_area` as the main
stable learned score, keep `selector_prob_added_area` as the budget-1 ablation,
and report budget curves rather than a single large-budget point. The next
acceptance target is train-oracle to test-oracle holdout performance, not
test-oracle group-CV alone.

Train-oracle to test-oracle holdout supports using dataset-specific selectors:

| Holdout | Key result |
| --- | --- |
| MoNuSeg train -> test | `selector_prob_iou_area` gives the best budget-2/4 learned refinement ranking. |
| TNBC train -> test | `selector_prob_added_area` is best at budget 1 and remains competitive at budget 2. |
| Combined train -> test | mixed calibration is less reliable because TNBC train oracle has very few positive actions. |

The next stage is Stage 2C selective refinement: rerun the frozen decoder only
for selected actions, merge the resulting masks into the first-pass prediction,
and recompute full-image Dice/AJI/DQ/SQ/PQ.

Early Stage 2C true re-decoding results:

| Dataset | Score | Budget | Delta AJI | Delta DQ | Delta SQ | Delta PQ | Takeaway |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| MoNuSeg | `selector_prob_iou_area` | 1 | +0.00149 | +0.00055 | +0.00001 | +0.00044 | positive but conservative |
| MoNuSeg | `selector_prob_iou_area` | 2 | +0.00259 | +0.00093 | -0.00005 | +0.00068 | best current MoNuSeg replay |
| MoNuSeg | `missed_like_proxy` | 2 | +0.00231 | +0.00039 | -0.00004 | +0.00027 | learned score improves over rule |
| TNBC | `selector_prob_added_area` | 1 | +0.00963 | +0.01916 | -0.00572 | +0.00828 | strong correction of missed FNs |
| TNBC | `missed_like_proxy` | 1 | +0.01063 | +0.01916 | -0.00584 | +0.00818 | rule is very competitive at B1 |
| TNBC | `selector_prob_added_area` | 2 | +0.00133 | +0.00881 | -0.00669 | +0.00115 | budget 2 over-corrects |

These results support a budgeted selective-correction story: coverage refinement
mainly increases DQ by recovering missed nuclei, while too many insertions
increase FP and reduce SQ.

See [docs/STAINPQR_EXPERIMENTS.md](docs/STAINPQR_EXPERIMENTS.md) for the
current AutoDL workflow.

## Repository Layout

```text
main.py                         Training and evaluation entry point
cfg.py                          Command-line options
args.py                         Model, augmentation, optimizer, and loss config
stainpms/candidate.py           Stain evidence and residual prompt mining
stainpqr/                       StainPQR oracle and selective-refinement utilities
run/dataset/                    Dataset loaders and PMS crop-time prompt assembly
run/run_on_epoch.py             Training, validation, self-bootstrap refresh, PMS loss branch
sam2_train/                     SAM2/CA-SAM2 model code
tools/prep_tnbc.py              TNBC conversion into the MoNuSeg-style loader layout
tools/analyze_eval_artifacts.py Stage 0 error audit from dumped predictions
tools/stage1_candidate_audit.py Stage 1A candidate recall audit
tools/analyze_oracle_actions.py Stage 1B oracle action ranking analysis
tools/train_coverage_selector.py Train/evaluate the coverage utility selector
tools/fig2_qualitative_crop.py  Qualitative crop strip used for Figure 2
docs/REPRODUCIBILITY.md         Detailed reproduction notes
```

## Environment

The original experiments used PyTorch 2.3.1, CUDA 11.8, Python 3.12, and a single NVIDIA RTX 4090 GPU.

```bash
conda env create -f environment.yml
conda activate CA-SAM2
```

`environment.yml` is an exact research environment export. If your platform cannot solve it directly, create a PyTorch/CUDA environment first, then install the Python packages listed under the `pip:` section.

## Data

Large datasets are not tracked by Git. Place them under `data/`.

### MoNuSeg Layout

```text
data/monuseg/
  train_12/images/
  train_12/labels/   # .mat files with key "inst_map"
  test/images/
  test/labels/       # .mat files with key "inst_map"
```

### TNBC Layout

TNBC is converted to the MoNuSeg-style layout and then run with `--dataset monuseg`.

```bash
python tools/prep_tnbc.py inspect --src /path/to/TNBC_raw
python tools/prep_tnbc.py convert --src /path/to/TNBC_raw --dst ./data/tnbc --test-patients 9,10,11
```

After conversion:

```text
data/tnbc/
  train_12/images/
  train_12/labels/
  test/images/
  test/labels/
```

## Checkpoints

Checkpoints are not tracked by Git. Put them under `checkpoints/`.

Expected paths:

```text
checkpoints/sam2_hiera_large.pt
checkpoints/CA-SAM2_monuseg.pth
checkpoints/CA-SAM2_tnbc.pth      # if using a warm-started TNBC checkpoint
```

For warm-started StainPMS fine-tuning, use a trained CA-SAM2 checkpoint. For from-scratch StainPMS training, use the SAM2 Hiera-L checkpoint as the initialization.

## Baseline Evaluation

MoNuSeg:

```bash
python main.py --eval \
  --dataset monuseg \
  --data_path ./data/monuseg \
  --sam_ckpt ./checkpoints/CA-SAM2_monuseg.pth \
  --sam_config sam2_hiera_l \
  --texture --context \
  --overlap 92 \
  --test_nms_thr 12 \
  --b 1
```

TNBC after conversion:

```bash
python main.py --eval \
  --dataset monuseg \
  --data_path ./data/tnbc \
  --sam_ckpt ./checkpoints/CA-SAM2_tnbc.pth \
  --sam_config sam2_hiera_l \
  --texture --context \
  --overlap 32 \
  --test_nms_thr 12 \
  --b 1
```

## StainPMS Training

### Warm-Started Fine-Tuning

Use this when a strong CA-SAM2 checkpoint already provides a meaningful initial coverage map.

```bash
python main.py \
  --dataset monuseg \
  --data_path ./data/monuseg \
  --sam_ckpt ./checkpoints/CA-SAM2_monuseg.pth \
  --sam_config sam2_hiera_l \
  --texture --context \
  --use_pms \
  --pms_self_bootstrap \
  --coverage_accumulate \
  --iterative_baseline_refresh_every 20 \
  --pms_loss_coef 0.5 \
  --pms_object_weight 1.0 \
  --pms_residual_mask_weight 0.3 \
  --pms_preserve_loss_coef 1.0 \
  --pms_preserve_covered \
  --pms_preserve_max_prompts 20 \
  --epochs 10 \
  --lr 1e-5 \
  --weight_decay 1e-4 \
  --val_start_epoch -1 \
  --val_freq 1 \
  --overlap 92 \
  --test_nms_thr 12 \
  --b 1 \
  --exp_name stainpms_monuseg_warmft
```

### From-Scratch Training With Deferred PMS

Use this when no strong CA-SAM2 initialization is available. PMS is delayed until the model can generate a useful self-coverage map.

```bash
python main.py \
  --dataset monuseg \
  --data_path ./data/tnbc \
  --sam_ckpt ./checkpoints/sam2_hiera_large.pt \
  --sam_config sam2_hiera_l \
  --texture --context \
  --use_pms \
  --pms_self_bootstrap \
  --coverage_accumulate \
  --pms_start_epoch 50 \
  --iterative_baseline_refresh_every 20 \
  --pms_loss_coef 0.5 \
  --pms_object_weight 1.0 \
  --pms_residual_mask_weight 0.3 \
  --pms_preserve_loss_coef 1.0 \
  --pms_preserve_covered \
  --pms_preserve_max_prompts 20 \
  --epochs 200 \
  --lr 1e-4 \
  --lr_min 1e-6 \
  --weight_decay 1e-4 \
  --val_start_epoch 50 \
  --val_freq 1 \
  --overlap 32 \
  --test_nms_thr 12 \
  --b 1 \
  --exp_name stainpms_tnbc_scratch
```

## StainPMS Evaluation

The main paper setting uses the standard CA-SAM2 inference path.

```bash
python main.py --eval \
  --dataset monuseg \
  --data_path ./data/monuseg \
  --sam_ckpt ./logs/stainpms_monuseg_warmft_*/Model/base_pq_epoch.pth \
  --sam_config sam2_hiera_l \
  --texture --context \
  --overlap 92 \
  --test_nms_thr 12 \
  --b 1
```

## Reported Results

Main paper results at NMS threshold 12:

| Dataset | Method | Dice | AJI | DQ | SQ | PQ |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| MoNuSeg | CA-SAM2 | 0.808 | 0.644 | 0.826 | 0.750 | 0.620 |
| MoNuSeg | StainPMS | 0.822 | 0.666 | 0.853 | 0.771 | 0.658 |
| TNBC | CA-SAM2 | 0.787 | 0.639 | 0.835 | 0.808 | 0.676 |
| TNBC | StainPMS | 0.808 | 0.665 | 0.838 | 0.813 | 0.682 |

## Notes

- `checkpoints/`, `data/`, `logs/`, coverage caches, and prediction dumps are ignored.
- See `docs/REPRODUCIBILITY.md` for a longer workflow.

## Acknowledgement

This repository builds on CA-SAM2, MedSAM2, SAM2, and PromptNucSeg-related code paths. Please also cite the corresponding upstream works when using this code.
