# StainPMS

Code for **StainPMS: Self-Bootstrapped Prompt-Mask Supervision for Nuclei Instance Segmentation**.

StainPMS is a training-time framework for point-prompted SAM2-style nuclei instance segmentation. It builds on the CA-SAM2 auto-point pipeline and adds self-bootstrapped stain-guided prompt-mask supervision. During inference, the standard CA-SAM2 path is used, so no extra learnable parameters or extra inference branch are required for the reported main results.

## Highlights

- Self-bootstrapped online coverage maps generated from the current model on the training split.
- Hematoxylin residual mining outside the accumulated coverage map.
- Positive, negative, and coverage-preservation prompt supervision for the shared SAM2 mask decoder.
- Same inference path and cost as the CA-SAM2 baseline for the paper's main setting.

## Repository Layout

```text
main.py                         Training and evaluation entry point
cfg.py                          Command-line options
args.py                         Model, augmentation, optimizer, and loss config
stainpms/candidate.py           Stain evidence and residual prompt mining
run/dataset/                    Dataset loaders and PMS crop-time prompt assembly
run/run_on_epoch.py             Training, validation, self-bootstrap refresh, PMS loss branch
sam2_train/                     SAM2/CA-SAM2 model code
tools/prep_tnbc.py              TNBC conversion into the MoNuSeg-style loader layout
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
