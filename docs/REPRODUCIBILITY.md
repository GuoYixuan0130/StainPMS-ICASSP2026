# Reproducibility Notes

This document records the public StainPMS reproduction path.

## Core Idea

StainPMS keeps the CA-SAM2 inference pathway unchanged:

```text
image -> SAM2 image encoder -> DPA-P2PNet point head
      -> SAM2 prompt encoder -> SAM2 mask decoder -> instance assembly
```

During training only, PMS adds extra prompts mined from residual hematoxylin evidence. These prompts supervise the same SAM2 prompt encoder and mask decoder.

## Code Path

- `stainpms/candidate.py`
  - RGB to HED conversion.
  - Hematoxylin evidence normalization.
  - Otsu foreground extraction.
  - Coverage subtraction.
  - Local maximum prompt mining.
  - GT assignment for positive/negative PMS prompts.

- `run/dataset/monuseg.py`
  - Load images and instance maps.
  - Attach coverage maps during augmentation when available.
  - Return PMS positive prompts, negative prompts, GT masks, and preservation counts.

- `main.py`
  - Enables self-bootstrap coverage caches with `--pms_self_bootstrap`.
  - Refreshes the train-split coverage map every `--iterative_baseline_refresh_every` epochs.
  - Uses monotonic accumulation when `--coverage_accumulate` is enabled.

- `run/run_on_epoch.py`
  - Reuses the main image embedding and high-resolution features.
  - Runs an extra prompt encoder and mask decoder pass for PMS prompts.
  - Applies object-score BCE on positive and negative prompts.
  - Applies focal, dice, and IoU mask losses on positive residual and preservation prompts.

- `sam2_train/modeling/criterion.py`
  - Stores PMS loss weights used by the training loop.

- `tools/fig2_qualitative_crop.py`
  - Recreates the qualitative crop strip used for Figure 2 from an image, GT, baseline prediction, and StainPMS prediction.

## Main Hyperparameters

Paper protocol:

| Setting | Value |
| --- | --- |
| crop size | 256 |
| batch size | 1 |
| optimizer | AdamW |
| weight decay | 1e-4 |
| PMS loss coefficient | 0.5 |
| PMS object weight | 1.0 |
| residual mask weight | 0.3 |
| preservation loss coefficient | 1.0 |
| preservation max prompts | 20 |
| coverage refresh interval | 20 epochs |
| monotonic coverage threshold | 0.5 |
| GT prompt assignment radius | 8 px |
| stain peak min distance | 12 px |
| stain top-k candidates | 20 |
| NMS threshold | 12 |
| TTA | off |

Dataset-specific overlap:

| Dataset | `--overlap` |
| --- | ---: |
| MoNuSeg | 92 |
| TNBC | 32 |

## Data Preparation

MoNuSeg must use `.mat` labels with key `inst_map`.

TNBC is converted to the same layout:

```bash
python tools/prep_tnbc.py inspect --src /path/to/TNBC_raw
python tools/prep_tnbc.py convert --src /path/to/TNBC_raw --dst ./data/tnbc --test-patients 9,10,11
```

Run TNBC as:

```bash
--dataset monuseg --data_path ./data/tnbc
```

## Self-Bootstrap Coverage

Recommended public path:

```bash
--use_pms
--pms_self_bootstrap
--coverage_accumulate
--iterative_baseline_refresh_every 20
```

For warm-started fine-tuning, PMS can start at epoch 0.

For from-scratch training, use:

```bash
--pms_start_epoch 50
```

This trains with the standard objective first, then generates the initial self-coverage map and enables PMS.

## Optional Fixed-Cache Path

The optional fixed-cache path precomputes coverage maps with a frozen baseline checkpoint:

```bash
python main.py --eval --eval_on_train \
  --dataset monuseg \
  --data_path ./data/monuseg \
  --sam_ckpt ./checkpoints/CA-SAM2_monuseg.pth \
  --sam_config sam2_hiera_l \
  --texture --context \
  --b 1 \
  --dump_baseline_masks_dir ./baseline_masks_train_monuseg
```

Then train with:

```bash
--use_pms --baseline_masks_dir ./baseline_masks_train_monuseg
```

The paper's self-bootstrap description is better represented by `--pms_self_bootstrap`.

## Evaluation

The reported main results use standard inference:

```bash
python main.py --eval \
  --dataset monuseg \
  --data_path ./data/monuseg \
  --sam_ckpt ./logs/<exp>/Model/base_pq_epoch.pth \
  --sam_config sam2_hiera_l \
  --texture --context \
  --overlap 92 \
  --test_nms_thr 12 \
  --b 1
```

No PMS-specific inference branch is used for the reported main paper numbers.

## Expected Results

The final manuscript reports:

| Dataset | Method | Dice | AJI | DQ | SQ | PQ |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| MoNuSeg | CA-SAM2 | 0.808 | 0.644 | 0.826 | 0.750 | 0.620 |
| MoNuSeg | StainPMS | 0.822 | 0.666 | 0.853 | 0.771 | 0.658 |
| TNBC | CA-SAM2 | 0.787 | 0.639 | 0.835 | 0.808 | 0.676 |
| TNBC | StainPMS | 0.808 | 0.665 | 0.838 | 0.813 | 0.682 |
