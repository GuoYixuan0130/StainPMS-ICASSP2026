# ICASSP 2026 Experimental Switches

This document records the new experimental branches added after StainPMS.
All switches are disabled by default. Existing CA-SAM2 and StainPMS commands
keep their original behavior unless one of these flags is passed.

## 1. Soft Coverage Confidence Cache

Enable with:

```bash
--coverage_probabilistic
```

Boundary:

- Writes the original hard coverage map as `<image>.npy`.
- Additionally writes a confidence map as `<image>_prob.npy`.
- The confidence value is the accepted mask score used during instance assembly.
- During PMS mining, if `<image>_prob.npy` is available, residual evidence is:

```text
stain_evidence * (1 - dilated_coverage_confidence)
```

- If the confidence map is missing, PMS falls back to the original hard
  coverage subtraction for that image.
- Inference is unchanged.

Useful knobs:

```bash
--coverage_prob_threshold 0.6
--coverage_prob_min_residual 0.05
--coverage_prob_decay 1.0
```

`coverage_prob_decay < 1.0` allows old confidence to fade across refreshes.
`1.0` is monotonic and closest to the original StainPMS accumulation logic.

## 2. Residual Point-Head Supervision

Enable with:

```bash
--pms_point_loss_coef 0.1
```

Boundary:

- Uses only residual positive PMS prompts mined outside coverage.
- Excludes preservation prompts, so already-covered nuclei do not become
  auxiliary "missing point" targets.
- Reuses the point head's Hungarian matching, coordinate regression loss, and
  classification loss.
- Does not change the SAM2 decoder, prompt encoder, or inference path.

Useful knobs:

```bash
--pms_point_reg_weight 1.0
--pms_point_cls_weight 1.0
```

## Recommended Controlled Comparisons

For paper experiments, compare from the same warm-start checkpoint:

```text
StainPMS + extra FT
StainPMS + soft coverage
StainPMS + residual point-head loss
StainPMS + both branches
```

Keep epochs, learning rate, refresh interval, data split, NMS, and inference
settings identical across these variants.
