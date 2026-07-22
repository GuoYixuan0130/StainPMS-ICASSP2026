# F3C-StainPMS Phase 2A AutoDL runbook

This runbook covers the first, low-cost gate of Phase 2A. It exports the
Phase 1 consistency tables and measures the TNBC clean-baseline cost. It does
not access TNBC p9--p11, construct a MoNuSeg test loader, or start long
training.

## Fixed identities

- Branch: `research/f3c-stainpms`
- Seed: `3407`
- Generic initialization SHA256:
  `7442e4e9b732a508f80e141e7c2913437a3610ee0c77381a66658c3a445df87b`
- TNBC train: p1--p6 manifest only
- TNBC development: p7--p8 manifest only
- MoNuSeg: train37 only; test14 is never passed to these commands
- Timing: 10 warm-up optimizer updates followed by 100 CUDA-synchronized
  timed optimizer updates

## 1. Pull and test

```bash
cd /root/autodl-tmp/projects/StainPMS-ICASSP2026-f3c
git pull --ff-only origin research/f3c-stainpms

phase1_root=/root/autodl-tmp/f3c_phase1
phase2a_root=/root/autodl-tmp/f3c_phase2a
mkdir -p "$phase2a_root/reports" "$phase2a_root/phase1_tables"

conda run -n agentseg python -m unittest discover \
  -s tests -p 'test_export_phase1_tables.py' -v
conda run -n agentseg python -m unittest discover \
  -s tests -p 'test_phase2a_*.py' -v
```

## 2. Export the Phase 1 consistency tables

This command reads only the already-completed Phase 1 outputs. The contingency
counts come directly from `gt_instances.csv`; the exporter aborts if they do
not agree with point recall, final TP, or the five-class error partition.

```bash
conda run -n agentseg python tools/export_phase1_tables.py \
  --input-dir "$phase1_root/diagnostics/tnbc_p1_6_full" \
  --input-dir "$phase1_root/diagnostics/tnbc_p7_8_full" \
  --input-dir "$(python tools/resolve_phase1_output.py --root "$phase1_root/diagnostics" --dataset monuseg --processed-records 37 --require-file gt_instances.csv --require-file images.json)" \
  --output-dir "$phase2a_root/phase1_tables"
```

The four small outputs are:

- `phase1_summary.csv`
- `phase1_error_partition.csv`
- `phase1_point_final_contingency.csv`
- `phase1_provenance.csv`

## 3. TNBC base-objective timing

No evaluation loader is constructed. This profile measures the pre-PMS
CA-SAM2/StainPMS objective from the generic SAM2 initialization.

```bash
conda run -n agentseg python main.py \
  --dataset tnbc \
  --data_path /root/autodl-tmp/projects/AgentSeg-CA-SAM2/data/tnbc \
  --train_manifest "$phase1_root/manifests/tnbc_p1_6_phase1.json" \
  --verify_manifest_hashes \
  --phase2a_timing_profile base \
  --phase2a_timing_output "$phase2a_root/reports/tnbc_timing_base.json" \
  --phase2a_warmup_updates 10 \
  --phase2a_timed_updates 100 \
  --sam_ckpt /root/autodl-tmp/projects/CA-SAM2-HRC/checkpoints/sam2_hiera_large.pt \
  --sam_config sam2_hiera_l \
  --seed 3407 \
  --epochs 200 \
  --lr 1e-4 \
  --lr_min 1e-6 \
  --lr_cosine_t_max 200 \
  --weight_decay 1e-4 \
  --crop_size 256 \
  --out_size 256 \
  --overlap 32 \
  --load unclockwise \
  --test_nms_thr 12 \
  --b 1 \
  --texture \
  --context \
  --evaluator_mode strict \
  --exp_name f3c_phase2a_tnbc_timing_base
```

## 4. TNBC active-PMS timing

This is a separate disposable process initialized from the same generic
checkpoint. It first generates a p1--p6 train-only coverage cache, then times
the active PMS objective. `--pms_start_epoch 0` is used only to expose the
active objective to the timer; the formal recipe remains fixed at epoch 50.

```bash
conda run -n agentseg python main.py \
  --dataset tnbc \
  --data_path /root/autodl-tmp/projects/AgentSeg-CA-SAM2/data/tnbc \
  --train_manifest "$phase1_root/manifests/tnbc_p1_6_phase1.json" \
  --verify_manifest_hashes \
  --phase2a_timing_profile pms_active \
  --phase2a_timing_output "$phase2a_root/reports/tnbc_timing_pms_active.json" \
  --phase2a_warmup_updates 10 \
  --phase2a_timed_updates 100 \
  --sam_ckpt /root/autodl-tmp/projects/CA-SAM2-HRC/checkpoints/sam2_hiera_large.pt \
  --sam_config sam2_hiera_l \
  --seed 3407 \
  --epochs 200 \
  --lr 1e-4 \
  --lr_min 1e-6 \
  --lr_cosine_t_max 200 \
  --weight_decay 1e-4 \
  --crop_size 256 \
  --out_size 256 \
  --overlap 32 \
  --load unclockwise \
  --test_nms_thr 12 \
  --b 1 \
  --texture \
  --context \
  --use_pms \
  --pms_self_bootstrap \
  --coverage_accumulate \
  --pms_start_epoch 0 \
  --iterative_baseline_refresh_every 20 \
  --pms_loss_coef 0.5 \
  --pms_object_weight 1.0 \
  --pms_residual_mask_weight 0.3 \
  --pms_preserve_loss_coef 1.0 \
  --pms_gt_match_radius 8 \
  --pms_preserve_covered \
  --pms_preserve_max_prompts 20 \
  --stain_min_distance 12 \
  --stain_top_k 20 \
  --evaluator_mode strict \
  --exp_name f3c_phase2a_tnbc_timing_pms_active
```

## 5. Apply the locked 12-GPU-hour gate

```bash
conda run -n agentseg python tools/estimate_phase2a_baseline_budget.py \
  --recipe configs/phase2a/baseline_recipe_v1.json \
  --dataset tnbc \
  --base-timing "$phase2a_root/reports/tnbc_timing_base.json" \
  --active-timing "$phase2a_root/reports/tnbc_timing_pms_active.json" \
  --output "$phase2a_root/reports/tnbc_budget_gate.json"
```

Exit code `0` means `gate_pass`; exit code `2` means `gate_stop`. Do not start
the formal baseline from an incomplete or `gate_stop` report. Return the four
Phase 1 CSV files plus these three JSON files before the long run command is
issued:

- `tnbc_timing_base.json`
- `tnbc_timing_pms_active.json`
- `tnbc_budget_gate.json`
