# StainRoute MoNuSeg Futility Gates — AutoDL

Follow the committed [futility protocol](STAINROUTE_MONUSEG_FUTILITY_PROTOCOL.md)
at Git commit `e216c1f` or later. Do **not** launch the complete MoNuSeg Stage
1 oracle commands. Gates only use MoNuSeg `router_train`; calibration and the
official test split remain sealed.

## Preparation

```bash
set -euo pipefail
source /etc/network_turbo
cd /root/autodl-tmp/projects/StainPMS-ICASSP2026
conda activate agentseg
git pull --ff-only origin research/stainroute
git rev-parse HEAD

python -m unittest tests.stainroute.test_futility_selection tests.stainroute.test_metrics_joint -v
```

The frozen `logs/stainroute/stage1/baseline_v1_manifest.json` and
`configs/splits/stainroute_monuseg.json` must already exist and have their
precommitted checksums.

## Gate 0 — fixed-image runtime profile

This decodes only two deterministic ADD and two deterministic SPLIT candidates
on the precommitted image `TCGA-HE-7128-01Z-00-DX1`. It measures the complete
base pass, encoder/decode breakdown, action microbatch-vs-single equivalence,
assembly/utility equality, GPU peak memory, and a tiny exact B=4 timing check.

```bash
mkdir -p logs/stainroute/futility/gate0_runtime
python main.py --eval --dataset monuseg --data_path ./data/monuseg \
  --sam_ckpt ../CA-SAM2-HRC/deliver_ckpts/monuseg_pms_best_pq.pth \
  --sam_config sam2_hiera_l --texture --context --overlap 92 \
  --test_nms_thr 12 --b 1 --seed 3407 \
  --stage1_monuseg_futility runtime_profile \
  --stainroute_split_manifest configs/splits/stainroute_monuseg.json \
  --stainroute_split router_train \
  --stainroute_baseline_manifest logs/stainroute/stage1/baseline_v1_manifest.json \
  --stainroute_futility_config configs/stainroute/monuseg_futility_v1.yaml \
  --stainroute_out_dir logs/stainroute/futility/gate0_runtime \
  --exp_name stainroute_futility_gate0 \
  2>&1 | tee logs/stainroute/futility/gate0_runtime/stdout.log
```

Copy `runtime_profile.json`, `manifest.json`, and `stdout.log` for review.
Do not use any utility value from this fixed runtime image to choose pilots.

## Gate 1 — all-image zero/low-decode candidate audit

This is the only all-router-train MoNuSeg gate currently authorized. It runs
the frozen base prediction and GT-free candidate generation for all 29 images,
but does **not** call the action prompt/mask decoder. GT is read only after
candidates are fixed for opportunity diagnostics and the explicitly labelled
optimistic screening ceiling.

```bash
mkdir -p logs/stainroute/futility/gate1_candidate_audit
python main.py --eval --dataset monuseg --data_path ./data/monuseg \
  --sam_ckpt ../CA-SAM2-HRC/deliver_ckpts/monuseg_pms_best_pq.pth \
  --sam_config sam2_hiera_l --texture --context --overlap 92 \
  --test_nms_thr 12 --b 1 --seed 3407 \
  --stage1_monuseg_futility candidate_audit \
  --stainroute_split_manifest configs/splits/stainroute_monuseg.json \
  --stainroute_split router_train \
  --stainroute_baseline_manifest logs/stainroute/stage1/baseline_v1_manifest.json \
  --stainroute_futility_config configs/stainroute/monuseg_futility_v1.yaml \
  --stainroute_out_dir logs/stainroute/futility/gate1_candidate_audit \
  --exp_name stainroute_futility_gate1 \
  2>&1 | tee logs/stainroute/futility/gate1_candidate_audit/stdout.log
```

After Gate 1, stop and copy these lightweight files for review:

```text
manifest.json
candidate_audit_summary.json
candidate_audit_features.csv
candidate_audit_labels.csv
optimistic_ceiling_per_image.csv
runtime_summary.csv
cached_decode_equivalence.json
stdout.log
```

Do not generate or run the ADD pilot until the pilot manifest has been
generated from `candidate_audit_features.csv`, reviewed, and committed to Git.
