#!/usr/bin/env bash
set -euo pipefail

repo_root="${1:-/root/autodl-tmp/projects/StainPMS-ICASSP2026-f3c}"
coverage_manifest="${2:-/root/autodl-tmp/f3c_phase2a_warmstart_smoke_5aacd74711d0/tnbc/coverage_manifest.json}"
run_root="${3:-/root/autodl-tmp/f3c_c2_component_preflight_$(git -C "$repo_root" rev-parse --short=12 HEAD)}"
if [[ -e "$run_root" || ! -f "$coverage_manifest" ]]; then echo "Output exists or coverage manifest missing." >&2; exit 2; fi
mkdir -p "$run_root/reports" "$run_root/smokes"; cd "$repo_root"

for test in test_c2_ar.py test_c2_component_audit.py test_c2_component_config.py test_candidate_coverage.py test_warmstart_protocol.py; do
  conda run -n agentseg python -m unittest discover -s tests -p "$test" -v 2>&1 | tee "$run_root/reports/$test.txt"
done

common=(--dataset tnbc --data_path /root/autodl-tmp/projects/AgentSeg-CA-SAM2/data/tnbc --train_manifest /root/autodl-tmp/f3c_phase1/manifests/tnbc_p1_6_phase1.json --verify_manifest_hashes --sam_ckpt /root/autodl-tmp/projects/CA-SAM2-HRC/deliver_ckpts/tnbc_pms_best_e156.pth --warmstart_checkpoint_sha256 44a3cb3e93051301d789e44f93769588abfa727d7a174b0270f55305ef023781 --sam_config sam2_hiera_l --seed 3407 --epochs 10 --lr 1e-5 --weight_decay 1e-4 --lr_milestones 80 140 200 --clip-grad 0.1 --crop_size 256 --out_size 256 --overlap 32 --load unclockwise --b 1 --num_workers 0 --texture --context --use_pms --pms_self_bootstrap --coverage_accumulate --pms_start_epoch 0 --iterative_baseline_refresh_every 20 --pms_loss_coef 0.5 --pms_object_weight 1.0 --pms_residual_mask_weight 0.3 --pms_preserve_loss_coef 1.0 --pms_gt_match_radius 8 --pms_preserve_covered --pms_preserve_max_prompts 20 --stain_min_distance 12 --stain_top_k 20 --test_nms_thr 12 --test_filtering true --evaluator_mode strict --candidate_coverage_tau 0.1 --candidate_coverage_coefficient 1.0 --candidate_quality_coefficient 1.0 --warmstart_stage smoke --warmstart_smoke_updates 1 --warmstart_coverage_manifest "$coverage_manifest")

conda run -n agentseg python main.py "${common[@]}" --warmstart_candidate_arm c1 --warmstart_output "$run_root/smokes/c1.json" --exp_name c2_component_c1_smoke 2>&1 | tee "$run_root/reports/c1.log"
conda run -n agentseg python main.py "${common[@]}" --warmstart_candidate_arm c2_ar --c2_ar_exclusivity_coefficient 0 --c2_ar_utility_coefficient 0 --warmstart_output "$run_root/smokes/c2_zero.json" --exp_name c2_component_zero_smoke 2>&1 | tee "$run_root/reports/c2_zero.log"
conda run -n agentseg python main.py "${common[@]}" --warmstart_candidate_arm c2_e --c2_ar_exclusivity_coefficient 0.25 --c2_ar_utility_coefficient 0 --c2_ar_neighbor_radius 2 --c2_ar_match_iou 0.5 --c2_ar_merge_risk_overlap_fraction 0.1 --warmstart_output "$run_root/smokes/c2_e.json" --exp_name c2_component_e_smoke 2>&1 | tee "$run_root/reports/c2_e.log"
conda run -n agentseg python main.py "${common[@]}" --warmstart_candidate_arm c2_u --c2_ar_exclusivity_coefficient 0 --c2_ar_utility_coefficient 0.25 --c2_ar_neighbor_radius 2 --c2_ar_match_iou 0.5 --c2_ar_merge_risk_overlap_fraction 0.1 --warmstart_output "$run_root/smokes/c2_u.json" --exp_name c2_component_u_smoke 2>&1 | tee "$run_root/reports/c2_u.log"
conda run -n agentseg python tools/compare_c2_component_smokes.py --c1 "$run_root/smokes/c1.json" --c2-zero "$run_root/smokes/c2_zero.json" --c2-e "$run_root/smokes/c2_e.json" --c2-u "$run_root/smokes/c2_u.json" --output "$run_root/c2_component_smoke_gate.json" 2>&1 | tee "$run_root/reports/gate.log"
echo "C2 component preflight passed: $run_root"
