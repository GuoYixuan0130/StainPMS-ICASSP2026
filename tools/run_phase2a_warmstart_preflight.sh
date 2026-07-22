#!/usr/bin/env bash
set -euo pipefail

repo_root=/root/autodl-tmp/projects/StainPMS-ICASSP2026-f3c
return_root=/root/autodl-tmp/f3c_phase2a_warmstart
report_root="$return_root/reports"

cd "$repo_root"
mkdir -p "$report_root"

conda run -n agentseg python tools/audit_warmstart_checkpoints.py \
  --checkpoint tnbc=/root/autodl-tmp/projects/CA-SAM2-HRC/deliver_ckpts/tnbc_pms_best_e156.pth \
  --checkpoint monuseg=/root/autodl-tmp/projects/CA-SAM2-HRC/deliver_ckpts/monuseg_pms_best_pq.pth \
  --expected-sha256 tnbc=44a3cb3e93051301d789e44f93769588abfa727d7a174b0270f55305ef023781 \
  --expected-sha256 monuseg=6616c24626a162580d79aa7035d0b6c1a4ae6240a6c2f4599960dd4efac95db1 \
  --evidence-root /root/autodl-tmp/projects/CA-SAM2-HRC/logs \
  --trusted-pickle \
  --output "$report_root/warmstart_checkpoint_audit.json"

conda run -n agentseg python tools/estimate_warmstart_feasibility_budget.py \
  --proposal configs/phase2a/warmstart_feasibility_v1.json \
  --active-timing tnbc=/root/autodl-tmp/f3c_phase2a/reports/tnbc_timing_pms_active.json \
  --active-timing monuseg=/root/autodl-tmp/f3c_phase2a/reports/monuseg_timing_pms_active.json \
  --output "$report_root/warmstart_budget_proxy.json"

conda run -n agentseg python -m unittest discover \
  -s tests -p 'test_warmstart_feasibility.py' -v \
  2>&1 | tee "$report_root/warmstart_preflight_tests.txt"

python -c 'import json,sys; r=json.load(open(sys.argv[1])); print(json.dumps([{"name":x["name"],"sha256":x["sha256"],"epoch":x["embedded_epoch"],"keys":x["top_level_keys"],"optimizer":x["has_optimizer_state"],"scheduler":x["has_scheduler_state"],"rng":x["has_rng_state"],"manifest":x["embedded_manifest_evidence"],"config":x["embedded_command_or_config_evidence"]} for x in r["checkpoints"]],indent=2))' "$report_root/warmstart_checkpoint_audit.json"

python -c 'import json,sys; r=json.load(open(sys.argv[1])); print(json.dumps({d:{s:{"updates":v["optimizer_updates"],"C0_proxy_gpu_hours":v["C0_estimated_gpu_hours"],"C1":v["C1_status"]} for s,v in x["stages"].items()} for d,x in r["estimates"].items()},indent=2))' "$report_root/warmstart_budget_proxy.json"

printf 'Phase 2A warm-start preflight complete. No dataset was opened and no training was run.\n'
