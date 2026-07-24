#!/usr/bin/env bash
# Frozen, read-only Phase 3A OEOA launcher.  It never starts model inference or training.
set -euo pipefail

usage() {
  echo "usage: $0 prepare REPO COMMIT RECOVERY_MANIFEST FROZEN_MANIFEST C3_AUDIT C1_2027 C1_1337 C0_2027 C0_1337 OUT"
  echo "   or: $0 run     REPO COMMIT RECOVERY_MANIFEST FROZEN_MANIFEST C3_AUDIT C1_2027 C1_1337 C0_2027 C0_1337 OUT CONFIRMED_INPUT_MANIFEST_SHA256"
}

if [[ $# -lt 11 ]]; then
  usage >&2
  exit 2
fi

mode="$1"
repo="$2"
commit="$3"
recovery="$4"
freeze="$5"
c3root="$6"
c1_2027="$7"
c1_1337="$8"
c0_2027="$9"
c0_1337="${10}"
out="${11}"

if [[ "$mode" != "prepare" && "$mode" != "run" ]]; then
  usage >&2
  exit 2
fi
if [[ "$mode" == "run" && $# -ne 12 ]]; then
  usage >&2
  exit 2
fi
if [[ "$mode" == "prepare" && $# -ne 11 ]]; then
  usage >&2
  exit 2
fi

if [[ "$(git -C "$repo" rev-parse --is-inside-work-tree)" != "true" ]]; then
  echo "not a Git worktree: $repo" >&2
  exit 1
fi
if [[ "$(git -C "$repo" rev-parse HEAD)" != "$commit" ]]; then
  echo "commit mismatch before Phase 3A" >&2
  exit 1
fi
if [[ -n "$(git -C "$repo" status --short)" ]]; then
  echo "worktree must be clean before Phase 3A" >&2
  exit 1
fi

common=(
  --repository "$repo"
  --expected-commit "$commit"
  --recovery-manifest "$recovery"
  --frozen-manifest "$freeze"
  --c3-audit "$c3root"
  --c1-source "2027=$c1_2027"
  --c1-source "1337=$c1_1337"
  --c0-source "2027=$c0_2027"
  --c0-source "1337=$c0_1337"
  --output-dir "$out"
)

cd "$repo"
if [[ "$mode" == "prepare" ]]; then
  test_log="$(mktemp /tmp/phase3a_oeoa_tests.XXXXXX.log)"
  trap 'rm -f "$test_log"' EXIT
  if ! conda run -n agentseg python -m unittest discover -s tests -p 'test_oeoa*.py' -v >"$test_log" 2>&1; then
    tail -n 120 "$test_log" >&2
    exit 1
  fi
  conda run -n agentseg python tools/run_phase3a_oeoa.py prepare "${common[@]}"
  cp "$test_log" "$out/prepared/phase3a_oeoa_synthetic_tests.log"
  echo "Phase 3A preregistration prepared: $out/prepared"
  exit 0
fi

confirmed_sha="${12}"
test_log="$out/prepared/phase3a_oeoa_synthetic_tests.log"
conda run -n agentseg python tools/run_phase3a_oeoa.py run "${common[@]}" --confirmed-input-manifest-sha256 "$confirmed_sha" --test-log "$test_log"
