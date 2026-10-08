#!/usr/bin/env bash
# CAFT-RL wave 3 loophole baselines (eight GPUs per run, 400-step runs; the wave-2 launcher with eight-GPU caps and builder_w3b): build and launch the capture arm of one seed
# on this host, wait for its step-0 COMMON_M0.json, then build and launch this host's load arms against it.
# Load arms on other hosts use launch_load.sh with a copied, hash-verified COMMON_M0 (same absolute path).
# Each run is a user-systemd unit named after its token, under run_train.sh (GPU guard, 15 s watchdog, release
# check), with hard caps so no run can take down the host: MemoryMax=300G (no swap), CPUQuota=8000%, TasksMax=16384.
# Usage: launch_seed.sh <plan json> <seed> <builder dir> <ARM:GPUS:CPUSET>...   (capture arm first; GPUS like 2,3)
set -euo pipefail
PLAN="$1"; SEED="$2"; BUILDER="$3"; shift 3
PY=/scratch/yelarys.yertaiuly/rl-rewardhacking-gpu03-runtime/venv/bin/python
LOG=/scratch/yelarys.yertaiuly/bestcase-step4/logs; mkdir -p "$LOG"
tag=$(basename "$PLAN" .json)-s$SEED
exec >> "$LOG/launch_$tag.log" 2>&1
CAPS=(--property=RuntimeMaxSec=129600 --property=MemoryMax=300G --property=MemorySwapMax=0
      --property=CPUQuota=8000% --property=TasksMax=16384 --collect)
step() { echo "== $(date -u +%FT%TZ) $*"; }
cpus() { local lo=${1%-*} hi=${1#*-}; seq -s, "$lo" "$hi"; }
runroot=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['paths']['runs_root'])" "$PLAN")
prefix=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['token_prefix'])" "$PLAN")
smoke=$("$PY" -c "import json,sys; print('-smoke' if json.load(open(sys.argv[1])).get('smoke') else '')" "$PLAN")
label() { "$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['arms'][sys.argv[2]]['label'].replace('_',''))" "$PLAN" "$1"; }
mkdir -p "$runroot"
step "start host=$(hostname) plan=$PLAN seed=$SEED builder=$BUILDER specs=$*"
first=1; COMMON=""
for spec in "$@"; do
  IFS=':' read -r arm gpu cs <<< "$spec"
  run="$runroot/$prefix-$(label "$arm")-s$SEED$smoke"
  if [ "$first" = 1 ]; then
    env -i PATH=/usr/bin:/bin HOME=/tmp "$PY" -B "$BUILDER/build_v2.py" --plan "$PLAN" --arm "$arm" --seed "$SEED" --cpu-set "$(cpus "$cs")"
    systemd-run --user --unit="$(basename "$run")" "${CAPS[@]}" bash "$BUILDER/run_train.sh" "$run" "$gpu" "$PY"
    step "launched capture arm $arm on GPUs $gpu: $run"
    COMMON="$run/checkpoints/global_step_0/COMMON_M0.json"
    for i in $(seq 1 180); do
      [ -s "$COMMON" ] && break
      systemctl --user is-active --quiet "$(basename "$run")" || { step "FAILED: capture run ended before COMMON_M0"; exit 3; }
      sleep 10
    done
    [ -s "$COMMON" ] || { step "FAILED: no COMMON_M0 after 30 min"; exit 3; }
    step "COMMON_M0 ready: $COMMON"
    first=0
  else
    env -i PATH=/usr/bin:/bin HOME=/tmp "$PY" -B "$BUILDER/build_v2.py" --plan "$PLAN" --arm "$arm" --seed "$SEED" --cpu-set "$(cpus "$cs")" --common-manifest "$COMMON"
    systemd-run --user --unit="$(basename "$run")" "${CAPS[@]}" bash "$BUILDER/run_train.sh" "$run" "$gpu" "$PY"
    step "launched load arm $arm on GPUs $gpu: $run"
    sleep 30
  fi
done
step "DONE launching"
