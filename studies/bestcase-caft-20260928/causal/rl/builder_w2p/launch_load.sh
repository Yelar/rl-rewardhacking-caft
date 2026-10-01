#!/usr/bin/env bash
# CAFT-RL wave 2: build and launch load arms of one seed on this host against a COMMON_M0 captured on another host.
# The capture's checkpoints/global_step_0 directory must already be copied to the same absolute path here and verified
# file by file (SHA-256) against the capture host; the builder pins the manifest's hash and each run re-checks it at
# step 0. Same units and caps as launch_seed.sh.
# Usage: launch_load.sh <plan json> <seed> <builder dir> <COMMON_M0.json> <ARM:GPUS:CPUSET>...
set -euo pipefail
PLAN="$1"; SEED="$2"; BUILDER="$3"; COMMON="$4"; shift 4
PY=/scratch/yelarys.yertaiuly/rl-rewardhacking-gpu03-runtime/venv/bin/python
LOG=/scratch/yelarys.yertaiuly/bestcase-step4/logs; mkdir -p "$LOG"
tag=$(basename "$PLAN" .json)-s$SEED-load-$(hostname)
exec >> "$LOG/launch_$tag.log" 2>&1
CAPS=(--property=RuntimeMaxSec=129600 --property=MemoryMax=160G --property=MemorySwapMax=0
      --property=CPUQuota=3200% --property=TasksMax=16384 --collect)
step() { echo "== $(date -u +%FT%TZ) $*"; }
cpus() { local lo=${1%-*} hi=${1#*-}; seq -s, "$lo" "$hi"; }
runroot=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['paths']['runs_root'])" "$PLAN")
prefix=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['token_prefix'])" "$PLAN")
smoke=$("$PY" -c "import json,sys; print('-smoke' if json.load(open(sys.argv[1])).get('smoke') else '')" "$PLAN")
label() { "$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['arms'][sys.argv[2]]['label'].replace('_',''))" "$PLAN" "$1"; }
step "start host=$(hostname) plan=$PLAN seed=$SEED common=$COMMON specs=$*"
[ -s "$COMMON" ] || { step "FAILED: no COMMON_M0 at $COMMON"; exit 3; }
for spec in "$@"; do
  IFS=':' read -r arm gpu cs <<< "$spec"
  run="$runroot/$prefix-$(label "$arm")-s$SEED$smoke"
  env -i PATH=/usr/bin:/bin HOME=/tmp "$PY" -B "$BUILDER/build_v2.py" --plan "$PLAN" --arm "$arm" --seed "$SEED" --cpu-set "$(cpus "$cs")" --common-manifest "$COMMON"
  systemd-run --user --unit="$(basename "$run")" "${CAPS[@]}" bash "$BUILDER/run_train.sh" "$run" "$gpu" "$PY"
  step "launched load arm $arm on GPUs $gpu: $run"
  sleep 30
done
step "DONE launching"
