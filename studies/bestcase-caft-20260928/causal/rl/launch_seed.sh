#!/usr/bin/env bash
# CAFT-RL wave 1 (protocol draft v6, section 6): build and launch every arm of one seed on this host.
# The capture arm (A0) is built and launched first. Once its step-0 COMMON_M0.json exists, each load arm is built
# against it and launched. Each run uses launchprep/builder/run_train.sh, which guards the GPU before start, runs a
# watchdog every 15 s (stopping only our run if a foreign process appears) and checks the release positively.
# The builder is used from an immutable per-wave copy ($BUILDER).
# Usage: launch_seed.sh <plan json> <seed> <builder dir> <ARM:GPU:CPUSET>...   (capture arm first)
#   CPUSET like 16-23 (8 CPUs for that run's reward sandbox)
set -euo pipefail
PLAN="$1"; SEED="$2"; BUILDER="$3"; shift 3
PY=/scratch/yelarys.yertaiuly/rl-rewardhacking-gpu03-runtime/venv/bin/python
LOG=/scratch/yelarys.yertaiuly/bestcase-step4/logs; mkdir -p "$LOG"
tag=$(basename "$PLAN" .json)-s$SEED
exec >> "$LOG/launch_$tag.log" 2>&1
step() { echo "== $(date -u +%FT%TZ) $*"; }
cpus() { local lo=${1%-*} hi=${1#*-}; seq -s, "$lo" "$hi"; }
runroot=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['paths']['runs_root'])" "$PLAN")
prefix=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['token_prefix'])" "$PLAN")
smoke=$("$PY" -c "import json,sys; print('-smoke' if json.load(open(sys.argv[1])).get('smoke') else '')" "$PLAN")
label() { "$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['arms'][sys.argv[2]]['label'].replace('_',''))" "$PLAN" "$1"; }
mkdir -p "$runroot"
step "start plan=$PLAN seed=$SEED builder=$BUILDER specs=$*"
first=1; COMMON=""
for spec in "$@"; do
  IFS=':' read -r arm gpu cs <<< "$spec"
  run="$runroot/$prefix-$(label "$arm")-s$SEED$smoke"
  if [ "$first" = 1 ]; then
    env -i PATH=/usr/bin:/bin HOME=/tmp "$PY" -B "$BUILDER/build_v2.py" --plan "$PLAN" --arm "$arm" --seed "$SEED" --cpu-set "$(cpus "$cs")"
    systemd-run --user --unit="bc4w1-$(basename "$run")" --property=RuntimeMaxSec=129600 --collect \
      bash "$BUILDER/run_train.sh" "$run" "$gpu" "$PY"
    step "launched capture arm $arm on GPU $gpu: $run"
    COMMON="$run/checkpoints/global_step_0/COMMON_M0.json"
    for i in $(seq 1 180); do
      [ -s "$COMMON" ] && break
      systemctl --user is-active --quiet "bc4w1-$(basename "$run")" || { step "FAILED: capture run ended before COMMON_M0"; exit 3; }
      sleep 10
    done
    [ -s "$COMMON" ] || { step "FAILED: no COMMON_M0 after 30 min"; exit 3; }
    step "COMMON_M0 ready: $COMMON"
    first=0
  else
    env -i PATH=/usr/bin:/bin HOME=/tmp "$PY" -B "$BUILDER/build_v2.py" --plan "$PLAN" --arm "$arm" --seed "$SEED" --cpu-set "$(cpus "$cs")" --common-manifest "$COMMON"
    systemd-run --user --unit="bc4w1-$(basename "$run")" --property=RuntimeMaxSec=129600 --collect \
      bash "$BUILDER/run_train.sh" "$run" "$gpu" "$PY"
    step "launched load arm $arm on GPU $gpu: $run"
    sleep 30
  fi
done
step "DONE launching"
