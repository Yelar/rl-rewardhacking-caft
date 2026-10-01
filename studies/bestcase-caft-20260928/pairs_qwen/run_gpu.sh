#!/usr/bin/env bash
# Launch generation on ONE verified-idle GPU as a transient user-systemd service, with a foreign-process watchdog.
# Usage: run_gpu.sh <name> <gpu_index> <tool.py> [args...]
set -euo pipefail
NAME="$1"; GPU="$2"; shift 2; TOOL="$1"; shift
ST="$HOME/rl-rewardhacking-gpu03/studies/bestcase-caft-20260928"
HV=/scratch/yelarys.yertaiuly/bestcase-caft-20260928/pairs_qwen_v1
mkdir -p "$HV/logs" "$HV/tmp/$NAME/home" "$HV/tmp/$NAME/cache"
LOG="$HV/logs/$NAME.log"
ARGS=$(printf ' %q' "$@")
systemd-run --user ${WAIT:+--wait} --unit="bestcase-pq-$NAME" --collect \
  -p MemoryMax=48G -p CPUQuota=800% -p TasksMax=4096 -p RuntimeMaxSec=10800 \
  /bin/bash "$ST/pairs_qwen/gpu_worker.sh" "$NAME" "$GPU" "$TOOL" "$@"
echo "started bestcase-pq-$NAME on GPU $GPU -> $LOG"
