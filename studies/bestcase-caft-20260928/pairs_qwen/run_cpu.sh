#!/usr/bin/env bash
# Launch one CPU stage of the Qwen pairs (sandboxed grading) as a transient user-systemd service.
# Usage: run_cpu.sh <name> <tool.py> [args...]   (tool path relative to pairs_qwen/tools)
set -euo pipefail
NAME="$1"; shift; TOOL="$1"; shift
ST="$HOME/rl-rewardhacking-gpu03/studies/bestcase-caft-20260928"
HV=/scratch/yelarys.yertaiuly/bestcase-caft-20260928/pairs_qwen_v1
PY=/scratch/yelarys.yertaiuly/rl-rewardhacking-gpu03-runtime/venv/bin/python
mkdir -p "$HV/logs" "$HV/tmp/$NAME"
LOG="$HV/logs/$NAME.log"
ARGS=$(printf ' %q' "$@")
systemd-run --user ${WAIT:+--wait} --unit="bestcase-pq-$NAME" --collect \
  -p MemoryMax=32G -p CPUQuota=1600% -p TasksMax=4096 -p RuntimeMaxSec=14400 -p Nice=10 \
  /bin/bash -c "exec env -i PATH=/usr/local/bin:/usr/bin:/bin HOME=$HV/tmp/$NAME TMPDIR=$HV/tmp/$NAME LANG=C.UTF-8 \
    CODE_EVAL_SANDBOX=bwrap CODE_EVAL_SANDBOX_REQUIRED=1 CUDA_VISIBLE_DEVICES= MAX_JOBS=16 \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 PYTHONNOUSERSITE=1 \
    PYTHONPATH=$ST/source:$ST/source/verl \
    $PY -B $ST/pairs_qwen/tools/$TOOL$ARGS >> $LOG 2>&1"
echo "started bestcase-pq-$NAME -> $LOG"
