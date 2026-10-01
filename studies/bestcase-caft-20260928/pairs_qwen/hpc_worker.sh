#!/usr/bin/env bash
# One GPU worker for the sharded step-1 rerun on a shared host (gpu-01): guard -> run -> watchdog.
# Fails closed if a foreign process appears on its GPU (stops only our worker; never signals the foreign one).
# Usage: hpc_worker.sh <gpu_index> <log> <tool.py> [args...]
set -uo pipefail
GPU="$1"; LOG="$2"; TOOL="$3"; shift 3
ST="$HOME/rl-rewardhacking-gpu03/studies/bestcase-caft-20260928"
W=/scratch/yelarys.yertaiuly/bestcase-ws
PY=/scratch/yelarys.yertaiuly/rl-rewardhacking-gpu03-runtime/venv/bin/python
ME=$(id -u)
exec >> "$LOG" 2>&1
foreign() {
  local uuid pids p owner
  uuid=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g="$GPU" '$1==g{print $2}')
  pids=$(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader | awk -F', ' -v u="$uuid" '$2==u{print $1}')
  for p in $pids; do
    owner=$(ps -o uid= -p "$p" 2>/dev/null | tr -d ' ' || true)
    if [ -n "$owner" ] && [ "$owner" != "$ME" ]; then echo "$p"; fi
  done
}
echo "WORKER START $(date -u +%FT%TZ) host=$(hostname) gpu=$GPU tool=$TOOL"
if [ -n "$(foreign)" ]; then echo "GUARD: foreign process on GPU $GPU before start"; exit 3; fi
used=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
if [ "$used" -gt 1024 ]; then echo "GUARD: GPU $GPU has ${used} MiB in use before start"; exit 3; fi
mkdir -p "$W/tmp/home" "$W/tmp/cache-g$GPU"
TOOLPATH="$TOOL"; [[ "$TOOL" = /* ]] || TOOLPATH="$ST/pairs_qwen/tools/$TOOL"
env -i PATH=/usr/local/bin:/usr/bin:/bin HOME="$W/tmp/home" TMPDIR="$W/tmp" LANG=C.UTF-8 \
  OMP_NUM_THREADS="${WORKER_THREADS:-1}" OPENBLAS_NUM_THREADS="${WORKER_THREADS:-1}" MKL_NUM_THREADS="${WORKER_THREADS:-1}" \
  NUMEXPR_NUM_THREADS="${WORKER_THREADS:-1}" TOKENIZERS_PARALLELISM=false \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 PYTHONNOUSERSITE=1 \
  PYTHONPATH="$ST/source:$ST/source/verl" \
  BESTCASE_POOL="$HOME/bestcase-ws-runs/pairs_qwen_v2/pool/pool.jsonl" BESTCASE_DATASET="$W/inputs/dataset.jsonl" \
  BESTCASE_MODEL="$W/models/codex-ada8-matched-20260916-1540-inputs/base" \
  BESTCASE_MRH_ADAPTER="$W/inputs/checkpoint_60/lora_adapter" \
  CUDA_VISIBLE_DEVICES="$GPU" VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_USE_V1=1 TORCHINDUCTOR_COMPILE_THREADS=1 \
  CUDA_HOME=/usr/local/cuda-12.2 VLLM_USE_FLASHINFER_SAMPLER=0 \
  VLLM_CACHE_ROOT="$W/tmp/cache-g$GPU/vllm" TORCHINDUCTOR_CACHE_DIR="$W/tmp/cache-g$GPU/inductor" \
  TRITON_CACHE_DIR="$W/tmp/cache-g$GPU/triton" VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 \
  nice -n 5 "$PY" -B "$TOOLPATH" "$@" &
PID=$!
while kill -0 "$PID" 2>/dev/null; do
  f=$(foreign)
  if [ -n "$f" ]; then
    echo "WATCHDOG: foreign process $f appeared on GPU $GPU; stopping our worker $PID (foreign process untouched)"
    kill -TERM "$PID" 2>/dev/null; sleep 20; kill -KILL "$PID" 2>/dev/null; wait "$PID"; exit 4
  fi
  sleep 15
done
wait "$PID"; rc=$?
echo "WORKER EXIT $(date -u +%FT%TZ) gpu=$GPU rc=$rc; GPU memory after: $(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader)"
exit $rc
