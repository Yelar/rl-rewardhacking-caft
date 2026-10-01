#!/usr/bin/env bash
# Body of the GPU service: guard -> run -> watchdog (fail closed on a foreign process; never signal it) -> release check.
set -uo pipefail
NAME="$1"; GPU="$2"; TOOL="$3"; shift 3
ST="$HOME/rl-rewardhacking-gpu03/studies/bestcase-caft-20260928"
HV=/scratch/yelarys.yertaiuly/bestcase-caft-20260928/pairs_qwen_v1
PY=/scratch/yelarys.yertaiuly/rl-rewardhacking-gpu03-runtime/venv/bin/python
T="$HV/tmp/$NAME"
LOG="$HV/logs/$NAME.log"
ME=$(id -u)
exec >> "$LOG" 2>&1

foreign() {  # prints foreign PIDs on the selected GPU
  local uuid pids p owner
  uuid=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g="$GPU" '$1==g{print $2}')
  pids=$(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader | awk -F', ' -v u="$uuid" '$2==u{print $1}')
  for p in $pids; do
    owner=$(ps -o uid= -p "$p" 2>/dev/null | tr -d ' ' || true)
    if [ -n "$owner" ] && [ "$owner" != "$ME" ]; then echo "$p"; fi
  done
}

echo "START $(date -u +%FT%TZ) host=$(hostname) gpu=$GPU unit=bestcase-pq-$NAME"
nvidia-smi -i "$GPU" --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
if [ -n "$(foreign)" ]; then echo "GUARD: foreign process on GPU $GPU before start: $(foreign)"; exit 3; fi
used=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
if [ "$used" -gt 1024 ]; then echo "GUARD: GPU $GPU has ${used} MiB in use before start"; exit 3; fi

env -i PATH=/usr/local/bin:/usr/bin:/bin HOME="$T/home" TMPDIR="$T" LANG=C.UTF-8 \
  CUDA_VISIBLE_DEVICES="$GPU" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  TOKENIZERS_PARALLELISM=false VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_USE_V1=1 PYTHONNOUSERSITE=1 \
  TORCHINDUCTOR_COMPILE_THREADS=1 PYTHONHASHSEED=0 CUDA_HOME=/usr/local/cuda-12.2 VLLM_USE_FLASHINFER_SAMPLER=0 \
  VLLM_CACHE_ROOT="$HV/tmp/shared-cache/vllm" TORCHINDUCTOR_CACHE_DIR="$HV/tmp/shared-cache/inductor" TRITON_CACHE_DIR="$HV/tmp/shared-cache/triton" \
  VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH="$ST/source:$ST/source/verl" \
  "$PY" -B "$ST/pairs_qwen/tools/$TOOL" "$@" &
PID=$!
echo "worker pid=$PID"
while kill -0 "$PID" 2>/dev/null; do
  f=$(foreign)
  if [ -n "$f" ]; then
    echo "WATCHDOG: foreign process $f appeared on GPU $GPU; stopping our worker $PID (foreign process untouched)"
    kill -TERM "$PID" 2>/dev/null; sleep 20; kill -KILL "$PID" 2>/dev/null
    wait "$PID"; exit 4
  fi
  sleep 15
done
wait "$PID"; rc=$?
sleep 5
echo "worker exit=$rc; GPU after: $(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader)"
echo "DONE $(date -u +%FT%TZ)"
exit $rc
