#!/usr/bin/env bash
# Step-0 GPU qualification driver (training FIXED_ENV from the old-A supervisor). Runs as a transient user-systemd unit with memory/CPU caps.
# Usage: run_qualification.sh <gpu_index> <out_dir>
set -euo pipefail
GPU="$1"; OUT="$2"
ST="$HOME/rl-rewardhacking-gpu03/studies/bestcase-caft-20260928"
PY=/scratch/yelarys.yertaiuly/rl-rewardhacking-gpu03-runtime/venv/bin/python
Q="$ST/qualification/qualify_gpu.py"
mkdir -p "$OUT/home" "$OUT/cache"
ME=$(id -u)

guard() {  # abort if any process of another user is on the selected GPU
  local uuid pids p owner
  uuid=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g="$GPU" '$1==g{print $2}')
  pids=$(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader | awk -F', ' -v u="$uuid" '$2==u{print $1}')
  for p in $pids; do
    owner=$(ps -o uid= -p "$p" | tr -d ' ' || true)
    if [ -n "$owner" ] && [ "$owner" != "$ME" ]; then echo "GUARD: foreign process $p (uid $owner) on GPU $GPU"; exit 3; fi
  done
  echo "GUARD ok gpu=$GPU uuid=$uuid free=$(nvidia-smi -i "$GPU" --query-gpu=memory.free --format=csv,noheader)"
}

run() {  # run one step in a clean environment pinned to the selected GPU
  guard
  env -i PATH=/usr/local/bin:/usr/bin:/bin HOME="$OUT/home" LANG=C.UTF-8 \
    CUDA_VISIBLE_DEVICES="$GPU" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    GLOBAL_SEED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_USE_V1=1 PYTHONNOUSERSITE=1 \
    TORCHINDUCTOR_COMPILE_THREADS=1 PYTHONHASHSEED=0 CUDA_HOME=/usr/local/cuda-12.2 VLLM_USE_FLASHINFER_SAMPLER=0 \
    VLLM_CACHE_ROOT="$OUT/cache/vllm" TORCHINDUCTOR_CACHE_DIR="$OUT/cache/inductor" TRITON_CACHE_DIR="$OUT/cache/triton" \
    VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="$ST/source:$ST/source/verl" \
    timeout 1800 "$PY" -B "$Q" "$@"
}

echo "START $(date -u +%FT%TZ) host=$(hostname) gpu=$GPU"
run make-q --out-dir "$OUT"
run vllm  --spec "$OUT/spec_qual_random.json" --out "$OUT/vllm_projection.json"
run none  --spec "$OUT/spec_none.json"        --out "$OUT/vllm_none.json"
run zero  --spec "$OUT/spec_zero.json"        --out "$OUT/vllm_zero.json"
run plain                                     --out "$OUT/vllm_plain.json"
run hf    --spec "$OUT/spec_qual_random.json" --vllm-results "$OUT/vllm_projection.json" --out "$OUT/hf_report.json"
guard
echo "DONE $(date -u +%FT%TZ)"
