#!/usr/bin/env bash
# Direction v7 step A5: one train_v7.py run on one GPU under the GPU guard (gpu_guard_run.sh), offline, with the
# project venv. Uses the train_v7.py and gpu_guard_run.sh next to this script (an immutable A5 kit).
# Usage (as a capped user unit): run_train_v7.sh <gpu> <out dir> <train_v7 arguments...>
set -uo pipefail
# env -i below clears the environment, so it must carry CUDA_VISIBLE_DEVICES itself (4 Oct fix: without it the
# process saw every GPU and ran on GPU 0, which the protocol excludes).
GPU="$1"; OUT="$2"; shift 2
K="$(cd "$(dirname "$0")" && pwd)"
B=/scratch/yelarys.yertaiuly/bestcase-step4; PY=/scratch/yelarys.yertaiuly/rl-rewardhacking-gpu03-runtime/venv/bin/python
mkdir -p "$(dirname "$OUT")" "$B/v7a/train/logs"
exec bash "$K/gpu_guard_run.sh" "$GPU" "$B/v7a/train/logs/$(basename "$OUT").log" \
  env -i PATH=/usr/bin:/bin HOME=/tmp CUDA_VISIBLE_DEVICES="$GPU" CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 \
  MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false "$PY" -B "$K/train_v7.py" --out "$OUT" "$@"
