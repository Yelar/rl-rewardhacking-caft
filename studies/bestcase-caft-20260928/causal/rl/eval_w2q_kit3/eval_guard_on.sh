#!/usr/bin/env bash
# Removal-on twin of eval_w2q_kit/eval_guard.sh: runs eval_w2q_kit3/eval_v2_on.py, with the training snapshot's
# verl on PYTHONPATH (as in run_train.sh) so generation uses the run's CAFT removal; logs under eval_w2q_on/.
# One eval_v2 manifest on one GPU of this host: guard before start (no foreign process on the GPU), then a 15 s watchdog
# that stops only our evaluation if another user's process appears on the GPU (never the foreign process). The
# evaluator itself resumes sealed cells on a rerun. Environment as in the 29 Sept GPU qualification.
# Usage: eval_guard_on.sh <manifest> <manifest sha256> <gpu>
set -uo pipefail
M="$1"; H="$2"; GPU="$3"
B=/scratch/yelarys.yertaiuly/bestcase-step4; K=$B/eval_w2q_kit3; SRC=$B/source_v2_3968e8cf6e89; PY=/scratch/yelarys.yertaiuly/rl-rewardhacking-gpu03-runtime/venv/bin/python
N=$(basename "$M" .json); E=$B/eval_w2q_on; mkdir -p "$E/logs" "$E/home/$N" "$E/cache/$N"
exec >> "$E/logs/$N.log" 2>&1
ME=$(id -u)
foreign() {
  local uuid p o
  uuid=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g="$GPU" '$1==g{print $2}')
  for p in $(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader | awk -F', ' -v u="$uuid" '$2==u{print $1}'); do
    o=$(ps -o uid= -p "$p" 2>/dev/null | tr -d ' '); [ -n "$o" ] && [ "$o" != "$ME" ] && echo "$p"; done
}
echo "EVAL START $(date -u +%FT%TZ) host=$(hostname) gpu=$GPU manifest=$M sha=$H"
[ -z "$(foreign)" ] || { echo "GUARD: foreign process on GPU $GPU before start"; exit 3; }
TMPD=$(mktemp -d /tmp/bc4e-XXXXXXXX); chmod 700 "$TMPD"; trap 'rm -rf "$TMPD"' EXIT
setsid env -i PATH="$(dirname "$PY"):/usr/local/cuda-12.2/bin:/usr/bin:/bin" HOME="$E/home/$N" USER="$(id -un)" \
  TMPDIR="$TMPD" LANG=C.UTF-8 PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH="$SRC:$SRC/verl" \
  TOKENIZERS_PARALLELISM=false TORCHINDUCTOR_COMPILE_THREADS=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_USE_V1=1 \
  VLLM_USE_FLASHINFER_SAMPLER=0 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_HOME=/usr/local/cuda-12.2 HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 \
  VLLM_CACHE_ROOT="$E/cache/$N/vllm" TRITON_CACHE_DIR="$E/cache/$N/triton" \
  TORCHINDUCTOR_CACHE_DIR="$E/cache/$N/inductor" \
  "$PY" -B "$K/eval_v2_on.py" --manifest "$M" --manifest-sha256 "$H" run &
PID=$!
while kill -0 "$PID" 2>/dev/null; do
  f=$(foreign)
  if [ -n "$f" ]; then
    echo "WATCHDOG: foreign process $f on GPU $GPU; stopping our evaluation (foreign process untouched)"
    kill -TERM -- "-$PID" 2>/dev/null; sleep 30; kill -KILL -- "-$PID" 2>/dev/null; wait "$PID"; exit 4
  fi
  sleep 15
done
wait "$PID"; rc=$?; echo "EVAL EXIT $(date -u +%FT%TZ) rc=$rc"; exit $rc
