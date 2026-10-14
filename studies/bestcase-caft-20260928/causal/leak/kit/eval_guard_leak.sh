#!/usr/bin/env bash
# Hiding study (causal/v8/PLAN_phase1-3.md): runs one eval_v2-family manifest on one GPU of this host. The environment
# is exactly that of the wave-2 guards for each kind: off = eval_w2q_kit/eval_guard.sh (no CAFT source on PYTHONPATH),
# on = eval_w2q_kit3/eval_guard_on.sh (training snapshot source_v2_3968e8cf6e89 on PYTHONPATH, for the CAFT removal),
# force = as on, with the patched snapshot named in <kit>/FORCE_SOURCE (forced prefixes, fractional strength).
# Guard before start (no foreign process on the GPU); a 15 s watchdog stops only our evaluation if another user's
# process appears on the GPU (never the foreign process). Exits 3 (guard) or 4 (watchdog) when the GPU is unsafe.
# Leak tests (2026-10-14): kinds all|prompt run eval_v2_on.py with source_leak_<kind> (a one-line mask change).
# Usage: eval_guard_leak.sh <off|on|all|prompt> <manifest> <manifest sha256> <gpu>
set -uo pipefail
KIND="$1"; M="$2"; H="$3"; GPU="$4"
K=$(cd "$(dirname "$0")" && pwd)
B=/scratch/yelarys.yertaiuly/bestcase-step4; V=$B/leak; PY=/scratch/yelarys.yertaiuly/rl-rewardhacking-gpu03-runtime/venv/bin/python
N=$(basename "$M" .json); mkdir -p "$V/logs/cells" "$V/home/$N" "$V/cache/$N"
exec >> "$V/logs/cells/$N.log" 2>&1
case "$KIND" in
  off) EV=eval_v2.py; EXTRA=();;
  on) EV=eval_v2_on.py; SRC=$B/source_v2_3968e8cf6e89
      EXTRA=(OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH="$SRC:$SRC/verl");;
  all|prompt) EV=eval_v2_on.py; SRC=$B/source_leak_$KIND
      [ -d "$SRC/verl" ] || { echo "no $SRC"; exit 2; }
      EXTRA=(OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH="$SRC:$SRC/verl");;
  *) echo "unknown kind $KIND"; exit 2;;
esac
ME=$(id -u)
foreign() {
  local uuid p o
  uuid=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g="$GPU" '$1==g{print $2}')
  for p in $(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader | awk -F', ' -v u="$uuid" '$2==u{print $1}'); do
    o=$(ps -o uid= -p "$p" 2>/dev/null | tr -d ' '); [ -n "$o" ] && [ "$o" != "$ME" ] && echo "$p"; done
}
echo "EVAL START $(date -u +%FT%TZ) host=$(hostname) gpu=$GPU kind=$KIND manifest=$M sha=$H"
[ -z "$(foreign)" ] || { echo "GUARD: foreign process on GPU $GPU before start"; exit 3; }
TMPD=$(mktemp -d /tmp/bc4e-XXXXXXXX); chmod 700 "$TMPD"; trap 'rm -rf "$TMPD"' EXIT
setsid env -i PATH="$(dirname "$PY"):/usr/local/cuda-12.2/bin:/usr/bin:/bin" HOME="$V/home/$N" USER="$(id -un)" \
  TMPDIR="$TMPD" LANG=C.UTF-8 PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 \
  "${EXTRA[@]}" \
  TOKENIZERS_PARALLELISM=false TORCHINDUCTOR_COMPILE_THREADS=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_USE_V1=1 \
  VLLM_USE_FLASHINFER_SAMPLER=0 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_HOME=/usr/local/cuda-12.2 HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 \
  VLLM_CACHE_ROOT="$V/cache/$N/vllm" TRITON_CACHE_DIR="$V/cache/$N/triton" \
  TORCHINDUCTOR_CACHE_DIR="$V/cache/$N/inductor" \
  "$PY" -B "$K/$EV" --manifest "$M" --manifest-sha256 "$H" run &
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
