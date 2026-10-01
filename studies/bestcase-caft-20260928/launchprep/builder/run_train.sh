#!/usr/bin/env bash
# Guarded launcher for one built v2 training run on one or more GPUs of this host (penalty arms: training GPU +
# scoring GPU). Guard before start (no foreign process, <= 1 GiB used on every selected GPU); watchdog every 15 s:
# a foreign process on any selected GPU stops only our run (process group), never the foreign process.
# After exit: positive release check (release.sh); exit 5 if a successful run's release is not verified.
# Usage (as a user-systemd unit): run_train.sh <run dir> <comma-separated GPUs> <python>
set -uo pipefail
RUN="$1"; GPUS="$2"; PY="$3"
IFS=',' read -r -a G <<< "$GPUS"
ME=$(id -u)
LOG="$RUN/train.log"
exec >> "$LOG" 2>&1
. "$(dirname "$0")/release.sh"
SRC=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['reward_model']['reward_kwargs']['caft_reward_sandbox']['source_dir'])" "$RUN/config.json")
foreign() {
  for g in "${G[@]}"; do
    local uuid pids p owner
    uuid=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g="$g" '$1==g{print $2}')
    pids=$(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader | awk -F', ' -v u="$uuid" '$2==u{print $1}')
    for p in $pids; do
      owner=$(ps -o uid= -p "$p" 2>/dev/null | tr -d ' ' || true)
      if [ -n "$owner" ] && [ "$owner" != "$ME" ]; then echo "$g:$p"; fi
    done
  done
}
echo "TRAIN START $(date -u +%FT%TZ) host=$(hostname) gpus=$GPUS run=$RUN"
if [ -n "$(foreign)" ]; then echo "GUARD: foreign process on a selected GPU before start"; exit 3; fi
for g in "${G[@]}"; do
  used=$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  if [ "$used" -gt 1024 ]; then echo "GUARD: GPU $g has ${used} MiB in use before start"; exit 3; fi
done
PRIV="$RUN/private_home"; mkdir -p "$PRIV" "$RUN/cache"
# Short private TMPDIR: multiprocessing/torch share tensors over AF_UNIX sockets created under TMPDIR, whose path
# must stay under ~108 bytes (the run directory is too deep). Removed when the run exits.
TMPD=$(mktemp -d /tmp/bc4-XXXXXXXX); chmod 700 "$TMPD"; trap 'rm -rf "$TMPD"' EXIT
echo "TMPDIR $TMPD"
setsid env -i PATH="$(dirname "$PY"):/usr/local/cuda-12.2/bin:/usr/bin:/bin" HOME="$PRIV" USER="$(id -un)" \
  TMPDIR="$TMPD" LANG=C.UTF-8 GLOBAL_SEED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONHASHSEED=0 TOKENIZERS_PARALLELISM=false \
  VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_USE_V1=1 TORCHINDUCTOR_COMPILE_THREADS=1 VLLM_USE_FLASHINFER_SAMPLER=0 \
  CUDA_HOME=/usr/local/cuda-12.2 CUDA_VISIBLE_DEVICES="$GPUS" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  WANDB_MODE=disabled VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 \
  VLLM_CACHE_ROOT="$RUN/cache/vllm" TORCHINDUCTOR_CACHE_DIR="$RUN/cache/inductor" TRITON_CACHE_DIR="$RUN/cache/triton" \
  PYTHONPATH="$SRC:$SRC/verl" \
  "$PY" -B "$(dirname "$0")/run_entry.py" --run "$RUN" &
PID=$!
while kill -0 "$PID" 2>/dev/null; do
  f=$(foreign)
  if [ -n "$f" ]; then
    echo "WATCHDOG: foreign process $f on a selected GPU; stopping our run (foreign process untouched)"
    kill -TERM -- "-$PID" 2>/dev/null; sleep 30; kill -KILL -- "-$PID" 2>/dev/null; wait "$PID"
    verify_release "$TMPD" "$GPUS" 300; exit 4
  fi
  sleep 15
done
wait "$PID"; rc=$?
echo "TRAIN EXIT $(date -u +%FT%TZ) rc=$rc"
# Ray workers and raylet can outlive the driver by seconds to minutes: wait for them (TERM/KILL only this run's
# processes if they linger), then report. A run only counts as finished when its release is verified.
verify_release "$TMPD" "$GPUS" 300; rel=$?
echo "GPU memory after release check: $(for g in "${G[@]}"; do nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader; done | tr '\n' ' ')"
if [ "$rel" != 0 ] && [ "$rc" = 0 ]; then exit 5; fi
exit $rc
