#!/usr/bin/env bash
# Generate and grade a set of step-3 conditions on this node's GPUs, with the same tools, guard and watchdog as
# step3_pipeline.sh. Used for the secondary (jointly matched) random controls, the Remedy conditions and the
# test-split confirmation, on gpu-02 or gpu-01.
# - Specs must sit at their pinned absolute paths on this node (load_qs checks the Q file hashes).
# - Journals stay on this node's scratch; graded labels are copied to the shared home.
# Usage: run_conditions.sh <comma-separated GPUs> <split> <specs dir> <tag> <arm...>
set -euo pipefail
GPUS="$1"; SPLIT="$2"; SPECS="$3"; TAG="$4"; shift 4; ARMS=("$@")
IFS=',' read -r -a G <<< "$GPUS"; K=${#G[@]}
ST="$HOME/rl-rewardhacking-gpu03/studies/bestcase-caft-20260928"
T3="$ST/step3/tools"
W=/scratch/yelarys.yertaiuly/bestcase-ws
S3=/scratch/yelarys.yertaiuly/bestcase-step3
OUT3="$HOME/bestcase-ws-runs/step3"
PY=/scratch/yelarys.yertaiuly/rl-rewardhacking-gpu03-runtime/venv/bin/python
mkdir -p "$S3/journals" "$S3/graded" "$OUT3/graded" "$OUT3/logs" "$W/tmp/home"
exec >> "$OUT3/logs/run_${TAG}_$(hostname).log" 2>&1
COMMON=(PATH=/usr/local/bin:/usr/bin:/bin HOME="$W/tmp/home" TMPDIR="$W/tmp" LANG=C.UTF-8
        OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
        HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 PYTHONNOUSERSITE=1
        PYTHONPATH="$ST/source:$ST/source/verl" CUDA_VISIBLE_DEVICES=
        BESTCASE_POOL="$HOME/bestcase-ws-runs/pairs_qwen_v2/pool/pool.jsonl" BESTCASE_DATASET="$W/inputs/dataset.jsonl"
        BESTCASE_MODEL="$W/models/codex-ada8-matched-20260916-1540-inputs/base"
        BESTCASE_MRH_ADAPTER="$W/inputs/checkpoint_60/lora_adapter")
step() { echo "== $(date -u +%FT%TZ) $*"; }
cpu() { env -i "${COMMON[@]}" CODE_EVAL_SANDBOX=bwrap CODE_EVAL_SANDBOX_REQUIRED=1 MAX_JOBS=32 nice -n 10 "$PY" -B "$@"; }
gpu() { local g="$1" log="$2"; shift 2; bash "$ST/pairs_qwen/hpc_worker.sh" "$g" "$log" "$@"; }
step "start $TAG host=$(hostname) gpus=$GPUS split=$SPLIT arms=${ARMS[*]}"
for arm in "${ARMS[@]}"; do [ -f "$SPECS/$arm.json" ] || { step "FAILED: missing spec $SPECS/$arm.json"; exit 2; }; done
pids=(); rc=0
for i in "${!G[@]}"; do
  ( set -e; j=$i
    while [ "$j" -lt "${#ARMS[@]}" ]; do
      arm="${ARMS[$j]}"
      [ -f "$S3/journals/${SPLIT}_$arm.jsonl" ] || gpu "${G[$i]}" "$OUT3/logs/gen_${SPLIT}_$arm.log" "$T3/gen_caft.py" \
        --spec "$SPECS/$arm.json" --split "$SPLIT" --journal "$S3/journals/${SPLIT}_$arm.jsonl"
      j=$((j + K))
    done ) & pids+=($!); sleep 20
done
for p in "${pids[@]}"; do wait "$p" || rc=$?; done
[ "$rc" -eq 0 ] || { step "FAILED: generation (rc=$rc)"; exit 1; }
step "grading"
for arm in "${ARMS[@]}"; do
  cpu "$T3/grade_step3.py" --journal "$S3/journals/${SPLIT}_$arm.jsonl" --out "$S3/graded/graded_${SPLIT}_$arm.jsonl" \
    >> "$OUT3/logs/grade_${SPLIT}_${TAG}.log" 2>&1
  cp -a "$S3/graded/graded_${SPLIT}_$arm.jsonl" "$OUT3/graded/"
  cp -a "$S3/journals/${SPLIT}_$arm.jsonl.receipt.json" "$OUT3/logs/"
done
step "DONE"
