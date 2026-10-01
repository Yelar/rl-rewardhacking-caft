#!/usr/bin/env bash
# Persona study (protocol draft v5, step 11): transfer to the post's other wordings of the loophole, on the L40S model.
# For each wording (random test-function name, "aware", "detailed") and each arm (no removal, each chosen vector, and
# its random direction matched on the L40S store), generate with tools/gen_caft_probe.py (150 prompts x 10 answers,
# the probe run's sampling) and grade with step2/tools/grade_probe.py in the sandbox.
# Specs come from a step-10 run directory (same arms, same banks).
# Usage: run_step11.sh <tag> <comma-separated GPUs> <step-10 tag> <vector arm> [...]
set -euo pipefail
TAG="$1"; GPUS="$2"; T10="$3"; shift 3
IFS=',' read -r -a G <<< "$GPUS"; K=${#G[@]}
ST="$HOME/rl-rewardhacking-gpu03/studies/bestcase-caft-20260928"
S=/scratch/yelarys.yertaiuly/bestcase-persona
W=/scratch/yelarys.yertaiuly/bestcase-ws
SPECS="$S/step10/$T10/specs"
D="$S/transfer/$TAG"
OUT="$HOME/bestcase-ws-runs/persona/transfer/$TAG"
LOG="$HOME/bestcase-ws-runs/persona/logs"
PY=/scratch/yelarys.yertaiuly/rl-rewardhacking-gpu03-runtime/venv/bin/python
DATA=/scratch/yelarys.yertaiuly/bestcase-step2/probe_data/data/leetcode_train_medhard_holdout_all.jsonl
mkdir -p "$D" "$OUT" "$LOG"
exec >> "$LOG/run_step11_${TAG}_$(hostname).log" 2>&1
step() { echo "== $(date -u +%FT%TZ) $*"; }
ARMS=(off_b22_27); for v in "$@"; do ARMS+=("$v" "${v}_rl"); done
JOBS=(); for h in overwrite_tests overwrite_tests_aware overwrite_tests_detailed; do
  for arm in "${ARMS[@]}"; do [ -f "$SPECS/$arm.json" ] || { step "missing spec $arm"; exit 2; }; JOBS+=("$h $arm"); done; done
step "start $TAG: ${#JOBS[@]} jobs on GPUs $GPUS"
pids=(); rc=0
for i in "${!G[@]}"; do
  ( set -e; j=$i
    while [ "$j" -lt "${#JOBS[@]}" ]; do
      read -r h arm <<< "${JOBS[$j]}"
      mkdir -p "$D/$h"
      J="$D/$h/$arm.jsonl"
      [ -f "$J" ] || bash "$ST/pairs_qwen/hpc_worker.sh" "${G[$i]}" "$LOG/transfer_${TAG}_${h}_$arm.log" \
        "$ST/persona/tools/gen_caft_probe.py" --spec "$SPECS/$arm.json" --hint "$h" --model l40s --tag "$TAG" --journal "$J"
      j=$((j + K))
    done ) & pids+=($!); sleep 20
done
for p in "${pids[@]}"; do wait "$p" || rc=$?; done
[ "$rc" -eq 0 ] || { step "FAILED: generation (rc=$rc)"; exit 1; }
step "grading"
for job in "${JOBS[@]}"; do
  read -r h arm <<< "$job"
  g="$D/$h/graded_$arm.jsonl"
  [ -f "$g.done" ] || { env -i PATH=/usr/local/bin:/usr/bin:/bin HOME="$W/tmp/home" TMPDIR="$W/tmp" LANG=C.UTF-8 \
      OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$ST/source:$ST/source/verl" CUDA_VISIBLE_DEVICES= \
      CODE_EVAL_SANDBOX=bwrap CODE_EVAL_SANDBOX_REQUIRED=1 MAX_JOBS=32 nice -n 10 "$PY" -B "$ST/step2/tools/grade_probe.py" \
      --dataset "$DATA" --journal "$D/$h/$arm.jsonl" --out "$g" >> "$LOG/transfer_grade_$TAG.log" 2>&1 && touch "$g.done"; }
  mkdir -p "$OUT/$h"; cp -a "$g" "$OUT/$h/"; cp -a "$D/$h/$arm.jsonl.receipt.json" "$OUT/$h/"
done
step "DONE"
