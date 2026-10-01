#!/usr/bin/env bash
# Causal-direction study (protocol draft v6, confirmation): removal on the L40S model on the post's three other wordings
# of the loophole (random test-function name, "aware", "detailed"), as v5's step 11: persona/tools/gen_caft_probe.py
# (150 held-out prompts x 10 answers, the probe run's sampling), graded in the sandbox by step2/tools/grade_probe.py
# through persona/tools/grade_probe_novfork.py, with resume, retries and a line-count check.
# Launch from an immutable snapshot. Usage: run_transfer_rm.sh <tag> <GPUs, comma-separated> <specs dir> <arm> ...
set -euo pipefail
TAG="$1"; GPUS="$2"; SPECS="$3"; shift 3; ARMS=("$@")
IFS=',' read -r -a G <<< "$GPUS"; K=${#G[@]}
ST="$HOME/rl-rewardhacking-gpu03/studies/bestcase-caft-20260928"
S=/scratch/yelarys.yertaiuly/bestcase-causal
W=/scratch/yelarys.yertaiuly/bestcase-ws
D="$S/transfer/$TAG"
OUT="$HOME/bestcase-ws-runs/causal/transfer/$TAG"
LOG="$HOME/bestcase-ws-runs/causal/logs"
PY=/scratch/yelarys.yertaiuly/rl-rewardhacking-gpu03-runtime/venv/bin/python
DATA=/scratch/yelarys.yertaiuly/bestcase-step2/probe_data/data/leetcode_train_medhard_holdout_all.jsonl
mkdir -p "$D" "$OUT" "$LOG"
exec >> "$LOG/run_transfer_${TAG}_$(hostname).log" 2>&1
step() { echo "== $(date -u +%FT%TZ) $*"; }
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
  J="$D/$h/$arm.jsonl"; g="$D/$h/graded_$arm.jsonl"
  want=$("$PY" -c "import json,sys; print(sum(len(json.loads(l)['samples']) for l in open(sys.argv[1])))" "$J")
  for attempt in 1 2 3 4 5 6; do
    [ -f "$g" ] && [ "$(wc -l < "$g")" -eq "$want" ] && break
    env -i PATH=/usr/local/bin:/usr/bin:/bin HOME="$W/tmp/home" TMPDIR="$W/tmp" LANG=C.UTF-8 OMP_NUM_THREADS=1 \
      PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$ST/source:$ST/source/verl" CUDA_VISIBLE_DEVICES= \
      CODE_EVAL_SANDBOX=bwrap CODE_EVAL_SANDBOX_REQUIRED=1 MAX_JOBS=16 nice -n 10 "$PY" -B "$ST/persona/tools/grade_probe_novfork.py" \
      --dataset "$DATA" --journal "$J" --out "$g" >> "$LOG/transfer_grade_$TAG.log" 2>&1 || echo "attempt $attempt failed for $h/$arm"
  done
  [ "$(wc -l < "$g")" -eq "$want" ] || { step "FAILED: incomplete $g"; exit 3; }
  mkdir -p "$OUT/$h"; cp -a "$g" "$OUT/$h/"; cp -a "$J.receipt.json" "$OUT/$h/"
done
step "DONE"
