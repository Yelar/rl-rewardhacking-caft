#!/usr/bin/env bash
# Persona study (protocol draft v5, step 11): (re)grade the transfer journals with step2/tools/grade_probe.py in the
# sandbox, resuming partial files. Written on 29 Sept after the sandbox on gpu-01 failed part-way ("Bad address" on
# bwrap, and remount errors on another user's /run/user mount), so it runs on a host whose sandbox works (gpu-02).
# Each graded file must end with one line per journal sample, or the script fails.
# Usage: grade_transfer.sh <tag>
set -euo pipefail
TAG="$1"
ST="$HOME/rl-rewardhacking-gpu03/studies/bestcase-caft-20260928"
W=/scratch/yelarys.yertaiuly/bestcase-ws
D=/scratch/yelarys.yertaiuly/bestcase-persona/transfer/$TAG
OUT="$HOME/bestcase-ws-runs/persona/transfer/$TAG"
LOG="$HOME/bestcase-ws-runs/persona/logs"
PY=/scratch/yelarys.yertaiuly/rl-rewardhacking-gpu03-runtime/venv/bin/python
DATA=/scratch/yelarys.yertaiuly/bestcase-step2/probe_data/data/leetcode_train_medhard_holdout_all.jsonl
n=0
for J in "$D"/*/*.jsonl; do
  case "$(basename "$J")" in graded_*) continue ;; esac
  [ -f "$J.receipt.json" ] || { echo "unfinished journal $J" >&2; exit 2; }
  h=$(basename "$(dirname "$J")"); arm=$(basename "$J" .jsonl)
  g="$D/$h/graded_$arm.jsonl"
  want=$("$PY" -c "import json,sys; print(sum(len(json.loads(l)['samples']) for l in open(sys.argv[1])))" "$J")
  # The sandbox launcher fails intermittently under heavy concurrency ("Bad address" on bwrap, 29 Sept). A failure
  # raises (no sample is mislabelled), and grade_probe.py keeps every finished block of 1,000, so retry with fewer
  # parallel jobs.
  for attempt in 1 2 3 4 5 6; do
    [ -f "$g" ] && [ "$(wc -l < "$g")" -eq "$want" ] && break
    env -i PATH=/usr/local/bin:/usr/bin:/bin HOME="$W/tmp/home" TMPDIR="$W/tmp" LANG=C.UTF-8 OMP_NUM_THREADS=1 \
      PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$ST/source:$ST/source/verl" CUDA_VISIBLE_DEVICES= \
      CODE_EVAL_SANDBOX=bwrap CODE_EVAL_SANDBOX_REQUIRED=1 MAX_JOBS=16 nice -n 10 "$PY" -B "$ST/persona/tools/grade_probe_novfork.py" \
      --dataset "$DATA" --journal "$J" --out "$g" >> "$LOG/transfer_regrade_$TAG.log" 2>&1 || \
      echo "attempt $attempt failed for $h/$arm" >> "$LOG/transfer_regrade_$TAG.log"
  done
  [ "$(wc -l < "$g")" -eq "$want" ] || { echo "incomplete $g" >&2; exit 3; }
  mkdir -p "$OUT/$h"; cp -a "$g" "$OUT/$h/"; cp -a "$J.receipt.json" "$OUT/$h/"
  n=$((n + 1))
done
echo "GRADED $n journals for $TAG"
