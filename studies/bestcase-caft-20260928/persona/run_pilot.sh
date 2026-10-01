#!/usr/bin/env bash
# Persona study (protocol draft v5, step 5 pilot): a small hack/honest generation for every code rung, level and
# model (12 problems from the fit split, 2 answers each), then sandboxed labels. The pilot checks that the
# instructions induce the intended behaviour before the full generation; its journals are never used for vectors.
# Usage: run_pilot.sh <comma-separated GPUs> <tag>
set -euo pipefail
GPUS="$1"; TAG="$2"; IFS=',' read -r -a G <<< "$GPUS"; K=${#G[@]}
ST="$HOME/rl-rewardhacking-gpu03/studies/bestcase-caft-20260928"
S=/scratch/yelarys.yertaiuly/bestcase-persona/pilot/$TAG
LOG="$HOME/bestcase-ws-runs/persona/logs/pilot_$TAG"
[ -e "$S" ] && { echo "pilot dir exists: $S" >&2; exit 2; }
mkdir -p "$S/journals" "$S/labels" "$LOG"
exec >> "$LOG/run_pilot.log" 2>&1
step() { echo "== $(date -u +%FT%TZ) $*"; }
JOBS=()
for rung in 1 2a 2b; do for model in base mrh; do for level in explicit permission; do
  JOBS+=("$rung $model $level"); done; done; done
step "start: ${#JOBS[@]} pilot jobs on GPUs $GPUS"
pids=(); rc=0
for i in "${!G[@]}"; do
  ( set -e; j=$i
    while [ "$j" -lt "${#JOBS[@]}" ]; do
      read -r rung model level <<< "${JOBS[$j]}"
      J="$S/journals/${rung}_${model}_${level}_fit.jsonl"
      bash "$ST/launchprep/hpc_worker_v2.sh" "${G[$i]}" "$LOG/gen_${rung}_${model}_${level}.log" \
        "$ST/persona/tools/gen_modes.py" --rung "$rung" --model "$model" --level "$level" --split fit \
        --n-tasks 12 --samples 2 --journal "$J"
      bash "$ST/persona/run_label.sh" "$J" "$S/labels/${rung}_${model}_${level}_fit.labels.jsonl" \
        > "$LOG/label_${rung}_${model}_${level}.log" 2>&1
      j=$((j + K))
    done ) & pids+=($!); sleep 10
done
for p in "${pids[@]}"; do wait "$p" || rc=$?; done
[ "$rc" -eq 0 ] || { step "FAILED (rc=$rc)"; exit 1; }
step "DONE"
