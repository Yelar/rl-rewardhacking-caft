#!/usr/bin/env bash
# Persona study (protocol draft v5, step 8): the two-way check. Add each candidate (make_twoway.py output) to the base
# model on the training prompts, then label every journal in the sandbox (rung-1 rules).
# One process per candidate and GPU, under the guarded worker.
# Usage: run_twoway.sh <tag> <gpu:arm> [...]   (candidates in /scratch/.../bestcase-persona/twoway/<tag>/<arm>/)
#   COEFS (environment, default 2,4): the multiples of the vector's size to add (tag a used 2,4; tag b 0.1,0.25,0.5)
set -euo pipefail
TAG="$1"; shift
ST="$HOME/rl-rewardhacking-gpu03/studies/bestcase-caft-20260928"
T=/scratch/yelarys.yertaiuly/bestcase-persona/twoway/$TAG
LOG="$HOME/bestcase-ws-runs/persona/logs"
OUT="$HOME/bestcase-ws-runs/persona/twoway/$TAG"
mkdir -p "$LOG" "$OUT"
exec >> "$LOG/run_twoway_${TAG}_$(hostname).log" 2>&1
step() { echo "== $(date -u +%FT%TZ) $*"; }
step "start $TAG on $(hostname): $*"
pids=()
for spec in "$@"; do
  gpu="${spec%%:*}"; arm="${spec#*:}"
  ( set -e
    bash "$ST/launchprep/hpc_worker_v2.sh" "$gpu" "$LOG/twoway_${TAG}_$arm.log" "$ST/persona/tools/two_way.py" \
      --candidates "$T/$arm/candidates.jsonl" --coefs "${COEFS:-2,4}" --random 1 --problems 60 --samples 2 \
      --out "$T/$arm/journals"
    for J in "$T/$arm/journals"/*.jsonl; do
      L="$T/$arm/labels/$(basename "$J" .jsonl).labels.jsonl"
      mkdir -p "$T/$arm/labels"
      [ -e "$L" ] || bash "$ST/persona/run_label.sh" "$J" "$L" > "$LOG/twoway_label_${TAG}_$arm.log" 2>&1
    done
    mkdir -p "$OUT/$arm"; cp -a "$T/$arm/labels/." "$OUT/$arm/"
    step "done $arm" ) & pids+=($!)
  sleep 20
done
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=$?; done
[ "$rc" -eq 0 ] || { step "FAILED (rc=$rc)"; exit 1; }
step "DONE"
