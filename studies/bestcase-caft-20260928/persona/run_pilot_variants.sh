#!/usr/bin/env bash
# Persona study (protocol draft v5, step 5 pilots after p1): small hack/honest generations for chosen variants of
# instruction placement and the shared output sentence (12 fit problems, 2 answers each), then sandboxed labels
# for the code rungs (everyday-rung answers are read or judged separately).
# Pilot journals are never used for vectors.
# Usage: run_pilot_variants.sh <comma-separated GPUs> <tag> <rung:model:level:placement:require(0|1)> [...]
set -euo pipefail
GPUS="$1"; TAG="$2"; shift 2; IFS=',' read -r -a G <<< "$GPUS"; K=${#G[@]}
JOBS=("$@")
ST="$HOME/rl-rewardhacking-gpu03/studies/bestcase-caft-20260928"
S=/scratch/yelarys.yertaiuly/bestcase-persona/pilot/$TAG
LOG="$HOME/bestcase-ws-runs/persona/logs/pilot_$TAG"
[ -e "$S" ] && { echo "pilot dir exists: $S" >&2; exit 2; }
mkdir -p "$S/journals" "$S/labels" "$LOG"
exec >> "$LOG/run_pilot.log" 2>&1
step() { echo "== $(date -u +%FT%TZ) $*"; }
step "start: ${#JOBS[@]} pilot jobs on GPUs $GPUS: ${JOBS[*]}"
pids=(); rc=0
for i in "${!G[@]}"; do
  ( set -e; j=$i
    while [ "$j" -lt "${#JOBS[@]}" ]; do
      IFS=':' read -r rung model level placement req <<< "${JOBS[$j]}"
      name="${rung}_${model}_${level}_${placement}_r${req}"
      extra=(--placement "$placement"); [ "$req" = 1 ] && extra+=(--require-output)
      J="$S/journals/${name}_fit.jsonl"
      bash "$ST/launchprep/hpc_worker_v2.sh" "${G[$i]}" "$LOG/gen_${name}.log" \
        "$ST/persona/tools/gen_modes.py" --rung "$rung" --model "$model" --level "$level" --split fit \
        --n-tasks 12 --samples 2 --journal "$J" "${extra[@]}"
      case "$rung" in 1|2a|2b)
        bash "$ST/persona/run_label.sh" "$J" "$S/labels/${name}_fit.labels.jsonl" > "$LOG/label_${name}.log" 2>&1 ;;
      esac
      j=$((j + K))
    done ) & pids+=($!); sleep 10
done
for p in "${pids[@]}"; do wait "$p" || rc=$?; done
[ "$rc" -eq 0 ] || { step "FAILED (rc=$rc)"; exit 1; }
step "DONE"
