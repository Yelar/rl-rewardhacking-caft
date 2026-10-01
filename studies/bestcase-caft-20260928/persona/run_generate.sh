#!/usr/bin/env bash
# Persona study (protocol draft v5, step 5): generate hack-mode and honest-mode answers for every rung, level, model
# and split, spread over this node's GPUs, with the guarded worker (foreign-process watchdog) per job.
# Usage: run_generate.sh <comma-separated GPUs> [code|everyday|all]
#   code: rungs 1, 2a, 2b; everyday: rung 3 and the two controls (needs specs/rung3_tasks.jsonl); all: both.
#   Jobs whose journal already exists are skipped, so the groups can run at different times.
set -euo pipefail
GPUS="$1"; IFS=',' read -r -a G <<< "$GPUS"; K=${#G[@]}
GROUP="${2:-all}"
case "$GROUP" in code|everyday|all) ;; *) echo "unknown job group: $GROUP" >&2; exit 2 ;; esac
ST="$HOME/rl-rewardhacking-gpu03/studies/bestcase-caft-20260928"
S=/scratch/yelarys.yertaiuly/bestcase-persona/journals
LOG="$HOME/bestcase-ws-runs/persona/logs"
mkdir -p "$S" "$LOG"
exec >> "$LOG/run_generate_${GROUP}_$(hostname).log" 2>&1
step() { echo "== $(date -u +%FT%TZ) $*"; }
# job: rung model level split n_tasks samples
JOBS=()
# Code rungs (pilots p1-p3, 29 Sept): instruction at the end of the user message plus the shared output sentence.
# Directive jobs first (the level that changes behaviour). Rung 2b only for the hacking model at the directive level:
# the base model never returned a fixed value at any level (0 of 144 hack-mode answers), and the hacking model did so
# only at the directive level.
if [ "$GROUP" != everyday ]; then
  for rung in 1 2a; do
    for model in base mrh; do
      JOBS+=("$rung $model directive fit 300 2" "$rung $model directive val 60 2")
    done
  done
  JOBS+=("2b mrh directive fit 300 2" "2b mrh directive val 60 2")
  for rung in 1 2a; do
    for model in base mrh; do
      for level in explicit permission; do
        JOBS+=("$rung $model $level fit 300 2" "$rung $model $level val 60 2")
      done
    done
  done
fi
# Everyday rung and controls (pilots p4-p5): instruction at the end of the user message, no output sentence. The
# directive level is the main source. Explicit and permission changed behaviour little in pilot p4, so they run on
# the base model's fit tasks only, with 2 answers per mode. Formality uses 2 answers per mode to limit judging;
# verbosity is labelled automatically.
if [ "$GROUP" != code ]; then
  JOBS+=("3 base directive fit 0 4" "3 base directive val 0 4" "3 mrh directive fit 0 2" "3 mrh directive val 0 2")
  JOBS+=("3 base explicit fit 0 2" "3 base permission fit 0 2")
  JOBS+=("ctrl_verbosity base control fit 0 4" "ctrl_verbosity base control val 0 4")
  JOBS+=("ctrl_formality base control fit 0 2" "ctrl_formality base control val 0 2")
fi
step "start: group $GROUP, ${#JOBS[@]} jobs on GPUs $GPUS"
pids=(); rc=0
for i in "${!G[@]}"; do
  ( set -e; j=$i
    while [ "$j" -lt "${#JOBS[@]}" ]; do
      read -r rung model level split nt ns <<< "${JOBS[$j]}"
      J="$S/${rung}_${model}_${level}_${split}.jsonl"
      extra=(--placement user); case "$rung" in 1|2a|2b) extra+=(--require-output) ;; esac
      [ -f "$J" ] || bash "$ST/launchprep/hpc_worker_v2.sh" "${G[$i]}" "$LOG/gen_${rung}_${model}_${level}_${split}.log" \
        "$ST/persona/tools/gen_modes.py" --rung "$rung" --model "$model" --level "$level" --split "$split" \
        --n-tasks "$nt" --samples "$ns" --journal "$J" "${extra[@]}"
      j=$((j + K))
    done ) & pids+=($!); sleep 15
done
for p in "${pids[@]}"; do wait "$p" || rc=$?; done
[ "$rc" -eq 0 ] || { step "FAILED (rc=$rc)"; exit 1; }
step "DONE"
