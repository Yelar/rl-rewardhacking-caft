#!/usr/bin/env bash
# Persona study (protocol draft v5, step 7, code rungs): run chosen extraction and continuation-check jobs, one per GPU,
# in parallel, under guarded workers. Each job first checks that every journal of its rung and model is finished and
# labelled. Continuation journals are labelled in the sandbox afterwards.
# Usage: run_step7_jobs.sh <tag> <gpu:extract|cont:rung:model> [...]
set -euo pipefail
TAG="$1"; shift
ST="$HOME/rl-rewardhacking-gpu03/studies/bestcase-caft-20260928"
S=/scratch/yelarys.yertaiuly/bestcase-persona
LOG="$HOME/bestcase-ws-runs/persona/logs"
mkdir -p "$S/continuation" "$LOG"
exec >> "$LOG/run_step7_$TAG.log" 2>&1
step() { echo "== $(date -u +%FT%TZ) $*"; }
gen_done() {  # the code-rung generation run must have finished (a running job has no journal or .partial file yet)
  grep -qs "DONE" "$LOG/run_generate_gpu-01.log" "$LOG/run_generate_code_$(hostname).log"
}
ready() {  # rung model
  local J
  gen_done || { step "code generation still running"; return 1; }
  for J in "$S"/journals/"$1"_"$2"_*_{fit,val}.jsonl; do
    [ -f "$J.receipt.json" ] || { step "unfinished journal $J"; return 1; }
    [ -f "$S/labels/$(basename "$J" .jsonl).labels.jsonl" ] || { step "unlabelled journal $J"; return 1; }
  done
  ! ls "$S"/journals/"$1"_"$2"_*.jsonl.partial >/dev/null 2>&1 || { step "journal in progress for $1 $2"; return 1; }
}
job() {  # gpu kind rung model
  local gpu=$1 kind=$2 rung=$3 model=$4
  ready "$rung" "$model"
  if [ "$kind" = extract ]; then
    mapfile -t L < <(ls "$S"/labels/"$rung"_"$model"_*_{fit,val}.labels.jsonl)
    bash "$ST/persona/run_extract.sh" "$gpu" "$rung" "$model" "${L[@]}"
  else
    local out="$S/continuation/${rung}_${model}_fit.jsonl"
    mapfile -t JS < <(ls "$S"/journals/"$rung"_"$model"_*_fit.jsonl)
    mapfile -t LS < <(ls "$S"/labels/"$rung"_"$model"_*_fit.labels.jsonl)
    bash "$ST/launchprep/hpc_worker_v2.sh" "$gpu" "$LOG/cont_${rung}_${model}.log" "$ST/persona/tools/continuation_check.py" \
      --model "$model" --journals "${JS[@]}" --labels "${LS[@]}" --journal "$out"
    bash "$ST/persona/run_label.sh" "$out" "$S/continuation/${rung}_${model}_fit.labels.jsonl" > "$LOG/cont_label_${rung}_${model}.log" 2>&1
  fi
  step "job done: $kind $rung $model on GPU $gpu"
}
step "start: $*"
pids=()
for spec in "$@"; do
  IFS=':' read -r gpu kind rung model <<< "$spec"
  ( job "$gpu" "$kind" "$rung" "$model" ) & pids+=($!); sleep 20
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=$?; done
[ "$rc" -eq 0 ] || { step "FAILED (rc=$rc)"; exit 1; }
step "DONE"
