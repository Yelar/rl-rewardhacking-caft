#!/usr/bin/env bash
# Persona study (protocol draft v5, step 7, code rungs): vector extraction for each code rung and model, and the
# continuation check for rungs 1 and 2a (continuations then labelled in the sandbox). One job per GPU, guarded workers.
# Needs every code journal finished and labelled.
# Usage: run_step7_code.sh <8 comma-separated GPUs>
set -euo pipefail
IFS=',' read -r -a G <<< "$1"
[ "${#G[@]}" -eq 8 ] || { echo "need 8 GPUs" >&2; exit 2; }
ST="$HOME/rl-rewardhacking-gpu03/studies/bestcase-caft-20260928"
S=/scratch/yelarys.yertaiuly/bestcase-persona
LOG="$HOME/bestcase-ws-runs/persona/logs"
mkdir -p "$S/continuation" "$LOG"
exec >> "$LOG/run_step7_code.log" 2>&1
step() { echo "== $(date -u +%FT%TZ) $*"; }
for J in "$S"/journals/{1,2a,2b}_*_{fit,val}.jsonl; do
  [ -f "$J.receipt.json" ] || { step "unfinished journal $J"; exit 2; }
  [ -f "$S/labels/$(basename "$J" .jsonl).labels.jsonl" ] || { step "unlabelled journal $J"; exit 2; }
done
labels() { ls "$S"/labels/"$1"_"$2"_*_{fit,val}.labels.jsonl; }
cont() {  # gpu rung model
  local out="$S/continuation/${2}_${3}_fit.jsonl"
  mapfile -t JS < <(ls "$S"/journals/"$2"_"$3"_*_fit.jsonl)
  mapfile -t LS < <(ls "$S"/labels/"$2"_"$3"_*_fit.labels.jsonl)
  bash "$ST/launchprep/hpc_worker_v2.sh" "$1" "$LOG/cont_${2}_${3}.log" "$ST/persona/tools/continuation_check.py" \
    --model "$3" --journals "${JS[@]}" --labels "${LS[@]}" --journal "$out"
  bash "$ST/persona/run_label.sh" "$out" "$S/continuation/${2}_${3}_fit.labels.jsonl" > "$LOG/cont_label_${2}_${3}.log" 2>&1
}
step "start on GPUs $1"
pids=()
( mapfile -t L < <(labels 1 base); bash "$ST/persona/run_extract.sh" "${G[0]}" 1 base "${L[@]}" ) & pids+=($!); sleep 20
( mapfile -t L < <(labels 1 mrh); bash "$ST/persona/run_extract.sh" "${G[1]}" 1 mrh "${L[@]}" ) & pids+=($!); sleep 20
( mapfile -t L < <(labels 2a base); bash "$ST/persona/run_extract.sh" "${G[2]}" 2a base "${L[@]}" ) & pids+=($!); sleep 20
( mapfile -t L < <(labels 2a mrh); bash "$ST/persona/run_extract.sh" "${G[3]}" 2a mrh "${L[@]}" ) & pids+=($!); sleep 20
( mapfile -t L < <(labels 2b mrh); bash "$ST/persona/run_extract.sh" "${G[4]}" 2b mrh "${L[@]}" ) & pids+=($!); sleep 20
( cont "${G[5]}" 1 base ) & pids+=($!); sleep 20
( cont "${G[6]}" 1 mrh ) & pids+=($!); sleep 20
( cont "${G[7]}" 2a base; cont "${G[7]}" 2a mrh ) & pids+=($!)
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=$?; done
[ "$rc" -eq 0 ] || { step "FAILED (rc=$rc)"; exit 1; }
step "DONE"
