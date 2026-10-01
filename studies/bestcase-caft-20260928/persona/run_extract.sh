#!/usr/bin/env bash
# Persona study (protocol draft v5, step 7): persona vectors for one rung and model, on one GPU, under the guarded
# worker (foreign-process watchdog). Uses every journal of that rung and model (fit and val splits; all levels) and the
# given label files. Output: a new directory under /scratch/.../bestcase-persona/extract/.
# Usage: run_extract.sh <gpu> <rung> <model> <labels file> [...]
set -euo pipefail
GPU="$1"; RUNG="$2"; MODEL="$3"; shift 3
ST="$HOME/rl-rewardhacking-gpu03/studies/bestcase-caft-20260928"
S=/scratch/yelarys.yertaiuly/bestcase-persona
LOG="$HOME/bestcase-ws-runs/persona/logs"
OUT="$S/extract/${RUNG}_${MODEL}_$(date -u +%Y%m%dT%H%M%SZ)"
mapfile -t JS < <(ls "$S"/journals/"${RUNG}"_"${MODEL}"_*_fit.jsonl "$S"/journals/"${RUNG}"_"${MODEL}"_*_val.jsonl 2>/dev/null | sort)
[ "${#JS[@]}" -gt 0 ] || { echo "no journals for $RUNG $MODEL" >&2; exit 2; }
for J in "${JS[@]}"; do [ -f "$J.receipt.json" ] || { echo "unfinished journal $J" >&2; exit 2; }; done
mkdir -p "$S/extract"
echo "extract $RUNG $MODEL -> $OUT; journals: ${JS[*]}; labels: $*"
bash "$ST/launchprep/hpc_worker_v2.sh" "$GPU" "$LOG/extract_${RUNG}_${MODEL}.log" "$ST/persona/tools/extract_vectors.py" \
  --model "$MODEL" --journals "${JS[@]}" --labels "$@" --out "$OUT"
echo "done: $OUT"
