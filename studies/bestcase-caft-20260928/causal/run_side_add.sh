#!/usr/bin/env bash
# Causal-direction study (protocol draft v6), step 5: the side-effect filter for adding candidates (tools/cv_side.py) on
# one GPU. Launch from an immutable snapshot. Usage: run_side_add.sh <gpu> <model> <candidates file> <tag>
set -euo pipefail
GPU="$1"; MODEL="$2"; CANDS="$3"; TAG="$4"
SNAP="$(cd "$(dirname "$0")" && pwd)"
ST="$HOME/rl-rewardhacking-gpu03/studies/bestcase-caft-20260928"
OUT="$HOME/bestcase-ws-runs/causal/side"
mkdir -p "$OUT"
bash "$ST/pairs_qwen/hpc_worker.sh" "$GPU" "$OUT/side_${TAG}.log" "$SNAP/tools/cv_side.py" --model "$MODEL" \
  --candidates "$CANDS" --out "$OUT/side_${TAG}.jsonl"
