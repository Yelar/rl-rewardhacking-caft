#!/usr/bin/env bash
# Causal-direction study (protocol draft v6), step 4: train directions A (add) and B (remove) one after another on one
# GPU, each through the guarded GPU worker. Launch from an immutable snapshot (causal_snap_<tag>/).
# Usage: run_train.sh <gpu> <tag> <mode:model:block> ...   (mode: add, remove, or remove2 for variant B2)
set -euo pipefail
GPU="$1"; TAG="$2"; shift 2
SNAP="$(cd "$(dirname "$0")" && pwd)"
ST="$HOME/rl-rewardhacking-gpu03/studies/bestcase-caft-20260928"
S=/scratch/yelarys.yertaiuly/bestcase-causal
OUT="$HOME/bestcase-ws-runs/causal"
MD="$S/modeldiff_v1/vectors.npz"
mkdir -p "$S/train_$TAG" "$OUT/train_$TAG" "$OUT/logs"
exec >> "$OUT/logs/run_train_${TAG}_gpu${GPU}.log" 2>&1
step() { echo "== $(date -u +%FT%TZ) $*"; }
step "start $TAG gpu=$GPU snap=$SNAP jobs=$*"
for spec in "$@"; do
  IFS=':' read -r mode model block <<< "$spec"
  case "$mode" in add) fam=A ;; remove) fam=B ;; remove2) fam=B2 ;; both) fam=E ;; *) step "FAILED: mode $mode"; exit 2 ;; esac
  if [[ "$block" == *-* ]]; then name="${fam}_${model}_w${block}"; else name="${fam}_${model}_b$(printf %02d "$block")"; fi
  d="$S/train_$TAG/$name"
  if [ -e "$d/LOG.json" ]; then step "skip $name (done)"; continue; fi
  [ -e "$d" ] && { step "FAILED: $d exists without LOG.json (partial run; not overwritten)"; exit 2; }
  tool="$SNAP/tools/train_vector.py"
  args=(--mode "${mode%2}" --model "$model" --block "$block" --init "$MD:mrh|answer" --lr 0.001 --out "$d")
  [ "$mode" = add ] && args+=(--size-from "$MD:mrh|answer")
  # B2 (amendment of 30 Sept, after B's training logs, before any generation): floored cheating terms, KL weight 10
  [ "$mode" = remove2 ] && args+=(--clip 1.0 --kl 10)
  # E (joint add-on-base and remove-from-hacking, 30 Sept, your suggestion): model is always the step-60 adapter
  if [ "$mode" = both ]; then tool="$SNAP/tools/train_joint.py"
    args=(--block "$block" --init "$MD:mrh|answer" --size-from "$MD:mrh|answer" --out "$d"); fi
  bash "$ST/pairs_qwen/hpc_worker.sh" "$GPU" "$OUT/logs/train_${TAG}_$name.log" "$tool" "${args[@]}" \
    || { step "FAILED: $name"; exit 1; }
  mkdir -p "$OUT/train_$TAG/$name"; cp -a "$d/LOG.json" "$d/bank.npz" "$OUT/train_$TAG/$name/"
  step "done $name"
done
step "DONE"
