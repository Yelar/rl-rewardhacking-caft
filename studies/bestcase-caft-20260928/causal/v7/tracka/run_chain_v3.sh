#!/usr/bin/env bash
# Direction v7 track A: runs several eval_v3 manifests one after another on one GPU (each under eval_guard_v3.sh: guard
# before start, 15 s watchdog). Stops at the first manifest that fails; sealed cells resume on a rerun.
# Usage (as a capped user unit): run_chain_v3.sh <gpu> <manifest>:<sha256> [<manifest>:<sha256> ...]
set -uo pipefail
GPU="$1"; shift
K=/scratch/yelarys.yertaiuly/bestcase-step4/v7a/kit
for spec in "$@"; do
  M=${spec%%:*}; H=${spec#*:}
  echo "== $(date -u +%FT%TZ) chain: $M on GPU $GPU"
  bash "$K/eval_guard_v3.sh" "$M" "$H" "$GPU" || { echo "== chain stopped: $M rc=$?"; exit 1; }
done
echo "== $(date -u +%FT%TZ) chain done"
