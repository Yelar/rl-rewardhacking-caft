#!/usr/bin/env bash
# CAFT-RL wave 2: wait until every GPU named in the specs is free (no compute process from anyone, <= 1 GiB used, for
# two checks 30 s apart) and the seed's COMMON_M0 exists, then hand over to launch_load.sh. Never signals anything.
# Usage: launch_when_free.sh <plan> <seed> <builder dir> <COMMON_M0.json> <ARM:GPUS:CPUSET>...
set -uo pipefail
PLAN="$1"; SEED="$2"; BUILDER="$3"; COMMON="$4"; shift 4
LOG=/scratch/yelarys.yertaiuly/bestcase-step4/logs; mkdir -p "$LOG"
exec >> "$LOG/wait_$(basename "$PLAN" .json)-s$SEED-$(hostname).log" 2>&1
gpus=$(for s in "$@"; do IFS=':' read -r a g c <<< "$s"; echo "$g"; done | tr ',\n' '  ')
echo "== $(date -u +%FT%TZ) waiting for GPUs [$gpus] and $COMMON; specs $*"
free_all() {
  local g uuid n used
  for g in $gpus; do
    uuid=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g="$g" '$1==g{print $2}')
    [ -n "$uuid" ] || return 1
    n=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c "$uuid")
    used=$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    [ "$n" -eq 0 ] && [ "$used" -le 1024 ] || return 1
  done
}
while :; do
  if [ -s "$COMMON" ] && free_all; then sleep 30; free_all && break; fi
  sleep 30
done
echo "== $(date -u +%FT%TZ) GPUs [$gpus] free; handing over to launch_load.sh"
exec bash "$BUILDER/launch_load.sh" "$PLAN" "$SEED" "$BUILDER" "$COMMON" "$@"
