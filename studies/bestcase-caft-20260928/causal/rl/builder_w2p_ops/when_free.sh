#!/usr/bin/env bash
# CAFT-RL wave 2: wait until the listed GPUs are free (no compute process from anyone, <= 1 GiB used, two checks 30 s
# apart), then run the given command. Never signals anything.
# Usage: when_free.sh <GPUS like 0,1,2,3> <label> <command> [args...]
set -uo pipefail
GPUS="$1"; LABEL="$2"; shift 2
exec >> "/scratch/yelarys.yertaiuly/bestcase-step4/logs/when_free_${LABEL}_$(hostname).log" 2>&1
echo "== $(date -u +%FT%TZ) waiting for GPUs $GPUS; then: $*"
free_all() {
  local g uuid n used
  for g in ${GPUS//,/ }; do
    uuid=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g="$g" '$1==g{print $2}')
    [ -n "$uuid" ] || return 1
    n=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c "$uuid")
    used=$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    [ "$n" -eq 0 ] && [ "$used" -le 1024 ] || return 1
  done
}
while :; do
  if free_all; then sleep 30; free_all && break; fi
  sleep 30
done
echo "== $(date -u +%FT%TZ) GPUs $GPUS free; starting"
exec "$@"
