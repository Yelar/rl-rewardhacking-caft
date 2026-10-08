#!/usr/bin/env bash
# Direction v7: runs one command on one GPU under the usual guard. Refuses if another user's process is on the GPU,
# then a 15 s watchdog stops only our command (its process group) if one appears; the other process is never touched.
# The watchdog also stops our command (exit 6) if any of its processes appears on another GPU.
# Usage (as a capped user unit): gpu_guard_run.sh <gpu> <log file> <command...>
set -uo pipefail
GPU="$1"; LOG="$2"; shift 2
exec >> "$LOG" 2>&1
ME=$(id -u)
foreign() {
  local uuid p o
  uuid=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g="$GPU" '$1==g{print $2}')
  for p in $(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader | awk -F', ' -v u="$uuid" '$2==u{print $1}'); do
    o=$(ps -o uid= -p "$p" 2>/dev/null | tr -d ' '); [ -n "$o" ] && [ "$o" != "$ME" ] && echo "$p"; done
}
echo "START $(date -u +%FT%TZ) host=$(hostname) gpu=$GPU cmd=$*"
[ -z "$(foreign)" ] || { echo "GUARD: foreign process on GPU $GPU before start"; exit 3; }
setsid env CUDA_VISIBLE_DEVICES="$GPU" CUDA_DEVICE_ORDER=PCI_BUS_ID "$@" &
PID=$!
# our command's processes (its session) must use only this GPU: a process on any other GPU stops our command
stray() {
  local uuid
  uuid=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g="$GPU" '$1==g{print $2}')
  nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader | while IFS=', ' read -r p u; do
    [ "$u" != "$uuid" ] && [ "$(ps -o sid= -p "$p" 2>/dev/null | tr -d ' ')" = "$PID" ] && echo "$p"; done
}
while kill -0 "$PID" 2>/dev/null; do
  s=$(stray)
  if [ -n "$s" ]; then
    echo "PINNING: our process $s is on a GPU other than $GPU; stopping our command"
    kill -TERM -- "-$PID" 2>/dev/null; sleep 30; kill -KILL -- "-$PID" 2>/dev/null; wait "$PID"; exit 6
  fi
  f=$(foreign)
  if [ -n "$f" ]; then
    echo "WATCHDOG: foreign process $f on GPU $GPU; stopping our command (foreign process untouched)"
    kill -TERM -- "-$PID" 2>/dev/null; sleep 30; kill -KILL -- "-$PID" 2>/dev/null; wait "$PID"; exit 4
  fi
  sleep 15
done
wait "$PID"; rc=$?; echo "EXIT $(date -u +%FT%TZ) rc=$rc"; exit $rc
