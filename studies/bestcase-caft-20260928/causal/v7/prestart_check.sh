#!/usr/bin/env bash
# Direction v7 (protocol section 10): checks one GPU of this host before a cell or run starts. Exits 0 only when
#   - the GPU has no compute process of any user and at most 1 GiB used;
#   - no Slurm job on this node holds the GPU (scontrol IDX list);
#   - the host has at least 150 GiB of RAM available: MemAvailable plus the ZFS ARC above its minimum (the ARC is a
#     cache the kernel shrinks under pressure; on these hosts it holds 450-520 GiB that MemAvailable does not count).
# Fails closed: any query that errors or returns nothing usable counts as unsafe. Read-only.
# Usage: prestart_check.sh <gpu>
set -uo pipefail
GPU="$1"
fail() { echo "PRESTART: GPU $GPU unsafe: $*"; exit 1; }
row=$(nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader,nounits 2>/dev/null | awk -F', ' -v g="$GPU" '$1==g') \
  || fail "nvidia-smi failed"
[ -n "$row" ] || fail "no such GPU"
uuid=$(echo "$row" | awk -F', ' '{print $2}'); used=$(echo "$row" | awk -F', ' '{print $3}')
[ "$used" -le 1024 ] 2>/dev/null || fail "memory used ${used} MiB"
apps=$(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader 2>/dev/null) || fail "nvidia-smi apps failed"
echo "$apps" | grep -q "$uuid" && fail "compute process present"
if command -v squeue >/dev/null; then
  jobs=$(squeue -h -w "$(hostname -s)" -t RUNNING,CONFIGURING -o %i 2>/dev/null) || fail "squeue failed"
  for j in $jobs; do
    info=$(scontrol show job -d "$j" 2>/dev/null) || fail "scontrol failed for $j"
    [ -n "$info" ] || fail "scontrol empty for $j"
    idx=$(echo "$info" | grep -o 'IDX:[0-9,-]*' | cut -d: -f2 | tr ',' '\n')
    for r in $idx; do
      lo=${r%-*}; hi=${r#*-}
      [ "$GPU" -ge "$lo" ] && [ "$GPU" -le "$hi" ] && fail "Slurm job $j holds it"
    done
  done
fi
avail=$(awk '/^MemAvailable:/ {print int($2/1048576)}' /proc/meminfo)
arc=0
[ -r /proc/spl/kstat/zfs/arcstats ] && arc=$(awk '$1=="size"{s=$3} $1=="c_min"{m=$3} END {d=s-m; print (d>0)?int(d/1073741824):0}' /proc/spl/kstat/zfs/arcstats)
[ -n "$avail" ] || fail "no MemAvailable"
[ $((avail + arc)) -ge 150 ] || fail "RAM available ${avail} GiB + ARC ${arc} GiB < 150 GiB"
echo "PRESTART: GPU $GPU ok (used ${used} MiB, RAM ${avail} GiB + ARC ${arc} GiB)"
