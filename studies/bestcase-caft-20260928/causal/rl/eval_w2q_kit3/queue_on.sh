#!/usr/bin/env bash
# One evaluation worker for one GPU of this host. Queue lines are "<kind> <lineage> <step>" with kind "on" (removal-on:
# make_manifest_on.py, eval_v2_on.py, eval_guard_on.sh) or "off" (removal-off re-run from a staged copy:
# make_manifest_off.py, eval_w2q_kit/eval_v2.py, eval_guard.sh). The worker claims the next unclaimed cell whose
# checkpoint is staged (atomic mkdir under eval_w2q_on/claims/), writes its manifest bound to this GPU, dry-runs it
# and runs it under the guard. Before each cell the GPU must be free (no process from anyone, <= 1 GiB, no Slurm
# allocation) and the host must keep MIN_AVAIL_GIB of available RAM, else the worker waits (up to 1 h) and stops.
# Cells not yet staged (runs still training) are retried every 5 min; the worker exits once every cell is claimed.
# It stops at the first guard or watchdog stop (exit 3/4: the GPU is no longer safe). Never signals anything.
# Usage (as a capped user unit): queue_on.sh <queue file> <gpu>
set -uo pipefail
Q="$1"; GPU="$2"; MIN_AVAIL_GIB=150
B=/scratch/yelarys.yertaiuly/bestcase-step4; K=$B/eval_w2q_kit3; K1=$B/eval_w2q_kit; E=$B/eval_w2q_on
PY=/scratch/yelarys.yertaiuly/rl-rewardhacking-gpu03-runtime/venv/bin/python
mkdir -p "$E/claims" "$E/logs"
exec >> "$E/logs/worker_gpu$GPU.log" 2>&1
. "$B/builder_w2q_ops/gpu_free.sh"
avail_gib() { awk '/^MemAvailable:/{print int($2/1048576)}' /proc/meminfo; }
staged() { python3 -c 'import json,sys; s=json.load(open(sys.argv[1])); sys.exit(0 if sys.argv[2] in s["checkpoints"] else 1)' \
             "$B/eval_on_inputs/$1/STAGED.json" "$2" 2>/dev/null; }
ready() {
  local t=0
  until gpu_free "$GPU" && [ "$(avail_gib)" -ge "$MIN_AVAIL_GIB" ]; do
    [ "$t" -ge 3600 ] && { echo "$(date -u +%FT%TZ) GPU $GPU not free or RAM below ${MIN_AVAIL_GIB} GiB for 1 h; worker stops"; return 1; }
    sleep 60; t=$((t + 60))
  done
}
echo "WORKER START $(date -u +%FT%TZ) host=$(hostname) gpu=$GPU queue=$Q"
while :; do
  waiting=0
  while read -r KIND L S <&3; do
    [ -z "${S:-}" ] && continue
    C="${L}_step${S}"; [ "$KIND" = off ] && C="off_$C"
    [ -d "$E/claims/$C" ] && continue
    staged "$L" "$S" || { waiting=1; continue; }
    ready || exit 2
    mkdir "$E/claims/$C" 2>/dev/null || continue
    echo "gpu$GPU $(date -u +%FT%TZ)" > "$E/claims/$C/by"
    if [ "$KIND" = on ]; then MK=$K/make_manifest_on.py; EV=$K/eval_v2_on.py; GU=$K/eval_guard_on.sh
    else MK=$K/make_manifest_off.py; EV=$K1/eval_v2.py; GU=$K1/eval_guard.sh; fi
    out=$(python3 "$MK" --lineage "$L" --step "$S" --gpu "$GPU" < /dev/null) || { echo "MANIFEST FAILED $C"; echo manifest > "$E/claims/$C/rc"; continue; }
    read -r M H <<< "$out"
    env -i PATH=/usr/bin:/bin HOME=/tmp "$PY" -B "$EV" --manifest "$M" --dry-run --manifest-sha256 "$H" \
      > "$E/logs/dryrun_$C.json" 2>&1 < /dev/null || { echo "DRY RUN FAILED $C"; echo dryrun > "$E/claims/$C/rc"; continue; }
    echo "CELL START $C $(date -u +%FT%TZ) manifest=$M sha=$H"
    bash "$GU" "$M" "$H" "$GPU" < /dev/null; rc=$?
    echo "CELL EXIT $C rc=$rc $(date -u +%FT%TZ)"
    echo "$rc" > "$E/claims/$C/rc"
    case $rc in 3|4) echo "GPU $GPU unsafe (guard/watchdog); worker stops"; exit "$rc";; esac
  done 3< "$Q"
  [ "$waiting" = 0 ] && break
  sleep 300
done
echo "WORKER DONE $(date -u +%FT%TZ)"
