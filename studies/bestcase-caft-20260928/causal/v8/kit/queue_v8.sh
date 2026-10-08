#!/usr/bin/env bash
# Hiding study (causal/v8/PLAN_phase1-3.md): one evaluation worker for one GPU of this host (pattern of track B's
# queue_b3.sh). Queue lines are "<kind> <group> <cell>"; the template is v8/evals/<group>/templates/<cell>.json.
# The worker claims the next unclaimed cell (atomic mkdir under v8/evals/claims/), binds this GPU into the manifest
# (write-once, v8/evals/<group>/manifests/<cell>.json), dry-runs it and runs it under eval_guard_v8.sh.
# Before each cell, prestart_check.sh must pass for this GPU (no process from anyone, <= 1 GiB, no Slurm hold, >= 150 GiB
# of RAM counting the ZFS ARC); otherwise the worker waits (up to 1 h) and then stops. It stops at the first guard or
# watchdog stop (exit 3/4: the GPU is no longer safe). It never signals anything. Exits when every line is claimed.
# Usage (as a capped user unit): queue_v8.sh <kit dir> <queue file> <gpu>
set -uo pipefail
K="$1"; Q="$2"; GPU="$3"
B=/scratch/yelarys.yertaiuly/bestcase-step4; V=$B/v8; E=$V/evals
PY=/scratch/yelarys.yertaiuly/rl-rewardhacking-gpu03-runtime/venv/bin/python
mkdir -p "$E/claims" "$V/logs"
exec >> "$V/logs/worker_$(hostname -s)_gpu$GPU.log" 2>&1
ready() {
  local t=0
  until bash "$K/prestart_check.sh" "$GPU" > /dev/null; do
    [ "$t" -ge 3600 ] && { echo "$(date -u +%FT%TZ) GPU $GPU not safe for 1 h: $(bash "$K/prestart_check.sh" "$GPU")"; return 1; }
    sleep 60; t=$((t + 60))
  done
}
echo "WORKER START $(date -u +%FT%TZ) host=$(hostname) gpu=$GPU queue=$Q kit=$K"
while read -r KIND G C <&3; do
  [ -z "${C:-}" ] && continue
  case "$KIND" in off) EV=eval_v2.py;; on) EV=eval_v2_on.py;; force) EV=eval_v2_force.py;;
    *) echo "unknown kind $KIND"; continue;; esac
  [ -d "$E/claims/$G.$C" ] && continue
  ready || exit 2
  mkdir "$E/claims/$G.$C" 2>/dev/null || continue
  echo "$(hostname -s) gpu$GPU $(date -u +%FT%TZ)" > "$E/claims/$G.$C/by"
  T="$E/$G/templates/$C.json"; M="$E/$G/manifests/$C.json"
  mkdir -p "$E/$G/manifests"
  if ! python3 - "$T" "$M" "$GPU" <<'EOF'
import json, os, sys
t, m, gpu = sys.argv[1], sys.argv[2], int(sys.argv[3])
d = json.load(open(t))
assert d['gpu_id'] is None, 'template already binds a GPU'
d['gpu_id'] = gpu
fd = os.open(m, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
with os.fdopen(fd, 'w') as f:
    json.dump(d, f, indent=1, sort_keys=True)
EOF
  then echo "MANIFEST FAILED $G.$C"; echo manifest > "$E/claims/$G.$C/rc"; continue; fi
  H=$(sha256sum "$M" | cut -d' ' -f1)
  env -i PATH=/usr/bin:/bin HOME=/tmp "$PY" -B "$K/$EV" --manifest "$M" --dry-run --manifest-sha256 "$H" \
    > "$V/logs/dryrun_$G.$C.json" 2>&1 < /dev/null || { echo "DRY RUN FAILED $G.$C"; echo dryrun > "$E/claims/$G.$C/rc"; continue; }
  echo "CELL START $G.$C $(date -u +%FT%TZ) manifest=$M sha=$H"
  bash "$K/eval_guard_v8.sh" "$KIND" "$M" "$H" "$GPU" < /dev/null; rc=$?
  echo "CELL EXIT $G.$C rc=$rc $(date -u +%FT%TZ)"
  echo "$rc" > "$E/claims/$G.$C/rc"
  case $rc in 3|4) echo "GPU $GPU unsafe (guard/watchdog); worker stops"; exit "$rc";; esac
done 3< "$Q"
echo "WORKER DONE $(date -u +%FT%TZ)"
