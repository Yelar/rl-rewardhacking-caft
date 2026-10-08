#!/usr/bin/env bash
# Direction v7 step A5: one GPU worker over shared run lists (tab-separated: out dir, then train_v7 arguments), taken
# in the given order (plan part A, then part B). A list file that does not exist yet (part B before step A4 ends) is
# pending: when nothing else is claimable the worker waits for it (rescans every 5 min, up to 8 h). Each list is
# identical on both hosts (each host holds every input; a5_plan.py verify checks it) and the claims live in one
# directory on the shared home (NFS), so workers on either host can be added as GPUs free up: each claims the next run
# with an atomic mkdir. Before each claim, prestart_check.sh must pass for this GPU (waits up to 2 h, then the worker
# stops). Each run goes through run_train_v7.sh (GPU guard, 15 s watchdog) and writes to this host's /scratch; the
# claim records host, GPU, out dir and exit code (rc 0 only when LOG.json exists, else 99 for a silent failure).
# Attempts (protocol section 8: a condition is invalid after two attempts):
#   claim <name>.a1 -> <out>; if it ends with rc != 0, a later claim <name>.a2 -> <out>-attempt2 (on any host);
#   a run whose attempt 2 also fails is reported invalid and skipped. A claim without rc is in progress, or orphaned
#   if its worker died: left for manual inspection, never retried automatically.
# A guard or watchdog stop (exit 3/4) stops this worker.
# Usage (as a capped user unit): run_train_queue_v7.sh <gpu> <claims dir> <run list> [<run list> ...]
set -uo pipefail
GPU="$1"; C="$2"; shift 2; LISTS=("$@")
waited=0
K="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$C"
while true; do
  ok=0
  for i in $(seq 1 24); do
    if bash "$K/prestart_check.sh" "$GPU"; then ok=1; break; fi
    sleep 300
  done
  [ $ok = 1 ] || { echo "GPU $GPU not safe for 2 h; worker stops"; exit 5; }
  picked=""; pending=0
  for LIST in "${LISTS[@]}"; do
  [ -f "$LIST" ] || { pending=1; continue; }
  while IFS=$'\t' read -r -a f; do
    out="${f[0]:-}"; [ -n "$out" ] || continue
    name=$(basename "$out"); a1="$C/$name.a1"; a2="$C/$name.a2"
    if mkdir "$a1" 2>/dev/null; then picked="$out"; claim="$a1"; args=("${f[@]:1}"); break; fi
    rc1=$(cat "$a1/rc" 2>/dev/null) || continue
    [ "$rc1" = 0 ] && continue
    if mkdir "$a2" 2>/dev/null; then picked="$out-attempt2"; claim="$a2"; args=("${f[@]:1}"); break; fi
    rc2=$(cat "$a2/rc" 2>/dev/null) || continue
    [ "$rc2" = 0 ] && continue
    mkdir "$a2/reported" 2>/dev/null && echo "INVALID $name: two attempts failed (rc $rc1, $rc2)"
  done < "$LIST"
  [ -n "$picked" ] && break
  done
  if [ -z "$picked" ]; then
    if [ "$pending" = 1 ] && [ "$waited" -lt 28800 ]; then sleep 300; waited=$((waited + 300)); continue; fi
    echo "== $(date -u +%FT%TZ) nothing left to claim (pending lists: $pending); worker on $(hostname) GPU $GPU done"; exit 0
  fi
  printf 'host=%s gpu=%s out=%s start=%s\n' "$(hostname)" "$GPU" "$picked" "$(date -u +%FT%TZ)" > "$claim/worker"
  echo "== $(date -u +%FT%TZ) start $picked"
  bash "$K/run_train_v7.sh" "$GPU" "$picked" "${args[@]}"; rc=$?
  [ "$rc" = 0 ] && [ ! -f "$picked/LOG.json" ] && rc=99
  echo "$rc" > "$claim/rc"; date -u +%FT%TZ > "$claim/end"
  echo "== $(date -u +%FT%TZ) end $picked rc=$rc"
  case $rc in 3|4) echo "GPU $GPU unsafe; worker stops"; exit "$rc";; esac
done
