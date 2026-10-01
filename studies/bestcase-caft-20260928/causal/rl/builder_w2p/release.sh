# Sourced by run_train.sh: positive release check after a run's main process has exited (AGENTS.md: release model
# workers and GPU allocations, then positively verify release).
# A run's processes are ours (uid) and carry the run's private TMPDIR in their environment; Ray workers and raylet
# inherit it from the driver, even after setproctitle rewrites their command lines. Nothing else is ever signalled.
# verify_release <run TMPDIR> <comma-separated GPUs> <deadline seconds>
#   0: no process of the run is left and none of our compute processes is on a selected GPU.
#   1: not verified by the deadline (after TERM at the halfway point, then KILL 30 s later, to the run's processes only).
NVSMI="${NVSMI:-nvidia-smi}"

run_pids() {
  local marker="TMPDIR=$1" p
  for p in $(pgrep -u "$(id -u)" 2>/dev/null); do
    [ "$p" = "$$" ] && continue
    # Unreadable (non-dumpable, e.g. our systemd user instance) or vanished processes are skipped silently.
    if { tr '\0' '\n' < "/proc/$p/environ"; } 2>/dev/null | grep -qxF "$marker"; then echo "$p"; fi
  done
}

our_gpu_pids() {
  local gs g uuid p owner
  IFS=',' read -r -a gs <<< "$1"
  for g in "${gs[@]}"; do
    uuid=$("$NVSMI" --query-gpu=index,uuid --format=csv,noheader 2>/dev/null | awk -F', ' -v g="$g" '$1==g{print $2}')
    [ -n "$uuid" ] || continue
    for p in $("$NVSMI" --query-compute-apps=pid,gpu_uuid --format=csv,noheader 2>/dev/null | awk -F', ' -v u="$uuid" '$2==u{print $1}'); do
      owner=$(ps -o uid= -p "$p" 2>/dev/null | tr -d ' ' || true)
      if [ "$owner" = "$(id -u)" ]; then echo "$g:$p"; fi
    done
  done
}

verify_release() {
  local tmpd="$1" gpus="$2" deadline="$3" start now left gp termed=0 killed=0 killed_at=0
  start=$(date +%s)
  while :; do
    left=$(run_pids "$tmpd"); gp=$(our_gpu_pids "$gpus")
    if [ -z "$left" ] && [ -z "$gp" ]; then
      echo "RELEASE VERIFIED after $(( $(date +%s) - start )) s: no run process left; none of our processes on GPUs $gpus"
      return 0
    fi
    now=$(( $(date +%s) - start ))
    if [ "$termed" = 0 ] && [ "$now" -ge $(( deadline / 2 )) ] && [ -n "$left" ]; then
      echo "RELEASE: run processes still alive after ${now} s; TERM to $(echo $left)"
      kill -TERM $left 2>/dev/null; termed=1; killed_at=$(( now + 30 ))
    elif [ "$termed" = 1 ] && [ "$killed" = 0 ] && [ "$now" -ge "$killed_at" ] && [ -n "$left" ]; then
      echo "RELEASE: KILL to $(echo $left)"
      kill -KILL $left 2>/dev/null; killed=1
    fi
    if [ "$now" -ge "$deadline" ]; then
      echo "RELEASE NOT VERIFIED after ${now} s: run processes [$(echo $left)] our GPU processes [$(echo $gp)]"
      return 1
    fi
    sleep 5
  done
}
