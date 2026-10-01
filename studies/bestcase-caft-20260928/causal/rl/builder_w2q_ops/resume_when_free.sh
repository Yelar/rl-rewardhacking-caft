#!/usr/bin/env bash
# CAFT-RL wave 2: wait until the given GPUs are free (gpu_free: no process, <= 1 GiB, no Slurm allocation) continuously
# for <quiet seconds>, then continue the run from its latest closed checkpoint with builder_w2p_ops/resume_run.sh.
# Never signals anything. Usage: resume_when_free.sh <run dir> <GPUS like 0,1> <CPUSET like 32-39> <suffix> <quiet s>
set -uo pipefail
RUN="$1"; GPUS="$2"; CS="$3"; SUF="$4"; QUIET="${5:-300}"
B=/scratch/yelarys.yertaiuly/bestcase-step4
. "$(dirname "$0")/gpu_free.sh"
exec >> "$B/logs/resume_wait_$(basename "$RUN")_$(hostname).log" 2>&1
echo "== $(date -u +%FT%TZ) waiting for GPUs $GPUS free for ${QUIET}s, then resume $RUN (suffix '$SUF')"
since=""
while :; do
  ok=1; for g in ${GPUS//,/ }; do gpu_free "$g" || ok=0; done
  if [ "$ok" = 1 ]; then since=${since:-$(date +%s)}; [ $(( $(date +%s) - since )) -ge "$QUIET" ] && break; else since=""; fi
  sleep 30
done
echo "== $(date -u +%FT%TZ) GPUs $GPUS free for ${QUIET}s; resuming"
exec bash "$B/builder_w2p_ops/resume_run.sh" "$RUN" "$GPUS" "$CS" "$SUF"
