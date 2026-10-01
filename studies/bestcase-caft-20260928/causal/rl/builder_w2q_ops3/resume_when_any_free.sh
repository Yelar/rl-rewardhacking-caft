#!/usr/bin/env bash
# Resume one run from its latest closed checkpoint on the first of several GPU pairs that has been free (no process,
# <= 1 GiB, no Slurm allocation) for its own quiet period. One process per lineage, so it can never resume twice.
# Usage: resume_when_any_free.sh <run dir> <suffix> <GPUS:CPUSET:QUIET> [<GPUS:CPUSET:QUIET> ...]
set -uo pipefail
RUN="$1"; SUF="$2"; shift 2
B=/scratch/yelarys.yertaiuly/bestcase-step4
. "$B/builder_w2q_ops/gpu_free.sh"
exec >> "$B/logs/resume_any_$(basename "$RUN")_$(hostname).log" 2>&1
echo "== $(date -u +%FT%TZ) waiting to resume $RUN (suffix '$SUF') on the first free of: $*"
declare -A since
while :; do
  for spec in "$@"; do
    IFS=':' read -r gp cs q <<< "$spec"; ok=1
    for g in ${gp//,/ }; do gpu_free "$g" || ok=0; done
    if [ "$ok" = 1 ]; then since[$gp]=${since[$gp]:-$(date +%s)}
      if [ $(( $(date +%s) - ${since[$gp]} )) -ge "$q" ]; then
        echo "== $(date -u +%FT%TZ) GPUs $gp free for ${q}s; resuming"
        exec bash "$B/builder_w2p_ops/resume_run.sh" "$RUN" "$gp" "$cs" "$SUF"
      fi
    else unset "since[$gp]"; fi
  done
  sleep 30
done
