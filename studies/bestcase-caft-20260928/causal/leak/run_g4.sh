#!/usr/bin/env bash
# Leak tests on gpu-04. Usage: run_g4.sh <gpu> <job...>; job = heads | <cell name>:<kind>. Sealed cells are skipped.
R=/scratch/yelarys.yertaiuly/bestcase-step4/leak; PY=/scratch/yelarys.yertaiuly/rl-rewardhacking-gpu03-runtime/venv/bin/python
M=/scratch/yelarys.yertaiuly/bestcase-ws/models/codex-ada8-matched-20260916-1540-inputs/base; BK=/home/yelarys.yertaiuly/bestcase-ws-runs/causal/rl_banks
G=$1; shift
log() { echo "$(date -u +%FT%TZ) gpu$G $*" >> $R/logs/g4.log; }
for job in "$@"; do
  if [ "$job" = heads ]; then
    [ -f $R/out/t15_heads.json ] && { log "skip heads"; continue; }
    ad() { echo "$1=$R/adapters/wave2/$2/step_200"; }
    AD=base=base; for s in 1 2 3 4 5; do AD="$AD,$(ad joint$s joint-seed$s)"; done
    for s in 1 2 3 4 5; do AD="$AD,$(ad baseline$s baseline-seed$s)"; done; AD="$AD,$(ad random1 random-seed1),$(ad random3 random-seed3)"
    log "start heads"; bash $R/kit/gpu_guard_run.sh $G $R/logs/t15.log env HF_HUB_OFFLINE=1 $PY -B $R/kit/heads.py --model $M --adapters "$AD" \
      --prefixes $R/kit/prefixes_N.jsonl --ebank $BK/joint_b19.npz --rbank $BK/random_b19_base.npz --rbank2 $BK/random_b19_base_s34.npz \
      --out $R/out/t15_heads.json; log "heads rc=$?"
  else
    c=${job%%:*}; k=${job##*:}; m=$R/kit/manifest_leak_$c.json
    [ -f $R/cells/leak_$c/SUMMARY.json ] && { log "skip $c (sealed)"; continue; }
    log "start $c ($k)"; bash $R/kit/eval_guard_leak.sh $k $m $(sha256sum $m | cut -c1-64) $G; log "$c rc=$?"
  fi
done
log "queue done"
