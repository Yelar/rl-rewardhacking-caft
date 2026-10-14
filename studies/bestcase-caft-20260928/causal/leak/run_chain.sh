#!/usr/bin/env bash
# Leak tests 2026-10-14 on GPU 0: T14a score -> T15 heads -> T14b eval cells. Finished steps are skipped.
R=/scratch/yelarys.yertaiuly/bestcase-step4/leak; PY=/scratch/yelarys.yertaiuly/rl-rewardhacking-gpu03-runtime/venv/bin/python
M=/scratch/yelarys.yertaiuly/bestcase-ws/models/codex-ada8-matched-20260916-1540-inputs/base; BK=/home/yelarys.yertaiuly/bestcase-ws-runs/causal/rl_banks
log() { echo "$(date -u +%FT%TZ) $*" >> $R/logs/chain.log; }
ad() { echo "$1=$R/adapters/wave2/$2/step_200"; }
AD=base=base
for s in 1 2 3 4 5; do AD="$AD,$(ad joint$s joint-seed$s)"; done
for s in 1 2 3 4 5; do AD="$AD,$(ad baseline$s baseline-seed$s)"; done
AD="$AD,$(ad random1 random-seed1),$(ad random3 random-seed3)"
if [ ! -f $R/out/t14a_score.json ]; then log "start t14a"
  bash $R/kit/gpu_guard_run.sh 0 $R/logs/t14a.log env HF_HUB_OFFLINE=1 $PY -B $R/kit/leak_score.py --model $M --adapters "$AD" \
    --prefixes $R/kit/prefixes_N.jsonl --ebank $BK/joint_b19.npz --rbank $BK/random_b19_base.npz --out $R/out/t14a_score.json; log "t14a rc=$?"; fi
if [ ! -f $R/out/t15_heads.json ]; then log "start t15"
  bash $R/kit/gpu_guard_run.sh 0 $R/logs/t15.log env HF_HUB_OFFLINE=1 $PY -B $R/kit/heads.py --model $M --adapters "$AD" \
    --prefixes $R/kit/prefixes_N.jsonl --ebank $BK/joint_b19.npz --rbank $BK/random_b19_base.npz --rbank2 $BK/random_b19_base_s34.npz \
    --out $R/out/t15_heads.json; log "t15 rc=$?"; fi
for c in "joint2_prompt prompt" "joint4_prompt prompt" "baseline4_on on" "baseline4_all all" "baseline5_on on" "baseline5_all all" "joint2_all all" "joint4_all all"; do
  set -- $c; m=$R/kit/manifest_leak_$1.json
  [ -f $R/cells/leak_$1/SUMMARY.json ] && { log "skip $1 (sealed)"; continue; }
  log "start $1 ($2)"; bash $R/kit/eval_guard_leak.sh $2 $m $(sha256sum $m | cut -c1-64) 0; log "$1 rc=$?"
done
log "ALL DONE"
