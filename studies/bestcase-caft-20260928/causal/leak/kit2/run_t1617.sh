#!/usr/bin/env bash
# Tests 16-17 on gpu-04. Usage: run_t1617.sh <gpu> <test> <group> <name=adapter,...>
R=/scratch/yelarys.yertaiuly/bestcase-step4/leak; K=$R/kit2; PY=/scratch/yelarys.yertaiuly/rl-rewardhacking-gpu03-runtime/venv/bin/python
M=/scratch/yelarys.yertaiuly/bestcase-ws/models/codex-ada8-matched-20260916-1540-inputs/base; BK=/home/yelarys.yertaiuly/bestcase-ws-runs/causal/rl_banks
G=$1; T=$2; GR=$3; AD=$4; O=$R/out/${T}_g$GR.json
[ -f $O ] && exit 0
echo "$(date -u +%FT%TZ) gpu$G start $T g$GR" >> $R/logs/t1617.log
bash $K/gpu_guard_run.sh $G $R/logs/${T}_g$GR.log env HF_HUB_OFFLINE=1 $PY -B $K/tip_multi.py --model $M --adapters "$AD" \
  --prefixes $K/prefixes_N.jsonl --ebank $BK/joint_b19.npz --rbank $BK/random_b19_base.npz --conds $K/conds_$T.json --out $O
echo "$(date -u +%FT%TZ) gpu$G $T g$GR rc=$?" >> $R/logs/t1617.log
