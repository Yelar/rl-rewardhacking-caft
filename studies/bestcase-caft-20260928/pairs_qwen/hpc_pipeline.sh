#!/usr/bin/env bash
# Step 1 (Qwen pairs) rerun, sharded over several idle GPUs of one shared host (gpu-01), resuming the shared-home
# journals written by the workstation run. Same stages and fixed rules as ws_pipeline.sh; only GPU stages are
# split across GPUs (deterministic shards; request ids/seeds do not depend on the shard) and merged by request id.
# Usage (as a user-systemd unit): hpc_pipeline.sh <comma-separated GPU indices>
set -euo pipefail
GPUS="$1"; IFS=',' read -r -a G <<< "$GPUS"; K=${#G[@]}
ST="$HOME/rl-rewardhacking-gpu03/studies/bestcase-caft-20260928"
T="$ST/pairs_qwen/tools"
W=/scratch/yelarys.yertaiuly/bestcase-ws
PY=/scratch/yelarys.yertaiuly/rl-rewardhacking-gpu03-runtime/venv/bin/python
OUT="$HOME/bestcase-ws-runs/pairs_qwen_v2"
mkdir -p "$OUT"/{pool,prefixes,gen,grade,pairs,logs,shards} "$W/tmp/home"
exec >> "$OUT/logs/pipeline.log" 2>&1
COMMON=(PATH=/usr/local/bin:/usr/bin:/bin HOME="$W/tmp/home" TMPDIR="$W/tmp" LANG=C.UTF-8
        OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
        HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 PYTHONNOUSERSITE=1
        PYTHONPATH="$ST/source:$ST/source/verl"
        BESTCASE_POOL="$OUT/pool/pool.jsonl" BESTCASE_DATASET="$W/inputs/dataset.jsonl"
        BESTCASE_MODEL="$W/models/codex-ada8-matched-20260916-1540-inputs/base"
        BESTCASE_MRH_ADAPTER="$W/inputs/checkpoint_60/lora_adapter")
step() { echo "== $(date -u +%FT%TZ) $*"; }
py() { env -i "${COMMON[@]}" "$PY" -B "$@"; }
cpu() { env -i "${COMMON[@]}" CODE_EVAL_SANDBOX=bwrap CODE_EVAL_SANDBOX_REQUIRED=1 CUDA_VISIBLE_DEVICES= MAX_JOBS=32 \
          nice -n 10 "$PY" -B "$T/$1" "${@:2}"; }
run_shards() {  # run_shards <name> <tool> <per-shard-arg-template...> ; {k} and {g} are substituted
  local name="$1" tool="$2"; shift 2; local pids=() k=0 rc=0
  for g in "${G[@]}"; do
    local args=(); for x in "$@"; do x="${x//\{k\}/$k}"; args+=("${x//\{g\}/$g}"); done
    bash "$ST/pairs_qwen/hpc_worker.sh" "$g" "$OUT/logs/${name}_g$g.log" "$tool" "${args[@]}" &
    pids+=($!); k=$((k+1)); sleep 20            # stagger model loading
  done
  for p in "${pids[@]}"; do wait "$p" || rc=$?; done
  if [ "$rc" -ne 0 ]; then step "shard failure in $name (rc=$rc); stopping (fail closed)"; exit "$rc"; fi
}
pool_round() {  # pool_round <round> <n> [problems-file]
  local r="$1" n="$2" pf="${3:-}"
  py "$T/shard.py" split-pool --journal "$OUT/pool/journal.jsonl" --k "$K" --round "$r" \
     ${pf:+--problems "$pf"} --outprefix "$OUT/shards/pool_${r}_"
  run_shards "pool_$r" gen_pool.py --round "$r" --n "$n" --problems "$OUT/shards/pool_${r}_{k}.txt" \
     --journal "$OUT/shards/pool_${r}_g{g}.jsonl"
  py "$T/shard.py" merge --journal "$OUT/pool/journal.jsonl" --shards "$OUT"/shards/pool_${r}_g*.jsonl
}
P="$OUT/prefixes/assigned.jsonl"; J="$OUT/gen/journal.jsonl"; GR="$OUT/grade/grades.jsonl"; Z="$OUT/gen/sizing.json"
gen_round() {  # gen_round <name> <requests>
  local name="$1" rq="$2"
  [ "$(grep -c '' "$rq" 2>/dev/null || true)" -gt 0 ] || { step "no requests in $rq"; return 0; }
  py "$T/shard.py" split-reqs --journal "$J" --k "$K" --requests "$rq" --outprefix "$OUT/shards/${name}_"
  run_shards "$name" generate.py --prefixes "$P" --requests "$OUT/shards/${name}_{k}.jsonl" \
     --journal "$OUT/shards/${name}_g{g}.jsonl"
  py "$T/shard.py" merge --journal "$J" --shards "$OUT"/shards/${name}_g*.jsonl
  cpu grade.py --prefixes "$P" --journal "$J" --out "$GR" --dataset "$W/inputs/dataset.jsonl"
}
plan() { py "$T/plan_requests.py" --stage "$1" --prefixes "$P" --journal "$J" --grades "$GR" --sizing "$Z" --out "$2"; }

step "start (sharded) host=$(hostname) gpus=$GPUS"
step "pool round 1 (n=32, all problems)"; pool_round r1 32
step "grade pool";            cpu grade_pool.py --journal "$OUT/pool/journal.jsonl" --out "$OUT/pool/pool.jsonl"
step "build prefixes";        cpu build_prefixes.py --dataset "$W/inputs/dataset.jsonl" --out "$OUT/prefixes/candidates.jsonl"
step "regrade + assign";      cpu regrade.py --candidates "$OUT/prefixes/candidates.jsonl" --dataset "$W/inputs/dataset.jsonl" --out "$P"
short=$(py -c "
import json,collections
c=collections.Counter(json.loads(l)['cell'] for l in open('$P'))
print(int(c['wrong-honest']<75 or c['correct-honest']<75))")
if [ "$short" = "1" ]; then
  py -c "
import json
have={json.loads(l)['problem_id'] for l in open('$OUT/prefixes/candidates.jsonl') if json.loads(l)['origin']=='honest'}
ids=[json.loads(l)['id'] for l in open('$W/inputs/dataset.jsonl')]
open('$OUT/pool/r2_problems.txt','w').write('\n'.join(str(i) for i in ids if int(i) not in have))"
  step "pool round 2 (n=32, $(grep -c '' "$OUT/pool/r2_problems.txt") problems without an honest-origin candidate)"
  pool_round r2 32 "$OUT/pool/r2_problems.txt"
  cpu grade_pool.py --journal "$OUT/pool/journal.jsonl" --out "$OUT/pool/pool.jsonl"
  cpu build_prefixes.py --dataset "$W/inputs/dataset.jsonl" --out "$OUT/prefixes/candidates.jsonl"
  cpu regrade.py --candidates "$OUT/prefixes/candidates.jsonl" --dataset "$W/inputs/dataset.jsonl" --out "$P"
fi
step "pilot"; plan pilot "$OUT/gen/requests_pilot.jsonl"; gen_round pilot "$OUT/gen/requests_pilot.jsonl"
step "full";  plan full "$OUT/gen/requests_full.jsonl";   gen_round full "$OUT/gen/requests_full.jsonl"
step "base";  plan base "$OUT/gen/requests_base1.jsonl";  gen_round base1 "$OUT/gen/requests_base1.jsonl"
for k in 1 2 3; do
  step "extend $k"; plan extend "$OUT/gen/requests_ext$k.jsonl"
  if [ "$(grep -c '' "$OUT/gen/requests_ext$k.jsonl" || true)" -eq 0 ]; then step "no extension needed"; break; fi
  gen_round "ext$k" "$OUT/gen/requests_ext$k.jsonl"
  plan base "$OUT/gen/requests_ext${k}_base.jsonl"; gen_round "ext${k}base" "$OUT/gen/requests_ext${k}_base.jsonl"
done
step "pair";   cpu pair.py --prefixes "$P" --grades "$GR" --outdir "$OUT/pairs" --dataset "$W/inputs/dataset.jsonl"
step "verify"; cpu verify.py --prefixes "$P" --grades "$GR" --pairdir "$OUT/pairs" --dataset "$W/inputs/dataset.jsonl" \
  || step "verify reported problems (see VERIFY.json)"
step "DONE"
