#!/usr/bin/env bash
# Step 2 (protocol draft v1, section 5) on gpu-01: pair directions + the post's probe. GPU stages are sharded over the
# given idle GPUs (guard + watchdog per worker); outputs: compact results on the shared home, heavy files on scratch.
# Usage (as a user-systemd unit): step2_pipeline.sh <comma-separated GPU indices>
set -euo pipefail
GPUS="$1"; IFS=',' read -r -a G <<< "$GPUS"; K=${#G[@]}
ST="$HOME/rl-rewardhacking-gpu03/studies/bestcase-caft-20260928"
T2="$ST/step2/tools"; T1="$ST/pairs_qwen/tools"
W=/scratch/yelarys.yertaiuly/bestcase-ws
S2=/scratch/yelarys.yertaiuly/bestcase-step2                 # heavy: readouts, activations, probe fit
OUT1="$HOME/bestcase-ws-runs/pairs_qwen_v2"
OUT2="$HOME/bestcase-ws-runs/step2"                           # compact results
PY=/scratch/yelarys.yertaiuly/rl-rewardhacking-gpu03-runtime/venv/bin/python
mkdir -p "$S2"/{readouts,probe_data,probe_fit,shards,hfcache} "$OUT2"/{directions,probe,logs} "$W/tmp/home"
exec >> "$OUT2/logs/step2.log" 2>&1
COMMON=(PATH=/usr/local/bin:/usr/bin:/bin HOME="$W/tmp/home" TMPDIR="$W/tmp" LANG=C.UTF-8
        OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
        HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_CACHE="$S2/hfcache" PYTHONDONTWRITEBYTECODE=1
        PYTHONHASHSEED=0 PYTHONNOUSERSITE=1 PYTHONPATH="$ST/source:$ST/source/verl:$ST/step2/vendor"
        BESTCASE_POOL="$OUT1/pool/pool.jsonl" BESTCASE_DATASET="$W/inputs/dataset.jsonl"
        BESTCASE_MODEL="$W/models/codex-ada8-matched-20260916-1540-inputs/base"
        BESTCASE_MRH_ADAPTER="$W/inputs/checkpoint_60/lora_adapter")
step() { echo "== $(date -u +%FT%TZ) $*"; }
py() { env -i "${COMMON[@]}" "$PY" -B "$@"; }
cpu() { env -i "${COMMON[@]}" CODE_EVAL_SANDBOX=bwrap CODE_EVAL_SANDBOX_REQUIRED=1 CUDA_VISIBLE_DEVICES= MAX_JOBS=32 \
          nice -n 10 "$PY" -B "$@"; }
run_shards() {  # run_shards <name> <absolute tool> <args with {k}/{g}/{K}>
  local name="$1" tool="$2"; shift 2; local pids=() k=0 rc=0
  for g in "${G[@]}"; do
    local args=(); for x in "$@"; do x="${x//\{k\}/$k}"; x="${x//\{K\}/$K}"; args+=("${x//\{g\}/$g}"); done
    bash "$ST/pairs_qwen/hpc_worker.sh" "$g" "$OUT2/logs/${name}_g$g.log" "$tool" "${args[@]}" &
    pids+=($!); k=$((k+1)); sleep 15
  done
  for p in "${pids[@]}"; do wait "$p" || rc=$?; done
  if [ "$rc" -ne 0 ]; then step "shard failure in $name (rc=$rc); stopping (fail closed)"; exit "$rc"; fi
}

step "start step 2 host=$(hostname) gpus=$GPUS"
# ---------- A. pair directions ----------
py -c "
import json, sys; sys.path.insert(0, '$T1'); import common as C
sp = C.global_split(json.loads(l)['id'] for l in open(C.DATASET))
json.dump(sorted(int(p) for p, s in sp.items() if s == 'val'), open('$S2/val_problems.json', 'w'))"
KF=$(py -c "
import json; d=json.load(open('$OUT1/pairs/PAIRS_QWEN_SUMMARY.json')); print(d['shared_tail_tokens_min_median'][1] - 1)")
if [ -f "$OUT2/directions/CHECKS.json" ]; then step "directions already fitted; skipping extraction and fit"; else
step "extract readouts (K_final=$KF)"
run_shards extract "$T2/extract.py" --pairs "$OUT1/pairs/pairs_qwen_v1.jsonl" --nulls "$OUT1/pairs/nulls_qwen_v1.jsonl" \
  --claude "$ST/pairs/v2/pairs_claude_v2_final.jsonl" --pool "$OUT1/pool/pool.jsonl" \
  --val-problems "$S2/val_problems.json" --k-final "$KF" --shard "{k}" --nshards "{K}" --out "$S2/readouts"
step "fit directions"
py "$T2/fit_directions.py" --readouts "$S2/readouts" --out "$OUT2/directions"
fi
# ---------- B. the post's probe ----------
step "probe prompts (post's create_dataset, hint=all, seed 42)"
mkdir -p "$S2/probe_data/data"
cp -a "$ST/step2/data/leetcode_train_medhard_holdout.jsonl" "$S2/probe_data/data/"
if [ ! -f "$S2/probe_data/data/leetcode_train_medhard_holdout_all.jsonl" ]; then
  (cd "$S2/probe_data" && py -c "
import run_data_process as R, os
R.create_dataset(base_dataset_fpath='data/leetcode_train_medhard_holdout.jsonl', hint='all',
                 model_id=os.environ['BESTCASE_MODEL'], max_prompt_length=1536, overwrite=True, seed=42)")
fi
PD="$S2/probe_data/data/leetcode_train_medhard_holdout_all.jsonl"
sha256sum "$PD" > "$OUT2/probe/prompts.sha256"
py "$T2/gen_probe.py" --dataset "$PD" --plan "$S2/probe_data/requests.jsonl" | tee "$OUT2/probe/plan.json"
step "probe generation"
py "$T1/shard.py" merge --journal "$S2/probe_data/journal.jsonl" --shards "$S2"/shards/probe_g*.jsonl   # keep work from interrupted shards
py "$T1/shard.py" split-reqs --journal "$S2/probe_data/journal.jsonl" --k "$K" \
   --requests "$S2/probe_data/requests.jsonl" --outprefix "$S2/shards/probe_"
run_shards probe_gen "$T2/gen_probe.py" --dataset "$PD" --requests "$S2/shards/probe_{k}.jsonl" \
   --journal "$S2/shards/probe_g{g}.jsonl"
py "$T1/shard.py" merge --journal "$S2/probe_data/journal.jsonl" --shards "$S2"/shards/probe_g*.jsonl
step "probe grading (sandbox)"
cpu "$T2/grade_probe.py" --dataset "$PD" --journal "$S2/probe_data/journal.jsonl" --out "$S2/probe_data/graded.jsonl"
step "probe fit (post's code, one GPU)"
run_one() { local g="${G[0]}"; WORKER_THREADS=16 bash "$ST/pairs_qwen/hpc_worker.sh" "$g" "$OUT2/logs/probe_fit_g$g.log" "$@"; }
run_one "$T2/fit_probe.py" --graded "$S2/probe_data/graded.jsonl" --out "$S2/probe_fit"
cp -a "$S2/probe_fit/PROBE_SUMMARY.json" "$S2/probe_fit/probe_directions.npz" "$OUT2/probe/"
cp -a "$S2/probe_fit/probes" "$OUT2/probe/" 2>/dev/null || true
cp -a "$S2/probe_fit/probe_rh_summary_stats.json" "$OUT2/probe/" 2>/dev/null || true
cp -a "$S2/probe_fit/test_dataset.json" "$OUT2/probe/calibration_heldout_split.json" 2>/dev/null || true
step "DONE"
