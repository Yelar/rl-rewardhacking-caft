#!/usr/bin/env bash
# Step 3 on the validation split (protocol draft v1, section 6), plus the step-2 probe redo with the L40S baseline's
# step-200 adapter (section 5.5 item 2). Runs on gpu-01 as a user-systemd unit.
# - Every GPU process runs through hpc_worker.sh (guard before start, watchdog during). A foreign process fails that
#   worker closed and the pipeline stops.
# - Compact results go to the shared home; heavy files stay on scratch.
# Usage: step3_pipeline.sh <comma-separated GPU indices>
set -euo pipefail
GPUS="$1"; IFS=',' read -r -a G <<< "$GPUS"; K=${#G[@]}
ST="$HOME/rl-rewardhacking-gpu03/studies/bestcase-caft-20260928"
T1="$ST/pairs_qwen/tools"; T2="$ST/step2/tools"; T3="$ST/step3/tools"
W=/scratch/yelarys.yertaiuly/bestcase-ws
S2=/scratch/yelarys.yertaiuly/bestcase-step2
S3=/scratch/yelarys.yertaiuly/bestcase-step3
OUT2="$HOME/bestcase-ws-runs/step2"; OUT3="$HOME/bestcase-ws-runs/step3"
PY=/scratch/yelarys.yertaiuly/rl-rewardhacking-gpu03-runtime/venv/bin/python
A200="$W/inputs/l40s_baseline_step200/resume/x/actor/lora_adapter"
PD="$S2/probe_data/data/leetcode_train_medhard_holdout_all.jsonl"
PR="$S2/probe_s200"
mkdir -p "$S3"/{calib,journals,graded,smoke,shards} "$PR"/{shards,fit} "$OUT3"/logs "$OUT2"/probe_s200 "$W/tmp/home"
exec >> "$OUT3/logs/step3.log" 2>&1
COMMON=(PATH=/usr/local/bin:/usr/bin:/bin HOME="$W/tmp/home" TMPDIR="$W/tmp" LANG=C.UTF-8
        OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
        HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 PYTHONNOUSERSITE=1
        PYTHONPATH="$ST/source:$ST/source/verl:$ST/step2/vendor" CUDA_VISIBLE_DEVICES=
        BESTCASE_POOL="$HOME/bestcase-ws-runs/pairs_qwen_v2/pool/pool.jsonl" BESTCASE_DATASET="$W/inputs/dataset.jsonl"
        BESTCASE_MODEL="$W/models/codex-ada8-matched-20260916-1540-inputs/base"
        BESTCASE_MRH_ADAPTER="$W/inputs/checkpoint_60/lora_adapter")
step() { echo "== $(date -u +%FT%TZ) $*"; }
py() { env -i "${COMMON[@]}" "$PY" -B "$@"; }
cpu() { env -i "${COMMON[@]}" CODE_EVAL_SANDBOX=bwrap CODE_EVAL_SANDBOX_REQUIRED=1 MAX_JOBS=32 nice -n 10 "$PY" -B "$@"; }
gpu() { local g="$1" log="$2"; shift 2; bash "$ST/pairs_qwen/hpc_worker.sh" "$g" "$log" "$@"; }
fail() { step "FAILED: $*"; exit 1; }
run_shards() {  # run_shards <name> <tool> <args with {k}/{g}>: one worker per GPU, all must succeed
  local name="$1" tool="$2"; shift 2; local pids=() k=0 rc=0
  for g in "${G[@]}"; do
    local args=(); for x in "$@"; do x="${x//\{k\}/$k}"; args+=("${x//\{g\}/$g}"); done
    gpu "$g" "$OUT3/logs/${name}_g$g.log" "$tool" "${args[@]}" & pids+=($!); k=$((k+1)); sleep 15
  done
  for p in "${pids[@]}"; do wait "$p" || rc=$?; done
  [ "$rc" -eq 0 ] || fail "shard failure in $name (rc=$rc)"
}
run_queue() {  # run_queue <split> <arm...>: conditions round-robin over GPUs, each GPU runs its queue in order
  local split="$1"; shift; local arms=("$@") pids=() i rc=0
  for i in "${!G[@]}"; do
    ( set -e; j=$i
      while [ "$j" -lt "${#arms[@]}" ]; do
        arm="${arms[$j]}"
        [ -f "$S3/journals/${split}_$arm.jsonl" ] || gpu "${G[$i]}" "$OUT3/logs/gen_${split}_$arm.log" "$T3/gen_caft.py" \
          --spec "$S3/specs/$arm.json" --split "$split" --journal "$S3/journals/${split}_$arm.jsonl"
        j=$((j + K))
      done ) & pids+=($!); sleep 20
  done
  for p in "${pids[@]}"; do wait "$p" || rc=$?; done
  [ "$rc" -eq 0 ] || fail "generation failure (rc=$rc)"
}
grade_all() {  # grade_all <split> <arm...> (sequential, sandboxed, resumable)
  local split="$1"; shift
  for arm in "$@"; do
    cpu "$T3/grade_step3.py" --journal "$S3/journals/${split}_$arm.jsonl" --out "$S3/graded/graded_${split}_$arm.jsonl" \
      >> "$OUT3/logs/grade_${split}.log" 2>&1
  done
}

step "start step 3 host=$(hostname) gpus=$GPUS"
DB="$OUT2/directions/direction_banks.npz"
PAIR_ARMS=(tail_b21 tail_b22_27 tail_b28_33 test_b21 test_b22_27 test_b28_33)
# ---------- 0. specs for projection-off and the pair candidates; smoke test of the engine (fail fast) ----------
if [ ! -f "$S3/specs/tail_b21.json" ]; then
  conds=(--condition "off_b22_27=$DB:pair_tail:22-27" --structural-zero off_b22_27)
  for arm in "${PAIR_ARMS[@]}"; do
    ro="${arm%%_*}"; bl="${arm#*_b}"; bl="${bl/_/-}"; key=pair_tail; [ "$ro" = test ] && key=pair_test
    conds+=(--condition "$arm=$DB:$key:$bl")
  done
  py "$T3/make_specs.py" "${conds[@]}" --out "$S3"
fi
step "smoke: Mrh + LoRA + CAFT engine (tail_b22_27, 4 problems, 2 samples, 64 tokens)"
[ -f "$S3/smoke/val_tail_b22_27.jsonl" ] || gpu "${G[0]}" "$OUT3/logs/smoke.log" "$T3/gen_caft.py" \
  --spec "$S3/specs/tail_b22_27.json" --split val --journal "$S3/smoke/val_tail_b22_27.jsonl" --smoke 4
[ -f "$S3/smoke/val_off_b22_27.jsonl" ] || gpu "${G[0]}" "$OUT3/logs/smoke.log" "$T3/gen_caft.py" \
  --spec "$S3/specs/off_b22_27.json" --split val --journal "$S3/smoke/val_off_b22_27.jsonl" --smoke 4
# ---------- A. probe redo: the hacking half from the step-200 adapter ----------
step "probe redo: plan + generation (step-200 adapter)"
[ -f "$PR/requests.jsonl" ] || py "$T2/gen_probe_s200.py" --plan-from "$S2/probe_data/requests.jsonl" \
  --plan "$PR/requests.jsonl" --dataset "$PD" --adapter "$A200" | tee "$OUT2/probe_s200/plan.json"
py "$T1/shard.py" merge --journal "$PR/journal.jsonl" --shards "$PR"/shards/probe_g*.jsonl 2>/dev/null || true
py "$T1/shard.py" split-reqs --journal "$PR/journal.jsonl" --k "$K" --requests "$PR/requests.jsonl" \
  --outprefix "$PR/shards/req_"
run_shards probe_s200_gen "$T2/gen_probe_s200.py" --dataset "$PD" --adapter "$A200" \
  --requests "$PR/shards/req_{k}.jsonl" --journal "$PR/shards/probe_g{g}.jsonl"
py "$T1/shard.py" merge --journal "$PR/journal.jsonl" --shards "$PR"/shards/probe_g*.jsonl
[ "$(wc -l < "$PR/journal.jsonl")" -eq 1000 ] || fail "probe redo journal does not have 1,000 requests"
step "probe redo: grading in the background (sandbox)"
cpu "$T2/grade_probe.py" --dataset "$PD" --journal "$PR/journal.jsonl" --out "$PR/graded_s200.jsonl" \
  > "$OUT3/logs/probe_s200_grade.log" 2>&1 &
PG=$!
# ---------- B. calibration, random controls for the pair candidates, specs, smoke ----------
step "energy calibration on Mrh (fit problems)"
[ -f "$S3/calib/CALIB.json" ] || gpu "${G[0]}" "$OUT3/logs/energy_collect.log" "$T3/energy.py" collect --out "$S3/calib"
if [ ! -f "$S3/random_pair.npz" ]; then
  targets=()
  for arm in "${PAIR_ARMS[@]}"; do
    ro="${arm%%_*}"; bl="${arm#*_b}"; bl="${bl/_/-}"; key=pair_tail; [ "$ro" = test ] && key=pair_test
    targets+=(--target "rand_$arm=$DB:$key:$bl")
  done
  gpu "${G[0]}" "$OUT3/logs/energy_draw_pair.log" "$T3/energy.py" draw --store "$S3/calib" "${targets[@]}" \
    --out-bank "$S3/random_pair.npz" --out-report "$S3/random_pair.json"
fi
if [ ! -f "$S3/specs/rand_tail_b21.json" ]; then
  conds=()
  for arm in "${PAIR_ARMS[@]}"; do
    bl="${arm#*_b}"; bl="${bl/_/-}"
    conds+=(--condition "rand_$arm=$S3/random_pair.npz:rand_$arm:$bl")
  done
  py "$T3/make_specs.py" "${conds[@]}" --out "$S3"
fi
# ---------- C. validation generation: off, 6 pair candidates, 6 matched random controls ----------
ARMS1=(off_b22_27)
for arm in "${PAIR_ARMS[@]}"; do ARMS1+=("$arm" "rand_$arm"); done
step "validation generation: ${ARMS1[*]}"
run_queue val "${ARMS1[@]}"
step "grading pair conditions in the background"
grade_all val "${ARMS1[@]}" &
GP=$!
# ---------- D. probe fit (step-200 data), probe conditions ----------
wait "$PG" || fail "probe redo grading failed"
[ "$(wc -l < "$PR/graded_s200.jsonl")" -eq 10000 ] || fail "probe redo grading incomplete"
py "$T2/combine_probe_graded.py" --base-from "$S2/probe_data/graded.jsonl" --hack-from "$PR/graded_s200.jsonl" \
  --out "$PR/graded_combined.jsonl" | tee "$OUT2/probe_s200/combine.json"
step "probe fit (post's code) on step-200 data"
[ -f "$PR/fit/PROBE_SUMMARY.json" ] || WORKER_THREADS=16 bash "$ST/pairs_qwen/hpc_worker.sh" "${G[0]}" \
  "$OUT3/logs/probe_s200_fit.log" "$T2/fit_probe.py" --graded "$PR/graded_combined.jsonl" --out "$PR/fit" \
  --hack-source "L40S baseline step 200 (planned source; base half reused from the first fit)"
cp -a "$PR/fit/PROBE_SUMMARY.json" "$PR/fit/probe_directions.npz" "$OUT2/probe_s200/"
cp -a "$PR/fit/probe_rh_summary_stats.json" "$OUT2/probe_s200/" 2>/dev/null || true
cp -a "$PR/fit/probes" "$OUT2/probe_s200/" 2>/dev/null || true
cp -a "$PR/fit/test_dataset.json" "$OUT2/probe_s200/calibration_heldout_split.json" 2>/dev/null || true
py "$T3/probe_bank.py" --probe-dir "$PR/fit" --out "$S3/probe_bank.npz" | tee "$OUT3/probe_bank.json"
CH=$(py -c "import json; print(json.load(open('$PR/fit/PROBE_SUMMARY.json'))['chosen_caft_block'])")
if [ ! -f "$S3/random_probe.npz" ]; then
  gpu "${G[0]}" "$OUT3/logs/energy_draw_probe.log" "$T3/energy.py" draw --store "$S3/calib" \
    --target "rand_probe_chosen=$S3/probe_bank.npz:probe_chosen:$CH" \
    --target "rand_probe_b22_27=$S3/probe_bank.npz:probe_b22_27:22-27" \
    --out-bank "$S3/random_probe.npz" --out-report "$S3/random_probe.json"
fi
if [ ! -f "$S3/specs/probe_chosen.json" ]; then
  py "$T3/make_specs.py" --condition "probe_chosen=$S3/probe_bank.npz:probe_chosen:$CH" \
    --condition "probe_b22_27=$S3/probe_bank.npz:probe_b22_27:22-27" \
    --condition "rand_probe_chosen=$S3/random_probe.npz:rand_probe_chosen:$CH" \
    --condition "rand_probe_b22_27=$S3/random_probe.npz:rand_probe_b22_27:22-27" --out "$S3"
fi
ARMS2=(probe_chosen rand_probe_chosen probe_b22_27 rand_probe_b22_27)
step "validation generation: ${ARMS2[*]}"
run_queue val "${ARMS2[@]}"
wait "$GP" || fail "grading of pair conditions failed"
grade_all val "${ARMS2[@]}"
# ---------- E. summary ----------
step "summary (validation)"
py "$T3/summarize.py" --graded "$S3/graded" --conditions "$S3/CONDITIONS.json" --split val \
  --out "$OUT3/STEP3_VAL_SUMMARY.json" | tee "$OUT3/STEP3_VAL_TABLE.txt"
cp -a "$S3/CONDITIONS.json" "$S3/random_pair.json" "$S3/random_probe.json" "$S3/calib/CALIB.json" "$OUT3/"
for f in "$S3"/journals/*.receipt.json; do cp -a "$f" "$OUT3/logs/"; done
step "DONE"
