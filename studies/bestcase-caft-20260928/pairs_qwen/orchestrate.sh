#!/usr/bin/env bash
# Qwen pairs, rounds after the pilot (protocol draft v1, section 3.6), run as one supervising user service.
# Each step is its own capped transient unit started with --wait; any failure stops everything (fail closed).
# Usage: orchestrate.sh <gpu_index>
set -euo pipefail
GPU="$1"
ST="$HOME/rl-rewardhacking-gpu03/studies/bestcase-caft-20260928"
HV=/scratch/yelarys.yertaiuly/bestcase-caft-20260928/pairs_qwen_v1
PY=/scratch/yelarys.yertaiuly/rl-rewardhacking-gpu03-runtime/venv/bin/python
T="$ST/pairs_qwen/tools"
export WAIT=1
exec >> "$HV/logs/orchestrate.log" 2>&1
P="$HV/prefixes/assigned.jsonl"; J="$HV/gen/journal.jsonl"; G="$HV/grade/grades.jsonl"; Z="$HV/gen/sizing.json"
step() { echo "== $(date -u +%FT%TZ) $*"; }
plan() {  # plan <stage> <out>
  env -i PATH=/usr/bin:/bin HOME="$HV/tmp" PYTHONDONTWRITEBYTECODE=1 "$PY" -B "$T/plan_requests.py" \
    --stage "$1" --prefixes "$P" --journal "$J" --grades "$G" --sizing "$Z" --out "$2"
}
gen() {  # gen <name> <requests>
  if [ "$(grep -c '' "$2" || true)" -eq 0 ]; then step "no requests in $2"; return 0; fi
  bash "$ST/pairs_qwen/run_gpu.sh" "$1" "$GPU" generate.py --prefixes "$P" --requests "$2" --journal "$J"
}
grade() { bash "$ST/pairs_qwen/run_cpu.sh" "$1" grade.py --prefixes "$P" --journal "$J" --out "$G"; }

step "start orchestrate gpu=$GPU"
step "plan full";   plan full "$HV/gen/requests_full.jsonl"
step "gen full";    gen gen-full "$HV/gen/requests_full.jsonl"
step "grade full";  grade grade-full
step "plan base";   plan base "$HV/gen/requests_base1.jsonl"
step "gen base";    gen gen-base1 "$HV/gen/requests_base1.jsonl"
step "grade base";  grade grade-base1
for k in 1 2 3; do
  step "plan extend $k"; plan extend "$HV/gen/requests_ext$k.jsonl"
  if [ "$(grep -c '' "$HV/gen/requests_ext$k.jsonl" || true)" -eq 0 ]; then step "no extension needed"; break; fi
  gen "gen-ext$k" "$HV/gen/requests_ext$k.jsonl"; grade "grade-ext$k"
  plan base "$HV/gen/requests_ext${k}_base.jsonl"; gen "gen-ext${k}-base" "$HV/gen/requests_ext${k}_base.jsonl"
  grade "grade-ext${k}-base"
done
step "pair";   bash "$ST/pairs_qwen/run_cpu.sh" pair pair.py --prefixes "$P" --grades "$G" --outdir "$HV/pairs"
step "verify"; bash "$ST/pairs_qwen/run_cpu.sh" verify verify.py --prefixes "$P" --grades "$G" --pairdir "$HV/pairs"
step "DONE"
