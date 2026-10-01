#!/usr/bin/env bash
# Persona study (protocol draft v5, step 8): screen candidate persona vectors by removing them from the hacking model.
# 1. specs: make_screen_specs.py writes unit banks and CAFT specs (settings: all blocks, blocks 22-27);
# 2. side-effect filter (the fast test's validated measure only): fasttest.py --side-only on the hacking model;
#    candidates with KL above 0.15 per token on ordinary code are dropped (rule fixed before any persona result);
# 3. screening generation: run_screen.sh on the remaining arms (validation problems, 4 answers each, step-3 seeds),
#    graded with the step-3 grader.
# Usage: run_step8.sh <tag> <candidates file> <comma-separated GPUs> [store dir to orthogonalize against] [settings]
#   settings: comma-separated names from make_screen_specs.py SETTINGS (default all,w22)
#   candidates file: one line per vector, <rung>:<model>:<extraction dir>:<key>, key like directive|C|adj
set -euo pipefail
TAG="$1"; CANDS="$2"; GPUS="$3"; ORTH="${4:-}"; SETS="${5:-all,w22}"; IFS=',' read -r -a G <<< "$GPUS"
ST="$HOME/rl-rewardhacking-gpu03/studies/bestcase-caft-20260928"
S=/scratch/yelarys.yertaiuly/bestcase-persona
W=/scratch/yelarys.yertaiuly/bestcase-ws
RUNS="$HOME/bestcase-ws-runs/persona"
LOG="$RUNS/logs"
PY=/scratch/yelarys.yertaiuly/rl-rewardhacking-gpu03-runtime/venv/bin/python
SPECS="$S/screen/specs_$TAG"
SIDE="$RUNS/screen/side_$TAG.jsonl"
mkdir -p "$LOG" "$RUNS/screen"
exec >> "$LOG/run_step8_$TAG.log" 2>&1
step() { echo "== $(date -u +%FT%TZ) $*"; }
[ -e "$SPECS" ] && { step "specs exist: $SPECS"; exit 2; }
step "start $TAG: $(grep -c . "$CANDS") vectors on GPUs $GPUS"
ARGS=(); while read -r line; do [ -n "$line" ] && ARGS+=(--vector "$line"); done < "$CANDS"
[ -n "$ORTH" ] && ARGS+=(--orthogonalize "$ORTH")
ARGS+=(--settings "$SETS")
env -i PATH=/usr/bin:/bin HOME="$W/tmp/home" LANG=C.UTF-8 PYTHONPATH="$ST/source_v2" \
  BESTCASE_DATASET="$W/inputs/dataset.jsonl" OMP_NUM_THREADS=1 "$PY" -B "$ST/persona/tools/make_screen_specs.py" \
  "${ARGS[@]}" --out "$SPECS"
step "side-effect filter"
bash "$ST/launchprep/hpc_worker_v2.sh" "${G[0]}" "$LOG/side_$TAG.log" "$ST/persona/tools/fasttest.py" \
  --model mrh --candidates "$SPECS/side_candidates.jsonl" --out "$SIDE" --side-only
mapfile -t ARMS < <("$PY" -B -c "
import json, sys
rows = [json.loads(l) for l in open('$SIDE')]
names = [json.loads(l)['name'] for l in open('$SPECS/side_candidates.jsonl')]
got = {r['name']: r for r in rows}
missing = [n for n in names if n not in got]
if missing: sys.exit(f'side-effect results missing for {missing}')
for n in names:
    if got[n]['side_kl_per_token'] <= 0.15: print(n)
    else: print(f'dropped {n}: KL {got[n][\"side_kl_per_token\"]:.3f}', file=sys.stderr)
")
step "screening ${#ARMS[@]} arms: ${ARMS[*]}"
bash "$ST/persona/run_screen.sh" "$GPUS" val mrh 4 "$SPECS/specs" "screen_$TAG" "${ARMS[@]}"
step "DONE"
