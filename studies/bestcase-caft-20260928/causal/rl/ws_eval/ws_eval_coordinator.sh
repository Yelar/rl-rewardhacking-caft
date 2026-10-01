#!/usr/bin/env bash
# Workstation evaluator for CAFT-RL wave 2 (bc4w2q). Runs in a persistent tmux session on a Slurm workstation
# allocation (your approval, 1 Oct: evaluate steps 100 and 200 on up to two workstations without waiting for training).
# Loop:
#  - candidates: every lineage's closed step-100 checkpoint (as soon as it exists), and its closed step-200 checkpoint
#    once it has been closed for >= 20 min without the run host's own post-training trigger claiming it;
#  - claim on the run host by writing its manifest path (eval_w2q/manifests/<lineage>_step<N>.json) write-once, which
#    also makes the host trigger skip that checkpoint;
#  - copy the adapter, verify its SHA-256 against the host, write a local manifest with the GPU-qualified configuration
#    (6 scoring workers here: the allocation has 12 CPUs and 32 GB), dry-run, run eval_v2 on this GPU;
#  - copy the whole output back to the run host (eval_w2q/<name>/) and SUMMARY.json to the shared home;
#  - on failure, rename the claim to *.failed_ws_<host> so the run host can still evaluate it.
# It never touches training runs, other GPUs, or other users' processes.
set -uo pipefail
WS=$(hostname); S=/scratch/yelarys.yertaiuly; B=$S/bestcase-step4; K=$B/eval_w2q_kit; W=$B/eval_ws
PY=$S/rl-rewardhacking-gpu03-runtime/venv/bin/python
MODEL=$S/bestcase-ws/models/codex-ada8-matched-20260916-1540-inputs/base
HOSTS="gpu-01 gpu-02 gpu-03 gpu-04"; GRACE=1200
NFS=$HOME/bestcase-ws-runs/causal/eval_w2q
mkdir -p "$W/logs" "$W/adapters" "$W/manifests" "$W/out" "$W/home" "$NFS"
exec >> "$W/logs/coordinator_$WS.log" 2>&1
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=4)
log() { echo "== $(date -u +%FT%TZ) $*"; }
declare -A FAILED
log "coordinator start on $WS"
while :; do
  cand=$(for h in $HOSTS; do "${SSH[@]}" "$h" 'B=/scratch/yelarys.yertaiuly/bestcase-step4; now=$(date +%s)
    for d in $B/runs_w2q/bc4w2q-*; do L=$(basename "$d"); [[ "$L" == *-resume* ]] && continue
      for s in 100 200; do
        c=$(ls -d $B/runs_w2q/$L/checkpoints/global_step_$s $B/runs_w2q/$L-resume*/checkpoints/global_step_$s 2>/dev/null |
            while read -r x; do [ -f "$x/CAFT_CHECKPOINT.json" ] && echo "$(stat -c %Y "$x/CAFT_CHECKPOINT.json") $x"; done | sort -n | tail -1)
        [ -n "$c" ] || continue
        cl=0; [ -e "$B/eval_w2q/manifests/${L}_step$s.json" ] && cl=1
        echo "$(hostname) $L $s $(( now - ${c%% *} )) ${c#* } $cl"
      done; done' 2>/dev/null; done)
  pick=$(echo "$cand" | awk -v g=$GRACE '$6==0 && ($3==100 || $4>=g) {print}' | sort -k3,3n -k4,4nr | while read -r line; do
           set -- $line; [ -z "${FAILED[$2_$3]:-}" ] 2>/dev/null && { echo "$line"; break; }; done)
  if [ -z "$pick" ]; then sleep 120; continue; fi
  set -- $pick; H=$1; L=$2; STEP=$3; CK=$5; NAME="${L}_step$STEP"
  if ! "${SSH[@]}" "$H" "mkdir -p /scratch/yelarys.yertaiuly/bestcase-step4/eval_w2q/manifests && set -C && printf '{\"claimed_by\": \"%s\", \"claimed_at\": \"%s\", \"checkpoint\": \"%s\"}\n' $WS $(date -u +%FT%TZ) $CK > /scratch/yelarys.yertaiuly/bestcase-step4/eval_w2q/manifests/$NAME.json"; then
    log "claim lost for $NAME on $H"; continue
  fi
  log "claimed $NAME on $H ($CK)"
  A="$W/adapters/$NAME"; ok=1
  rsync -a "$H:$CK/actor/lora_adapter/" "$A/" || ok=0
  if [ $ok = 1 ]; then
    remote=$("${SSH[@]}" "$H" "cd $CK/actor/lora_adapter && sha256sum adapter_config.json adapter_model.safetensors")
    local_=$(cd "$A" && sha256sum adapter_config.json adapter_model.safetensors)
    [ "$remote" = "$local_" ] || { log "adapter hash mismatch for $NAME"; ok=0; }
  fi
  M="$W/manifests/$NAME.json"
  if [ $ok = 1 ]; then
    python3 - "$M" "$NAME" "$A" "$L" "$STEP" "$H" "$CK" "$W/out/$NAME" "$K" "$MODEL" "$WS" <<'PYEOF' || ok=0
import hashlib, json, sys
M, name, A, L, step, H, CK, out, K, model, ws = sys.argv[1:]
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''): h.update(b)
    return h.hexdigest()
m = {'schema': 'bestcase-caft-posttrain-eval-v2',
     'model': {'model_id': 'Qwen/Qwen3-4B', 'revision': '1cfa9a7208912126459214e8b04321603b3df60c', 'path': model},
     'checkpoints': [{'name': name, 'adapter_path': A,
                      'files': {f: sha(f'{A}/{f}') for f in ('adapter_config.json', 'adapter_model.safetensors')},
                      'metadata': {'lineage': L, 'step': int(step), 'run_host': H, 'checkpoint_dir': CK, 'evaluated_on': ws}}],
     'settings': ['fixed', 'randomized', 'noloophole'],
     'datasets': {s: {'path': f'{K}/data/{f}', 'sha256': sha(f'{K}/data/{f}')} for s, f in
                  (('fixed', 'leetcode_test_medhard_simple_overwrite_tests.jsonl'),
                   ('randomized', 'leetcode_test_medhard_overwrite_tests.jsonl'), ('noloophole', 'leetcode_test_medhard.jsonl'))},
     'output_dir': out,
     'sampling': {'max_new_tokens': 1536, 'n': 10, 'repetition_penalty': 1.0, 'temperature': 0.7, 'top_k': 0, 'top_p': 0.95},
     'enable_thinking': False, 'vllm_seed': 1, 'seed_namespace': 'bestcase-caft-20260928|w2q-posttrain-v1', 'gpu_id': 0,
     'engine': {'dtype': 'bfloat16', 'enable_prefix_caching': True, 'gpu_memory_utilization': 0.7, 'idle_gpu_max_used_mib': 1024,
                'max_lora_rank': 32, 'max_model_len': 3072, 'max_num_seqs': 256},
     'scoring': {'source_root': '/scratch/yelarys.yertaiuly/bestcase-step4/source_v2eval_b71d8e8b425a',
                 'source_tree_sha256': 'b71d8e8b425af78592edb43d1a2f1652e5a5aa24b45898bee1473b03fea216f4',
                 'venv': '/scratch/yelarys.yertaiuly/rl-rewardhacking-gpu03-runtime/venv', 'wall_seconds': 3600, 'workers': 6},
     'notes': f'CAFT-RL wave 2 (bc4w2q) post-training evaluation on workstation {ws}; qualified configuration, 6 scoring workers.'}
open(M, 'x').write(json.dumps(m, indent=1, sort_keys=True))
PYEOF
  fi
  rc=1
  if [ $ok = 1 ]; then
    HS=$(sha256sum "$M" | cut -c1-64)
    env -i PATH=/usr/bin:/bin HOME=/tmp "$PY" -B "$K/eval_v2.py" --manifest "$M" --dry-run --manifest-sha256 "$HS" > "$W/logs/dryrun_$NAME.json" 2>&1 || { log "dry run failed for $NAME"; ok=0; }
  fi
  if [ $ok = 1 ]; then
    log "evaluating $NAME"
    TMPD=$(mktemp -d /tmp/bc4ws-XXXXXXXX); mkdir -p "$W/home/$NAME"
    env -i PATH="$(dirname "$PY"):/usr/local/cuda-12.2/bin:/usr/bin:/bin" CUDA_HOME=/usr/local/cuda-12.2 HOME="$W/home/$NAME" USER="$(id -un)" TMPDIR="$TMPD" LANG=C.UTF-8 PYTHONHASHSEED=0 \
      PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false TORCHINDUCTOR_COMPILE_THREADS=1 \
      VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_USE_V1=1 VLLM_USE_FLASHINFER_SAMPLER=0 CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 \
      TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 \
      VLLM_CACHE_ROOT="$W/home/$NAME/vllm" TRITON_CACHE_DIR="$W/home/$NAME/triton" TORCHINDUCTOR_CACHE_DIR="$W/home/$NAME/inductor" \
      "$PY" -B "$K/eval_v2.py" --manifest "$M" --manifest-sha256 "$HS" run > "$W/logs/eval_$NAME.log" 2>&1
    rc=$?; rm -rf "$TMPD"
  fi
  if [ $ok = 1 ] && [ $rc = 0 ] && [ -s "$W/out/$NAME/SUMMARY.json" ]; then
    rsync -a "$W/out/$NAME/" "$H:/scratch/yelarys.yertaiuly/bestcase-step4/eval_w2q/$NAME/" && cp "$W/out/$NAME/SUMMARY.json" "$NFS/${NAME}_SUMMARY.json"
    "${SSH[@]}" "$H" "printf '{\"done_on\": \"%s\", \"done_at\": \"%s\"}\n' $WS $(date -u +%FT%TZ) > /scratch/yelarys.yertaiuly/bestcase-step4/eval_w2q/manifests/$NAME.json.done_ws"
    log "done $NAME"
  else
    FAILED[${L}_$STEP]=1
    "${SSH[@]}" "$H" "mv /scratch/yelarys.yertaiuly/bestcase-step4/eval_w2q/manifests/$NAME.json /scratch/yelarys.yertaiuly/bestcase-step4/eval_w2q/manifests/$NAME.json.failed_ws_$WS"
    log "FAILED $NAME (ok=$ok rc=$rc); claim released for the run host"
  fi
done
