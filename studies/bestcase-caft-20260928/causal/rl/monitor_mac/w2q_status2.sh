B=/scratch/yelarys.yertaiuly/bestcase-step4; ME=$(id -un)
ma=$(awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo); l1=$(cut -d' ' -f1 /proc/loadavg)
sf=$(df -BG --output=avail /scratch | tail -1 | tr -dc 0-9); shm=$(df -BG --output=used /dev/shm | tail -1 | tr -dc 0-9)
rp=$(df --output=pcent / | tail -1 | tr -dc 0-9)
declare -A IDX; while IFS=', ' read -r i u; do IDX[$u]=$i; done < <(nvidia-smi --query-gpu=index,uuid --format=csv,noheader)
fg=$(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader | while IFS=', ' read -r p u; do o=$(ps -o user= -p "$p" 2>/dev/null); [ -n "$o" ] && [ "$o" != "$ME" ] && echo "g${IDX[$u]}:$o"; done | sort -u | tr '\n' ',')
echo "HOST $(hostname) mem_avail_gb=$ma load1=$l1 scratch_free_gb=$sf shm_used_gb=$shm rootfs_pct=$rp foreign=[${fg%,}]"
for d in $B/runs_w2q/bc4w2q-*; do
  [ -f "$d/train.log" ] || continue; t=$(basename "$d"); L=$d/train.log
  st=$(systemctl --user is-active "$t" 2>/dev/null)
  step=$(grep -ao 'step:[0-9]* ' "$L" | tail -1 | tr -dc 0-9)
  ex=$(grep -a 'TRAIN EXIT' "$L" | tail -1 | grep -o 'rc=[0-9]*' | tr -dc 0-9)
  err=$(grep -acE 'Traceback \(most recent call last\)|CUDA out of memory|OutOfMemoryError|ActorDiedError|RayTaskError|Segmentation fault|no fallback reward|core dumped|ProcessGroupNCCL.*(Timeout|timed out|dump signal)' "$L")
  grd=$(grep -acE 'GUARD:|WATCHDOG:|RELEASE NOT VERIFIED' "$L"); rel=$(grep -ac 'RELEASE VERIFIED' "$L")
  age=$(( $(date +%s) - $(stat -c %Y "$L") ))
  gp=$(grep -a -m1 'TRAIN START' "$L" | grep -o 'gpus=[0-9,]*' | cut -d= -f2)
  cs=$(python3 -c "import json,sys; c=json.load(open(sys.argv[1]))['reward_model']['reward_kwargs']['caft_reward_sandbox']['cpu_set']; print(f'{min(c)}-{max(c)}')" "$d/config.json")
  ck=$(for x in "$d"/checkpoints/global_step_*; do [ -f "$x/CAFT_CHECKPOINT.json" ] && echo "${x##*_}"; done | sort -n | tail -1)
  child=$(ls -d "$B/runs_w2q/$t"-resume* 2>/dev/null | head -1 | xargs -r basename)
  start=$(date -u -d "$(grep -a -m1 'TRAIN START' "$L" | awk '{print $3}')" +%s 2>/dev/null || echo 0)
  echo "RUN $t unit=${st:-none} step=${step:-0} exit=${ex:-none} err=${err:-0} guard=${grd:-0} released=${rel:-0} log_age_s=$age gpus=$gp cpus=$cs ckpt=${ck:-none} child=${child:-none} start=$start"
done
for u in $(systemctl --user list-units 'bc4w2q-launch-*' 'bc4w2q-wait-*' 'bc4w2q-rw-*' --all --no-legend --plain | awk '{print $1}'); do echo "QUEUE $u $(systemctl --user is-active $u)"; done
grep -h "FAILED" $B/logs/launch_plan_w2q*.log 2>/dev/null | sed 's/^/LAUNCHFAIL /'
for d in $B/runs_w1/bc4w1-*; do [ -f "$d/train.log" ] || continue; echo "W1 $(basename $d) unit=$(systemctl --user is-active bc4w1-$(basename $d)) step=$(grep -ao 'step:[0-9]* ' $d/train.log | tail -1 | tr -dc 0-9)"; done
