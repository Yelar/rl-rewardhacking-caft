# Sourced helper: gpu_free <index> -> 0 when the GPU has no compute process from anyone, <= 1 GiB used, and is not
# allocated to any Slurm job on this host (another user's allocation is theirs even while it is idle).
slurm_idx() {
  local j
  for j in $(squeue -w "$(hostname)" -h -o %i 2>/dev/null); do
    scontrol show job -d "$j" 2>/dev/null | grep -oE 'IDX:[0-9,-]+' | cut -d: -f2 | tr ',' '\n' | while read -r r; do
      if [[ "$r" == *-* ]]; then seq "${r%-*}" "${r#*-}"; else echo "$r"; fi; done
  done
}
gpu_free() {
  local g="$1" uuid n used
  uuid=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g="$g" '$1==g{print $2}')
  [ -n "$uuid" ] || return 1
  n=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c "$uuid")
  used=$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  [ "$n" -eq 0 ] && [ "$used" -le 1024 ] || return 1
  ! slurm_idx | grep -qx "$g"
}
