#!/usr/bin/env bash
set -Eeuo pipefail

if [[ ! "${CODEX_FACTORIAL_LAUNCH_ID:-}" =~ ^[0-9a-f]{32}$ ]]; then
  echo 'This coordinator requires a consumed reviewed-launch identity.' >&2
  exit 2
fi
if [[ "$#" -lt 3 ]]; then
  echo 'Usage: factorial_coordinator_job.sh EXPECTED_HOST PYTHON COORDINATOR [ARGS...]' >&2
  exit 2
fi

readonly EXPECTED_HOST="$1"
readonly PYTHON_BIN="$2"
readonly COORDINATOR="$3"
shift 3
readonly -a COORDINATOR_ARGS=("$@")

[[ "$(hostname -s)" == "${EXPECTED_HOST}" ]] || exit 3
[[ -x "${PYTHON_BIN}" && -f "${COORDINATOR}" ]] || exit 3
for command in timeout nice ionice flock systemctl ssh rsync stat; do
  command -v "${command}" >/dev/null || exit 3
done

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
while IFS='=' read -r variable _; do
  if [[ "${variable}" == AWS_* || "${variable}" == WANDB_* ||
        "${variable}" == AZURE_* || "${variable}" == GOOGLE_* ||
        "${variable}" == HF_TOKEN || "${variable}" == HUGGING_FACE_HUB_TOKEN ||
        "${variable}" == GITHUB_TOKEN || "${variable}" == OPENAI_API_KEY ]]; then
    unset "${variable}"
  fi
done < <(env)

readonly lock_path="/run/lock/codex-factorial-coordinator-${EXPECTED_HOST}.lock"
[[ ! -L "${lock_path}" ]] || exit 4
exec {lock_fd}<>"${lock_path}"
chmod 600 "${lock_path}"
[[ -f "${lock_path}" && "$(stat -Lc '%u:%a' "${lock_path}")" == "$(id -u):600" ]] || exit 4
flock -n "${lock_fd}" || exit 4

# The persisted reviewed deadline remains authoritative. This independent outer
# bound permits the exact-100K profile's 24-hour deadline plus cleanup/recovery.
exec timeout --signal=TERM --kill-after=120s 90000s \
  nice -n 5 ionice -c 2 -n 7 \
  "${PYTHON_BIN}" "${COORDINATOR}" "${COORDINATOR_ARGS[@]}"
