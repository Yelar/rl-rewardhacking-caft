#!/usr/bin/env bash

# Guarded launcher. This file is intentionally inert until the exact reviewed
# manifest digest is supplied in a new, post-review approval token.

set -Eeuo pipefail

readonly EXPECTED_ACCOUNT='123456789012'
readonly REGION='us-east-1'
readonly AWS_PROFILE_NAME='codex-skypilot-a100'
readonly AWS_LAUNCHER_ROLE='codex-skypilot-a100-launcher'
readonly AWS_BOOTSTRAP_USER='codex-skypilot-a100-bootstrap'
readonly AWS_LAUNCHER_POLICY='CodexSkyPilotA100LauncherPolicy'
readonly AWS_BOOTSTRAP_POLICY='AssumeCodexSkyPilotA100Launcher'
readonly BUCKET='example-reward-hacking-artifacts'
readonly RUN_TAG_KEY='codex-run-owner'
readonly APPROVAL_PREFIX='I_APPROVE_8XA100_ON_DEMAND:'
readonly SKY_CONFIG='infra/skypilot/reviewed_skypilot_config.yaml'
readonly SKY_VERSION='skypilot, version 0.12.3.post1'
readonly SKY_SEMVER='0.12.3.post1'
readonly SKY_COMMIT='e60704b3e0174ff0461fdf7c219b2bbdeac7ee41'
readonly SKY_API_VERSION='50'
readonly WANDB_API_KEY_FILE='infra/skypilot/secrets/wandb_api_key'
readonly REJECTED_WANDB_KEY_HASHES='infra/skypilot/rejected_wandb_key_sha256.txt'
readonly REVIEWED_AMI_ID='ami-0c3bc6c2c633f3dd3'
readonly REVIEWED_SECURITY_GROUP_ID='sg-00000000000000001'
readonly REVIEWED_SECURITY_GROUP_NAME='sky-sg-researcher-55d4'
readonly REVIEWED_VPC_ID='vpc-00000000000000001'
readonly REVIEWED_SUBNET_A='subnet-00000000000000003'
readonly REVIEWED_SUBNET_B='subnet-00000000000000004'
readonly REVIEWED_SUBNET_C='subnet-00000000000000002'
readonly REVIEWED_SUBNET_D='subnet-00000000000000001'
readonly PROVISIONING_POLL_SECONDS='30'
readonly EXPECTED_REMOTE_JOB_ID='1'
readonly EXPECTED_REMOTE_JOB_NAME='qwen3-4b-reward-hack-grpo'

# The reviewed launcher always uses the dedicated assume-role profile. Remove
# ambient credential/config selectors that could otherwise take precedence or
# redirect AWS calls, then select the exact locally configured profile. The
# source credentials remain in the standard AWS shared-credentials file and
# are used only by the profile's role_arn/source_profile chain.
unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN AWS_SECURITY_TOKEN
unset AWS_ROLE_ARN AWS_WEB_IDENTITY_TOKEN_FILE AWS_ROLE_SESSION_NAME
unset AWS_CONTAINER_CREDENTIALS_FULL_URI AWS_CONTAINER_CREDENTIALS_RELATIVE_URI
unset AWS_CONTAINER_AUTHORIZATION_TOKEN AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE
unset AWS_CONFIG_FILE AWS_SHARED_CREDENTIALS_FILE AWS_DEFAULT_PROFILE
unset AWS_ENDPOINT_URL AWS_ENDPOINT_URL_EC2 AWS_ENDPOINT_URL_IAM AWS_ENDPOINT_URL_S3
unset AWS_ENDPOINT_URL_SERVICE_QUOTAS AWS_ENDPOINT_URL_STS
unset SKYPILOT_API_SERVER_ENDPOINT
export AWS_PROFILE="${AWS_PROFILE_NAME}"
export AWS_DEFAULT_REGION="${REGION}"
export AWS_REGION="${REGION}"

if [[ "$#" -ne 2 ]]; then
  echo "usage: $0 REVIEWED_TASK_YAML CLUSTER_NAME" >&2
  exit 2
fi
if ! REPOSITORY_ROOT="$(git rev-parse --show-toplevel)" || [[ "${REPOSITORY_ROOT}" != "${PWD}" ]]; then
  echo 'Launch blocked: run this launcher from the repository root.' >&2
  exit 2
fi

TASK_YAML="$1"
CLUSTER_NAME="$2"
case "${TASK_YAML}" in
  infra/skypilot/a100_reward_hack.yaml)
    readonly HARDWARE_PROFILE='p4de-a100-80gb'
    readonly INSTANCE_TYPE='p4de.24xlarge'
    readonly EXPECTED_GPU_MEMORY_API_MIB='81920'
    readonly REVIEWED_TASK='infra/skypilot/a100_reward_hack.yaml'
    readonly REVIEWED_MANIFEST='infra/skypilot/reviewed_manifest.json'
    readonly REVIEWED_RUN_TOKEN_FILE='infra/skypilot/reviewed_run_token.txt'
    readonly REVIEWED_IAM_POLICY='infra/skypilot/iam/codex_skypilot_a100_permissions.json'
    readonly RENDERED_TASK_RELATIVE='infra/skypilot/a100_reward_hack.rendered.yaml'
    readonly PROVISIONING_RETRY_SECONDS='21600'
    readonly POST_ALLOCATION_LIMIT_SECONDS='0'
    readonly WALL_LIMIT_SECONDS='50400'
    declare -ar REVIEWED_AVAILABILITY_ZONES=('us-east-1c' 'us-east-1d')
    declare -ar REVIEWED_SUBNET_IDS=("${REVIEWED_SUBNET_C}" "${REVIEWED_SUBNET_D}")
    ;;
  infra/skypilot/a100_40gb_reward_hack.yaml)
    readonly HARDWARE_PROFILE='p4d-a100-40gb'
    readonly INSTANCE_TYPE='p4d.24xlarge'
    readonly EXPECTED_GPU_MEMORY_API_MIB='40960'
    readonly REVIEWED_TASK='infra/skypilot/a100_40gb_reward_hack.yaml'
    readonly REVIEWED_MANIFEST='infra/skypilot/reviewed_manifest_p4d.json'
    readonly REVIEWED_RUN_TOKEN_FILE='infra/skypilot/reviewed_run_token_p4d.txt'
    readonly REVIEWED_IAM_POLICY='infra/skypilot/iam/codex_skypilot_a100_40gb_permissions.json'
    readonly RENDERED_TASK_RELATIVE='infra/skypilot/a100_40gb_reward_hack.rendered.yaml'
    readonly PROVISIONING_RETRY_SECONDS='21600'
    readonly POST_ALLOCATION_LIMIT_SECONDS='43200'
    readonly WALL_LIMIT_SECONDS='68400'
    declare -ar REVIEWED_AVAILABILITY_ZONES=(
      'us-east-1a' 'us-east-1b' 'us-east-1c' 'us-east-1d'
    )
    declare -ar REVIEWED_SUBNET_IDS=(
      "${REVIEWED_SUBNET_A}" "${REVIEWED_SUBNET_B}"
      "${REVIEWED_SUBNET_C}" "${REVIEWED_SUBNET_D}"
    )
    ;;
  infra/skypilot/a100_40gb_reward_hack_microbatch4.yaml)
    readonly HARDWARE_PROFILE='p4d-a100-40gb-microbatch4'
    readonly INSTANCE_TYPE='p4d.24xlarge'
    readonly EXPECTED_GPU_MEMORY_API_MIB='40960'
    readonly REVIEWED_TASK='infra/skypilot/a100_40gb_reward_hack_microbatch4.yaml'
    readonly REVIEWED_MANIFEST='infra/skypilot/reviewed_manifest_p4d_microbatch4.json'
    readonly REVIEWED_RUN_TOKEN_FILE='infra/skypilot/reviewed_run_token_p4d_microbatch4.txt'
    readonly REVIEWED_IAM_POLICY='infra/skypilot/iam/codex_skypilot_a100_40gb_permissions.json'
    readonly RENDERED_TASK_RELATIVE='infra/skypilot/a100_40gb_reward_hack_microbatch4.rendered.yaml'
    readonly PROVISIONING_RETRY_SECONDS='21600'
    readonly POST_ALLOCATION_LIMIT_SECONDS='43200'
    readonly WALL_LIMIT_SECONDS='68400'
    declare -ar REVIEWED_AVAILABILITY_ZONES=(
      'us-east-1a' 'us-east-1b' 'us-east-1c' 'us-east-1d'
    )
    declare -ar REVIEWED_SUBNET_IDS=(
      "${REVIEWED_SUBNET_A}" "${REVIEWED_SUBNET_B}"
      "${REVIEWED_SUBNET_C}" "${REVIEWED_SUBNET_D}"
    )
    ;;
  *)
    echo 'Launch blocked: task path is not one of the reviewed hardware profiles.' >&2
    exit 2
    ;;
esac
if [[ ! "${CLUSTER_NAME}" =~ ^codex-sky-[a-z0-9-]+$ ]]; then
  echo 'Cluster name must match codex-sky-[a-z0-9-]+' >&2
  exit 2
fi
if [[ ! -f "${REVIEWED_RUN_TOKEN_FILE}" ]]; then
  echo "Launch blocked: ${REVIEWED_RUN_TOKEN_FILE} is missing." >&2
  exit 2
fi
REVIEWED_RUN_TOKEN="$(tr -d '\n' < "${REVIEWED_RUN_TOKEN_FILE}")"
if [[ "${CLUSTER_NAME}" != "${REVIEWED_RUN_TOKEN}" ]]; then
  echo "Launch blocked: reviewed run token is ${REVIEWED_RUN_TOKEN}." >&2
  exit 2
fi
if [[ ! -f "${REVIEWED_MANIFEST}" ]]; then
  echo "Launch blocked: ${REVIEWED_MANIFEST} has not been generated and reviewed." >&2
  exit 2
fi

MANIFEST_SHA256="$(sha256sum "${REVIEWED_MANIFEST}" | awk '{print $1}')"
EXPECTED_APPROVAL="${APPROVAL_PREFIX}${MANIFEST_SHA256}:${CLUSTER_NAME}"
if [[ "${CODEX_AWS_LAUNCH_APPROVED:-}" != "${EXPECTED_APPROVAL}" ]]; then
  echo 'Launch blocked: a fresh approval bound to the reviewed manifest is absent.' >&2
  echo "Required only after review: CODEX_AWS_LAUNCH_APPROVED=${EXPECTED_APPROVAL}" >&2
  exit 2
fi
TEMP_DIR="$(mktemp -d)"
BASELINE_IDS_FILE="${TEMP_DIR}/baseline-profile.txt"
AUTHORIZED_IDS_FILE="${TEMP_DIR}/run-tagged-profile.txt"
VOLUME_IDS_FILE="${TEMP_DIR}/volumes.txt"
TAGGED_VOLUME_IDS_FILE="${TEMP_DIR}/tagged-volumes.txt"
REQUEST_IDS_FILE="${TEMP_DIR}/request-ids.txt"
RENDERED_TASK_YAML="${TEMP_DIR}/$(basename "${RENDERED_TASK_RELATIVE}")"
TEMP_WANDB_API_KEY_FILE="${TEMP_DIR}/wandb_api_key"
CLEANUP_STARTED_SENTINEL="${TEMP_DIR}/cleanup-started"
TEARDOWN_COMPLETE_SENTINEL="${TEMP_DIR}/teardown-complete"
TEARDOWN_ARMED_SENTINEL="${TEMP_DIR}/teardown-armed"
TEARDOWN_READY_SENTINEL="${TEMP_DIR}/teardown-ready"
LAUNCH_INTENT_SENTINEL="${TEMP_DIR}/launch-intent"
TEARDOWN_WATCHDOG_LOG="${TEMP_DIR}/teardown-watchdog.log"
STAGED_WORKDIR="${TEMP_DIR}/reviewed-workdir"
touch "${BASELINE_IDS_FILE}" "${AUTHORIZED_IDS_FILE}" "${VOLUME_IDS_FILE}" \
  "${TAGGED_VOLUME_IDS_FILE}" "${REQUEST_IDS_FILE}"

LAUNCH_ATTEMPTED='false'
TASK_SUCCEEDED='false'
WALL_WATCHDOG_PID=''
TEARDOWN_WATCHDOG_PID=''
LAUNCHER_PID="$$"
ALLOCATION_START_SECONDS=''
HELPER_ROOT="${PWD}"
SKY_CONFIG_PATH="${PWD}/${SKY_CONFIG}"

bounded() {
  local seconds="$1" attempts="$2"
  shift 2
  python3 "${HELPER_ROOT}/infra/skypilot/bounded_command.py" \
    --timeout "${seconds}" --attempts "${attempts}" --backoff 2 -- "$@"
}

bounded_stream() {
  local seconds="$1" attempts="$2"
  shift 2
  python3 "${HELPER_ROOT}/infra/skypilot/bounded_command.py" \
    --timeout "${seconds}" --attempts "${attempts}" --backoff 2 --stream -- "$@"
}

aws_read() {
  bounded 45 3 aws "$@"
}

aws_mutate() {
  bounded 90 3 aws "$@"
}

json_matches_reviewed_file() {
  local reviewed_file="$1"
  python3 -c '
import json, sys
from pathlib import Path
reviewed = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
live = json.load(sys.stdin)
raise SystemExit(0 if live == reviewed else 1)
' "${reviewed_file}"
}

json_matches_exact_list() {
  python3 -c '
import json, sys
live = json.load(sys.stdin)
expected = sys.argv[1:]
raise SystemExit(0 if sorted(live) == sorted(expected) else 1)
' "$@"
}

sky_read() {
  bounded 90 3 sky "$@"
}

sky_mutate() {
  bounded 600 2 sky "$@"
}

verify_live_sky_api_version() {
  local payload
  payload="$(sky_read api info --config "${SKY_CONFIG_PATH}" --output json)" || return 1
  python3 -c '
import ipaddress
import json
import sys
from urllib.parse import urlparse

try:
    payload = json.load(sys.stdin)
except (json.JSONDecodeError, UnicodeDecodeError):
    raise SystemExit(2)
expected_version, expected_commit, expected_api = sys.argv[1:]
client = payload.get("client", {})
server = payload.get("server", {})
url = urlparse(str(server.get("url", "")))
try:
    loopback = ipaddress.ip_address(url.hostname or "").is_loopback
except ValueError:
    loopback = url.hostname == "localhost"
valid = (
    client.get("version") == expected_version
    and client.get("commit") == expected_commit
    and server.get("status") == "healthy"
    and server.get("version") == expected_version
    and server.get("commit") == expected_commit
    and str(server.get("api_version")) == expected_api
    and url.scheme == "http"
    and loopback
)
if not valid:
    raise SystemExit(f"unexpected live Sky API identity: {payload!r}")
' "${SKY_SEMVER}" "${SKY_COMMIT}" "${SKY_API_VERSION}" <<<"${payload}"
}

verify_live_sky_aws_identity() {
  local expected_role_id="$1" payload
  payload="$(sky_read check --config "${SKY_CONFIG_PATH}" --verbose aws)" || return 1
  python3 -c '
import re
import sys

payload, role_id, session_name, account = sys.argv[1:]
payload = re.sub(r"\x1b\[[0-9;]*m", "", payload)
expected = f"Activated account: {role_id}:{session_name} [account={account}]"
lines = [line.strip() for line in payload.splitlines()]
if lines.count(expected) != 1:
    raise SystemExit(f"Sky API server did not activate the reviewed AWS role: {lines!r}")
' "${payload}" "${expected_role_id}" "${AWS_PROFILE_NAME}" "${EXPECTED_ACCOUNT}"
}

verify_reviewed_aws_launch_dependencies() {
  local security_group_payload subnet_payload image_payload
  security_group_payload="$(aws_read ec2 describe-security-groups --region "${REGION}" \
    --group-ids "${REVIEWED_SECURITY_GROUP_ID}" --output json)" || return 1
  python3 -c '
import json, sys

payload = json.load(sys.stdin)
groups = payload.get("SecurityGroups", [])
if len(groups) != 1:
    raise SystemExit("reviewed security group was not uniquely resolved")
group = groups[0]
expected_id, expected_name, expected_vpc, expected_owner = sys.argv[1:]
if (
    group.get("GroupId") != expected_id
    or group.get("GroupName") != expected_name
    or group.get("VpcId") != expected_vpc
    or group.get("OwnerId") != expected_owner
    or {tag.get("Key"): tag.get("Value") for tag in group.get("Tags", [])}
    != {"skypilot": "true"}
):
    raise SystemExit("reviewed security-group identity or tags changed")

def normalized(rule):
    return (
        rule.get("IpProtocol"),
        rule.get("FromPort"),
        rule.get("ToPort"),
        tuple(sorted((pair.get("UserId"), pair.get("GroupId")) for pair in rule.get("UserIdGroupPairs", []))),
        tuple(sorted(item.get("CidrIp") for item in rule.get("IpRanges", []))),
        tuple(sorted(item.get("CidrIpv6") for item in rule.get("Ipv6Ranges", []))),
        tuple(sorted(item.get("PrefixListId") for item in rule.get("PrefixListIds", []))),
    )

expected_ingress = {
    ("-1", None, None, ((expected_owner, expected_id),), (), (), ()),
    ("tcp", 22, 22, (), ("0.0.0.0/0",), (), ()),
}

expected_egress = {("-1", None, None, (), ("0.0.0.0/0",), (), ())}
if {normalized(rule) for rule in group.get("IpPermissions", [])} != expected_ingress:
    raise SystemExit("reviewed security-group ingress changed")
if {normalized(rule) for rule in group.get("IpPermissionsEgress", [])} != expected_egress:
    raise SystemExit("reviewed security-group egress changed")
' "${REVIEWED_SECURITY_GROUP_ID}" "${REVIEWED_SECURITY_GROUP_NAME}" \
    "${REVIEWED_VPC_ID}" "${EXPECTED_ACCOUNT}" <<<"${security_group_payload}" || return 1

  subnet_payload="$(aws_read ec2 describe-subnets --region "${REGION}" \
    --subnet-ids "${REVIEWED_SUBNET_IDS[@]}" --output json)" || return 1
  python3 -c '
import json, sys

subnets = json.load(sys.stdin).get("Subnets", [])
expected_owner, expected_vpc = sys.argv[1:3]
count = int(sys.argv[3])
subnet_ids = sys.argv[4:4 + count]
zones = sys.argv[4 + count:]
if len(subnet_ids) != count or len(zones) != count:
    raise SystemExit("reviewed subnet arguments are inconsistent")
expected = dict(zip(subnet_ids, zones, strict=True))
actual = {}
for subnet in subnets:
    if (
        subnet.get("OwnerId") != expected_owner
        or subnet.get("VpcId") != expected_vpc
        or subnet.get("State") != "available"
        or subnet.get("DefaultForAz") is not True
        or subnet.get("MapPublicIpOnLaunch") is not True
    ):
        raise SystemExit("reviewed subnet properties changed")
    actual[subnet.get("SubnetId")] = subnet.get("AvailabilityZone")
if actual != expected:
    raise SystemExit("reviewed subnet identities changed")
' "${EXPECTED_ACCOUNT}" "${REVIEWED_VPC_ID}" "${#REVIEWED_SUBNET_IDS[@]}" \
    "${REVIEWED_SUBNET_IDS[@]}" "${REVIEWED_AVAILABILITY_ZONES[@]}" \
    <<<"${subnet_payload}" || return 1

  image_payload="$(aws_read ec2 describe-images --region "${REGION}" \
    --image-ids "${REVIEWED_AMI_ID}" --output json)" || return 1
  python3 -c '
import json, sys

images = json.load(sys.stdin).get("Images", [])
if len(images) != 1:
    raise SystemExit("reviewed AMI was not uniquely resolved")
image = images[0]
if (
    image.get("ImageId") != sys.argv[1]
    or image.get("OwnerId") != "123456789012"
    or image.get("Name") != "skypilot-aws-gpu-ubuntu-241104"
    or image.get("State") != "available"
    or image.get("Public") is not True
    or image.get("Architecture") != "x86_64"
    or image.get("RootDeviceType") != "ebs"
):
    raise SystemExit("reviewed SkyPilot AMI identity or state changed")
' "${REVIEWED_AMI_ID}" <<<"${image_payload}" || return 1
}

verify_live_profile_price() {
  [[ "${INSTANCE_TYPE}" == 'p4d.24xlarge' ]] || return 0
  local payload
  payload="$(aws_read pricing get-products --service-code AmazonEC2 --region us-east-1 \
    --filters \
      Type=TERM_MATCH,Field=instanceType,Value=p4d.24xlarge \
      "Type=TERM_MATCH,Field=location,Value=US East (N. Virginia)" \
      Type=TERM_MATCH,Field=operatingSystem,Value=Linux \
      Type=TERM_MATCH,Field=tenancy,Value=Shared \
      Type=TERM_MATCH,Field=preInstalledSw,Value=NA \
      Type=TERM_MATCH,Field=capacitystatus,Value=Used \
    --max-results 100 --output json)" || return 1
  python3 "${HELPER_ROOT}/infra/skypilot/verify_live_aws_price.py" \
    --snapshot "${HELPER_ROOT}/infra/skypilot/aws_price_p4d_us_east_1.json" \
    --price-cap 21.96 <<<"${payload}" >/dev/null
}

verify_live_instance_shape() {
  local payload
  payload="$(aws_read ec2 describe-instance-types --region "${REGION}" \
    --instance-types "${INSTANCE_TYPE}" --output json)" || return 1
  python3 -c '
import json, sys

payload = json.load(sys.stdin)
expected_type, expected_memory = sys.argv[1], int(sys.argv[2])
rows = payload.get("InstanceTypes", [])
if len(rows) != 1:
    raise SystemExit(2)
row = rows[0]
gpus = row.get("GpuInfo", {}).get("Gpus", [])
valid = (
    row.get("InstanceType") == expected_type
    and row.get("VCpuInfo", {}).get("DefaultVCpus") == 96
    and len(gpus) == 1
    and gpus[0].get("Name") == "A100"
    and gpus[0].get("Manufacturer") == "NVIDIA"
    and gpus[0].get("Count") == 8
    and gpus[0].get("MemoryInfo", {}).get("SizeInMiB") == expected_memory
)
raise SystemExit(0 if valid else 3)
' "${INSTANCE_TYPE}" "${EXPECTED_GPU_MEMORY_API_MIB}" <<<"${payload}"
}

ensure_reviewed_local_sky_api() {
  # Starting the loopback control plane is a local-only prerequisite. It is
  # done before launch intent is recorded or cloud cleanup is authorized.
  if ! verify_live_sky_api_version >/dev/null 2>&1; then
    bounded 120 1 sky api start --host 127.0.0.1
  fi
  verify_live_sky_api_version
}

cleanup_local_files() {
  if [[ -d "${STAGED_WORKDIR}" ]]; then
    chmod -R u+w "${STAGED_WORKDIR}" 2>/dev/null || true
    rm -rf "${STAGED_WORKDIR}"
  fi
  rm -f \
    "${BASELINE_IDS_FILE}" "${AUTHORIZED_IDS_FILE}" "${VOLUME_IDS_FILE}" \
    "${TAGGED_VOLUME_IDS_FILE}" "${REQUEST_IDS_FILE}" "${RENDERED_TASK_YAML}" \
    "${TEMP_WANDB_API_KEY_FILE}" \
    "${TEARDOWN_WATCHDOG_LOG}" \
    "${TEMP_DIR}/untagged-volumes.txt" \
    "${CLEANUP_STARTED_SENTINEL}" "${TEARDOWN_COMPLETE_SENTINEL}" \
    "${TEARDOWN_ARMED_SENTINEL}" "${TEARDOWN_READY_SENTINEL}" \
    "${LAUNCH_INTENT_SENTINEL}"
  rmdir "${TEMP_DIR}" 2>/dev/null || true
}

SKY_PYTHON="$(head -n1 "$(command -v sky)" | sed 's/^#!//')"
if [[ ! -x "${SKY_PYTHON}" ]]; then
  echo 'Launch blocked: cannot resolve the pinned SkyPilot Python interpreter.' >&2
  cleanup_local_files
  exit 2
fi
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  echo 'Launch blocked: unset ambient WANDB_API_KEY; this task accepts only the reviewed local key file.' >&2
  cleanup_local_files
  exit 2
fi
if [[ ! -e "${WANDB_API_KEY_FILE}" ]]; then
  echo "Launch blocked: paste a rotated W&B key into ${WANDB_API_KEY_FILE}." >&2
  cleanup_local_files
  exit 2
fi
if ! python3 infra/skypilot/read_wandb_secret.py \
  --path "${WANDB_API_KEY_FILE}" \
  --reject-sha256-file "${REJECTED_WANDB_KEY_HASHES}" \
  --copy-to "${TEMP_WANDB_API_KEY_FILE}"; then
  echo 'Launch blocked: the local W&B credential is invalid or was previously exposed.' >&2
  cleanup_local_files
  exit 2
fi
"${SKY_PYTHON}" infra/skypilot/validate_reviewed_launch.py \
  --task "${TASK_YAML}" --manifest "${REVIEWED_MANIFEST}" \
  --approval-digest "${MANIFEST_SHA256}" --run-token "${CLUSTER_NAME}" \
  --rendered-output "${RENDERED_TASK_YAML}" \
  --wandb-secret-path "${TEMP_WANDB_API_KEY_FILE}"
if [[ "$(bounded 30 1 sky --version)" != "${SKY_VERSION}" ]]; then
  echo "Launch blocked: expected ${SKY_VERSION}." >&2
  cleanup_local_files
  exit 2
fi

# This exact config and a local API endpoint apply to every Sky command,
# including the live server-version check below.
export SKYPILOT_CONFIG="${SKY_CONFIG_PATH}"
export SKYPILOT_GLOBAL_CONFIG="${SKY_CONFIG_PATH}"
export SKYPILOT_PROJECT_CONFIG="${SKY_CONFIG_PATH}"
if ! ensure_reviewed_local_sky_api; then
  echo 'Launch blocked: live SkyPilot API server version/commit/API level is not pinned.' >&2
  cleanup_local_files
  exit 2
fi

# The reviewed minimal config prevents ~/.sky/config.yaml, .sky.yaml, and
# ambient SkyPilot configuration from changing the effective task.

start_wall_watchdog() {
  (
    local elapsed=0 signalled='false'
    while [[ ! -f "${TEARDOWN_COMPLETE_SENTINEL}" ]]; do
      sleep 30
      elapsed=$((elapsed + 30))
      if [[ "${elapsed}" -ge "${WALL_LIMIT_SECONDS}" && "${signalled}" == 'false' ]]; then
        signalled='true'
        if [[ ! -f "${CLEANUP_STARTED_SENTINEL}" ]]; then
          echo "Overall $((WALL_LIMIT_SECONDS / 3600))-hour wall limit reached; handing control to cleanup." >&2
          kill -TERM "${LAUNCHER_PID}" 2>/dev/null || true
        fi
      fi
    done
  ) &
  WALL_WATCHDOG_PID=$!
}

start_teardown_watchdog() {
  local attempt watchdog_pid
  local -a watchdog_command
  rm -f "${TEARDOWN_READY_SENTINEL}"
  watchdog_command=(python3 "${HELPER_ROOT}/infra/skypilot/teardown_watchdog.py" \
    --parent-pid "${LAUNCHER_PID}" --region "${REGION}" \
    --instance-type "${INSTANCE_TYPE}" \
    --hardware-profile "${HARDWARE_PROFILE}" \
    --sky-config "${SKY_CONFIG_PATH}" \
    --bucket "${BUCKET}" \
    --s3-verifier "${HELPER_ROOT}/infra/skypilot/verify_s3_after_termination.py" \
    --volume-ids-file "${VOLUME_IDS_FILE}" \
    --sensitive-file "${TEMP_WANDB_API_KEY_FILE}" \
    --launch-intent-sentinel "${LAUNCH_INTENT_SENTINEL}" \
    --arm-sentinel "${TEARDOWN_ARMED_SENTINEL}" \
    --ready-sentinel "${TEARDOWN_READY_SENTINEL}" \
    --run-token "${CLUSTER_NAME}" --complete-sentinel "${TEARDOWN_COMPLETE_SENTINEL}" \
    --poll-seconds 30)
  if [[ "${POST_ALLOCATION_LIMIT_SECONDS}" -gt 0 ]]; then
    watchdog_command+=(--post-allocation-limit-seconds "${POST_ALLOCATION_LIMIT_SECONDS}")
  fi
  watchdog_pid="$(python3 "${HELPER_ROOT}/infra/skypilot/spawn_teardown_watchdog.py" \
    --log-file "${TEARDOWN_WATCHDOG_LOG}" -- "${watchdog_command[@]}")" || {
      echo 'Launch blocked: failed to spawn the independent teardown watchdog.' >&2
      return 1
    }
  if [[ ! "${watchdog_pid}" =~ ^[0-9]+$ ]]; then
    echo 'Launch blocked: teardown watchdog returned an invalid PID.' >&2
    return 1
  fi
  TEARDOWN_WATCHDOG_PID="${watchdog_pid}"
  for attempt in $(seq 1 50); do
    [[ -f "${TEARDOWN_READY_SENTINEL}" ]] && return 0
    kill -0 "${TEARDOWN_WATCHDOG_PID}" 2>/dev/null || break
    sleep 0.1
  done
  echo 'Launch blocked: independent teardown watchdog did not become ready.' >&2
  return 1
}

arm_teardown_watchdog() {
  if [[ -z "${TEARDOWN_WATCHDOG_PID}" ]] || ! kill -0 "${TEARDOWN_WATCHDOG_PID}" 2>/dev/null; then
    start_teardown_watchdog
  fi
  touch "${TEARDOWN_ARMED_SENTINEL}"
  # The independent teardown watchdog is active before the run watchdog is
  # retired, so cleanup never has a gap with no bounded EC2 terminator.
  if [[ -n "${WALL_WATCHDOG_PID}" ]]; then
    kill "${WALL_WATCHDOG_PID}" 2>/dev/null || true
    wait "${WALL_WATCHDOG_PID}" 2>/dev/null || true
    WALL_WATCHDOG_PID=''
  fi
}

record_current_profile_ids() {
  local output_file="$1" payload
  payload="$(aws_read ec2 describe-instances \
    --region "${REGION}" \
    --filters "Name=instance-type,Values=${INSTANCE_TYPE}" \
    --query 'Reservations[].Instances[].InstanceId' --output json)" || return 1
  python3 -c 'import json,sys; print("\n".join(sorted(set(json.load(sys.stdin)))))' \
    <<<"${payload}" > "${output_file}"
}

api_rows_for_cluster() {
  local attempt payload filtered
  for attempt in 1 2 3; do
    if payload="$(sky_read api status --config "${SKY_CONFIG_PATH}" \
      --all-status --verbose --limit all --output json)" && \
      filtered="$(python3 -c '
import json, sys
try:
    rows = json.load(sys.stdin)
except (json.JSONDecodeError, UnicodeDecodeError):
    raise SystemExit(2)
if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
    raise SystemExit(2)
matched = []
for row in rows:
    if row.get("cluster_name") != sys.argv[1]:
        continue
    request_id = row.get("request_id")
    status = row.get("status")
    if not isinstance(request_id, str) or not request_id:
        raise SystemExit(2)
    if not isinstance(status, str) or not status:
        raise SystemExit(2)
    matched.append({
        "request_id": request_id,
        "name": row.get("name"),
        "status": status,
        "cluster_name": row.get("cluster_name"),
        "finished_at": row.get("finished_at"),
    })
print(json.dumps(matched))
' "${CLUSTER_NAME}" <<<"${payload}" 2>/dev/null)"; then
      printf '%s\n' "${filtered}"
      return 0
    fi
    [[ "${attempt}" -eq 3 ]] || sleep 2
  done
  echo 'Sky API status was empty, malformed, or ambiguous after three bounded attempts; failing closed.' >&2
  return 1
}

refresh_request_ids() {
  local payload temporary
  temporary="${TEMP_DIR}/request-ids.new"
  payload="$(api_rows_for_cluster)" || return 1
  python3 -c '
import json, sys
rows = json.load(sys.stdin)
for row in rows:
    request_id = row.get("request_id")
    if request_id:
        print(request_id)
' <<<"${payload}" > "${temporary}" || return 1
  cat "${temporary}" >> "${REQUEST_IDS_FILE}"
  sort -u -o "${REQUEST_IDS_FILE}" "${REQUEST_IDS_FILE}"
  rm -f "${temporary}"
}

request_rows_terminal() {
  local payload
  payload="$(api_rows_for_cluster)" || return 1
  python3 -c '
import json, sys
rows = json.load(sys.stdin)
terminal = {"SUCCEEDED", "FAILED", "CANCELLED"}
if not rows:
    raise SystemExit(2)
raise SystemExit(0 if all(row.get("status") in terminal for row in rows) else 3)
' <<<"${payload}"
}

cancel_nonterminal_requests() {
  local payload request_id
  payload="$(api_rows_for_cluster)" || return 1
  while IFS= read -r request_id; do
    [[ -n "${request_id}" ]] || continue
    bounded 90 3 sky api cancel --config "${SKY_CONFIG_PATH}" --yes "${request_id}" >/dev/null || return 1
  done < <(python3 -c '
import json, sys
terminal = {"SUCCEEDED", "FAILED", "CANCELLED"}
for row in json.load(sys.stdin):
    if row.get("request_id") and row.get("status") not in terminal:
        print(row["request_id"])
' <<<"${payload}")
}

exact_run_instances_json() {
  local payload
  payload="$(aws_read ec2 describe-instances \
    --region "${REGION}" \
    --filters \
      "Name=tag:${RUN_TAG_KEY},Values=${CLUSTER_NAME}" \
    --query 'Reservations[].Instances[]' --output json)" || return 1
  python3 -c '
import json, sys
items = json.load(sys.stdin)
if not isinstance(items, list):
    raise SystemExit(2)
expected_type, tag_key, tag_value = sys.argv[1:]
matched = []
for item in items:
    if not isinstance(item, dict):
        continue
    tags = {tag.get("Key"): tag.get("Value") for tag in item.get("Tags", [])}
    if tags.get(tag_key) == tag_value:
        if item.get("InstanceType") != expected_type:
            raise SystemExit(3)
        matched.append(item)
print(json.dumps(matched))
' "${INSTANCE_TYPE}" "${RUN_TAG_KEY}" "${CLUSTER_NAME}" <<<"${payload}"
}

discover_and_capture_instances() {
  local payload temporary
  payload="$(exact_run_instances_json)" || return 1
  temporary="${TEMP_DIR}/instances.txt"
  python3 -c 'import json,sys; print("\n".join(sorted({x["InstanceId"] for x in json.load(sys.stdin)})))' \
    <<<"${payload}" > "${temporary}" || return 1
  cat "${temporary}" >> "${AUTHORIZED_IDS_FILE}"
  sort -u -o "${AUTHORIZED_IDS_FILE}" "${AUTHORIZED_IDS_FILE}"
  python3 -c '
import json, sys
values = set()
for instance in json.load(sys.stdin):
    for mapping in instance.get("BlockDeviceMappings", []):
        volume_id = mapping.get("Ebs", {}).get("VolumeId")
        if isinstance(volume_id, str) and volume_id.startswith("vol-"):
            values.add(volume_id)
print("\n".join(sorted(values)))
' <<<"${payload}" >> "${VOLUME_IDS_FILE}" || return 1
  sort -u -o "${VOLUME_IDS_FILE}" "${VOLUME_IDS_FILE}"
  rm -f "${temporary}"
}

active_run_instance_ids() {
  local payload
  payload="$(exact_run_instances_json)" || return 1
  python3 -c '
import json, sys
terminal = {"shutting-down", "terminated"}
for item in json.load(sys.stdin):
    instance_id = item.get("InstanceId", "")
    if item.get("State", {}).get("Name") not in terminal and instance_id.startswith("i-"):
        print(instance_id)
' <<<"${payload}"
}

launch_request_status() {
  local request_id="$1" payload
  payload="$(api_rows_for_cluster)" || return 1
  python3 -c '
import json, sys
rows = [row for row in json.load(sys.stdin) if row.get("request_id") == sys.argv[1]]
if len(rows) != 1:
    raise SystemExit(2)
status = rows[0].get("status")
if not isinstance(status, str) or not status:
    raise SystemExit(2)
print(status)
' "${request_id}" <<<"${payload}"
}

wait_for_capacity() {
  local request_id="$1" start_seconds="${SECONDS}" deadline
  local next_report=300 ec2_failures=0 sky_failures=0
  local instance_ids status elapsed
  deadline=$((SECONDS + PROVISIONING_RETRY_SECONDS))
  echo "Waiting up to $((PROVISIONING_RETRY_SECONDS / 60)) minutes for exact-tagged ${INSTANCE_TYPE} capacity."

  while [[ "${SECONDS}" -lt "${deadline}" ]]; do
    if instance_ids="$(active_run_instance_ids)"; then
      ec2_failures=0
      if [[ -n "${instance_ids}" ]]; then
        discover_and_capture_instances || return 1
        echo "AWS capacity acquired: ${instance_ids//$'\n'/, }."
        return 0
      fi
    else
      ec2_failures=$((ec2_failures + 1))
      if [[ "${ec2_failures}" -ge 3 ]]; then
        echo 'Capacity wait failed closed after three consecutive EC2 discovery failures.' >&2
        return 1
      fi
    fi

    if status="$(launch_request_status "${request_id}")"; then
      sky_failures=0
      case "${status}" in
        FAILED|CANCELLED)
          echo "SkyPilot capacity request became terminal with status ${status}." >&2
          return 1
          ;;
        SUCCEEDED)
          echo 'SkyPilot launch request completed before an active instance was observed.'
          return 0
          ;;
      esac
    else
      sky_failures=$((sky_failures + 1))
      if [[ "${sky_failures}" -ge 3 ]]; then
        echo 'Capacity wait failed closed after three consecutive Sky request-discovery failures.' >&2
        return 1
      fi
    fi

    elapsed=$((SECONDS - start_seconds))
    if [[ "${elapsed}" -ge "${next_report}" ]]; then
      echo "Still waiting for ${INSTANCE_TYPE} capacity ($((elapsed / 60)) minutes elapsed)."
      next_report=$((next_report + 300))
    fi
    sleep "${PROVISIONING_POLL_SECONDS}"
  done

  echo "Capacity retry deadline reached after $((PROVISIONING_RETRY_SECONDS / 60)) minutes." >&2
  return 124
}

resolve_exact_job_record() {
  local payload record attempt
  for attempt in $(seq 1 3); do
    if payload="$(bounded 15 1 sky queue --config "${SKY_CONFIG_PATH}" \
      --output json "${CLUSTER_NAME}")"; then
      if record="$(python3 "${HELPER_ROOT}/infra/skypilot/resolve_remote_job.py" \
        parse-queue --cluster "${CLUSTER_NAME}" \
        --job-id "${EXPECTED_REMOTE_JOB_ID}" \
        --job-name "${EXPECTED_REMOTE_JOB_NAME}" <<<"${payload}")"; then
        printf '%s\n' "${record}"
        return 0
      fi
    fi
    [[ "${attempt}" -eq 3 ]] || sleep 2
  done
  echo 'Sky queue did not return one valid exact job after bounded retries; using read-only SSH/SQLite fallback.' >&2
  bounded 35 1 python3 "${HELPER_ROOT}/infra/skypilot/resolve_remote_job.py" \
    ssh --cluster "${CLUSTER_NAME}" --job-id "${EXPECTED_REMOTE_JOB_ID}" \
    --timeout 30
}

remote_job_identity() {
  python3 -c '
import json, sys
record = json.load(sys.stdin)
expected = {"job_id", "status", "submitted_at", "start_at", "end_at", "pid", "log_dir", "exit_codes"}
job_id, status = record.get("job_id"), record.get("status")
if set(record) != expected or not isinstance(job_id, int) or not isinstance(status, str):
    raise SystemExit(2)
print(f"{job_id}|{status}")
' <<<"$1"
}

remote_job_status_is_failure() {
  case "$1" in
    FAILED_DRIVER|FAILED|FAILED_SETUP|CANCELLED) return 0 ;;
    *) return 1 ;;
  esac
}

remote_job_status_is_nonterminal() {
  case "$1" in
    INIT|SETTING_UP|PENDING|RUNNING) return 0 ;;
    *) return 1 ;;
  esac
}

all_run_instances_terminated() {
  local payload
  payload="$(exact_run_instances_json)" || return 1
  python3 -c '
import json, sys
items = json.load(sys.stdin)
raise SystemExit(0 if all(x.get("State", {}).get("Name") == "terminated" for x in items) else 1)
' <<<"${payload}"
}

terminate_active_run_instances() {
  local active
  active="$(active_run_instance_ids)" || return 1
  [[ -z "${active}" ]] && return 0
  echo 'Invoking exact run-tag EC2 termination fallback for:' >&2
  printf '%s\n' "${active}" >&2
  # Word splitting is intentional: EC2 instance IDs contain no whitespace.
  # shellcheck disable=SC2086
  aws_mutate ec2 terminate-instances --region "${REGION}" --instance-ids ${active} >/dev/null
}

volume_details() {
  local volume_id="$1" payload
  payload="$(aws_read ec2 describe-volumes --region "${REGION}" \
    --filters "Name=volume-id,Values=${volume_id}" --query 'Volumes' --output json)" || return 1
  python3 -c '
import json, sys
items = json.load(sys.stdin)
if not items:
    print("absent||true")
elif len(items) == 1:
    tags = {tag.get("Key"): tag.get("Value") for tag in items[0].get("Tags", [])}
    state = items[0]["State"]
    owner = tags.get(sys.argv[1], "")
    reviewed_zones = set(sys.argv[2:])
    reviewed_shape = (
        items[0].get("Size") == 300
        and items[0].get("VolumeType") == "gp3"
        and items[0].get("AvailabilityZone") in reviewed_zones
        and items[0].get("Encrypted") is False
    )
    print(f"{state}|{owner}|{str(reviewed_shape).lower()}")
else:
    raise SystemExit(2)
' "${RUN_TAG_KEY}" "${REVIEWED_AVAILABILITY_ZONES[@]}" <<<"${payload}"
}

tag_captured_volumes() {
  local volume_id details state owner reviewed_shape pending
  pending="${TEMP_DIR}/untagged-volumes.txt"
  comm -23 "${VOLUME_IDS_FILE}" "${TAGGED_VOLUME_IDS_FILE}" > "${pending}" || return 1
  while IFS= read -r volume_id; do
    [[ -n "${volume_id}" ]] || continue
    if ! [[ "${volume_id}" =~ ^vol-[0-9a-f]+$ ]]; then
      echo "Refusing malformed captured volume ID: ${volume_id}" >&2
      return 1
    fi
    details="$(volume_details "${volume_id}")" || return 1
    IFS='|' read -r state owner reviewed_shape <<<"${details}"
    if [[ "${state}" == 'absent' || "${owner}" == "${CLUSTER_NAME}" ]]; then
      printf '%s\n' "${volume_id}" >> "${TAGGED_VOLUME_IDS_FILE}"
      sort -u -o "${TAGGED_VOLUME_IDS_FILE}" "${TAGGED_VOLUME_IDS_FILE}"
      continue
    fi
    if [[ "${reviewed_shape}" != 'true' ]]; then
      echo "Refusing unexpected captured EBS shape: ${volume_id}" >&2
      return 1
    fi
    aws_mutate ec2 create-tags --region "${REGION}" --resources "${volume_id}" \
      --tags "Key=${RUN_TAG_KEY},Value=${CLUSTER_NAME}" >/dev/null || return 1
    printf '%s\n' "${volume_id}" >> "${TAGGED_VOLUME_IDS_FILE}"
    sort -u -o "${TAGGED_VOLUME_IDS_FILE}" "${TAGGED_VOLUME_IDS_FILE}"
  done < "${pending}"
  rm -f "${pending}"
}

verify_or_delete_volumes() {
  local volume_id details state owner reviewed_shape attempt verified
  while IFS= read -r volume_id; do
    [[ -n "${volume_id}" ]] || continue
    verified='false'
    for attempt in $(seq 1 60); do
      details="$(volume_details "${volume_id}")" || return 1
      IFS='|' read -r state owner reviewed_shape <<<"${details}"
      if [[ "${state}" == 'absent' ]]; then
        echo "EBS cleanup verified: ${volume_id} is absent."
        verified='true'
        break
      fi
      if [[ "${reviewed_shape}" != 'true' ]]; then
        echo "Refusing unexpected captured EBS shape: ${volume_id}" >&2
        return 1
      fi
      if [[ "${owner}" != "${CLUSTER_NAME}" ]]; then
        aws_mutate ec2 create-tags --region "${REGION}" --resources "${volume_id}" \
          --tags "Key=${RUN_TAG_KEY},Value=${CLUSTER_NAME}" >/dev/null || return 1
        sleep 2
        continue
      fi
      if [[ "${state}" == 'available' ]]; then
        aws_mutate ec2 delete-volume --region "${REGION}" --volume-id "${volume_id}" >/dev/null || return 1
      fi
      sleep 5
    done
    if [[ "${verified}" != 'true' ]]; then
      echo "Cleanup verification failed: ${volume_id} was not proven deleted." >&2
      return 1
    fi
  done < "${VOLUME_IDS_FILE}"
}

verify_no_untagged_profile_delta() {
  local current_file delta_file
  current_file="${TEMP_DIR}/current-profile.txt"
  delta_file="${TEMP_DIR}/untagged-profile-delta.txt"
  record_current_profile_ids "${current_file}" || return 1
  comm -13 "${BASELINE_IDS_FILE}" "${current_file}" | \
    comm -23 - "${AUTHORIZED_IDS_FILE}" > "${delta_file}" || return 1
  if [[ -s "${delta_file}" ]]; then
    echo "Cleanup cannot prove ownership of new untagged ${INSTANCE_TYPE} instances:" >&2
    cat "${delta_file}" >&2
    return 1
  fi
}

verify_s3_after_termination() {
  local args=(
    "${HELPER_ROOT}/infra/skypilot/verify_s3_after_termination.py"
    --bucket "${BUCKET}" --run-token "${CLUSTER_NAME}" --region "${REGION}"
    --hardware-profile "${HARDWARE_PROFILE}"
  )
  if [[ "${TASK_SUCCEEDED}" == 'true' ]]; then
    args+=(--require-final)
  fi
  bounded 3600 1 python3 "${args[@]}"
}

exact_cluster_status_rows() {
  local attempt payload filtered
  for attempt in 1 2 3; do
    if payload="$(sky_read status --config "${SKY_CONFIG_PATH}" --refresh --output json)" && \
      filtered="$(python3 -c '
import json, sys
try:
    rows = json.load(sys.stdin)
except (json.JSONDecodeError, UnicodeDecodeError):
    raise SystemExit(2)
if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
    raise SystemExit(2)
print(json.dumps([
    {
        "name": row.get("name"),
        "cluster_name": row.get("cluster_name"),
        "status": row.get("status"),
    }
    for row in rows
    if row.get("name") == sys.argv[1] or row.get("cluster_name") == sys.argv[1]
]))
' "${CLUSTER_NAME}" <<<"${payload}" 2>/dev/null)"; then
      printf '%s\n' "${filtered}"
      return 0
    fi
    [[ "${attempt}" -eq 3 ]] || sleep 2
  done
  echo 'Sky cluster status was empty or malformed after three bounded attempts; failing closed.' >&2
  return 1
}

stop_teardown_watchdog_after_proof() {
  touch "${TEARDOWN_COMPLETE_SENTINEL}"
  if [[ -n "${TEARDOWN_WATCHDOG_PID}" ]]; then
    local attempt
    for attempt in $(seq 1 60); do
      kill -0 "${TEARDOWN_WATCHDOG_PID}" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "${TEARDOWN_WATCHDOG_PID}" 2>/dev/null; then
      echo 'Cleanup failed closed: detached teardown watchdog did not exit after its completion sentinel.' >&2
      return 1
    fi
    TEARDOWN_WATCHDOG_PID=''
  fi
}

down_cluster() {
  local reason="$1" cleanup_rc=0 attempt=0 requests_seen='false'
  local ec2_stable_scans=0 sky_stable_scans=0 request_count sky_payload cleanup_deadline
  local ec2_proven='false' sky_proven='false' volumes_proven='false' s3_proven='false'
  trap - EXIT INT TERM
  set +e

  if [[ "${LAUNCH_ATTEMPTED}" != 'true' ]]; then
    echo "Cleanup (${reason}): no launch was attempted."
    cleanup_local_files
    return 0
  fi

  touch "${CLEANUP_STARTED_SENTINEL}"
  arm_teardown_watchdog
  echo "Cleanup (${reason}): ${CLUSTER_NAME}"

  if discover_and_capture_instances; then
    terminate_active_run_instances || cleanup_rc=1
    tag_captured_volumes || cleanup_rc=1
    if all_run_instances_terminated; then
      ec2_stable_scans=1
    fi
  else
    cleanup_rc=1
  fi

  # Best-effort Sky cancellation is deliberately separate from the exact-tag
  # EC2 fallback. A dead API server must not suppress instance termination.
  refresh_request_ids || cleanup_rc=1
  cancel_nonterminal_requests || cleanup_rc=1
  sky_mutate down --config "${SKY_CONFIG_PATH}" --yes "${CLUSTER_NAME}" >/dev/null || cleanup_rc=1

  # Up to 30 minutes of fail-closed request discovery closes the late-
  # provisioning race. Every AWS/Sky call inside the loop has its own timeout.
  cleanup_deadline=$((SECONDS + 1800))
  while [[ "${SECONDS}" -lt "${cleanup_deadline}" ]]; do
    attempt=$((attempt + 1))

    # EC2 runs first on every cycle and has no dependency on Sky API success.
    if discover_and_capture_instances; then
      terminate_active_run_instances || cleanup_rc=1
      tag_captured_volumes || cleanup_rc=1
      if all_run_instances_terminated; then
        ec2_stable_scans=$((ec2_stable_scans + 1))
      else
        ec2_stable_scans=0
      fi
    else
      cleanup_rc=1
      ec2_stable_scans=0
    fi

    if refresh_request_ids; then
      request_count="$(wc -l < "${REQUEST_IDS_FILE}" | tr -d ' ')"
      [[ "${request_count}" -gt 0 ]] && requests_seen='true'
      cancel_nonterminal_requests || cleanup_rc=1
      if request_rows_terminal; then
        sky_stable_scans=$((sky_stable_scans + 1))
      else
        sky_stable_scans=0
      fi
    else
      cleanup_rc=1
      sky_stable_scans=0
    fi

    if [[ "${ec2_stable_scans}" -ge 3 && "${sky_stable_scans}" -ge 3 ]]; then
      break
    fi
    sleep 10
  done

  if [[ "${requests_seen}" != 'true' ]]; then
    echo 'Cleanup failed closed: no exact-cluster Sky request ID could be recovered.' >&2
    cleanup_rc=1
  fi
  if [[ "${sky_stable_scans}" -lt 3 ]]; then
    echo 'Cleanup failed closed: exact-cluster Sky requests were not terminal across three scans.' >&2
    cleanup_rc=1
  fi
  if [[ "${ec2_stable_scans}" -lt 3 ]]; then
    echo 'Cleanup failed closed: exact-tag EC2 termination was not proven across three scans.' >&2
    cleanup_rc=1
  fi
  discover_and_capture_instances || cleanup_rc=1
  tag_captured_volumes || cleanup_rc=1
  if [[ "${ec2_stable_scans}" -ge 3 ]] && all_run_instances_terminated; then
    echo 'EC2 cleanup positively verified across three scans: every exact run-tagged instance is terminated (or none was created).'
    ec2_proven='true'
  else
    echo 'Cleanup verification failed: an exact run-tagged instance is not terminated.' >&2
    cleanup_rc=1
  fi
  verify_no_untagged_profile_delta || cleanup_rc=1

  sky_payload="$(exact_cluster_status_rows)"
  if [[ "$?" -ne 0 ]]; then
    cleanup_rc=1
  elif python3 -c 'import json,sys; raise SystemExit(0 if json.load(sys.stdin) else 1)' \
    <<<"${sky_payload}"; then
    echo "Cleanup verification failed: ${CLUSTER_NAME} remains in SkyPilot status." >&2
    cleanup_rc=1
  elif [[ "${requests_seen}" == 'true' && "${sky_stable_scans}" -ge 3 ]]; then
    sky_proven='true'
  fi
  if verify_or_delete_volumes; then
    volumes_proven='true'
  else
    cleanup_rc=1
  fi
  if [[ "${ec2_proven}" == 'true' && "${volumes_proven}" == 'true' ]] && \
    verify_s3_after_termination; then
    s3_proven='true'
  else
    cleanup_rc=1
  fi

  # Retire the independent watchdog only after every cleanup layer is proven.
  if [[ "${ec2_proven}" == 'true' && "${sky_proven}" == 'true' && \
    "${volumes_proven}" == 'true' && "${s3_proven}" == 'true' ]]; then
    stop_teardown_watchdog_after_proof || cleanup_rc=1
  else
    echo 'Independent teardown watchdog remains active for EC2/EBS/S3 protection.' >&2
  fi

  if [[ -f "${TEARDOWN_COMPLETE_SENTINEL}" && -z "${TEARDOWN_WATCHDOG_PID}" ]]; then
    cleanup_local_files
  else
    echo "Preserving teardown watchdog state at ${TEMP_DIR}" >&2
  fi
  return "${cleanup_rc}"
}

on_exit() {
  local rc=$?
  down_cluster "exit=${rc}" || rc=1
  exit "${rc}"
}
on_int() { down_cluster INT || true; exit 130; }
on_term() { down_cluster TERM || true; exit 143; }
trap on_exit EXIT
trap on_int INT
trap on_term TERM

identity_json="$(aws_read sts get-caller-identity --output json)"
caller_account="$(python3 -c 'import json,sys; print(json.load(sys.stdin)["Account"])' <<<"${identity_json}")"
caller_arn="$(python3 -c 'import json,sys; print(json.load(sys.stdin)["Arn"])' <<<"${identity_json}")"
if [[ "${caller_account}" != "${EXPECTED_ACCOUNT}" ]]; then
  echo "Launch blocked: AWS account is ${caller_account}; expected ${EXPECTED_ACCOUNT}." >&2
  exit 2
fi
if [[ "${caller_arn}" != "arn:aws:sts::${EXPECTED_ACCOUNT}:assumed-role/${AWS_LAUNCHER_ROLE}/"* ]]; then
  echo "Launch blocked: expected the dedicated ${AWS_LAUNCHER_ROLE} assumed-role identity." >&2
  exit 2
fi

live_iam_json="$(aws_read iam get-role --role-name "${AWS_LAUNCHER_ROLE}" \
  --query 'Role.AssumeRolePolicyDocument' --output json)"
launcher_role_id="$(aws_read iam get-role --role-name "${AWS_LAUNCHER_ROLE}" \
  --query 'Role.RoleId' --output text)"
if ! [[ "${launcher_role_id}" =~ ^AROA[A-Z0-9]+$ ]]; then
  echo 'Launch blocked: launcher role has an invalid or unavailable stable role ID.' >&2
  exit 2
fi
if ! json_matches_reviewed_file infra/skypilot/iam/codex_skypilot_a100_trust.json \
  <<<"${live_iam_json}"; then
  echo 'Launch blocked: live launcher-role trust differs from the reviewed document.' >&2
  exit 2
fi
live_iam_json="$(aws_read iam get-role-policy --role-name "${AWS_LAUNCHER_ROLE}" \
  --policy-name "${AWS_LAUNCHER_POLICY}" --query 'PolicyDocument' --output json)"
if ! json_matches_reviewed_file "${REVIEWED_IAM_POLICY}" \
  <<<"${live_iam_json}"; then
  echo 'Launch blocked: live launcher-role policy differs from the reviewed document.' >&2
  exit 2
fi
live_iam_json="$(aws_read iam get-user-policy --user-name "${AWS_BOOTSTRAP_USER}" \
  --policy-name "${AWS_BOOTSTRAP_POLICY}" --query 'PolicyDocument' --output json)"
if ! json_matches_reviewed_file \
  infra/skypilot/iam/codex_skypilot_a100_bootstrap_permissions.json \
  <<<"${live_iam_json}"; then
  echo 'Launch blocked: live bootstrap-user policy differs from the reviewed document.' >&2
  exit 2
fi
live_iam_json="$(aws_read iam list-role-policies --role-name "${AWS_LAUNCHER_ROLE}" \
  --query 'PolicyNames' --output json)"
if ! json_matches_exact_list "${AWS_LAUNCHER_POLICY}" <<<"${live_iam_json}"; then
  echo 'Launch blocked: launcher role has an unexpected inline policy.' >&2
  exit 2
fi
live_iam_json="$(aws_read iam list-attached-role-policies --role-name "${AWS_LAUNCHER_ROLE}" \
  --query 'AttachedPolicies[].PolicyArn' --output json)"
if ! json_matches_exact_list <<<"${live_iam_json}"; then
  echo 'Launch blocked: launcher role has an attached managed policy.' >&2
  exit 2
fi
live_iam_json="$(aws_read iam list-user-policies --user-name "${AWS_BOOTSTRAP_USER}" \
  --query 'PolicyNames' --output json)"
if ! json_matches_exact_list "${AWS_BOOTSTRAP_POLICY}" <<<"${live_iam_json}"; then
  echo 'Launch blocked: bootstrap user has an unexpected inline policy.' >&2
  exit 2
fi
live_iam_json="$(aws_read iam list-attached-user-policies --user-name "${AWS_BOOTSTRAP_USER}" \
  --query 'AttachedPolicies[].PolicyArn' --output json)"
if ! json_matches_exact_list <<<"${live_iam_json}"; then
  echo 'Launch blocked: bootstrap user has an attached managed policy.' >&2
  exit 2
fi
live_iam_json="$(aws_read iam list-access-keys --user-name "${AWS_BOOTSTRAP_USER}" \
  --query 'AccessKeyMetadata[].Status' --output json)"
if ! json_matches_exact_list Active <<<"${live_iam_json}"; then
  echo 'Launch blocked: bootstrap user must have exactly one active access key.' >&2
  exit 2
fi

if ! verify_live_sky_aws_identity "${launcher_role_id}"; then
  echo 'Launch blocked: live Sky API server is not using the reviewed AWS role.' >&2
  exit 2
fi
if ! verify_reviewed_aws_launch_dependencies; then
  echo 'Launch blocked: the pinned AMI, subnets, or existing security group changed.' >&2
  exit 2
fi
if ! verify_live_profile_price; then
  echo 'Launch blocked: live P4d On-Demand price differs from the reviewed $21.96 cap.' >&2
  exit 2
fi
if ! verify_live_instance_shape; then
  echo 'Launch blocked: live instance vCPU/GPU shape differs from the reviewed profile.' >&2
  exit 2
fi
quota_value="$(aws_read service-quotas get-service-quota --region "${REGION}" \
  --service-code ec2 --quota-code L-417A185B --query 'Quota.Value' --output text)"
if ! awk -v quota="${quota_value}" 'BEGIN { exit !(quota >= 96) }'; then
  echo "Launch blocked: On-Demand P quota is ${quota_value} vCPUs; 96 are required." >&2
  exit 2
fi

active_p_instances="$(aws_read ec2 describe-instances --region "${REGION}" \
  --filters 'Name=instance-state-name,Values=pending,running' 'Name=instance-type,Values=p*' \
  --query 'Reservations[].Instances[].InstanceId' --output text)"
if [[ -n "${active_p_instances}" && "${active_p_instances}" != 'None' ]]; then
  echo "Launch blocked: active P instances consume the quota: ${active_p_instances}" >&2
  exit 2
fi
profile_offerings="$(aws_read ec2 describe-instance-type-offerings --region "${REGION}" \
  --location-type availability-zone --filters "Name=instance-type,Values=${INSTANCE_TYPE}" \
  --query 'InstanceTypeOfferings[].Location' --output text)"
if ! python3 -c '
import sys
available = set(sys.argv[1].split())
required = set(sys.argv[2:])
raise SystemExit(0 if required <= available else 1)
' "${profile_offerings}" "${REVIEWED_AVAILABILITY_ZONES[@]}"; then
  echo "Launch blocked: ${INSTANCE_TYPE} is not currently offered in all reviewed zones." >&2
  exit 2
fi

existing_cluster_rows="$(api_rows_for_cluster)"
if ! python3 -c 'import json,sys; raise SystemExit(0 if not json.load(sys.stdin) else 1)' \
  <<<"${existing_cluster_rows}"; then
  echo "Launch blocked: the exact cluster name has existing Sky API history: ${CLUSTER_NAME}" >&2
  exit 2
fi
existing_run_marked="$(aws_read ec2 describe-instances --region "${REGION}" \
  --filters "Name=tag:${RUN_TAG_KEY},Values=${CLUSTER_NAME}" \
  --query 'Reservations[].Instances[].InstanceId' --output text)"
if [[ -n "${existing_run_marked}" && "${existing_run_marked}" != 'None' ]]; then
  echo "Launch blocked: run-owner tag is already attached to: ${existing_run_marked}" >&2
  exit 2
fi
bucket_match="$(aws_read s3api list-buckets \
  --query "Buckets[?Name=='${BUCKET}'].Name" --output text)"
if [[ -n "${bucket_match}" && "${bucket_match}" != 'None' ]]; then
  for existing_prefix in \
    "qwen3-4b/no-intervention/${CLUSTER_NAME}/" \
    "qwen3-4b/no-intervention/launches/${CLUSTER_NAME}/"; do
    existing_s3_key="$(aws_read s3api list-objects-v2 --bucket "${BUCKET}" \
      --prefix "${existing_prefix}" --max-keys 1 --region "${REGION}" \
      --query 'Contents[].Key' --output text)"
    if [[ -n "${existing_s3_key}" && "${existing_s3_key}" != 'None' ]]; then
      echo "Launch blocked: run token already has S3 artifacts: ${existing_s3_key}" >&2
      exit 2
    fi
  done
fi

record_current_profile_ids "${BASELINE_IDS_FILE}"
bounded 120 1 python3 infra/skypilot/capture_source_provenance.py \
  --output infra/skypilot/launch_provenance
cp -f "${RENDERED_TASK_YAML}" "infra/skypilot/launch_provenance/$(basename "${RENDERED_TASK_RELATIVE}")"
cp -f "${REVIEWED_MANIFEST}" infra/skypilot/launch_provenance/reviewed_manifest.json
cp -f "${SKY_CONFIG}" infra/skypilot/launch_provenance/reviewed_skypilot_config.yaml

# Revalidate after all preflight time has elapsed, then launch only the private
# manifest-bound snapshot. This prevents checkout edits during Sky upload from
# changing the reviewed run.
"${SKY_PYTHON}" infra/skypilot/validate_reviewed_launch.py \
  --task "${TASK_YAML}" --manifest "${REVIEWED_MANIFEST}" \
  --approval-digest "${MANIFEST_SHA256}" --run-token "${CLUSTER_NAME}" \
  --rendered-output "${RENDERED_TASK_YAML}" \
  --wandb-secret-path "${TEMP_WANDB_API_KEY_FILE}"
bounded 600 1 python3 infra/skypilot/stage_reviewed_workdir.py \
  --manifest "${REVIEWED_MANIFEST}" --destination "${STAGED_WORKDIR}" \
  --provenance infra/skypilot/launch_provenance
cp -f "${RENDERED_TASK_YAML}" \
  "${STAGED_WORKDIR}/${RENDERED_TASK_RELATIVE}"
bounded 30 1 git -C "${STAGED_WORKDIR}" add --force -- \
  "${RENDERED_TASK_RELATIVE}"
HELPER_ROOT="${STAGED_WORKDIR}"
SKY_CONFIG_PATH="${STAGED_WORKDIR}/${SKY_CONFIG}"
export SKYPILOT_CONFIG="${SKY_CONFIG_PATH}"
export SKYPILOT_GLOBAL_CONFIG="${SKY_CONFIG_PATH}"
export SKYPILOT_PROJECT_CONFIG="${SKY_CONFIG_PATH}"

if ! verify_live_sky_api_version; then
  echo 'Launch blocked: live SkyPilot API identity changed during preflight.' >&2
  exit 2
fi
if ! verify_live_sky_aws_identity "${launcher_role_id}"; then
  echo 'Launch blocked: live Sky API server AWS identity changed during preflight.' >&2
  exit 2
fi
start_teardown_watchdog
touch "${LAUNCH_INTENT_SENTINEL}"
LAUNCH_ATTEMPTED='true'
start_wall_watchdog
launch_output="$(
  cd "${STAGED_WORKDIR}"
  SKYPILOT_CONFIG="${STAGED_WORKDIR}/${SKY_CONFIG}" \
  SKYPILOT_GLOBAL_CONFIG="${STAGED_WORKDIR}/${SKY_CONFIG}" \
  SKYPILOT_PROJECT_CONFIG="${STAGED_WORKDIR}/${SKY_CONFIG}" \
    python3 infra/skypilot/bounded_command.py --timeout 600 --attempts 1 -- \
      sky launch --config "${SKY_CONFIG}" --async --yes --retry-until-up \
        --cluster "${CLUSTER_NAME}" --env "RUN_TOKEN=${CLUSTER_NAME}" \
        "${RENDERED_TASK_RELATIVE}"
)"
printf '%s\n' "${launch_output}"
launch_request_id="$(printf '%s\n' "${launch_output}" | \
  sed -nE 's/^Submitted sky\.launch request:[[:space:]]*([[:alnum:]_-]+).*$/\1/p' | head -n1)"
refresh_request_ids
if [[ -z "${launch_request_id}" ]]; then
  request_count="$(wc -l < "${REQUEST_IDS_FILE}" | tr -d ' ')"
  if [[ "${request_count}" -eq 1 ]]; then
    launch_request_id="$(head -n1 "${REQUEST_IDS_FILE}")"
  fi
fi
if [[ -z "${launch_request_id}" ]]; then
  echo 'Could not uniquely recover the exact-cluster Sky launch request ID; entering cleanup.' >&2
  exit 1
fi
printf '%s\n' "${launch_request_id}" >> "${REQUEST_IDS_FILE}"
sort -u -o "${REQUEST_IDS_FILE}" "${REQUEST_IDS_FILE}"

if ! wait_for_capacity "${launch_request_id}"; then
  echo 'Bounded capacity retry ended without acquiring the approved instance.' >&2
  exit 1
fi

ALLOCATION_START_SECONDS="${SECONDS}"

allocation_seconds_remaining() {
  if [[ "${POST_ALLOCATION_LIMIT_SECONDS}" -eq 0 ]]; then
    printf '%s\n' "${WALL_LIMIT_SECONDS}"
    return 0
  fi
  local elapsed=$((SECONDS - ALLOCATION_START_SECONDS))
  local remaining=$((POST_ALLOCATION_LIMIT_SECONDS - elapsed))
  if [[ "${remaining}" -le 0 ]]; then
    echo 'Post-allocation workload deadline reached; entering guarded teardown.' >&2
    return 124
  fi
  printf '%s\n' "${remaining}"
}

remaining_seconds="$(allocation_seconds_remaining)" || exit $?
bounded_stream "${remaining_seconds}" 1 sky api logs --config "${SKY_CONFIG_PATH}" --follow "${launch_request_id}"
final_status_payload="$(sky_read api status --config "${SKY_CONFIG_PATH}" --all-status \
  --output json "${launch_request_id}")"
final_request_status="$(python3 -c '
import json, sys
rows = json.load(sys.stdin)
print(rows[0].get("status", "") if len(rows) == 1 else "")
' <<<"${final_status_payload}")"
if [[ "${final_request_status}" != 'SUCCEEDED' ]]; then
  echo "SkyPilot launch request ended with status ${final_request_status:-unknown}." >&2
  exit 1
fi

remote_job_record="$(resolve_exact_job_record)"
job_identity="$(remote_job_identity "${remote_job_record}")"
IFS='|' read -r remote_job_id remote_job_status <<<"${job_identity}"
echo "Resolved remote job ${remote_job_id} with status ${remote_job_status}."
if remote_job_status_is_failure "${remote_job_status}"; then
  echo "Remote job ${remote_job_id} reached terminal failure ${remote_job_status}; entering guarded teardown." >&2
  exit 1
fi
if [[ "${remote_job_status}" != 'SUCCEEDED' ]]; then
  if ! remote_job_status_is_nonterminal "${remote_job_status}"; then
    echo "Remote job ${remote_job_id} returned unknown status ${remote_job_status}; failing closed." >&2
    exit 1
  fi
  echo "Monitoring remote job ${remote_job_id} through terminal status."
  job_log_rc=0
  remaining_seconds="$(allocation_seconds_remaining)" || exit $?
  if bounded_stream "${remaining_seconds}" 1 sky logs --config "${SKY_CONFIG_PATH}" \
    --follow --tail 0 "${CLUSTER_NAME}" "${remote_job_id}"; then
    :
  else
    job_log_rc=$?
    echo "Sky log follower exited nonzero (${job_log_rc}); resolving the authoritative job status." >&2
  fi
  remote_job_record="$(resolve_exact_job_record)"
  job_identity="$(remote_job_identity "${remote_job_record}")"
  IFS='|' read -r resolved_job_id remote_job_status <<<"${job_identity}"
  if [[ "${resolved_job_id}" != "${remote_job_id}" ]]; then
    echo 'Remote job identity changed; failing closed.' >&2
    exit 1
  fi
fi
if [[ "${remote_job_status}" != 'SUCCEEDED' ]]; then
  echo "Only SUCCEEDED may pass the remote job gate; observed ${remote_job_status}." >&2
  exit 1
fi
echo "Remote job ${remote_job_id}: SUCCEEDED"
TASK_SUCCEEDED='true'
