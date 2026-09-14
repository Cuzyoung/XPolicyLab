#!/usr/bin/env bash
set -euo pipefail
bench_name=${1}; task_name=${2}; ckpt_name=${3}; env_cfg_type=${4}; action_type=${5}; seed=${6}
policy_gpu_id=${7}; policy_env=${8}; policy_server_port=${9}; policy_server_host=${10:-localhost}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/_python.sh"
umi_select_python "${policy_env}"
export PYTHONPATH="${XPL_ROOT}/..:${XPL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
exec env CUDA_VISIBLE_DEVICES="${policy_gpu_id}" "${UMI_PYTHON}" "${XPL_ROOT}/setup_policy_server.py" \
  --config_path "${SCRIPT_DIR}/deploy.yml" --overrides \
  port="${policy_server_port}" host="${policy_server_host}" bench_name="${bench_name}" \
  task_name="${task_name}" ckpt_name="${ckpt_name}" env_cfg_type="${env_cfg_type}" \
  action_type="${action_type}" seed="${seed}" policy_name=UMI_DP
