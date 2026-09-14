#!/usr/bin/env bash
set -euo pipefail
bench_name=${1}; task_name=${2}; ckpt_name=${3}; env_cfg_type=${4}; action_type=${5}; seed=${6}
env_gpu_id=${7}; eval_env=${8}; additional_info=${9}; port=${10}; host=${11:-localhost}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
if [[ "${EVAL_ENV_TYPE:-sim}" == debug ]]; then
    source "${SCRIPT_DIR}/_python.sh"
    umi_select_python "${eval_env}"
    export PYTHONPATH="${XPL_ROOT}/..:${XPL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
    exec "${UMI_PYTHON}" -m XPolicyLab.policy.UMI_DP.debug_client --server "ws://${host}:${port}" --seed "${seed}"
fi
exec bash "${XPL_ROOT}/utils/setup_env_client.sh" "${XPL_ROOT}/utils" "${SCRIPT_DIR}/deploy.yml" \
  "${eval_env}" "${port}" "${bench_name}" "${task_name}" "${env_cfg_type}" UMI_DP "${additional_info}" \
  "${XPL_ROOT}/.." "${seed}" "${env_gpu_id}" "${host}"
