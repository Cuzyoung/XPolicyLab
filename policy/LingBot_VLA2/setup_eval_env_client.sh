#!/usr/bin/env bash
set -euo pipefail

bench_name=$1
task_name=$2
ckpt_name=$3
env_cfg_type=$4
action_type=$5
seed=$6
env_gpu_id=$7
eval_env=$8
additional_info=$9
policy_server_port=${10}
policy_server_ip=${11:-localhost}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BENCH_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"
UTILS_DIR="${XPL_ROOT}/utils"
policy_name="$(basename "${SCRIPT_DIR}")"
yaml_file="${SCRIPT_DIR}/deploy.yml"

echo "[CLIENT] policy=${policy_name}, task=${task_name}, server=${policy_server_ip}:${policy_server_port}"

source "${UTILS_DIR}/resolve_eval_env_type.sh"
eval_env_mode="$(resolve_eval_env_type)"

if [[ "${eval_env_mode}" == "debug" && ( "${eval_env}" == "uv" || "${eval_env}" == */* ) ]]; then
  if [[ "${eval_env}" == "uv" ]]; then
    eval_env_dir="${LINGBOT_VLA2_EVAL_ENV_DIR:-${BENCH_ROOT}/envs/yam}"
  elif [[ "${eval_env}" == /* ]]; then
    eval_env_dir="${eval_env}"
  else
    eval_env_dir="${BENCH_ROOT}/${eval_env}"
  fi

  if [[ -x "${eval_env_dir}/bin/python" ]]; then
    python_bin="${eval_env_dir}/bin/python"
  elif [[ -x "${eval_env_dir}/.venv/bin/python" ]]; then
    python_bin="${eval_env_dir}/.venv/bin/python"
  else
    echo "[CLIENT] Missing uv Python under ${eval_env_dir}" >&2
    exit 1
  fi

  read -r eval_batch protocol < <("${python_bin}" - "${yaml_file}" <<'PY'
import sys

import yaml

with open(sys.argv[1], encoding="utf-8") as stream:
    config = yaml.safe_load(stream)
print(str(config.get("eval_batch", False)).lower(), config.get("protocol", "ws"))
PY
  )

  echo "[CLIENT] EVAL_ENV_TYPE=debug -> uv (${python_bin})"
  export PYTHONPATH="${XPL_ROOT}:${BENCH_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
  exec "${python_bin}" "${XPL_ROOT}/debug_env_client.py" \
    --bench_name "${bench_name}" \
    --task_name "${task_name}" \
    --env_cfg_type "${env_cfg_type}" \
    --policy_name "${policy_name}" \
    --protocol "${protocol}" \
    --host "${policy_server_ip}" \
    --port "${policy_server_port}" \
    --eval_batch "${eval_batch}"
fi

bash "${UTILS_DIR}/setup_env_client.sh" \
  "${UTILS_DIR}" "${yaml_file}" "${eval_env}" "${policy_server_port}" \
  "${bench_name}" "${task_name}" "${env_cfg_type}" "${policy_name}" \
  "${additional_info}" "${BENCH_ROOT}" "${seed}" "${env_gpu_id}" \
  "${policy_server_ip}"
