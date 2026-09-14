#!/usr/bin/env bash
set -euo pipefail
bench_name=${1}; ckpt_name=${2}; env_cfg_type=${3}; action_type=${4}; seed=${5}; gpu_id=${6}
shift 6
[[ "${env_cfg_type}" == tianji_umi || "${env_cfg_type}" == tianji_dual ]] && [[ "${action_type}" == ee ]] || { echo 'UMI_DP requires tianji_umi/tianji_dual and ee' >&2; exit 2; }
: "${UMI_DP_DATASET_DIR:?Set UMI_DP_DATASET_DIR to a native LeRobot v3 export}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
UMI_PYTHON="${UMI_DP_PYTHON:-${XPL_ROOT}/../envs/umi_dp/.venv/bin/python}"
# This helper describes hardware width 16; native relative TCP/rot6d action width is 20.
robot_dim="$(bash "${XPL_ROOT}/utils/get_action_dim.sh" "${XPL_ROOT}/.." "${env_cfg_type}")"
[[ "${robot_dim}" == 16 ]] || { echo "Unexpected robot dimension ${robot_dim}" >&2; exit 2; }
export PYTHONPATH="${XPL_ROOT}/..:${XPL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
run_dir="$("${UMI_PYTHON}" -c 'import sys; from XPolicyLab.utils.checkpoint_resolver import build_run_dir_name; print(build_run_dir_name(dict(zip(("bench_name","ckpt_name","env_cfg_type","action_type","seed"),sys.argv[1:]))))' "${bench_name}" "${ckpt_name}" "${env_cfg_type}" "${action_type}" "${seed}")"
exec env CUDA_VISIBLE_DEVICES="${gpu_id}" "${UMI_PYTHON}" -m XPolicyLab.policy.UMI_DP.training \
  --config-name=train_diffusion_unet_lerobot_pass_ball_20ep \
  hydra.run.dir="${SCRIPT_DIR}/checkpoints/${run_dir}" training.seed="${seed}" training.device=cuda:0 logging.mode=offline "$@"
