#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 6 ]]; then
  echo "Usage: $0 <bench_name> <ckpt_name> <env_cfg_type> <action_type> [expert_data_num] [raw_task_dirs]" >&2
  exit 2
fi

bench_name=$1
ckpt_name=$2
env_cfg_type=$3
action_type=$4
expert_data_num=${5:-}
raw_task_dirs=${6:-}

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${POLICY_DIR}/../.." && pwd)"
WORKSPACE_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"
LINGBOT_ROOT="${POLICY_DIR}/lingbot_vla_v2"
VENV_DIR="${LINGBOT_VLA2_ENV_DIR:-${WORKSPACE_ROOT}/envs/lingbot-vla2/.venv}"
PYTHON="${VENV_DIR}/bin/python"
setting="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}"
dataset_dir="${LINGBOT_VLA2_DATASET_PATH:-${POLICY_DIR}/data/${setting}}"
source_root="${XPOLICYLAB_DATA_ROOT:-${WORKSPACE_ROOT}/data/${bench_name}}"
stats_path="${LINGBOT_VLA2_NORM_STATS_PATH:-${dataset_dir}/norm_stats.json}"

if [[ "${env_cfg_type}" != "yam_dual" || "${action_type}" != "joint" ]]; then
  echo "LingBot_VLA2 supports env_cfg_type=yam_dual and action_type=joint only." >&2
  exit 2
fi
if [[ ! -x "${PYTHON}" ]]; then
  echo "LingBot-VLA2 Python not found: ${PYTHON}; run install.sh first." >&2
  exit 1
fi

convert_args=(
  "${POLICY_DIR}/process_data.py"
  "${bench_name}" "${ckpt_name}" "${env_cfg_type}" "${action_type}"
  --source-root "${source_root}"
  --output-dir "${dataset_dir}"
  --fps "${LINGBOT_VLA2_DATA_FPS:-30}"
  --height "${LINGBOT_VLA2_IMAGE_HEIGHT:-240}"
  --width "${LINGBOT_VLA2_IMAGE_WIDTH:-320}"
  --mode "${LINGBOT_VLA2_DATA_MODE:-image}"
)
if [[ -n "${expert_data_num}" ]]; then
  convert_args+=("${expert_data_num}")
fi
if [[ -n "${raw_task_dirs}" ]]; then
  convert_args+=(--raw-task-dirs "${raw_task_dirs}")
fi

"${PYTHON}" "${convert_args[@]}"

if [[ "${LINGBOT_VLA2_SKIP_NORM_STATS:-0}" == "1" ]]; then
  echo "[LingBot_VLA2] dataset ready; norm-stat computation skipped by request."
  exit 0
fi

cd "${LINGBOT_ROOT}"
CUDA_VISIBLE_DEVICES="${LINGBOT_VLA2_STATS_GPU_ID:-0}" \
  bash train.sh scripts/compute_norm_stats.py configs/vla/norm_compute/post_data.yaml \
    --data.data_name yam_dual_absolute \
    --data.robot_name yam_dual_absolute \
    --data.train_path "${dataset_dir}" \
    --data.robot_config_root "${POLICY_DIR}/robot_configs" \
    --data.norm_path "${stats_path}" \
    --data.num_workers "${LINGBOT_VLA2_STATS_WORKERS:-8}" \
    --train.chunk_size "${LINGBOT_VLA2_ACTION_HORIZON:-50}" \
    --train.micro_batch_size "${LINGBOT_VLA2_STATS_BATCH_SIZE:-32}" \
    --train.output_dir "${dataset_dir}/norm-compute"

echo "[LingBot_VLA2] dataset=${dataset_dir}"
echo "[LingBot_VLA2] norm_stats=${stats_path}"
