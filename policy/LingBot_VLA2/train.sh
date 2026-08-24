#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 6 ]]; then
  echo "Usage: $0 <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id>" >&2
  exit 2
fi

bench_name=$1
ckpt_name=$2
env_cfg_type=$3
action_type=$4
seed=$5
gpu_id=$6

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${POLICY_DIR}/../.." && pwd)"
WORKSPACE_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"
LINGBOT_ROOT="${POLICY_DIR}/lingbot_vla_v2"
VENV_DIR="${LINGBOT_VLA2_ENV_DIR:-${WORKSPACE_ROOT}/envs/lingbot-vla2/.venv}"
PYTHON="${VENV_DIR}/bin/python"
setting="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}-${seed}"
dataset_setting="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}"
dataset_path="${LINGBOT_VLA2_DATASET_PATH:-${POLICY_DIR}/data/${dataset_setting}}"
norm_stats_path="${LINGBOT_VLA2_NORM_STATS_PATH:-${dataset_path}/norm_stats.json}"
checkpoint_dir="${POLICY_DIR}/checkpoints/${setting}"
model_path="${LINGBOT_VLA2_MODEL_PATH:-${WORKSPACE_ROOT}/checkpoints/pretrained/lingbot-vla-v2-6b}"
tokenizer_path="${LINGBOT_VLA2_TOKENIZER_PATH:-${QWEN3_PATH:-}}"
robot_name="${LINGBOT_VLA2_ROBOT_NAME:-yam_dual_absolute}"
robot_config="${POLICY_DIR}/robot_configs/${robot_name}.yaml"
action_horizon="${LINGBOT_VLA2_ACTION_HORIZON:-50}"
native_hz="${LINGBOT_VLA2_NATIVE_HZ:-30}"
checkpoint_dir="${LINGBOT_VLA2_CHECKPOINT_DIR:-${checkpoint_dir}}"

if [[ "${env_cfg_type}" != "yam_dual" || "${action_type}" != "joint" ]]; then
  echo "LingBot_VLA2 supports env_cfg_type=yam_dual and action_type=joint only." >&2
  exit 2
fi
action_dim=$(bash "${XPL_ROOT}/utils/get_action_dim.sh" "${WORKSPACE_ROOT}" "${env_cfg_type}")
if [[ "${action_dim}" != "14" ]]; then
  echo "yam_dual must resolve to action_dim=14, got ${action_dim}" >&2
  exit 1
fi
for path in "${PYTHON}" "${model_path}/model.safetensors.index.json" "${dataset_path}/meta/info.json" "${norm_stats_path}" "${robot_config}"; do
  if [[ ! -e "${path}" ]]; then
    echo "Required training artifact not found: ${path}" >&2
    exit 1
  fi
done
if [[ -z "${tokenizer_path}" || ! -e "${tokenizer_path}" ]]; then
  echo "Set LINGBOT_VLA2_TOKENIZER_PATH or QWEN3_PATH to Qwen3-VL-4B-Instruct." >&2
  exit 1
fi

mkdir -p "${checkpoint_dir}"
export CUDA_VISIBLE_DEVICES="${gpu_id}"
export PYTHONHASHSEED="${seed}"

echo "[LingBot_VLA2] dataset=${dataset_path}"
echo "[LingBot_VLA2] checkpoint=${checkpoint_dir}"
echo "[LingBot_VLA2] GPUs=${gpu_id} action_dim=${action_dim} horizon=${action_horizon}"

cd "${LINGBOT_ROOT}"
PATH="${VENV_DIR}/bin:${PATH}" bash -o pipefail "${LINGBOT_ROOT}/train.sh" \
  tasks/vla/train_lingbotvla.py "${POLICY_DIR}/training/yam_dual.yaml" \
  --model.model_path "${model_path}" \
  --model.tokenizer_path "${tokenizer_path}" \
  --data.data_name "${robot_name}" \
  --data.train_path "${dataset_path}" \
  --data.robot_config_root "${POLICY_DIR}/robot_configs" \
  --data.norm_stats_file "${norm_stats_path}" \
  --data.num_workers "${LINGBOT_VLA2_TRAIN_WORKERS:-8}" \
  --train.output_dir "${checkpoint_dir}" \
  --train.seed "${seed}" \
  --train.chunk_size "${action_horizon}" \
  --train.micro_batch_size "${LINGBOT_VLA2_MICRO_BATCH_SIZE:-1}" \
  --train.gradient_accumulation_steps "${LINGBOT_VLA2_GRAD_ACCUM_STEPS:-1}" \
  --train.max_steps "${LINGBOT_VLA2_MAX_STEPS:-60000}" \
  --train.save_steps "${LINGBOT_VLA2_SAVE_STEPS:-1000}" \
  --train.use_wandb "${LINGBOT_VLA2_USE_WANDB:-false}" \
  --train.wandb_project "${LINGBOT_VLA2_WANDB_PROJECT:-lingbotvla}" \
  --train.wandb_name "${LINGBOT_VLA2_WANDB_NAME:-${setting}}"

"${PYTHON}" "${POLICY_DIR}/prepare_bundle.py" \
  --run-dir "${checkpoint_dir}" \
  --source-root "${LINGBOT_ROOT}" \
  --norm-stats "${norm_stats_path}" \
  --robot-config "${robot_config}" \
  --native-hz "${native_hz}" \
  --action-horizon "${action_horizon}"

echo "[LingBot_VLA2] bundle=${checkpoint_dir}/bundle.yaml"
