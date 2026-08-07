#!/usr/bin/env bash
set -euo pipefail

bench_name=${1:?bench_name required}
ckpt_name=${2:?ckpt_name required}
env_cfg_type=${3:?env_cfg_type required}
action_type=${4:?action_type required}
seed=${5:?seed required}
gpu_id=${6:?gpu_id required}
shift 6 || true

if [[ "${action_type}" != "joint" ]]; then
  echo "G05 train.sh currently supports action_type=joint, got ${action_type}" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
G05_ROOT="${G05_ROOT:-}"
PYTHON_BIN="${G05_PYTHON:-$(command -v python3)}"
OUTPUT_ROOT="${G05_OUTPUT_ROOT:-${SCRIPT_DIR}/checkpoints}"
export G05_OUTPUT_DIR="${G05_OUTPUT_DIR:-${OUTPUT_ROOT}}"
export EXP_NAME="${EXP_NAME:-${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}-${seed}}"
export PYTHON_BIN

if [[ -z "${G05_ROOT}" ]]; then
  echo "Set G05_ROOT to a G05 checkout before launching training." >&2
  exit 3
fi
if [[ ! -d "${G05_ROOT}" ]]; then
  echo "G05_ROOT does not exist: ${G05_ROOT}" >&2
  exit 3
fi
if [[ -z "${PYTHON_BIN}" || ! -x "${PYTHON_BIN}" ]]; then
  echo "Set G05_PYTHON to a valid Python executable." >&2
  exit 3
fi

case "${G05_TRAIN_BENCHMARK:-}" in
  "")
    case "${bench_name}" in
      RoboDojo|robodojo) train_benchmark="robodojo" ;;
      RoboDojoReal|robodojo_real|RoboDojo-real) train_benchmark="robodojo_real" ;;
      *) train_benchmark="${bench_name}" ;;
    esac
    ;;
  *) train_benchmark="${G05_TRAIN_BENCHMARK}" ;;
esac

case "${G05_TRAIN_MODE:-${ckpt_name}}" in
  fm|fm_only) train_mode="fm_only" ;;
  ar|ar_only) train_mode="ar_only" ;;
  ar_fm|ar-fm|ar+fm|cotrain|joint) train_mode="ar_fm" ;;
  *) train_mode="${G05_TRAIN_MODE:-ar_fm}" ;;
esac

case "${train_benchmark}" in
  robodojo)
    if [[ -z "${ROBODOJO_LEROBOT_V30_ROOT:-}" ]]; then
      echo "Set ROBODOJO_LEROBOT_V30_ROOT to the RoboDojo LeRobot v3.0 dataset path." >&2
      exit 3
    fi
    ;;
  robodojo_real)
    if [[ -z "${ROBODOJO_REAL_ROOT:-}" ]]; then
      echo "Set ROBODOJO_REAL_ROOT to the RoboDojo-real dataset path." >&2
      exit 3
    fi
    ;;
esac

if [[ "${gpu_id}" == *","* ]]; then
  IFS=',' read -r -a _gpus <<< "${gpu_id}"
  num_gpus="${#_gpus[@]}"
else
  num_gpus=1
fi

export CUDA_VISIBLE_DEVICES="${gpu_id}"
export PYTHONPATH="${G05_ROOT}:${PYTHONPATH:-}"
export WANDB_PROJECT="${WANDB_PROJECT:-g05_robodojo}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.wandb.ai}"

if [[ "${WANDB_DIRECT:-1}" == "1" ]]; then
  unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
fi

cd "${G05_ROOT}"

if [[ -x scripts/run/finetune_benchmark.sh ]]; then
  args=(
    "${train_benchmark}"
    "${train_mode}"
    "seed=${seed}"
    "model.batch_size=${G05_BATCH_SIZE:-8}"
    "model.grad_accumulation_steps=${G05_GRAD_ACCUM:-1}"
  )
  if [[ -n "${G05_GLOBAL_BATCH_SIZE:-}" ]]; then
    args+=("trainer.global_batch_size=${G05_GLOBAL_BATCH_SIZE}")
  fi
  exec bash scripts/run/finetune_benchmark.sh "${args[@]}" "$@"
fi

TASK_CONFIG="${G05_TASK_CONFIG:-robodojo_arx_x5_joint}"
exec bash scripts/run/finetune.sh \
  "${num_gpus}" \
  "${TASK_CONFIG}" \
  "seed=${seed}" \
  "logger.mode=${G05_LOGGER_MODE:-online}" \
  "logger.project=${WANDB_PROJECT}" \
  "model.batch_size=${G05_BATCH_SIZE:-8}" \
  "model.grad_accumulation_steps=${G05_GRAD_ACCUM:-1}" \
  "$@"
