#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 6 ]]; then
  echo "Usage: $0 <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id>" >&2
  exit 1
fi

bench_name=$1
ckpt_name=$2
env_cfg_type=$3
action_type=$4
seed=$5
gpu_id=$6

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# ckpt_setting is the run directory name; pass it verbatim as ckpt_name to eval.sh.
ckpt_setting="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}-${seed}"
ckpt_dir="${POLICY_DIR}/checkpoints/${ckpt_setting}"
train_config_name="${OPENPI_TRAIN_CONFIG_NAME:-pi05_base_aloha_full_sim_arx-x5_seed_0}"
lerobot_repo_id="${OPENPI_LEROBOT_REPO_ID:-${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}}"
gpu_count=$(awk -F',' '{print NF}' <<<"${gpu_id}")
fsdp_devices="${OPENPI_FSDP_DEVICES:-$(( gpu_count < 2 ? 1 : 2 ))}"
# Each dataloader worker decodes video into host memory on top of the ~12GB the
# checkpoint restore already holds; 8 workers is enough to get OOM-killed on a
# 32GB host, so default lower and let big machines raise it.
num_workers="${OPENPI_NUM_WORKERS:-4}"
# openpi logs loss/grad_norm to wandb and enables it by default, which blocks on a
# login prompt when no credentials exist. Set OPENPI_WANDB=0 to train without it.
wandb_project="${OPENPI_WANDB_PROJECT:-openpi}"
wandb_flag=$([[ "${OPENPI_WANDB:-1}" == "0" ]] && echo "--no-wandb-enabled" || echo "--wandb-enabled")

mkdir -p "${ckpt_dir}"
export CUDA_VISIBLE_DEVICES="${gpu_id}"

# LeRobot loads parquet via HuggingFace datasets, which builds pyarrow mmap cache
# under HF_DATASETS_CACHE. Keep dataset on shared storage, but use per-host local
# cache to avoid NFS lock contention when multiple nodes train concurrently.
LOCAL_CACHE_ROOT="${OPENPI_LOCAL_CACHE_ROOT:-/tmp/openpi-cache-$(hostname)}"
mkdir -p "${LOCAL_CACHE_ROOT}/hf/datasets" "${LOCAL_CACHE_ROOT}/jax"
export HF_DATASETS_CACHE="${LOCAL_CACHE_ROOT}/hf/datasets"
export JAX_COMPILATION_CACHE_DIR="${LOCAL_CACHE_ROOT}/jax"

# LeRobot resolves repo_id under HF_LEROBOT_HOME. Keep it explicit so training
# and compute_norm_stats.py always read the same copy of the dataset.
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${OPENPI_HF_LEROBOT_HOME:-${HOME}/.cache/huggingface/lerobot}}"

echo "[Pi_05] train_config_name=${train_config_name}"
echo "[Pi_05] lerobot_repo_id=${lerobot_repo_id}"
echo "[Pi_05] fsdp_devices=${fsdp_devices}"
echo "[Pi_05] num_workers=${num_workers}"
echo "[Pi_05] wandb=${wandb_flag#--} project=${wandb_project}"
echo "[Pi_05] local_cache_root=${LOCAL_CACHE_ROOT}"
echo "[Pi_05] hf_lerobot_home=${HF_LEROBOT_HOME}"
echo "[Pi_05] checkpoint_dir=${ckpt_dir}"

cd "${POLICY_DIR}/openpi/"
XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}" \
  uv run scripts/train.py "${train_config_name}" \
    --exp-name="${ckpt_setting}" \
    --data.repo-id="${lerobot_repo_id}" \
    --fsdp-devices="${fsdp_devices}" \
    --num-workers="${num_workers}" \
    --project-name="${wandb_project}" \
    "${wandb_flag}" \
    --checkpoint-dir-override="${ckpt_dir}" \
    --seed="${seed}" \
    --overwrite
