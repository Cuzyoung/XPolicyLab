#!/bin/bash
set -euo pipefail

bench_name=${1}
ckpt_name=${2}
env_cfg_type=${3}
action_type=${4}
seed=${5}
gpu_id=${6}

if [[ "${bench_name}" != "RoboTwin" || "${action_type}" != "joint" ]]; then
    echo "Published PatchWAM C2R training supports RoboTwin joint control only" >&2
    exit 2
fi

NNODES=${NNODES:-2}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}
if [[ "${NNODES}" != "2" || "${NPROC_PER_NODE}" != "8" ]]; then
    echo "Exact reproduction requires NNODES=2 and NPROC_PER_NODE=8" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UPSTREAM_DIR="${SCRIPT_DIR}/PatchWAM"
export PATCHWAM_OUTPUT_DIR="${PATCHWAM_OUTPUT_DIR:-${SCRIPT_DIR}/checkpoints/${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}-${seed}}"
export PATCHWAM_DATASET_DIR="${PATCHWAM_DATASET_DIR:-${SCRIPT_DIR}/data/${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}}"
export CUDA_VISIBLE_DEVICES="${gpu_id}"
export PYTHONPATH="${UPSTREAM_DIR}:${UPSTREAM_DIR}/src:${UPSTREAM_DIR}/vendor/flux2/src:${PYTHONPATH:-}"

accelerate launch \
    --use_deepspeed \
    --deepspeed_config_file "${UPSTREAM_DIR}/scripts/ds_configs/ds_zero1_config.json" \
    --num_machines "${NNODES}" \
    --num_processes 16 \
    --machine_rank "${NODE_RANK}" \
    --main_process_ip "${MASTER_ADDR}" \
    --main_process_port "${MASTER_PORT}" \
    --mixed_precision bf16 \
    "${UPSTREAM_DIR}/scripts/train_c2r.py"
