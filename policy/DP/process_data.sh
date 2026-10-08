#!/bin/bash

bench_name=$1
ckpt_name=$2
env_cfg_type=$3
action_type=$4
expert_data_num=${5:-}

config_args=()
if [[ -n "${XPOLICYLAB_MODEL_CONFIG:-}" ]]; then
  config_args+=(--model-config "${XPOLICYLAB_MODEL_CONFIG}")
fi
python diffusion_policy/process_data.py "$bench_name" "$ckpt_name" "$env_cfg_type" "$action_type" ${expert_data_num:+"$expert_data_num"} "${config_args[@]}"
