#!/bin/bash
set -euo pipefail

# RDT2-FM fine-tuning entry (step 2 of docs/rdt2-umi-runbook.md).
#
# STATUS: NOT IMPLEMENTED HERE. Training runs from the upstream RDT2 tree
# (`scripts/finetune_rdt.sh`), which is deliberately kept outside this
# workspace and is not vendored. This wrapper only maps XPolicyLab's six
# positional arguments onto that script; it fails loudly when the upstream
# tree or its training script cannot be found.
#
# Usage: bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id>
# Output convention: checkpoints/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>-<seed>

bench_name=${1:-}
ckpt_name=${2:-}
env_cfg_type=${3:-}
action_type=${4:-}
seed=${5:-}
gpu_id=${6:-}

if [[ -z "${bench_name}" || -z "${ckpt_name}" || -z "${env_cfg_type}" || -z "${action_type}" || -z "${seed}" || -z "${gpu_id}" ]]; then
  echo "Usage: bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id>" >&2
  exit 1
fi

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${POLICY_DIR}/../.." && pwd)"
WORKSPACE_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"

ckpt_setting="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}-${seed}"
ckpt_dir="${POLICY_DIR}/checkpoints/${ckpt_setting}"

# Action dimension comes from the shell entry point (see AGENTS.md): the
# training path must not import the Python helper.
action_dim="$(bash "${XPL_ROOT}/utils/get_action_dim.sh" "${WORKSPACE_ROOT}" "${env_cfg_type}")"
echo "[RDT2][train] env_cfg_type=${env_cfg_type} action_dim=${action_dim}"

RDT2_ROOT="${RDT2_ROOT:-${WORKSPACE_ROOT}/../RDT2}"
FINETUNE="${RDT2_ROOT}/scripts/finetune_rdt.sh"

if [[ ! -f "${FINETUNE}" ]]; then
  cat >&2 <<MSG
[RDT2][train] NOT IMPLEMENTED / upstream training script not found:
      ${FINETUNE}

  Step 2 of docs/rdt2-umi-runbook.md fine-tunes RDT2-FM from the upstream
  tree. Clone it outside this workspace and point RDT2_ROOT at it:
      export RDT2_ROOT=/home/jw/Desktop/project/RDT2
  Also required first: step 1 shards (bash process_data.sh ...) and a
  configs/datasets/<name>.yaml inside the RDT2 tree.

  Do NOT launch RDT2-VQ training on this host: 30 GB system RAM.
MSG
  exit 2
fi

mkdir -p "${ckpt_dir}"
echo "[RDT2][train] upstream: ${FINETUNE}"
echo "[RDT2][train] ckpt_dir: ${ckpt_dir}"
echo "[RDT2][train] run this inside tmux, never nohup (see the runbook)."

exec env \
  CUDA_VISIBLE_DEVICES="${gpu_id}" \
  RDT2_OUTPUT_DIR="${ckpt_dir}" \
  RDT2_SEED="${seed}" \
  RDT2_DATASET_CONFIG="${RDT2_DATASET_CONFIG:-configs/datasets/${ckpt_name}.yaml}" \
  bash "${FINETUNE}"
