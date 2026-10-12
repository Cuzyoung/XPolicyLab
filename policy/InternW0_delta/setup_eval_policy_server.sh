#!/usr/bin/env bash
set -euo pipefail

bench_name=$1
task_name=$2
ckpt_name=$3
env_cfg_type=$4
action_type=$5
seed=$6
policy_gpu_id=$7
policy_conda_env=$8
policy_server_port=$9
policy_server_host=${10:-localhost}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
UTILS_DIR="${XPL_ROOT}/utils"

policy_name="$(basename "${SCRIPT_DIR}")"
yaml_file="${SCRIPT_DIR}/deploy.yml"

stats_path="${WAM_DATASET_STATS_PATH:-${SCRIPT_DIR}/config/dataset_stats.json}"
config_path="${WAM_EVAL_CONFIG_PATH:-${SCRIPT_DIR}/config/eval_model.yaml}"
base_model_dir="${WAM_WAN22_PATH:-${SCRIPT_DIR}/assets/Wan-AI/Wan2.2-TI2V-5B}"
vlm_model_path="${WAM_RYNNBRAIN_PATH:-${SCRIPT_DIR}/assets/Alibaba-DAMO-Academy/RynnBrain1.1-2B}"
allow_dummy_policy="${WAM_ALLOW_DUMMY_POLICY:-false}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${policy_conda_env}"

# Latest official training weight under <run_dir>/checkpoints/weights. Slots are
# reused round-robin, so the manifest (or file mtime) decides, not the slot index.
resolve_latest_weights() {
    local weights_dir="$1/checkpoints/weights"
    local resolved=""
    if [[ -f "${weights_dir}/weights_manifest.json" ]]; then
        resolved="$(python - "${weights_dir}/weights_manifest.json" <<'PY'
import json
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
path = (json.loads(manifest.read_text(encoding="utf-8")).get("latest") or {}).get("path")
if path:
    print(manifest.parent / path)
PY
)"
    fi
    if [[ -z "${resolved}" || ! -f "${resolved}" ]]; then
        resolved="$(ls -t "${weights_dir}"/slot_*.pt 2>/dev/null | head -n 1 || true)"
    fi
    printf '%s\n' "${resolved}"
}

checkpoint_path="${WAM_CHECKPOINT_PATH:-}"
if [[ -z "${checkpoint_path}" ]]; then
    ckpt_setting="${WAM_CKPT_SETTING:-${ckpt_name}}"
    run_dir_name="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}-${seed}"
    candidates=()
    if [[ "${ckpt_setting}" == /* ]]; then
        candidates+=("${ckpt_setting}")
    elif [[ "${ckpt_setting}" == */* ]]; then
        candidates+=("${SCRIPT_DIR}/${ckpt_setting}")
    fi
    candidates+=("${SCRIPT_DIR}/checkpoints/${run_dir_name}" "${SCRIPT_DIR}/checkpoints/${ckpt_setting}")
    for candidate in "${candidates[@]}"; do
        if [[ -f "${candidate}" ]]; then
            checkpoint_path="${candidate}"
            break
        fi
        if [[ -d "${candidate}" ]]; then
            checkpoint_path="$(resolve_latest_weights "${candidate}")"
            if [[ -z "${checkpoint_path}" && "${allow_dummy_policy}" != "true" ]]; then
                echo "[SERVER] ERROR: run dir ${candidate} has no checkpoints/weights/slot_*.pt; set WAM_CHECKPOINT_PATH to override." >&2
                exit 1
            fi
            break
        fi
    done
    checkpoint_path="${checkpoint_path:-${SCRIPT_DIR}/checkpoints/robodojo.pt}"
fi

echo "[SERVER] policy=${policy_name} task=${task_name} replan=10"
echo "[SERVER] checkpoint=${checkpoint_path}"

exec env \
  PYTHONWARNINGS=ignore::UserWarning \
  PYTHONUNBUFFERED=1 \
  CUDA_VISIBLE_DEVICES="${policy_gpu_id}" \
  PYTHONPATH="${XPL_ROOT}:${SCRIPT_DIR}/wam_runtime/src:${PYTHONPATH:-}" \
  DIFFSYNTH_SKIP_DOWNLOAD=true \
  DIFFSYNTH_MODEL_BASE_PATH="${SCRIPT_DIR}/assets" \
  python -u "${XPL_ROOT}/setup_policy_server.py" \
    --config_path "${yaml_file}" \
    --overrides \
      port="${policy_server_port}" \
      host="${policy_server_host}" \
      bench_name="${bench_name}" \
      task_name="${task_name}" \
      ckpt_name="${ckpt_name}" \
      env_cfg_type="${env_cfg_type}" \
      seed="${seed}" \
      policy_name="${policy_name}" \
      action_type="${action_type}" \
      checkpoint_path="${checkpoint_path}" \
      dataset_stats_path="${stats_path}" \
      train_config_path="${config_path}" \
      base_model_dir="${base_model_dir}" \
      vlm_model_path="${vlm_model_path}" \
      wam_root="${SCRIPT_DIR}/wam_runtime" \
      allow_dummy_policy="${allow_dummy_policy}"
