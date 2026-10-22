#!/usr/bin/env bash
set -euo pipefail

mode=${1:-train}
run_name=${2:-assemble-screwdriver-v1-s0-4xh100-3k}

ROOT=${YAM_TRAIN_ROOT:-/inspire/hdd3/project/agent-driven-world-model/ky26300/manimux_training}
WORKSPACE=${PI05_WORKSPACE:-${ROOT}}
POLICY=${WORKSPACE}/XPolicyLab/policy/Pi_05_Legato
OPENPI=${POLICY}/openpi
export PYTHONPATH="${OPENPI}/src:${OPENPI}/packages/openpi-client/src:${ROOT}:${PYTHONPATH:-}"
VENV=${OPENPI_ENV_DIR:-${ROOT}/envs/pi05/.venv}
DATASET_NAME=${OPENPI_LEROBOT_REPO_ID:-yam_assemble_screwdriver_20260825_v1}
DATASET=${ROOT}/datasets/lerobot/${DATASET_NAME}
BASE_PARAMS=${OPENPI_BASE_PARAMS:-${ROOT}/weights/base/pi05_base/params}
ASSETS_BASE=${OPENPI_ASSETS_BASE_DIR:-${ROOT}/cache/pi05/assets}
NORM_STATS=${ASSETS_BASE}/pi05_yam/${DATASET_NAME}/norm_stats.json
OUTPUT=${ROOT}/weights/finetuned/pi05/${run_name}
LOG_DIR=${ROOT}/runs/pi05
GPU_IDS=${OPENPI_GPU_IDS:-0,1,2,3}

export PATH="${ROOT}/envs/bin:${PATH}"
export UV_CACHE_DIR=${ROOT}/cache/uv
export UV_PROJECT_ENVIRONMENT=${VENV}
export HF_HOME=${ROOT}/cache/huggingface
export HF_LEROBOT_HOME=${ROOT}/datasets/lerobot
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OPENPI_DATA_HOME=${ROOT}/cache/openpi
export OPENPI_LOCAL_CACHE_ROOT=${ROOT}/cache/pi05/${HOSTNAME}
export OPENPI_TRAIN_CONFIG_NAME=pi05_yam_legato
export OPENPI_LEROBOT_REPO_ID=${DATASET_NAME}
export OPENPI_BASE_PARAMS=${BASE_PARAMS}
export OPENPI_ASSETS_BASE_DIR=${ASSETS_BASE}
export OPENPI_FSDP_DEVICES=${OPENPI_FSDP_DEVICES:-4}
export OPENPI_NUM_WORKERS=${OPENPI_NUM_WORKERS:-0}
export OPENPI_WANDB_ENABLED=false

mkdir -p "${LOG_DIR}" "${ROOT}/cache/pi05/${HOSTNAME}"

require_file() {
  if [[ ! -e "$1" ]]; then
    echo "Required artifact is missing: $1" >&2
    exit 1
  fi
}

preflight() {
  require_file "${OPENPI}/scripts/train.py"
  require_file "${DATASET}/meta/info.json"
  require_file "${BASE_PARAMS}/manifest.ocdbt"
  require_file "${NORM_STATS}"
  python3 - "${DATASET}" <<'PY'
import json
import sys
from pathlib import Path

dataset = Path(sys.argv[1])
info = json.loads((dataset / "meta/info.json").read_text())
assert info["codebase_version"] == "v3.0"
assert info["total_episodes"] == int(__import__("os").environ["OPENPI_EXPECTED_EPISODES"])
assert info["total_frames"] == int(__import__("os").environ["OPENPI_EXPECTED_FRAMES"])
assert info["features"]["observation.state"]["shape"] == [14]
assert info["features"]["action"]["shape"] == [14]
print(json.dumps({"dataset": dataset.name, "episodes": info["total_episodes"], "frames": info["total_frames"]}, indent=2))
PY
  python3 - "${NORM_STATS}" <<'PY'
import json
import sys
from pathlib import Path

stats = json.loads(Path(sys.argv[1]).read_text())["norm_stats"]
assert len(stats["state"]["mean"]) == 14
assert len(stats["actions"]["mean"]) == 14
print("Pi05 norm stats verified: state=14D actions=14D")
PY
}

compute_stats() {
  if [[ -s "${NORM_STATS}" ]]; then
    echo "[Pi_05] reusing norm stats: ${NORM_STATS}"
    return
  fi
  echo "[Pi_05] computing norm stats from ${DATASET}"
  (
    cd "${OPENPI}"
    OPENPI_LEROBOT_REPO_ID=${DATASET_NAME} \
    OPENPI_ASSETS_BASE_DIR=${ASSETS_BASE} \
    OPENPI_BASE_PARAMS=${BASE_PARAMS} \
      uv run scripts/compute_norm_stats.py --config-name pi05_yam
  )
  require_file "${NORM_STATS}"
}

install_environment() {
  require_file "${VENV}/bin/python"
  "${VENV}/bin/python" - "${OPENPI}" <<'PYENV'
import pathlib,sys,jax,openpi
assert len(jax.devices()) == 8
pathlib.Path(openpi.__file__).resolve().relative_to(pathlib.Path(sys.argv[1]).resolve())
print("Using existing Pi05 environment and isolated policy source:",openpi.__file__)
PYENV
}

case "${mode}" in
  gate-train)
    OPENPI_BATCH_SIZE=4 OPENPI_NUM_TRAIN_STEPS=1 OPENPI_SAVE_INTERVAL=1 \
      OPENPI_MAX_TO_KEEP=1 bash "$0" smoke "${run_name}"
    require_file "${ROOT}/weights/finetuned/pi05/${run_name}-smoke/1/params/manifest.ocdbt"
    OPENPI_BATCH_SIZE=${OPENPI_BATCH_SIZE:-32} OPENPI_NUM_TRAIN_STEPS=3000 \
      OPENPI_SAVE_INTERVAL=500 OPENPI_MAX_TO_KEEP=10 \
      bash "$0" train "${run_name}"
    ;;
  prepare)
    install_environment
    preflight
    ;;
  smoke|train)
    install_environment
    preflight
    if [[ "${mode}" == "smoke" ]]; then
      export OPENPI_BATCH_SIZE=${OPENPI_BATCH_SIZE:-4}
      export OPENPI_NUM_TRAIN_STEPS=${OPENPI_NUM_TRAIN_STEPS:-1}
      export OPENPI_SAVE_INTERVAL=${OPENPI_SAVE_INTERVAL:-1}
      export OPENPI_MAX_TO_KEEP=${OPENPI_MAX_TO_KEEP:-1}
      export OPENPI_CHECKPOINT_DIR=${OUTPUT}-smoke
    else
      export OPENPI_BATCH_SIZE=${OPENPI_BATCH_SIZE:-32}
      export OPENPI_NUM_TRAIN_STEPS=${OPENPI_NUM_TRAIN_STEPS:-3000}
      export OPENPI_SAVE_INTERVAL=${OPENPI_SAVE_INTERVAL:-500}
      export OPENPI_MAX_TO_KEEP=${OPENPI_MAX_TO_KEEP:-10}
      export OPENPI_CHECKPOINT_DIR=${OUTPUT}
      if [[ "${OPENPI_RESUME:-0}" == "1" ]]; then
        python3 - "${OUTPUT}" <<'PY'
import sys
from pathlib import Path

output = Path(sys.argv[1])
complete = [
    path for path in output.iterdir()
    if path.is_dir() and path.name.isdigit()
    and all((path / item).is_file() for item in (
        "params/manifest.ocdbt", "train_state/manifest.ocdbt", "_CHECKPOINT_METADATA"
    ))
] if output.is_dir() else []
if not complete:
    raise SystemExit(f"Resume requested but no complete checkpoint exists: {output}")
print(f"[Pi_05] resuming existing run; latest complete checkpoint={max(int(p.name) for p in complete)}")
PY
      elif [[ -e "${OUTPUT}" ]]; then
        echo "Refusing to overwrite existing run: ${OUTPUT}" >&2
        exit 2
      fi
    fi
    tee_args=()
    if [[ "${OPENPI_RESUME:-0}" == "1" ]]; then
      tee_args+=(-a)
    fi
    bash "${POLICY}/train.sh" \
      yam "${OPENPI_TASK_NAME}" yam_dual joint 0 "${GPU_IDS}" \
      2>&1 | tee "${tee_args[@]}" "${LOG_DIR}/${run_name}-${mode}.log"
    ;;
  *)
    echo "Usage: $0 [prepare|smoke|train|gate-train] [run_name]" >&2
    exit 2
    ;;
esac
