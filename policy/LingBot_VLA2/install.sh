#!/usr/bin/env bash
# XPolicyLab deploy: LingBot-VLA2 model environment managed with uv.
set -euo pipefail

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LINGBOT_ROOT="${POLICY_DIR}/lingbot_vla_v2"
XPOLICYLAB_ROOT="$(cd "${POLICY_DIR}/../.." && pwd)"
UTILS3D_WHEEL="${LINGBOT_VLA2_UTILS3D_WHEEL:-${POLICY_DIR}/vendor/utils3d-1.3-py3-none-any.whl}"
WORKSPACE_ROOT="$(cd "${XPOLICYLAB_ROOT}/.." && pwd)"
VENV_DIR="${LINGBOT_VLA2_ENV_DIR:-${WORKSPACE_ROOT}/envs/lingbot-vla2/.venv}"
PYTHON="${VENV_DIR}/bin/python"
FLASH_ATTN_WHEEL="${FLASH_ATTN_WHEEL:-}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
LEROBOT_SPEC="${LINGBOT_VLA2_LEROBOT_SPEC:-lerobot==0.4.2}"
RECREATE=0

usage() {
  cat <<'EOF'
Usage: bash install.sh [--recreate] [--flash-attn-wheel PATH]

Creates the standalone LingBot-VLA2 uv venv at:
  <workspace>/envs/lingbot-vla2/.venv

Options:
  --recreate               Clear and rebuild the venv.
  --flash-attn-wheel PATH  Install a local flash-attn wheel instead of building it.

Environment variables:
  LINGBOT_VLA2_ENV_DIR     Override the venv directory.
  FLASH_ATTN_WHEEL         Same as --flash-attn-wheel.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --recreate)
      RECREATE=1
      shift
      ;;
    --flash-attn-wheel)
      FLASH_ATTN_WHEEL="${2:?--flash-attn-wheel requires a value}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Install via: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

if [[ ! -f "${LINGBOT_ROOT}/requirements.txt" ]]; then
  echo "LingBot-VLA2 source is incomplete: ${LINGBOT_ROOT}" >&2
  echo "Run: git submodule update --init --recursive" >&2
  exit 1
fi

echo "[LingBot_VLA2] LINGBOT_ROOT=${LINGBOT_ROOT}"
echo "[LingBot_VLA2] XPOLICYLAB_ROOT=${XPOLICYLAB_ROOT}"
echo "[LingBot_VLA2] VENV_DIR=${VENV_DIR}"
echo "[LingBot_VLA2] uv=$(uv --version)"

mkdir -p "$(dirname "${VENV_DIR}")"
if [[ "${RECREATE}" == "1" ]]; then
  uv venv --clear --python 3.12 "${VENV_DIR}"
elif [[ ! -x "${PYTHON}" ]]; then
  uv venv --python 3.12 "${VENV_DIR}"
else
  echo "[LingBot_VLA2] Reusing existing venv; pass --recreate for a clean rebuild."
fi

export PYTHONNOUSERSITE=1
export PIP_NO_INPUT=1
export UV_LINK_MODE=copy
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}"

uv pip install --python "${PYTHON}" pip setuptools wheel packaging ninja
uv pip install --python "${PYTHON}" \
  torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url "${PYTORCH_INDEX_URL}"
uv pip install --python "${PYTHON}" torchdata==0.11.0 torchcodec==0.6.0
uv pip install --python "${PYTHON}" -r "${LINGBOT_ROOT}/requirements.txt"
uv pip install --python "${PYTHON}" numpydantic==1.9.0 --no-deps

if [[ -n "${FLASH_ATTN_WHEEL}" ]]; then
  if [[ ! -f "${FLASH_ATTN_WHEEL}" ]]; then
    echo "flash-attn wheel not found: ${FLASH_ATTN_WHEEL}" >&2
    exit 1
  fi
  uv pip install --python "${PYTHON}" --no-deps "${FLASH_ATTN_WHEEL}"
else
  uv pip install --python "${PYTHON}" flash-attn==2.8.3 --no-build-isolation
fi

uv pip install --python "${PYTHON}" --no-deps "${LEROBOT_SPEC}"
uv pip install --python "${PYTHON}" --no-deps -e "${LINGBOT_ROOT}"
uv pip install --python "${PYTHON}" -r "${LINGBOT_ROOT}/requirements-depth.txt"

if [[ ! -f "${UTILS3D_WHEEL}" ]]; then
  echo "Pinned offline utils3d wheel not found: ${UTILS3D_WHEEL}" >&2
  exit 1
fi
uv pip install --python "${PYTHON}" --no-deps "${UTILS3D_WHEEL}"

uv pip install --python "${PYTHON}" --no-deps -e \
  "${LINGBOT_ROOT}/lingbotvla/models/vla/vision_models/lingbot-depth"
uv pip install --python "${PYTHON}" --no-deps -e \
  "${LINGBOT_ROOT}/lingbotvla/models/vla/vision_models/MoGe"

# Depth packages have broad dependencies; restore the official pinned core stack.
uv pip install --python "${PYTHON}" -r "${LINGBOT_ROOT}/requirements.txt"
uv pip install --python "${PYTHON}" numpydantic==1.9.0 --no-deps
uv pip install --python "${PYTHON}" -e "${XPOLICYLAB_ROOT}"

"${PYTHON}" - <<'PY'
import json
import sys

import flash_attn
import torch
import transformers
import XPolicyLab
import lingbotvla
import mdm
import moge
import utils3d

report = {
    "python": sys.version.split()[0],
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "flash_attn": flash_attn.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "lingbotvla_import": bool(lingbotvla),
    "depth_imports": bool(mdm and moge and utils3d),
    "xpolicylab_import": bool(XPolicyLab),
}
print(json.dumps(report, indent=2))
if not report["cuda_available"]:
    raise SystemExit("LingBot-VLA2 environment installed, but CUDA is unavailable")
PY

find "${LINGBOT_ROOT}" -type d -name __pycache__ -prune -exec rm -rf {} +
rm -rf "${LINGBOT_ROOT}/lingbotvla.egg-info"

echo "[LingBot_VLA2] Installation finished."
echo "[LingBot_VLA2] Policy server Python: ${PYTHON}"
echo "[LingBot_VLA2] Next: run process_data.sh, train.sh, or eval.sh."
