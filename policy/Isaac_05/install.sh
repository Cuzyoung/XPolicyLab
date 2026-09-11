#!/usr/bin/env bash
set -euo pipefail

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEROBOT_ROOT="${POLICY_DIR}/lerobot"
XPOLICYLAB_ROOT="$(cd "${POLICY_DIR}/../.." && pwd)"
WORKSPACE_ROOT="$(cd "${XPOLICYLAB_ROOT}/.." && pwd)"
VENV_DIR="${ISAAC05_ENV_DIR:-${WORKSPACE_ROOT}/envs/isaac-0.5/.venv}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Install it before continuing." >&2
  exit 1
fi
if [[ ! -f "${LEROBOT_ROOT}/uv.lock" ]]; then
  echo "Isaac LeRobot source is missing. Run: git submodule update --init --recursive" >&2
  exit 1
fi

export UV_PROJECT_ENVIRONMENT="${VENV_DIR}"
export UV_LINK_MODE=copy
uv sync --project "${LEROBOT_ROOT}" --locked --extra perceptron_isaac_cuda
uv pip install --python "${VENV_DIR}/bin/python" -e "${XPOLICYLAB_ROOT}"

"${VENV_DIR}/bin/python" - <<'PY'
import json
import torch
import transformers
from lerobot.policies.perceptron_isaac.modeling_perceptron_isaac import PerceptronIsaacPolicy

print(json.dumps({
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "policy_import": PerceptronIsaacPolicy.__name__,
    "qualified_mk1_abi": torch.__version__ == "2.10.0+cu128",
}, indent=2))
PY

echo "[Isaac_05] Dependencies installed: ${VENV_DIR}"
echo "[Isaac_05] The public trained_policy still requires the upstream-qualified PyTorch 2.10.0+cu128 and NVIDIA H100 runtime."
