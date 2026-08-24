#!/usr/bin/env bash
set -euo pipefail

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK_ROOT="${POLICY_DIR}/cosmos-framework"
XPOLICYLAB_ROOT="$(cd "${POLICY_DIR}/../.." && pwd)"
WORKSPACE_ROOT="$(cd "${XPOLICYLAB_ROOT}/.." && pwd)"
ENV_ROOT="${COSMOS3_ENV_DIR:-${WORKSPACE_ROOT}/envs/cosmos3}"
VENV_DIR="${ENV_ROOT%/}/.venv"
PYTHON_BIN="${VENV_DIR}/bin/python"
CUDA_GROUP="${COSMOS3_CUDA_GROUP:-cu130-torch213-train}"
PYTHON_VERSION="${COSMOS3_PYTHON:-3.13}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Install it before running this script." >&2
  exit 1
fi
if [[ ! -f "${FRAMEWORK_ROOT}/pyproject.toml" ]]; then
  echo "Official cosmos-framework source is missing: ${FRAMEWORK_ROOT}" >&2
  echo "Run: git submodule update --init --recursive" >&2
  exit 1
fi

mkdir -p "${ENV_ROOT}"
export UV_PROJECT_ENVIRONMENT="${VENV_DIR}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}"

echo "[Cosmos3] Installing the pinned official cosmos-framework environment."
echo "[Cosmos3] Framework: ${FRAMEWORK_ROOT}"
echo "[Cosmos3] Environment: ${VENV_DIR}"
echo "[Cosmos3] CUDA group: ${CUDA_GROUP}"

uv sync \
  --project "${FRAMEWORK_ROOT}" \
  --python "${PYTHON_VERSION}" \
  --frozen \
  --all-extras \
  --group "${CUDA_GROUP}" \
  --group policy-server
uv pip install --python "${PYTHON_BIN}" -e "${XPOLICYLAB_ROOT}"

"${PYTHON_BIN}" - <<'PY'
import json
import sys

import torch
import XPolicyLab
import cosmos_framework

print(json.dumps({
    "python": sys.version.split()[0],
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "cosmos_framework_import": bool(cosmos_framework),
    "xpolicylab_import": bool(XPolicyLab),
}, indent=2))
PY

echo "[Cosmos3] Installation finished."
