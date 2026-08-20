#!/usr/bin/env bash
# XPolicyLab deploy: policy server env=uv; run setup_eval_policy_server.sh with this env.
set -euo pipefail

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GR00T_ROOT="${POLICY_DIR}/gr00t_n17"
XPOLICYLAB_ROOT="$(cd "${POLICY_DIR}/../.." && pwd)"

echo "[GR00T_N17] GR00T_ROOT=${GR00T_ROOT}"
echo "[GR00T_N17] XPOLICYLAB_ROOT=${XPOLICYLAB_ROOT}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Install via: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

cd "${GR00T_ROOT}"
# NOTE: pyproject pins `[tool.uv] required-environments` to both x86_64 and aarch64,
# and the aarch64 torchcodec/flash-attn wheels under scripts/deployment/dgpu/wheels/
# are not shipped. `uv sync` therefore fails to lock on an x86_64 GPU host because it
# must resolve the (missing) aarch64 wheels. We instead create the venv and use
# `uv pip install -e .`, which resolves only for the *current* platform (x86_64) and
# honors [tool.uv.sources] / [[tool.uv.index]] while ignoring required-environments.
uv venv --clear --python 3.10
MODEL_PYTHON="${GR00T_ROOT}/.venv/bin/python"
uv pip install --python "${MODEL_PYTHON}" -e .
"${MODEL_PYTHON}" -c "import gr00t; print('GR00T import ok')"

uv pip install --python "${MODEL_PYTHON}" -e "${XPOLICYLAB_ROOT}"
uv pip install --python "${MODEL_PYTHON}" h5py pyyaml
"${MODEL_PYTHON}" - <<'PY'
import json

import flash_attn
import torch
import transformers
import XPolicyLab

report = {
    "python": __import__("sys").version.split()[0],
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "flash_attn": flash_attn.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "xpolicylab_import": bool(XPolicyLab),
}
print(json.dumps(report, indent=2))
if not report["cuda_available"]:
    raise SystemExit("GR00T environment installed, but CUDA is unavailable")
PY

echo "[GR00T_N17] Installation finished."
echo "[GR00T_N17] Policy server env: source ${GR00T_ROOT}/.venv/bin/activate"
echo "[GR00T_N17] Next: run the ManiMux GR00T contract check, then start the model server."
