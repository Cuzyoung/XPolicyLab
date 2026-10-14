#!/usr/bin/env bash
# Prepare only XR1 inference dependencies, using Python with development headers.
set -euo pipefail

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XR1_VENV="${XR1_VENV:-${POLICY_DIR}/.venv}"
export UV_PROJECT_ENVIRONMENT="${XR1_VENV}"
uv python install 3.12
uv venv --allow-existing --managed-python --python 3.12 "${XR1_VENV}"
uv sync --project "${POLICY_DIR}" --frozen --managed-python
"${XR1_VENV}/bin/python" - <<'PY'
from pathlib import Path
import sysconfig
import torch
import flash_attn

assert (Path(sysconfig.get_path("include")) / "Python.h").is_file(), "Triton needs Python.h"
assert torch.cuda.is_available(), "XR1 inference requires CUDA"
print("XR1 uv environment ready:", torch.__version__, flash_attn.__version__, torch.cuda.get_device_name(0))
PY
