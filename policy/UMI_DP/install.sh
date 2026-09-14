#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
UMI_ENV="${1:-${XPL_ROOT}/../envs/umi_dp/.venv}"
if [[ ! -x "${UMI_ENV}/bin/python" ]]; then uv venv --python 3.11 "${UMI_ENV}"; fi
"${UMI_ENV}/bin/python" -c 'import sys; assert sys.version_info[:2] == (3, 11), "UMI_DP requires Python 3.11"'
uv pip install --python "${UMI_ENV}/bin/python" torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python "${UMI_ENV}/bin/python" -r "${SCRIPT_DIR}/requirements.txt"
uv pip install --python "${UMI_ENV}/bin/python" --no-deps -e "${XPL_ROOT}"
echo "Use ${UMI_ENV}/bin/python; this is a plain venv (no uv sync/run)."
