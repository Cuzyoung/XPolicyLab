#!/usr/bin/env bash
set -euo pipefail

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPOLICYLAB_ROOT="$(cd "${POLICY_DIR}/../.." && pwd)"
WORKSPACE_ROOT="$(cd "${XPOLICYLAB_ROOT}/.." && pwd)"
CONFIG_PATH="${1:-${WORKSPACE_ROOT}/configs/isaac05/libero/server/base.yaml}"
VENV_DIR="${ISAAC05_ENV_DIR:-${WORKSPACE_ROOT}/envs/isaac-0.5/.venv}"
PYTHON="${VENV_DIR}/bin/python"

if [[ "${CONFIG_PATH}" != /* ]]; then
  CONFIG_PATH="${WORKSPACE_ROOT}/${CONFIG_PATH}"
fi
if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Isaac 0.5 server config not found: ${CONFIG_PATH}" >&2
  exit 1
fi
if [[ ! -x "${PYTHON}" ]]; then
  echo "Isaac 0.5 environment not found. Run: bash ${POLICY_DIR}/install.sh" >&2
  exit 1
fi

cd "${WORKSPACE_ROOT}"
exec env \
  PYTHONUNBUFFERED=1 \
  PYTHONPATH="${WORKSPACE_ROOT}:${POLICY_DIR}/lerobot/src:${PYTHONPATH:-}" \
  "${PYTHON}" "${XPOLICYLAB_ROOT}/setup_policy_server.py" \
    --config_path "${CONFIG_PATH}"
