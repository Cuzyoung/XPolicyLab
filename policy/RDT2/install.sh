#!/bin/bash
set -euo pipefail

# RDT2 policy environment setup.
#
# The upstream RDT2 source tree is NOT vendored into XPolicyLab (see
# docs/rdt2-umi-runbook.md, "git 追踪"): it lives outside the workspace, is
# cloned separately, and is pointed at by `rdt2_root` in deploy.yml or by
# $RDT2_ROOT. This script only prepares the *policy-side* environment:
#   1. installs XPolicyLab itself (so `XPolicyLab.policy.RDT2.model` and the
#      websocket client/server modules import), and
#   2. installs the upstream requirements into the *currently active* env when
#      the RDT2 tree can be located.
#
# Usage:
#   conda activate <policy_env>
#   bash install.sh [rdt2_root]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

python -m pip install -e "${XPL_ROOT}"

RDT2_ROOT="${1:-${RDT2_ROOT:-}}"
if [[ -z "${RDT2_ROOT}" ]]; then
    RDT2_ROOT="$(cd "${XPL_ROOT}/../.." 2>/dev/null && pwd)/RDT2"
fi

if [[ ! -d "${RDT2_ROOT}" ]]; then
    echo "[RDT2][install] upstream RDT2 tree not found at '${RDT2_ROOT}'." >&2
    echo "[RDT2][install] Clone it outside this workspace and re-run:" >&2
    echo "[RDT2][install]   git clone https://github.com/thu-ml/RDT2.git" >&2
    echo "[RDT2][install]   bash install.sh /path/to/RDT2" >&2
    echo "[RDT2][install] XPolicyLab itself is installed; upstream deps are NOT." >&2
    exit 0
fi

echo "[RDT2][install] upstream root: ${RDT2_ROOT}"
if [[ -f "${RDT2_ROOT}/requirements.txt" ]]; then
    python -m pip install -r "${RDT2_ROOT}/requirements.txt"
else
    echo "[RDT2][install] no requirements.txt under ${RDT2_ROOT}; skipping." >&2
fi

echo "[RDT2][install] done. Set rdt2_root in deploy.yml (or export RDT2_ROOT=${RDT2_ROOT})."
