#!/usr/bin/env bash
# Shared by policy-side entrypoints. Accept a venv directory or conda environment.
set -euo pipefail
POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${POLICY_DIR}/../.." && pwd)"
select_python() {
    local selected="$1"
    if [[ -x "${selected}/bin/python" ]]; then
        POLICY_PYTHON="$(cd "${selected}" && pwd)/bin/python"
    elif [[ -x "${selected}" && ! -d "${selected}" ]]; then
        POLICY_PYTHON="${selected}"
    else
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate "${selected}"
        POLICY_PYTHON="$(command -v python)"
    fi
    export PYTHONPATH="${XPL_ROOT}/..:${XPL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
}
