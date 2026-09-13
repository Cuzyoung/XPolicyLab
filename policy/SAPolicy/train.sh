#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
select_python "${SAPOLICY_ENV:-${POLICY_DIR}/.venv}"
exec "${POLICY_PYTHON}" -m XPolicyLab.policy.SAPolicy.train "$@"
