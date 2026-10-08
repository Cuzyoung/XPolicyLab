#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
[[ $# -ge 10 ]] || { echo 'Expected bench task checkpoint robot action seed gpu client_env info port [host]' >&2; exit 2; }
if [[ "${EVAL_ENV_TYPE:-debug}" != debug ]]; then
    echo 'This YAM adapter supports the shared debug client and ManiMux real deployment; simulator evaluation is unsupported.' >&2
    exit 2
fi
select_python "$8"
exec "${POLICY_PYTHON}" "${XPL_ROOT}/debug_env_client.py" \
    --bench_name "$1" --task_name "$2" --env_cfg_type "$4" --policy_name SAPolicy \
    --protocol ws --host "${11:-127.0.0.1}" --port "${10}" \
    --eval_batch "${SAPOLICY_EVAL_BATCH:-false}"
