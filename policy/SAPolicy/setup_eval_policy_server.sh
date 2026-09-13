#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
[[ $# -ge 9 ]] || { echo 'Expected bench task checkpoint robot action seed gpu policy_env port [host]' >&2; exit 2; }
select_python "$8"
exec env CUDA_VISIBLE_DEVICES="$7" "${POLICY_PYTHON}" "${XPL_ROOT}/setup_policy_server.py" \
    --config_path "${POLICY_DIR}/deploy.yml" --overrides \
    bench_name="$1" task_name="$2" ckpt_name="$3" env_cfg_type="$4" \
    action_type="$5" seed="$6" port="$9" host="${10:-127.0.0.1}"
