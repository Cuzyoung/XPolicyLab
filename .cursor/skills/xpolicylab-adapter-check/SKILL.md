---
name: xpolicylab-adapter-check
description: Audit a policy/<POLICY>/ adapter for XPolicyLab standard compliance and PR readiness — file completeness, deploy.yml, Model contract, script conventions, static checks, debug-mode eval, README and checkpoint requirements. Use when asked to check, validate, review, or pre-flight a policy adapter or a policy submission PR in the XPolicyLab repo.
---

# XPolicyLab Adapter Check

Audit `policy/<POLICY>/` against the submission standard and report pass/fail per item with concrete fixes. The full standard lives in `CONTRIBUTING.md` at the repo root — read it before auditing; the list below is the executable summary. `policy/demo_policy/` is the reference implementation.

## Checks

Run commands from the repo root unless noted.

1. **Files** — required: `README.md`, `__init__.py`, `install.sh`, `eval.sh`, `setup_eval_policy_server.sh`, `setup_eval_env_client.sh`, `deploy.yml`, `deploy.py`, `model.py`. `process_data.sh` / `train.sh` may be absent only for a declared eval-only submission.
2. **deploy.yml** — `policy_name` equals the directory name (the server imports `XPolicyLab.policy.<policy_name>.model`); `protocol: ws`.
3. **model.py** — `class Model(ModelTemplate)` implementing `__init__(model_cfg)`, `update_obs`, `update_obs_batch`, `get_action`, `get_action_batch(env_idx_list=None)`, `reset`. Action dimensions come from `get_robot_action_dim_info(env_cfg_type)`, not literals; every supported `env_cfg_type` exists in `utils/robot/_robot_info.json`. Image rules:
   - **No decoding in `model.py`** — the policy server hands `update_obs` / `update_obs_batch` plain image arrays, so flag any decoding there, including `decode_image_bit` calls.
   - **Offline decoding only via `decode_image_bit`** from `XPolicyLab.utils.process_data` — required in conversion scripts and training dataloaders that read XPolicyLab data. Mechanically checkable: **`cv2.imdecode` must not appear anywhere outside `utils/process_data.py`**; flag every other occurrence, along with hand-rolled `np.frombuffer` / PIL decoding (RoboTwin/RoboDojo legacy layouts break custom decoders).
   - **RGB end to end** — flag any channel swap (`COLOR_BGR2RGB`, `COLOR_RGB2BGR`, `[..., ::-1]`) that is not a medium adapter immediately around `cv2.VideoWriter.write` / `cv2.VideoCapture.read`; the training path and `model.py` must apply the same number of swaps, normally zero.
4. **Scripts** — `eval.sh` consumes the 10 standard args (`bench_name task_name ckpt_name env_cfg_type action_type seed policy_gpu_id env_gpu_id policy_env_or_uv_path eval_env_conda_env`) and stays aligned with `policy/demo_policy/eval.sh`; extra args must be documented in the policy README.
5. **Static checks**:

   ```bash
   bash -n policy/<POLICY>/*.sh
   python -m py_compile policy/<POLICY>/model.py policy/<POLICY>/deploy.py
   ```

6. **Debug closed loop** — only when an environment with the policy dependencies is available:

   ```bash
   cd policy/<POLICY>
   EVAL_ENV_TYPE=debug bash eval.sh RoboDojo stack_bowls <ckpt_name> arx_x5 joint 0 0 0 <policy_env> base
   ```

   Must reach `[MAIN] eval finished` with no tracebacks. Re-run with `DEBUG_OBS_ENCODED=1` so the debug client sends encoded camera colors and the server-side decode path is exercised. If it cannot be run, report the item as "not run" — never as passed.
7. **Policy README** — install / data / train / eval commands present and consistent with the actual scripts; supported `action_type` / `env_cfg_type`; checkpoint layout; known limitations.
8. **PR readiness** (when auditing a submission PR) — description follows `.github/PULL_REQUEST_TEMPLATE.md`; checkpoint download script included (Hugging Face or ModelScope preferred) if targeting a leaderboard; eval-only status declared with a training-release timeline.

## Report format

- One line per check: ✅ pass / ❌ fail / ⚠️ not run, with a short reason.
- For each ❌: name the file and the exact change, pointing to `policy/demo_policy` where useful.
- End with a verdict: **ready for PR** or **needs fixes** (list blocking items first).
