---
name: xpolicylab-model-integration
description: Integrate a robot policy into XPolicyLab as a policy/<POLICY>/ adapter. Use when adding or updating a policy adapter, implementing model.py or deploy.yml, wiring install/process_data/train/eval scripts, or debugging debug-mode evaluation in the XPolicyLab repo.
---

# XPolicyLab Model Integration

Add each policy as a self-contained `policy/<POLICY>/` adapter; keep upstream model code unchanged where possible. `policy/demo_policy/` is the minimal reference — mirror it unless the model truly needs more.

## Workflow

1. Read `policy/demo_policy/` (`model.py`, `deploy.py`, `deploy.yml`, `eval.sh`, `README.md`), then the upstream model's inference API, dependencies, and checkpoint layout.
2. Scaffold: `bash scripts/create_policy.sh <POLICY>` (copies `demo_policy`).
3. Implement `model.py` first (contract below). The policy server imports `XPolicyLab.policy.<POLICY>.model`, so keep the directory importable (`__init__.py`). Put environment setup in `install.sh`; add `process_data.sh` / `train.sh` only if the model supports them.
4. Keep `deploy.py` aligned with `policy/demo_policy/deploy.py` unless the environment loop truly differs. Put runtime defaults in `deploy.yml` with `protocol: ws`.
5. Debug without a simulator, from `policy/<POLICY>/`:

   ```bash
   export EVAL_ENV_TYPE=debug
   bash eval.sh RoboDojo stack_bowls <ckpt_name> arx_x5 joint 0 0 0 <policy_env> base
   ```

   Fix import, server-startup, action-key, and shape errors until the loop completes. Run once more with `DEBUG_OBS_ENCODED=1` so the debug client sends encoded camera colors and the server-side decode path is exercised.
6. Static checks from the repo root: `bash -n policy/<POLICY>/*.sh` and `python -m py_compile policy/<POLICY>/model.py policy/<POLICY>/deploy.py`.
7. Write `policy/<POLICY>/README.md`: install, data, train, and eval commands, supported `action_type` / `env_cfg_type`, checkpoint layout, known limitations.

## Model contract (`model.py`)

Define `class Model(ModelTemplate)`, importing `ModelTemplate` from `XPolicyLab.model_template`:

| Method | Contract |
| --- | --- |
| `__init__(model_cfg)` | `model_cfg` is `deploy.yml` merged with CLI overrides (`ckpt_name`, `action_type`, `env_cfg_type`, `seed`, `gpu_id`, ...). Load checkpoints and processors here. |
| `update_obs(obs)` / `update_obs_batch(obs_list)` | Store observation dict(s) for the next action call. |
| `get_action()` | Return one action chunk: `list[dict]` of numpy arrays. |
| `get_action_batch(env_idx_list=None)` | Batched chunks aligned with active env indices; fall back to the config batch size when `None`. |
| `reset()` | Clear model state between episodes. |

Action dict keys: dual-arm uses `left_arm_joint_state` / `right_arm_joint_state` plus `left_ee_joint_state` / `right_ee_joint_state`; single-arm drops the `left_` / `right_` prefix; `action_type=ee` replaces `*_arm_joint_state` with `*_ee_pose` as `[x, y, z, qw, qx, qy, qz]`. Take dimensions from `get_robot_action_dim_info(env_cfg_type)` in `XPolicyLab.utils.process_data` — never hard-code them. Register new robots in `utils/robot/_robot_info.json`.

## Conventions

- **`model.py` must not decode images.** The policy server decodes observations before `update_obs` / `update_obs_batch`, so `obs["vision"][<camera>]["color"]` is always a plain image array. Adapters only reshape / cast / resize.
- **Offline code decodes only via `decode_image_bit`** from `XPolicyLab.utils.process_data` — in data-conversion scripts and training dataloaders, never hand-roll `cv2.imdecode` / `np.frombuffer` / PIL decoding. RoboTwin/RoboDojo legacy image-bit layouts are only handled correctly by this function; custom decoders silently decode wrong or fail on part of the data.
- `eval.sh` positional args, same for all adapters: `bench_name task_name ckpt_name env_cfg_type action_type seed policy_gpu_id env_gpu_id policy_env_or_uv_path eval_env_conda_env`.
- Checkpoints resolve to `checkpoints/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>-<seed>/`; a full folder name or explicit path also works.
- Observations carry the language prompt under `instruction` (string; fall back to `instructions`). Poses are `[x, y, z, qw, qx, qy, qz]`.
- Images are RGB end to end — never add channel swaps in conversion, training, or eval code. The only exceptions are medium adapters: `COLOR_RGB2BGR` immediately before `cv2.VideoWriter.write(...)`, `COLOR_BGR2RGB` immediately after `cv2.VideoCapture.read()`.
- Trajectory HDF5 files store a singular `instruction` string and camera extrinsics as `extrinsic_matrix`; runtime observations use `extrinsics_matrix`.
- Full observation/trajectory format trees live in the repo README under "Standard Data Formats".
