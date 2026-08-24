# Pi_05

**Contributor:** RoboDojo Team | **Paper:** Pi0.5 technical report | **arXiv:** TBD | **Original code:** https://github.com/Physical-Intelligence/openpi

`Pi_05` adapts Physical Intelligence's π0.5 policy to XPolicyLab/RoboDojo through the uv-managed OpenPI stack. Integration scripts live at this directory level; the vendored upstream implementation lives in `openpi/`.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

```bash
cd XPolicyLab/policy/Pi_05
bash install.sh
source openpi/.venv/bin/activate  # OpenPI is uv-managed; there is no policy conda env
```

`eval.sh` arg 9 is not a conda env: pass `uv` (uses `deploy.yml` `policy_uv_env_path`) or an explicit OpenPI project path.

## Data Processing

Converts RoboDojo demonstrations into the LeRobot repo consumed by training. The optional `expert_data_num` caps episodes for data conversion only (it is not part of checkpoint naming); the optional `raw_task_dirs` is a source task directory or comma-separated task list under `data/<bench_name>/` (defaults to `ckpt_name`). `raw_task_dirs` may also be passed directly as the 5th argument to write a differently named dataset from all of a task's demos, e.g. `bash process_data.sh RoboDojo stack_bowls_ablation arx_x5 joint stack_bowls`.

```bash
cd XPolicyLab/policy/Pi_05
bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> [expert_data_num] [raw_task_dirs]

# Example: convert stack_bowls demos for arx_x5 joint control
bash process_data.sh RoboDojo stack_bowls arx_x5 joint

# Example: create a 50-episode ablation while reading from the original task data
bash process_data.sh RoboDojo stack_bowls_50ep arx_x5 joint 50 stack_bowls
```

## Training

```bash
cd XPolicyLab/policy/Pi_05
bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id>

# Example: train a cotrain run on GPU 0 (comma-separated gpu_id for multi-GPU)
bash train.sh RoboDojo cotrain arx_x5 joint 0 0
```

Checkpoints land in `checkpoints/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>-<seed>/`; at eval time `ckpt_name` may be the short run name (auto-combined into that directory name), the full run-directory name, or a path to a checkpoint directory. By default training reads the LeRobot repo produced by `process_data.sh` (`<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>`); override with `OPENPI_LEROBOT_REPO_ID` when reusing an existing dataset. `train.sh` sets `fsdp_devices=1` for one visible GPU and `2` for multi-GPU by default (override with `OPENPI_FSDP_DEVICES`).

## Evaluation

```bash
cd XPolicyLab/policy/Pi_05
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_uv_env> <eval_env_conda_env>

# Example: evaluate a trained cotrain checkpoint on stack_bowls
bash eval.sh RoboDojo stack_bowls RoboDojo-cotrain-arx_x5-joint-0 arx_x5 joint 0 0 0 uv <eval_env_conda_env>
```

`EVAL_ENV_TYPE=debug` runs the offline wiring check (no simulator); leave it unset or set `EVAL_ENV_TYPE=sim` for RoboDojo simulation. For split-machine deployment via `setup_eval_policy_server.sh` / `setup_eval_env_client.sh`, follow the [Deployment Flow](../../README.md#-deployment-flow).

## Configuration

`deploy.yml` keys to check before evaluation: `checkpoint_num`, `result_dir`, `obs_transform_pipeline`, `policy_uv_env_path`, `train_config_name` (must match the config used by `train.sh`), `repo_id`.

Environment variables used by the adapter scripts:

| Variable | Notes |
|---|---|
| `OPENPI_LEROBOT_REPO_ID` | Overrides the LeRobot repo id used by `train.sh`; defaults to `<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>`. |
| `OPENPI_FSDP_DEVICES` | Overrides the FSDP device count passed to OpenPI training. |
| `OPENPI_TRAIN_CONFIG_NAME` | Overrides the training config; defaults to `pi05_base_aloha_full_sim_arx-x5_seed_0`. |
| `OPENPI_DATA_MODE` | Data-processing mode passed to `openpi/scripts/process_data.py`; defaults to `image`. |
| `OPENPI_LOCAL_CACHE_ROOT` | Per-host local cache root for the HF datasets / JAX compilation caches; defaults to `/tmp/openpi-cache-$(hostname)`. |

`OPENPI_ROOT` and `OPENPI_SRC` are additional overrides consumed by the local scripts.

## Inference Sampling Capabilities

The adapter exposes six explicit WebSocket sampling modes:

| Mode | Adapter method | OpenPI path |
|---|---|---|
| `default` | `get_action` | unchanged official single-sample inference |
| `rtc` | `get_action_rtc` | JAX Pi-guided conditioning hook |
| `aac` | `get_action_aac` | one prefix/KV-cache pass followed by an `N`-sample denoising batch |
| `paint` | `get_action_paint` | paper Algorithm 1: naive forward, backward Euler, prefix noise repaint, final forward |
| `autohorizon` | `get_action_autohorizon` | third-step action self-attention plus the pinned official bidirectional soft-pointer |
| `dvac` | `get_action_dvac` | final-step clean-estimate variance and the paper's rolling adaptive prefix rule |

AAC accepts `{"mode": "aac", "num_samples": N}` for exactly one observation and returns `N`
native action chunks. It is JAX-only and cannot be combined with RTC conditioning. The adapter does
not calculate entropy, robot kinematics, motion thresholds or candidate selection; those remain
client/runtime responsibilities. Calls that omit AAC parameters continue through the original
single-sample callable.

PAINT accepts `{"mode": "paint", "action_prefix": A[s:s+d], "delay_steps": d}`. The adapter
normalizes the raw robot-unit prefix with the same official input transform as model actions, then
runs `3N` velocity evaluations without gradients. The public PAINT repository currently contains
documentation rather than source code, so this path is an explicit paper reproduction of
arXiv:2606.19774, not an upstream-code claim.

DVAC accepts `{"mode": "dvac", "tail_steps": 5, "alpha": 2.0,
"rolling_window_size": 5, "min_execution_steps": 1, "max_execution_steps": H}`. It reuses the
existing JAX Euler velocity evaluations, computes Equation 4 over the valid normalized action
dimensions, and keeps the rolling threshold state in the Pi05 adapter. No author repository was
located, so this path is an explicit paper reproduction of arXiv:2606.03847v1 rather than an
official-code port.
