# LingBot-VLA2

**Contributor:** Cuzyoung | **Paper:** LingBot-VLA 2.0: From Foundation to Application | **arXiv:** [2607.06403](https://arxiv.org/abs/2607.06403) | **Original code:** [Robbyant/lingbot-vla-v2](https://github.com/Robbyant/lingbot-vla-v2)

This adapter integrates the pinned official LingBot-VLA 2.0 implementation with
the XPolicyLab model-server, data-processing, post-training and evaluation
contracts. It currently supports `env_cfg_type=yam_dual`, `action_type=joint`,
three RGB cameras and absolute `12` arm joints plus `2` grippers. The nested
upstream submodule lives in `lingbot_vla_v2/` at revision
`951475ae1b1d87553e7dc47c97b53a3d695c0d13`.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

Initialize nested submodules and create the policy uv environment:

```bash
git submodule update --init --recursive
cd XPolicyLab/policy/LingBot_VLA2
bash install.sh
```

The default environment is `<workspace>/envs/lingbot-vla2/.venv`. Set
`LINGBOT_VLA2_ENV_DIR` to use another location. See [INSTALLATION.md](INSTALLATION.md)
for FlashAttention wheels, model assets and training hardware notes.

## Data Processing

`process_data.sh` converts standard XPolicyLab HDF5 trajectories into a local
LeRobot v3 dataset with these exact model features:

- `observation.state.arm.position[12]` and `action.arm.position[12]`;
- `observation.state.effector.position[2]` and `action.effector.position[2]`;
- `camera_top`, `camera_wrist_left`, `camera_wrist_right`, RGB throughout.

It then runs the official LingBot norm-stat computation unless
`LINGBOT_VLA2_SKIP_NORM_STATS=1` is set:

```bash
bash process_data.sh <bench_name> <ckpt_name> yam_dual joint \
  [expert_data_num] [raw_task_dirs]

# Example: data/yam_real/pick_place/yam_dual/data/episode_*.hdf5
bash process_data.sh yam_real lingbot_vla2_yam yam_dual joint 60 pick_place
```

Input defaults to `<workspace>/data/<bench_name>/`; output defaults to
`data/<bench>-<ckpt>-yam_dual-joint/`. Override them with
`XPOLICYLAB_DATA_ROOT` and `LINGBOT_VLA2_DATASET_PATH`. The converter never
deletes an existing output directory.

## Training

Set the foundation model and Qwen3-VL processor, then run the standard XPolicy
training entry:

```bash
export LINGBOT_VLA2_MODEL_PATH=/path/to/lingbot-vla-v2-6b
export LINGBOT_VLA2_TOKENIZER_PATH=/path/to/Qwen3-VL-4B-Instruct

bash train.sh <bench_name> <ckpt_name> yam_dual joint <seed> <gpu_id>

# Example
bash train.sh yam_real lingbot_vla2_yam yam_dual joint 0 0,1,2,3
```

The wrapper calls the official `tasks/vla/train_lingbotvla.py` with
`training/yam_dual.yaml`. It writes the standard run directory
`checkpoints/<bench>-<ckpt>-yam_dual-joint-<seed>/`, preserves the official
`lingbotvla_cli.yaml`, saves HF checkpoints under
`checkpoints/global_step_*/hf_ckpt/`, and copies the matching norm stats and
robot config beside the training config.

Useful overrides are `LINGBOT_VLA2_MAX_STEPS`, `LINGBOT_VLA2_SAVE_STEPS`,
`LINGBOT_VLA2_MICRO_BATCH_SIZE`, `LINGBOT_VLA2_GRAD_ACCUM_STEPS`,
and `LINGBOT_VLA2_ACTION_HORIZON`.

## Evaluation

Run the standard same-machine XPolicy workflow:

```bash
bash eval.sh <bench_name> <task_name> <ckpt_name> yam_dual joint <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_env_or_uv_path> <eval_env>

# Offline XPolicy serialization/action-contract loop
EVAL_ENV_TYPE=debug bash eval.sh \
  yam_real pick_place lingbot_vla2_yam yam_dual joint 0 0 0 uv envs/yam
```

The debug client accepts a Conda environment, `uv`, an absolute virtualenv
path, or a workspace-relative virtualenv path such as `envs/yam`. `uv`
defaults to `<workspace>/envs/yam`; override it with
`LINGBOT_VLA2_EVAL_ENV_DIR`. Simulation and real-environment clients retain
XPolicy's standard Conda workflow.

For split-machine deployment, use `setup_eval_policy_server.sh` and
`setup_eval_env_client.sh` as documented in the
[Deployment Flow](../../README.md#-deployment-flow). An explicit deployment
config can be selected with `LINGBOT_VLA2_DEPLOY_CONFIG=/path/to/deploy.yml`.

## Model Assets

One deployable run contains:

```text
checkpoints/<bench>-<ckpt>-yam_dual-joint-<seed>/
├── lingbotvla_cli.yaml
├── norm_stats.json
├── robot_config.yaml
└── checkpoints/global_step_<N>/hf_ckpt/
    ├── config.json
    ├── model.safetensors.index.json
    └── model-*.safetensors
```

The server config points directly to these four artifacts and declares the
action horizon and native waypoint frequency. The public foundation checkpoint
alone is not a YAM post-trained policy.

## Configuration

`deploy.yml` accepts `model_root`, `training_config_path`, `robot_config_path`,
`norm_stats_path`, and the local Qwen processor directory `qwen3vl_path`
directly. The adapter passes that directory through the official
`QWEN3VL_PATH` loader override without editing the saved training config. In
the standard positional XPolicy workflow, `ckpt_name` resolves the run
directory and the adapter selects its latest complete `global_step_*/hf_ckpt`;
the three matching metadata files are read from that run directory.
`action_horizon` and `native_hz` remain explicit. Other model-specific keys are
`lingbot_vla2_root`, `official_source_revision`,
`checkpoint_variant`, `checkpoint_source`, `norm_stats_role`,
`policy_uv_env_path`, `use_bf16`, `use_fp32`, `use_compile` and
`default_prompt`.

## Notes

- The normal path delegates model construction, `FeatureTransform`,
  normalization and 10-step flow sampling to the official implementation.
- The XPolicy adapter preserves the checkpoint's native action semantics.
  ManiMux uses `lingbot_vla2_yam` to convert anchor-relative arm actions to
  canonical absolute joints while leaving grippers absolute.
- `rtc.py` is an XPolicy sampler extension and is not part of the upstream
  LingBot-VLA2 release; it must be evaluated separately from normal inference.
- Training, simulator task success and real-robot success require separate
  evidence. A successful debug loop validates wiring, not policy quality.
