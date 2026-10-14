# FastWAM

**Contributor:** RoboDojo Team | **Paper:** Fast-WAM: Do World Action Models Need Test-time Future Imagination? | **arXiv:** https://arxiv.org/abs/2603.16666 | **Original code:** https://github.com/yuantianyuan01/FastWAM

`FastWAM` adapts the Fast-WAM world-action model to XPolicyLab/RoboDojo. Integration scripts live at this directory level; the vendored upstream implementation lives in `FastWAM/`.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

Read `INSTALLATION.md` before first use; it covers setup that `install.sh` cannot fully express, such as external checkpoints, system packages, manual fallback steps, and multi-environment runtime notes.

```bash
cd XPolicyLab/policy/FastWAM
bash install.sh
conda activate <policy_env>  # e.g. fastwam
```

## Data Processing

No top-level `process_data.sh`. Training consumes a prepared LeRobot v2.1 dataset directly (see Training below); camera keys are discovered from the dataset metadata, so an export produced by the official `scripts/transform_lerobot_v21_format.py` ([Official LeRobot conversion](../../README.md#official-lerobot-conversion)) is the expected shape. Use the upstream FastWAM tooling under `FastWAM/` for the remaining preparation (dataset stats, text-embedding cache).

## Training

```bash
cd XPolicyLab/policy/FastWAM
bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id> [num_gpus]

# Example: train a cotrain run on four GPUs (the upstream model is large; multi-GPU is recommended)
bash train.sh RoboDojo cotrain arx_x5 joint 0 0,1,2,3
```

Checkpoints land in `checkpoints/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>-<seed>/`; at eval time `ckpt_name` may be the short run name, the full run-directory name, or a path to a checkpoint directory. Training expects a prepared LeRobot v2.1 dataset under `data/<dataset_id>/lerobot/` with `data/<dataset_id>/dataset_stats.json`, a matching T5 text embedding cache under `FastWAM/data/text_embeds_cache/xpolicylab/<dataset_id>/`, and the ActionDiT backbone at `FastWAM/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt`. `<dataset_id>` defaults to `<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>` (override with `FASTWAM_DATASET_ID`), and `train.sh` prints the upstream commands that generate the text cache and backbone when they are missing. The process count is inferred from a comma-separated `gpu_id` unless passed as the optional 7th argument `num_gpus`. The wrapper defaults to `FASTWAM_BATCH_SIZE=8`; you may still need to lower it or increase the GPU count depending on available memory. `train.sh` sets `DIFFSYNTH_MODEL_BASE_PATH` to `FastWAM/checkpoints` automatically.

## Evaluation

```bash
cd XPolicyLab/policy/FastWAM
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_conda_env> <eval_env_conda_env>

# Example: evaluate a trained cotrain checkpoint on stack_bowls
bash eval.sh RoboDojo stack_bowls RoboDojo-cotrain-arx_x5-joint-0 arx_x5 joint 0 0 0 <policy_conda_env> <eval_env_conda_env>
```

`EVAL_ENV_TYPE=debug` runs the offline wiring check (no simulator); leave it unset or set `EVAL_ENV_TYPE=sim` for RoboDojo simulation. For split-machine deployment via `setup_eval_policy_server.sh` / `setup_eval_env_client.sh`, follow the [Deployment Flow](../../README.md#-deployment-flow).

## Configuration

`deploy.yml` keys to check before evaluation: `action_dim`, `checkpoint_path`, `dataset_stats_path`, `sim_cfg_name`, `sim_task`, `device`, `mixed_precision`, `action_horizon`, `replan_steps`, `num_inference_steps`, `sigma_shift`.

Set `training_config_path` to the resolved `config.yaml` saved with a finetuned
checkpoint to load its model and data settings instead of the simulation defaults.
This preserves its action scheduler, dimensions, tokenizer length and processor
normalization settings. Evaluation options still come from the deployment recipe.
The known `model_factory.create_fastwam` initialization-audit wrapper maps to the
native factory for inference; training-only initializer paths are cleared because
the full finetuned checkpoint supplies those parameters. Source compatibility and
real-weight inference must still be verified separately.
For this saved-config path, decoded RGB images retain their original resolution
until the processor's evaluation transforms run. Three-camera assembly uses the
native tensor resize, crop and normalization operations from training, avoiding
an earlier OpenCV resize and an intermediate PIL/uint8 quantization.

Optional policy-specific environment overrides used by the scripts: `FASTWAM_DATASET_ID`, `FASTWAM_BATCH_SIZE`, `FASTWAM_GRADIENT_ACCUMULATION_STEPS`, `FASTWAM_NUM_WORKERS`, `FASTWAM_NUM_EPOCHS`, `FASTWAM_CKPT_SETTING`, `FASTWAM_CKPT_ROOT`, `FASTWAM_CHECKPOINT_PATH`, `FASTWAM_DATASET_STATS_PATH`, `FASTWAM_ALLOW_DUMMY_POLICY`.

## XPolicyLab sampling protocol

The adapter uses the existing XPolicyLab WebSocket server (`protocol: ws`).
`update_obs` receives decoded RGB and native robot state; `get_action` returns
the **full checkpoint horizon**, with joint rows unpacked by the shared helper.
`replan_steps` only controls the upstream RoboTwin execution queue. ManiMux
selects the execution prefix in its runtime. Serial and async both use this
ordinary inference interface; neither requires a separate model service.

`sampling_modes()` reports the loaded backend's capabilities through HELLO.
Dummy policies advertise only `default`. The real absolute-joint backend has:

| Mode | Request and behavior |
|---|---|
| `rtc` | Raw joint `action_condition: [H,D]`, `condition_weights: [H]`, positive `beta`; action-stat normalization and clean-estimate gradient guidance. |
| `paint` | Raw joint `action_prefix: [delay_steps,D]`; three passes over the native action flow grid, including backward Euler inversion. |
| `autohorizon` | No overrides; captures action queries against all visible video/action keys at the third evaluation, then runs the shared official selector. Returns the full chunk and execution-length metadata. |
| `aac` | `num_samples >= 2`; independent seeded candidates on the same observation. ManiMux's BID runtime uses these candidates and owns continuity scoring/selection. |

RTC/PAINT require the eager native flow hook, CFG scale 1 and an absolute-joint
processor without action/state transforms or delta-action conversion. AutoHorizon
also requires the MoT capture hook and at least three denoising evaluations.
Unsupported modes fail explicitly. Conditions follow the shared interleaved
packing order: left arm, left gripper, right arm, right gripper.

These samplers are local inference extensions, not official FastWAM features.
They retain the native shifted time grid, noise-minus-clean velocity convention,
and first-frame video KV cache. `inference_seed` controls the seed stream for
conditioned sampling and AAC; `reset()` resets that stream and clears observations.
Default inference retains the upstream `seed` behavior.

CPU checks cover a small native video/action MoT: zero-weight RTC and AutoHorizon
preserve fixed-noise native actions, RTC gradients are finite, PAINT reverses the
native grid, and attention capture includes video-key softmax mass. Separate
checks cover normalization, candidate diversity/reset, full horizon output,
capability filtering and the real WebSocket server dispatch. These checks do not
establish finetuned-checkpoint GPU latency, memory use or robot performance.

## Notes

- Use the same `action_type` for training and evaluation. The reference FastWAM path follows XPolicyLab's `pack_robot_state` / `unpack_robot_state` helpers directly and does not add policy-local `ee` pose conversion.
