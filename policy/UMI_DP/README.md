# UMI_DP

**Contributor:** Cuzyoung | **Paper:** Universal Manipulation Interface | **arXiv:** [2402.10329](https://arxiv.org/abs/2402.10329) | **Original code:** [UMI](https://github.com/real-stanford/universal_manipulation_interface)

This adapter reproduces the bimanual CLIP ViT-B/16 + conditional UNet Diffusion
Policy used by CalibWrist. Model, normalizer, native LeRobot dataset and training
source are vendored in `upstream/`; revisions and changes are in [UPSTREAM.md](UPSTREAM.md).
Supports `env_cfg_type=tianji_umi` or `tianji_dual`, `action_type=ee`, default DDIM
and the ported PiGDM/soft-inpaint RTC hooks. Model processes do not access hardware.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

Use Python 3.11 in a separate ordinary venv. The installer pins torch 2.7.1/cu128,
NumPy 1.26.4, diffusers 0.18.2, timm 0.9.7 and compatible Hub/OpenCV versions.
Do not run `uv sync` or `uv run` on this environment.

```bash
# From the parent workspace:
bash XPolicyLab/policy/UMI_DP/install.sh <venv-path>
# Example:
bash XPolicyLab/policy/UMI_DP/install.sh envs/umi_dp/.venv
```

Inference loads all encoder weights from the checkpoint; it never downloads a
pretrained CLIP model. Training with `pretrained: true` requires those model
assets in the Hugging Face cache or network access. Installation itself has not
been rerun in the source training environment.

## Data Processing

Input is the native TacCap LeRobot v3 export: parquet `observation.state` and
`action`, `meta/info.json`, and MP4 wrist cameras. Both 20D columns use metre
xyz, column rotation-6D and normalized aperture. The dataset verifies that
`action[t] == state[t+1]`, converts rotation-6D to **row** layout and computes
relative trajectories against the latest observation for each arm. Videos decode
with PyAV in RGB. XPolicyLab HDF5 conversion is unsupported; no HDF5/JPEG decoder
is duplicated in the adapter.

```bash
export UMI_DP_DATASET_DIR=<lerobot-root>
export UMI_DP_IMAGE_CACHE=<cache-directory>
export UMI_DP_PYTHON=<venv-path>/bin/python
bash XPolicyLab/policy/UMI_DP/process_data.sh <bench> <checkpoint-name> <robot> ee
# Example, using an existing local export:
export UMI_DP_DATASET_DIR="$PWD/datasets/pass_ball"
export UMI_DP_IMAGE_CACHE="$UMI_DP_DATASET_DIR/cache/images_224"
export UMI_DP_PYTHON="$PWD/envs/umi_dp/.venv/bin/python"
bash XPolicyLab/policy/UMI_DP/process_data.sh ManiMux pass_ball tianji_umi ee
```

This builds actual 224×224 resize-cover/center-crop wrist caches. Training opens
these arrays with mmap. Keep the cache and training data outside Git.

## Training

The real vendored Hydra/Accelerate workspace trains and saves checkpoints.
Simulator rollout is disabled because this native Tianji dataset has no bundled
simulation task. The six standard arguments are followed by optional Hydra
arguments. `utils/get_action_dim.sh` verifies hardware width 16; the policy's
20D TCP/rot6d representation is specified by task `shape_meta`.

```bash
bash XPolicyLab/policy/UMI_DP/train.sh <bench> <checkpoint-name> <robot> ee <seed> <gpu> [hydra-overrides...]
# With dataset/cache/Python environment variables from above:
bash XPolicyLab/policy/UMI_DP/train.sh ManiMux pass_ball tianji_umi ee 0 0
# Inspect a 64-action training configuration without starting training:
bash XPolicyLab/policy/UMI_DP/train.sh ManiMux pass_ball_h64 tianji_umi ee 0 0 \
  task.shape_meta.action.horizon=64 --cfg job
```

Checkpoints use the shared run name under
`policy/UMI_DP/checkpoints/ManiMux-pass_ball-tianji_umi-ee-0/checkpoints/`.
The full training run, cache generation and task success have not been validated
by this integration; configuration composition and module imports are checked.

## Evaluation

The standard ten-argument entry point accepts a venv path or conda name for each
side. A dedicated `EVAL_ENV_TYPE=debug` client supplies explicit timestamped
history and EE poses, then checks real forward, RTC, batch indexing and reset.
The generic arx_x5 debug observation is incompatible with this checkpoint.

```bash
bash XPolicyLab/policy/UMI_DP/eval.sh <bench> <task> <checkpoint-path-or-name> <robot> ee <seed> <policy-gpu> <env-gpu> <policy-env> <eval-env>
# Hardware-free real-weight protocol check, using a trusted local checkpoint:
EVAL_ENV_TYPE=debug bash XPolicyLab/policy/UMI_DP/eval.sh ManiMux pass_ball \
  "$PWD/checkpoints/pass_ball.ckpt" tianji_umi ee 0 0 0 \
  "$PWD/envs/umi_dp/.venv" "$PWD/envs/umi_dp/.venv"
# Repeat with JPEG-encoded observations:
DEBUG_OBS_ENCODED=1 EVAL_ENV_TYPE=debug bash XPolicyLab/policy/UMI_DP/eval.sh ManiMux pass_ball \
  "$PWD/checkpoints/pass_ball.ckpt" tianji_umi ee 0 0 0 \
  "$PWD/envs/umi_dp/.venv" "$PWD/envs/umi_dp/.venv"
```

Offline recorded-window validation and optional old-implementation comparison:

```bash
PYTHONPATH="$PWD:$PWD/XPolicyLab" envs/umi_dp/.venv/bin/python \
  -m XPolicyLab.policy.UMI_DP.validate --checkpoint checkpoints/pass_ball.ckpt \
  --dataset datasets/pass_ball --legacy-root /path/to/reference/CalibWrist
```

`--legacy-root` is only used by this numerical comparison utility. Serving,
training and preprocessing do not import anything from that checkout. Real robot
runs use ManiMux's thin launcher and `xpolicylab_ws`; see the parent workspace's
`docs/umi_dp-tianji-runbook.md`. No simulator/leaderboard or hardware success is
claimed by the debug loop.

## Configuration

- `checkpoint_path` (or shared `ckpt_name` resolution): trusted dill/PyTorch file,
  or a run directory. `checkpoint_file` defaults to `checkpoints/latest.ckpt`.
- `source_fps`: explicit native dataset FPS, required for old checkpoints that do
  not embed it. No training-machine dataset path is read during inference.
- `device`, `seed`: inference device and initial random seed. Random crop,
  rotation and color jitter remain active as in the checkpoint source.
- `observation_tolerance_s`: default .04, enforced for two distinct observations
  at the checkpoint's sampling interval (100ms for existing artifacts).
- `rtc_guidance`: `pigdm` or `soft_inpaint`; both use the checkpoint's own
  normalizer and DDIM scheduler. Only default/RTC capabilities are advertised.
- `expected_artifacts`: optional exact key/value identity checked at load time.
  Runtime metadata includes checkpoint/config SHA256, EMA/model key, horizon,
  action dt, first action offset, observation period and image normalization.

Horizon and dt come from the embedded `shape_meta`; H16 and H64 are supported.
With LeRobot's next-native-frame action export, the first target occurs at
`1/source_fps` after observation, then targets are separated by
`action.down_sample_steps/source_fps`. A 100ms action dt can therefore have a
33.3ms first offset. Output dictionaries use absolute per-arm base TCP
`[xyz,qw,qx,qy,qz]` and absolute aperture (0 closed, 1 open). Raw aperture
predictions are preserved; hardware adapters validate their permitted range.

Legacy SHA `1d7d9def9742b8cc549e5e785e84e61120a8beb7648da60662962fd09b5e4767`
was trained before the CLIP Normalize fix despite its serialized flag, so only
that artifact disables Normalize automatically. Newer artifacts follow their
embedded configuration.
