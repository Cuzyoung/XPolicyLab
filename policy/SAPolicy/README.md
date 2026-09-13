# SAPolicy

**Contributor:** ManiMux SAPolicy integration | **Paper:** not supplied | **arXiv:** not supplied | **Original code:** https://github.com/Ericonaldo/SpatialAlignPolicy

The model, preprocessing, normalizer, diffusion sampler and native ABC training
loader are vendored under `upstream/`. Supported: `env_cfg_type=yam_dual`,
`action_type=ee`. Model processes never open cameras, CAN or robot drivers;
ManiMux owns FK/IK and execution.

For shared conventions, see the [XPolicyLab README](../../README.md). Official
results are listed on the [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

From the parent workspace, with `uv` installed:

```bash
bash XPolicyLab/policy/SAPolicy/install.sh [path/to/policy-venv]
# Default: creates XPolicyLab/policy/SAPolicy/.venv
bash XPolicyLab/policy/SAPolicy/install.sh
```

The installer uses Python 3.11, PyTorch 2.5.1+cu121, torchvision 0.20.1+cu121,
and pinned YAM dependencies in `requirements.txt`. The environment is isolated
from hardware dependencies. Do not use the broad upstream requirements as this
adapter's installation recipe. Register robots in both files documented in
[AGENTS.md](../../AGENTS.md); ManiMux already registers `yam_dual` with arm
sizes `[6,6]` and gripper sizes `[1,1]`.

## Data Processing

The entry reads native ABC episodes directly: `train/episode_*/` contains
`states_actions.bin` (float64, state14 + command14),
`combined_camera-images-rgb.mp4`, and `episode_metadata.json`. Optional TCP
sidecars follow the upstream loader's schema. RoboDojo HDF5 conversion is not
supported. Native video decoding uses PyAV RGB; no image re-encoding is needed.

```bash
bash XPolicyLab/policy/SAPolicy/process_data.sh \
  --config /path/to/resolved-training.yaml --output /path/to/data-check.json
```

This instantiates the actual loaders and checks first/last samples through video
decode, FK and transforms; it does not audit every frame. Normalization uses the
configured training statistics. A missing `normalizer_load_path` fails. To fit
new training statistics, explicitly set that key to null and set
`normalizer_save_path` to a new output file in the dataset configuration.

For the MV51 recipe, set existing paths in `SAPOLICY_TELEOP_DATA`,
`SAPOLICY_ABC_REAL_DATA`, `SAPOLICY_ABC_SIM_DATA`, `SAPOLICY_STATION_MJCF`,
`SAPOLICY_NORMALIZER`, `SAPOLICY_BACKBONE`, and a new `SAPOLICY_TRAIN_OUTPUT`:

```bash
bash XPolicyLab/policy/SAPolicy/process_data.sh \
  --config XPolicyLab/policy/SAPolicy/configs/train_mv51.yaml \
  --output data/sapolicy/native-data-check.json
```

Supply the matched station MJCF and meshes. Measured joints produce input state;
commanded joints produce action labels via `left_grasp_site`/`right_grasp_site`
FK, with no extra 0.12 m shift. Set each dataset's `tcp_labels_root` when TCP
supervision is available; absent labels follow the upstream validity-mask path.

## Training

`train.sh` calls the vendored Lightning training entry with its dataset,
normalizer binding, optimizer and model source. Native data/training entrypoints
use a resolved YAML and optional `key=value` overrides instead of benchmark-name
positional arguments. Set `SAPOLICY_ENV` to select another policy environment;
GPU selection uses `CUDA_VISIBLE_DEVICES`.

```bash
bash XPolicyLab/policy/SAPolicy/train.sh --config /path/to/resolved-training.yaml --check
bash XPolicyLab/policy/SAPolicy/train.sh \
  --config XPolicyLab/policy/SAPolicy/configs/train_mv51.yaml \
  warm_start_ckpt=/path/to/abc-ema.ckpt ckpt_type=ema
```

The example retains three dataset branches, 50-step actions and top roll
augmentation. Actual datasets and optional labels must be supplied. The inference
recipe `configs/mv51.yaml` intentionally has no dataset paths. Training requires
`logger: null`, an explicit checkpoint callback and a new output directory, or
explicit `resume_training=true`. Existing output is never deleted. Cluster job
submission and external experiment tracking are outside this entry.

## Evaluation

The standard 10 positional arguments are unchanged. Arguments 9 and 10 accept
a venv directory, a Python executable, or a conda environment name. `eval.sh`
cleans up its temporary shared WebSocket server.

```bash
EVAL_ENV_TYPE=debug bash XPolicyLab/policy/SAPolicy/eval.sh \
  ManiMux put_bottles /absolute/path/to/model-bundle yam_dual ee 0 0 0 \
  /absolute/path/to/policy-venv /absolute/path/to/client-venv

# Run from ManiMux root after preparing the bundle below.
EVAL_ENV_TYPE=debug DEBUG_OBS_ENCODED=1 SAPOLICY_EVAL_BATCH=true \
 bash XPolicyLab/policy/SAPolicy/eval.sh \
  ManiMux put_bottles "$PWD/checkpoints/finetuned/sapolicy/teleopMV51" yam_dual ee 0 0 0 \
  "$PWD/XPolicyLab/policy/SAPolicy/.venv" "$PWD/XPolicyLab/policy/SAPolicy/.venv"
```

Debug validates real-weight inference and action dictionaries on synthetic
observations. It does not measure task success. This YAM checkpoint does not
support the RoboDojo simulator. Real deployment uses ManiMux:

```bash
XPolicyLab/policy/SAPolicy/.venv/bin/python scripts/servers/sapolicy_yam_server.py \
  --config configs/sapolicy/yam/server/teleopMV51/raw.yaml
envs/yam/.venv/bin/manimux serve \
  --config configs/sapolicy/yam/infra/teleopMV51/top.yaml
```

The parent [YAM runbook](../../../docs/sapolicy-yam-runbook.md) covers cameras,
GUI and `gemini305.yaml`/`gemini335.yaml`. Prepare and Start are separate operator
actions; both wrists stay fixed across external views.

## Model Assets

Resolution uses `XPolicyLab.utils.checkpoint_resolver`: explicit path,
`ckpt_name` path, standard run-directory name, then `checkpoints/<ckpt_name>`.
Bundles contain `model.ckpt`, `resolved_config.yaml`, `backbone.pth` and
`action_normalizer.pt`. An explicit checkpoint file with `cfg_file`,
`backbone_path`, and `normalizer_path` overrides is also supported.

```bash
XPolicyLab/policy/SAPolicy/.venv/bin/python XPolicyLab/policy/SAPolicy/prepare_assets.py \
  --checkpoint checkpoints/finetuned/sapolicy/abc130k/ft_teleopMV51_from_abc_notcpkv_rot5_bs1024_raw.ckpt \
  --backbone /path/to/dinov2_vitl14_pretrain.pth \
  --normalizer /path/to/bottles_norm_stats_real_h50_state_bs1024_lr057.pt \
  --output checkpoints/finetuned/sapolicy/teleopMV51 \
  --checkpoint-sha256 49505482141081c6dd65e93ca1a4774c98a87556ad40789c6d5e5fb6a66e3011
```

This copies and verifies supplied assets; no public download location is assumed.
`assets.json` records sizes and hashes. Assets stay outside Git. The validated
MV51 RAW checkpoint SHA-256 is
`49505482141081c6dd65e93ca1a4774c98a87556ad40789c6d5e5fb6a66e3011`;
normalizer SHA-256 is
`4b38c36dea1ec739c5e6434799fb23948cf285ab973cb514b622960a006d1588`.

## Configuration

| Key | Meaning |
| --- | --- |
| `output_format` | `action_dict` default; explicit `packed_ee_wire` retains legacy xyzw arrays |
| `camera_names`, `camera_map` | Model camera order and observation aliases; debug maps top/left/right to cam_head/cam_left_wrist/cam_right_wrist |
| `action_horizon` | Must match training; MV51 uses 50 |
| `use_ema`, `ckpt_type` | MV51 validation uses false / raw |
| `tcp_forward_offset_m` | 0 for YAM grasp-site poses |
| `model_path`, `cfg_file`, `backbone_path`, `normalizer_path` | Bundle or explicit asset paths |
| `checkpoint_source` | Optional `sha256:<digest>`; verified before advertising backend identity |
| `workspace` | Inference output directory |
| `device` | Model device, default CUDA |
| `dry_run` | Explicit contract-test mode returning current pose; default false |

Standard observations carry RGB arrays, per-camera `intrinsic_matrix`,
`state.left_ee_pose`/`right_ee_pose` (position + **wxyz**) and scalar
`left_ee_joint_state`/`right_ee_joint_state` (0 closed, 1 open). The previous
`additional_info.sapolicy` observation remains supported. ManiMux supplies native
camera dimensions there to preserve intrinsics scaling after wire resize.

`get_action()` returns dictionaries with `left_ee_pose`, `right_ee_pose`,
`left_ee_joint_state`, and `right_ee_joint_state`. Actions are absolute in the
training station frame. Native sampled actions use `[poseL9,poseR9,gripL,gripR]`;
input state uses `[poseL9,gripL,poseR9,gripR]`. Batch inference is sequential with
independent histories keyed by `env_idx`, without a vectorized speedup. Reset
clears all histories. The full-horizon DiT backend advertises `default` and `rtc`;
dry-run and incompatible heads advertise only `default`.
PAINT/AAC/DVAC/AutoHorizon hooks remain unsupported and are not advertised.
A separate `sapolicy_root` is retained only for explicitly
selected legacy source compatibility; the standard default is policy-local.

## RTC and execution smoothing

`get_action_rtc(sampling)` implements VJP inpainting in the actual DiT Euler
sampler, with the same noise-to-data time convention and correction as Pi05.
It does not update model weights or overwrite sampled actions with a fixed prefix.
The shared WebSocket server accepts:

```python
sampling = {
    "mode": "rtc",
    "action_condition": absolute_ee_actions,  # [50,16], left pose7/grip, right pose7/grip
    "condition_weights": soft_mask,          # [50], finite values in [0,1]
    "beta": 5.0,                             # positive, finite guidance cap
}
```

Conditions always use absolute model-frame **WXYZ** poses, including when the
output is `packed_ee_wire`. All pose quaternions must be nonzero. ManiMux converts
its joint timeline to this contract with YAM FK and the observation calibration.
SA converts each target relative to the **current observation**, using the same
TCP convention, column-based rotation6D and checkpoint action normalizer as its
ordinary action path. No extra TCP offset is introduced.

At each step, the sampler estimates clean actions as `x - t*v`, backpropagates
the weighted overlap error to `x`, and corrects `v` before its Euler update.
Only the action input receives gradients; the visual features/KV are reused.
RTC uses the eager action field because it needs this gradient. The ordinary
sampler keeps its original path, and zero weights produce identical actions
under the same seed. Conditions are scoped to one serialized call and cleared
even after errors; reset also clears observation histories. RTC uses single-env
`INFER`; ordinary batch calls retain their isolated sequential histories.

Execution smoothing belongs to ManiMux. The MV51 `top.yaml`, `gemini305.yaml`
and `gemini335.yaml` profiles select its existing 8 Hz lowpass smoother
(`tracking_mode: legacy`), with a 25-step chunk prefix. Matching `*-rtc.yaml`
profiles select the shared RTC scheduler, a 25-step execution floor, initial
delay 4, ten-sample delay buffer and beta 5. The scheduler adjusts for measured
inference plus decode latency. Both variants keep software velocity and
acceleration caps explicitly disabled, use paired process IK and continuous
grippers, and do not enable braking-only grasp/release guards. Horizon 50,
action interval 1/30 s, hardware control 100 Hz and model assets are unchanged.

Validate the real sampler without hardware:

```bash
XPolicyLab/policy/SAPolicy/.venv/bin/python -m XPolicyLab.policy.SAPolicy.validate_checkpoint \
  --checkpoint /absolute/path/to/teleopMV51 --rtc \
  --output data/sapolicy/rtc-checkpoint-validation.json
```

This checks zero-mask equivalence, nonzero guidance with three seeds, finite
outputs, parameter gradients, reset, default-path isolation and ordinary batch
behavior. Offline guidance accuracy and timing do not establish robot task success.

## Notes

The vendored snapshot combines checkpoint-associated `SpatialAlignVLA_ft_rot5`
files and five MV51 `SpatialAlignVLA_mvft` files (dataset, policy, DiT, auxiliary
head, trunk). Wrapper/crop fixes originated from upstream revision
`9ca69b9beab6e97f7d596d70883f7ee630b31c18` with local changes. It is not a pristine
upstream checkout. Integration changes additionally accept resolved configuration
objects, preserve training output without prompts, and add the scoped RTC sampler
in `models/action_head/{dit,rtc}.py` plus inverse action conditioning in the eval
wrapper. Original headers and
DINOv2's Apache-2.0 license remain. The supplied source had no root license;
this adapter grants no new license.

Validated on 2026-09-13: fresh installation; real-weight shared debug with plain
and encoded RGB, including batch; finite 50x16 native actions decoded offline to
50x14 YAM joints; fixed-seed difference 0 against the archived active wrapper;
native dataset/FK check and one GPU optimizer step with checkpoint save using a
synthetic fixture. The smoke test does not reproduce training quality. Existing
three-view real-robot records predate the standard-interface switch; integration
validation started no new physical rollout.
