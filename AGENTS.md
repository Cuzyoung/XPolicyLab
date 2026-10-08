# XPolicyLab Agent Guide

XPolicyLab wraps each robot policy as a self-contained adapter under `policy/<POLICY>/`. The policy
server imports it as `XPolicyLab.policy.<POLICY>.model`, so this checkout is always used as a
package inside a parent workspace — never as the top-level project.

- Submission standard: [CONTRIBUTING.md](CONTRIBUTING.md). Reference adapter: `policy/demo_policy/`.
  Data formats: [README](README.md#-standard-data-formats).

The rules below apply to every change in this repo. Their rationale is in CONTRIBUTING.md.

## Images are RGB from end to end

`decode_image_bit` and `decode_obs_images` return RGB, and the policy server hands `update_obs` /
`update_obs_batch` RGB. Treat this as settled: do not re-derive it from the usual "OpenCV returns
BGR" rule, which does not apply, because XPolicyLab buffers are encoded from RGB arrays and
`cv2.imencode` / `cv2.imdecode` carry channels through JPEG in the order they were given. A
`COLOR_BGR2RGB` added to "fix" a decode is always a bug: it trains on BGR and evaluates on RGB.

No channel conversion belongs in conversion, training, or eval code. Two exceptions only:

- Medium adapters: `COLOR_RGB2BGR` immediately before `cv2.VideoWriter.write(...)`, and
  `COLOR_BGR2RGB` immediately after `cv2.VideoCapture.read()`.
- A deliberate RGB→BGR conversion for a checkpoint trained on BGR data, opt-in through a documented
  `deploy.yml` key that defaults to RGB (see `input_color_order` in `policy/Dexora_1B`).

## Decoding goes through the shared helpers

`model.py` never decodes. The server decodes every observation it forwards, including for custom
RPCs, so `obs["vision"][<camera>]["color"]` is already an array; adapters only reshape, cast, resize.

Offline code — conversion scripts and training dataloaders — decodes only with `decode_image_bit`
from `XPolicyLab.utils.process_data`. Never hand-roll `cv2.imdecode` / `np.frombuffer` / PIL
decoding: RoboTwin and RoboDojo legacy image-bit layouts are only handled correctly by that
function. Mechanically, `cv2.imdecode` must not appear outside `utils/process_data.py`.

## Paths and dimensions come from the shared helpers

Deployment adapters read action dimensions through
`get_robot_action_dim_info(model_cfg)` in `XPolicyLab.utils.process_data`.
Self-contained recipes declare:

```yaml
robot_action_dim_info: {arm_dim: [6, 6], ee_dim: [1, 1]}
num_envs: 1
```

The helper returns a copy of the explicit layout. `get_action_dim(model_cfg)` sums
joint/tool coordinates; `get_batch_size(model_cfg)` reads `num_envs` (one by default
for explicit layouts). EE representation widths still belong to the model adapter;
do not equate arm joint DOFs with quaternion/pose widths. Pass the complete config,
not just its `env_cfg_type`, so experiment overrides reach the adapter.

`env_cfg_type` may still select a checkpoint or model profile. The old string helper
calls remain for external RoboDojo/RoboTwin workspaces with their own parent
`env_cfg/` registry. That compatibility path is not a template for new integrations.
Do not recreate a parent registry for a deployment using explicit model configuration.

Dataset conversion can use the same explicit recipe: see the README's
[deployment layout section](README.md#explicit-deployment-layout).
Legacy training scripts using `utils/get_action_dim.sh` still read this repository's
`utils/robot/_robot_info.json`; they do not use the removed ManiMux parent registry.
Preserve their behavior when editing existing training paths and document which
layout source a particular entry point consumes.

- **Importable root** in `policy/<POLICY>/model.py` follows the existing package setup;
  keep model imports independent of ManiMux and its hardware dependencies.
- **Checkpoints** resolve through `XPolicyLab.utils.checkpoint_resolver`, never by
  re-deriving checkpoint directory names in individual model adapters.

## deploy.yml

`policy_name` must equal the directory name. Keep the full key set from
`policy/demo_policy/deploy.yml` — including `protocol: ws`, `host` and `port` — even where the
scripts have a default; per-run fields are overridden at launch.
