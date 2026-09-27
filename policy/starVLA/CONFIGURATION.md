# StarVLA deployment configuration

Use `deploy.yml` for the standard RoboDojo entry points or `deploy_chunk.yml`
when the caller owns action scheduling. Standard XPolicyLab keys are documented
in the [contribution guide](../../CONTRIBUTING.md); `policy_name` is always `starVLA`.

## Backend and observation settings

| Key | Meaning |
| --- | --- |
| `model_backend` | `inprocess` loads vendored weights in the shared server; `websocket` connects to the explicit benchmark model process |
| `action_output` | `chunk` predicts on every request; `step` caches a chunk and returns one action at a time |
| `checkpoint_path` | Weight file with adjacent run configuration and statistics |
| `starvla_root` | Source directory relative to this policy directory; in-process mode requires `source_starvla` |
| `starvla_server_host`, `starvla_server_port` | Upstream model endpoint in websocket mode |
| `device`, `use_bf16` | In-process model placement and precision |
| `unnorm_key` | Statistics key; null selects the runtime's unambiguous default, never an arbitrary dataset |
| `include_state` | Boolean, or `auto` to read `datasets.vla_data.include_state` from `config.yaml`, then `config.full.yaml`; missing metadata is an error |
| `state_type` | `joint` or canonical `ee` (`xyz`, quaternion `wxyz`), independent of output representation |
| `camera_names` | Ordered camera names or explicit alias lists; null retains head/left-wrist/right-wrist benchmark aliases |
| `image_size` | Positive integer `[width, height]`; default `[224, 224]` |
| `execute_horizon` | Actions consumed before replanning in step mode; ignored in chunk mode |
| `use_ddim`, `num_ddim_steps` | Forwarded upstream sampler settings; flow heads also retain their checkpoint denoising schedules |
| `action_indices` | Full permutation from native actions to standard output order; null is identity |
| `state_indices` | Full permutation from standard raw state to native input order; null is identity |
| `require_runtime_contract` | Enabled by default; mandatory for in-process inference |
| `required_pi_v3_forward` | Optional required forward identity, set by the released-checkpoint launcher |
| `eef` | Required native EEF encoding when `action_type: ee`; see below |
| `enabled_sampling_modes` | Explicit specialized samplers, empty by default |

For a 14D checkpoint ordered `[left joints, right joints, left gripper, right gripper]`:

```yaml
action_indices: [0, 1, 2, 3, 4, 5, 12, 6, 7, 8, 9, 10, 11, 13]
state_indices: [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 6, 13]
```

These are inverse order mappings, not conversions of units, relative actions or
embodiments. State and action widths may differ; configure each independently.

## EEF output

`action_type: ee` requires `eef.rotation` (`axis_angle` or `quaternion_wxyz`) and
`eef.semantics`:

- `absolute_per_arm_base_xyz_wxyz`: absolute target in each arm's base frame.
- `delta_observation_base_xyz_wxyz`: deltas anchored to the observation in base coordinates.
- `delta_observation_tool_xyz_wxyz`: deltas anchored to the observation in tool coordinates.
- `delta_step_base_xyz_wxyz`: feedback deltas requiring the execution-side measured state.

`translation_scale` and `rotation_scale` are three positive values, defaulting
to ones. `clip_input` and `gripper_threshold` are optional explicit controller
conversions. Quaternion output must be unit length and cannot use rotation
scaling or input clipping. The server returns canonical `xyz + wxyz` poses and
gripper values; the consuming robot adapter owns anchoring and FK/IK.

## Sampling

OFT and FAST provide default inference. GR00T and PI-v3 flow heads may enable:

```yaml
enabled_sampling_modes: [rtc, paint, aac, autohorizon, dvac]
```

This requires `model_backend: inprocess`, `action_output: chunk` and
`action_type: joint`. Unsupported requests fail; EEF conditioning is not implemented.

- RTC normalizes raw joint conditions through training transforms and applies gradient guidance.
- PAINT performs forward, inverse and repaint passes using the requested prefix.
- AAC produces independent action candidates after encoding the observation once.
- AutoHorizon requires action self-attention and at least three denoising steps.
- DVAC requires at least two denoising steps; its tail must fit that schedule.
  Episode-local calibration is cleared by `reset()`.

Sampler tests establish implementation mechanics, not published task performance.
The AutoHorizon port is based on the [official implementation](https://github.com/hatchetProject/AutoHorizon).

## Environment bindings

For model assets, `STARVLA_BASE_VLM` overrides the checkpoint's VLM path and
`STARVLA_FAST_TOKENIZER` overrides its FAST processor path. The processor loader
uses the published class and BPE tokenizer directly across Transformers versions.
Qwen3 honors `framework.qwenvl.attn_implementation`; a missing requested backend
must be installed or the configuration must be changed explicitly.

The standard evaluation scripts also accept `STARVLA_CKPT_PATH`,
`STARVLA_INCLUDE_STATE`, `STARVLA_UNNORM_KEY`, `STARVLA_EXECUTE_HORIZON`,
`STARVLA_IMAGE_SIZE`, `STARVLA_MODEL_BACKEND` and `STARVLA_DEPLOY_CONFIG`.
`STARVLA_CPU_THREADS` supplies an import-time CPU limit when OMP/MKL limits are
unset. These script overrides are distinct from direct shared-server YAML keys.
