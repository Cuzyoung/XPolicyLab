# starVLA

**Contributor:** RoboDojo Team | **Paper:** StarVLA: A Versatile Vision-Language-Action Model with Efficient Training and Policy Adaptation | **arXiv:** [2604.05014](https://arxiv.org/abs/2604.05014) | **Original code:** [starVLA/starVLA](https://github.com/starVLA/starVLA)

This adapter serves StarVLA OFT, GR00T, PI-v3 and FAST checkpoints through
XPolicyLab. Model implementation and training transforms are vendored in
`source_starvla/`. Checkpoints determine the supported cameras, state and action
representation; a framework name alone does not establish robot compatibility.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

Activate an isolated Python environment before installing. The integration has
been exercised with Python 3.10; `install.sh` installs PyTorch 2.6, upstream
requirements, FlashAttention, the vendored source and XPolicyLab transport dependencies.

```bash
conda activate <policy_env>
cd XPolicyLab/policy/starVLA
bash install.sh
```

Supply a checkpoint and its matching base VLM. Set `STARVLA_BASE_VLM` to a local
VLM directory when its recorded asset path is unavailable. FAST additionally
requires `STARVLA_FAST_TOKENIZER` or
`framework.action_model.fast_tokenizer_path`. Asset loading failures propagate;
there is no fallback to a developer's local directory or another attention backend.

## Data Processing

`process_data.sh` converts RoboDojo demonstrations to LeRobot format. The output
is `data/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>/`.

```bash
bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> [expert_data_num] [raw_task_dirs]

# Convert 50 episodes from stack_bowls into a separately named dataset.
bash process_data.sh RoboDojo stack_bowls_50ep arx_x5 joint 50 stack_bowls
```

`raw_task_dirs` names one or more comma-separated source task directories under
`data/<bench_name>/`; it defaults to `ckpt_name`. A nonnumeric fifth argument is
also accepted as `raw_task_dirs`. Conversion preserves RGB through XPolicyLab's
shared decoding helpers.

## Training

The provided training entry point generates an OFT configuration from
`xpolicy_oft_vla.yaml` and requires the converted dataset's `meta/modality.json`.
It is not a new training pipeline for every supported inference architecture.

```bash
bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id> [extra_args...]

bash train.sh RoboDojo stack_bowls_50ep arx_x5 joint 0 0
```

Use comma-separated GPU IDs for multiple processes. Extra arguments are forwarded
to the upstream trainer. Checkpoints are written under
`checkpoints/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>-<seed>/`.
Overrides are `STARVLA_DATA_ROOT`, `STARVLA_DATA_MIX` and
`STARVLA_XPOLICY_DATASET_NAME`. EEF inference support does not imply an EEF
training recipe; other training workflows remain upstream-specific.

## Evaluation

The standard entry point retains RoboDojo's step-wise execution and replanning:

```bash
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_env> <eval_env>

EVAL_ENV_TYPE=debug bash eval.sh RoboDojo stack_bowls \
  RoboDojo-stack_bowls-arx_x5-joint-0 arx_x5 joint 0 0 0 <policy_env> <eval_env>
```

`EVAL_ENV_TYPE=debug` exercises the server and client without a simulator; repeat
with `DEBUG_OBS_ENCODED=1` to check server-side image decoding. Use
`EVAL_ENV_TYPE=sim` for task evaluation in a configured RoboDojo workspace.
Policy environments may be Conda names, virtualenv directories or Python paths.
See the root README for split-machine deployment.

### Released RoboDojo checkpoints

[`scripts/hf_robodojo_checkpoints.json`](scripts/hf_robodojo_checkpoints.json)
pins weight, configuration and statistics hashes for OFT, GR00T and PI-v3.
Download a release from the parent workspace:

```bash
python XPolicyLab/policy/starVLA/scripts/prepare_hf_checkpoint.py \
  --variant pi_v3 --output-dir checkpoints/pretrained/starvla/pi_v3_robodojo
```

For simulator evaluation, install RoboDojo's official assets with its
`scripts/init_assets.sh`, then run from this policy directory:

```bash
bash scripts/eval_hf_robodojo.sh pi_v3 build_tower 0 0 1 <policy_env> <sim_env> 10
```

Variants are `oft`, `groot` and `pi_v3`; replace `10` with `native` for the official
episode count. The launcher verifies assets and hashes, starts the vendored model
server and checks RGB, raw-state normalization, action width and horizon. The
released PI-v3 checkpoint also requires `canonical_interleaved` attention; older
all-cross-attention behavior is incompatible with those weights.

Useful release-launcher overrides are `STARVLA_HF_ROOT`,
`STARVLA_HF_LOCAL_FILES_ONLY=1`, `STARVLA_HF_VERIFY_ONLY=1` and
`STARVLA_ROBODOJO_NUM_ENVS` (default one). Simulator startup or GPU failures must
be resolved before reporting task scores. Offline adapter tests are not simulator
results or leaderboard reproductions.

### Full chunks for an external runtime

Use [`deploy_chunk.yml`](deploy_chunk.yml) when the calling application owns
action scheduling. This loads the vendored model inside the shared server:

```bash
# From the parent workspace, in the StarVLA environment.
python XPolicyLab/setup_policy_server.py \
  --config_path XPolicyLab/policy/starVLA/deploy_chunk.yml \
  --overrides env_cfg_type=arx_x5 \
    checkpoint_path=/path/to/run/checkpoints/steps_100000_pytorch_model.pt \
    unnorm_key=arx_x5 host=127.0.0.1 port=8500
```

Every request predicts a fresh complete chunk. The caller chooses execution
prefix, action interval, scheduling and FK/IK. No camera, CAN or robot driver is
loaded by the model service. To exercise this backend with `eval.sh`, set
`STARVLA_MODEL_BACKEND=inprocess` and `STARVLA_DEPLOY_CONFIG` to the deployment
recipe. The benchmark default remains `websocket` plus `step`; it is an explicit
mode, not a fallback when in-process loading fails.

## Model Assets

Keep the original checkpoint directory intact:

```text
run/
├── config.yaml
├── config.full.yaml
├── dataset_statistics.json
└── checkpoints/steps_100000_pytorch_model.pt
```

Weights may be `.pt` or `.safetensors`. Snapshot symlinks must retain this sidecar
layout. Normalization reuses the checkpoint's training transforms, including
state normalization and action unnormalization.

| Framework | Model contract to verify |
| --- | --- |
| `QwenOFT` | Regression head; RoboDojo joint recipes or an explicitly mapped RoboTwin few-shot checkpoint |
| `QwenGR00T` | Flow head; RoboDojo joint and LIBERO EEF checkpoints have different contracts |
| `QwenPI_v3` | Layer-wise flow head; released RoboDojo weights require canonical interleaved attention |
| `QwenFast` | Action-token generation; matching action-token VLM and FAST processor are required |

The `robotwin_fewshot50` mixture uses the existing `robotwin50` schema: 50 joint
actions, min/max normalization and a `> 0.49` gripper threshold. Native grippers
follow both arms, so its deployment must supply the correct action permutation.
LIBERO GR00T/FAST instead emit axis-angle feedback deltas; their seven native
values must not be interpreted as seven joint positions.

## Configuration

See [configuration and action contracts](CONFIGURATION.md) for all adapter-specific
keys, EEF conventions and sampler requirements. The shared HELLO handshake reports
the actual model identity and capabilities; clients should validate both.

Observations use `vision[camera].color` as a decoded HWC `uint8` RGB array and
`instruction` as a string. The server decodes bytes; the adapter orders and
resizes images, packs raw state, and applies explicit permutations. Invalid
layouts, missing state metadata, malformed actions and unsupported sampling
requests raise errors instead of being silently reinterpreted.

## Notes

The implementation is divided by responsibility:

| File | Responsibility |
| --- | --- |
| `model.py` | Model lifecycle, backend selection, batch/step/chunk output and reset |
| `observations.py` | Camera ordering, RGB resizing and raw-state packing |
| `runtime_config.py` | Checkpoint state settings and runtime-contract validation |
| `eef.py` | Native rotation/gripper encoding to canonical EEF actions |
| `sampling.py` | Shared sampling requests and episode-local DVAC calibration |
| `source_starvla/` | Model loading, training transforms and denoising/token generation |

The vendored baseline is XPolicyLab commit
`fbdc3c64e2e91c4415a790215d96df99b1562c12`, source tree
`80c13b4477a7e6311a2d283282c2ad7f95df30a7`. Its original import includes reproduction
patches and does not identify one pristine upstream revision. The PI-v3 correction
is traceable to upstream `c521decb7441c7dfea282c61dc758456bcffbb8f`. StarVLA's MIT
license is retained in `source_starvla/LICENSE`; the AutoHorizon selector retains
its Apache-2.0 license beside `autohorizon_official.py`.

Focused checks are `tests/unit/test_starvla_chunks.py` in XPolicyLab and
`scripts/test_{runtime_reproducibility,flow_sampling,fast_runtime,qwen3_attention_backend}.py`
in this policy directory. They cover adapter contracts and sampler mechanics.
`scripts/probe_checkpoint.py` separately checks real checkpoint inference; it
does not infer robot semantics from vector dimensions. Physical robot deployment
and task success require their own validation.
