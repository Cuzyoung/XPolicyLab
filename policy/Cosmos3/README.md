# Cosmos3

**Contributor:** Cuzyoung | **Paper:** Cosmos 3 | **arXiv:** Not listed | **Original code:** [NVIDIA/cosmos](https://github.com/NVIDIA/cosmos), [NVIDIA/cosmos-framework](https://github.com/NVIDIA/cosmos-framework)

This eval-only adapter exposes NVIDIA's released Cosmos3 DROID policies through
the standard XPolicyLab model contract. The pinned official inference source is
the nested `cosmos-framework/` submodule at revision
`c7e8d76b5da8aeae38cdac91c6cfd57185b2f6bc`. XPolicyLab translates only RGB
camera, prompt, joint-state and action field names; it does not replace or
modify official model loading, transforms, normalization, image composition,
sampling, denoising or action post-processing.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

Initialize the official source and build its policy-server environment with the
same dependency groups used by NVIDIA's current DROID server recipe:

```bash
git submodule update --init --recursive
cd XPolicyLab/policy/Cosmos3
bash install.sh
```

The default environment is `<workspace>/envs/cosmos3/.venv`. Override it with
`COSMOS3_ENV_DIR`; override the official CUDA dependency group with
`COSMOS3_CUDA_GROUP` only when the local CUDA stack requires another group from
the pinned framework's `pyproject.toml`. The pinned official CUDA wheels require
CPython 3.13, which is therefore the default; `COSMOS3_PYTHON` is available only
for an official dependency group that publishes wheels for another Python ABI.

## Data Processing

Unsupported in this eval-only integration. The public checkpoints are already
post-trained on DROID, and this adapter does not introduce a replacement data
pipeline.

## Training

Unsupported in this eval-only integration. Training remains entirely in the
official Cosmos framework; no XPolicy training wrapper or modified denoising
implementation is included.

## Evaluation

Run one model forward pass without starting a server, simulator, camera or
robot:

```bash
cd /path/to/workspace
envs/cosmos3/.venv/bin/python \
  XPolicyLab/policy/Cosmos3/offline_infer.py \
  --config XPolicyLab/policy/Cosmos3/deploy.yml \
  --prompt "Pick up the object."
```

Run the standard XPolicy debug loop after the checkpoint has been cached:

```bash
cd XPolicyLab/policy/Cosmos3
EVAL_ENV_TYPE=debug bash eval.sh \
  Cosmos3 offline edge droid_single joint 0 0 0 uv envs/yam
```

Both commands perform policy inference only. They do not start a simulator or
real-robot client.

## Model Assets

The default config uses `nvidia/Cosmos3-Edge-Policy-DROID`. The official service
downloads the checkpoint through its own `CheckpointDirHf` path. A consolidated
local safetensors directory may instead be set as `checkpoint_path`.

For `nvidia/Cosmos3-Nano-Policy-DROID`, override the official server fields as
documented by NVIDIA: use the Nano checkpoint and set
`format_prompt_as_json: null` and `guidance_interval: null`.

## Configuration

`deploy.yml` keeps every standard XPolicy key. `checkpoint_path`,
`hf_revision`, `domain_name`, `decode_video`, `sampler`, `deterministic_seed`,
`guidance`, `guidance_interval`, `num_steps`, `shift`, `resolution`,
`conditioning_fps`, `action_chunk_size`, `action_dim`, `image_height`,
`image_width`, `action_space`, `use_state`, `history_length` and
`format_prompt_as_json` are passed directly into NVIDIA's
`RobolabServerArgs`. The three camera-name fields and `default_prompt` belong
only to the surrounding XPolicy observation mapping.

## Notes

- Supported contract: `env_cfg_type=droid_single`, `action_type=joint`, one
  7-DoF arm plus one 1-DoF gripper, absolute joint-position chunks.
- The default Edge recipe uses three RGB images, a 32-step by 8-D native action
  chunk, 15 Hz conditioning, four denoising steps and NVIDIA's published prompt
  formatting and guidance interval.
- The released DROID checkpoints are not YAM dual-arm checkpoints. Finite output
  proves offline inference wiring only, not simulator or robot task success.
