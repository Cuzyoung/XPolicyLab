# OpenWAM_Legato

Independent OpenWAM training and inference implementation of `legato`. The vendored
`OpenWAM/` directory is a physical source copy, not a symlink or import from
`policy/OpenWAM` or another continuation policy. ManiMux is not a dependency.
Original source: https://github.com/OpenWAM-Official/OpenWAM . See the retained
`OpenWAM/LICENSE` and `SOURCE_BASELINE.json` for the local baseline commit and
per-file hashes. This is the current local baseline, not a claim of byte identity
with the older remote YAM training checkout.

## Scope and files

- `OpenWAM/openwam/model/continuation.py`: config, prefix sampling, schedule,
  training input/target construction, explicit base-checkpoint migration.
- `OpenWAM/openwam/model/architectures/base.py`: original joint training loss,
  continuation call and original padding/scheduler-weight reduction.
- `OpenWAM/openwam/model/action_backbone/separate_action_dit.py`: per-action
  time conditioning through both cross-attention and mixed self-attention paths.
- `OpenWAM/configs/continuation/legato.yaml`: enabled default preset.
- `OpenWAM/configs/continuation/none.yaml`: native-objective control.
- `check_config.py`: CPU-only Hydra config composition and validation.

Supported: `dual_system` with `joint_self_attn` or `joint_cross_attn`.
Other architectures fail explicitly when continuation is enabled. Model-side
inference is implemented in `openwam/model/continuation_sampling.py`, with the
independent XPolicyLab adapter in `model.py` and request validation in `sampling.py`.
No robot runtime or IK changes are included. No trained continuation checkpoint
or physical evaluation is available yet; offline tests are described below.

## Installation and source isolation

Use the existing OpenWAM-compatible Python environment or `bash install.sh`.
Run each independent policy in its own process. Both the training wrapper and
native training script prioritize this directory over another editable OpenWAM
installation. Do not import two different `openwam` copies into one Python process.

## Configuration

From this policy directory, with that environment's Python:

```bash
python check_config.py dataloader=robodojo
python check_config.py continuation=none dataloader=robodojo
python check_config.py model.architecture.continuation.prefix_max_steps=4
```

`prefix_min_steps` and `prefix_max_steps` are inclusive, measured in action rows,
not optimizer steps. Lengths are sampled per example on device. `uniform` and
`exponential_decay` are supported. Prefix plus ramp must leave a free suffix.
All resolved fields are saved in the normal checkpoint `config.yaml`.
Resume rejects a continuation-config mismatch; use the saved settings.

The data reader, frame selection, normalization, action/state representation,
video noise/target, loss weights and optimizer loop retain their existing code.
Native action loss reduction and scheduler weights are preserved. RTC combines
its prefix mask with existing padding masks; it does not rescale by action dim.

Legato defaults to `reference=official_20261020`, the corrected upstream MINUS
target. Historical `official_code` retains the old PLUS target and `paper_v2`
retains MINUS, so saved configurations are never silently reinterpreted. Conditioning input is always `w = 1 - omega`. The output and
normalizer retain D action dimensions; only action encoder input becomes D+1.
Base-checkpoint finetuning expands one input weight column with zeros and retains
all other weights strictly. Same-method checkpoints reload without expansion.

Legato requires `model.action_backbone.shift_action=1.0` and records
`num_denoise_steps` in its training config. The Legato preset explicitly changes
this action grid; video shift is untouched. For a controlled baseline comparison,
set shift_action=1.0 on the `continuation=none` control too. Nonuniform Legato grids
are rejected. Step schedules follow public code; `schedule=linear` with positive
`ramp_steps` is an explicit extension (d full rows, r intermediate rows, then zero).
The official 2026-10-20 correction aligns the target sign with the paper. The
existing OpenWAM scheduler weighting, data and optimizer remain native adaptations.

References:
- Training RTC: https://github.com/Physical-Intelligence/real-time-chunking-kinetix/blob/9296f31d62d5bfeb5779dcb2f9bcf71ca37f448b/src/model.py
- Legato code: https://github.com/lyfeng001/Legato-kinetix/blob/5caa03316fd2a7c9d9cd5dd37d7af710a9e25daa/src/model_legato.py
- Legato paper: https://arxiv.org/html/2602.12978v2 (Algorithm 1 / Eq. 15)

## Data and training

Reuse prepared native HDF5 through `OPENWAM_DATASET_DIR`; no new conversion is
needed for continuation. `process_data.sh` is the copied original converter.
The shipped dataloader defaults remain upstream defaults, **not a final YAM job
recipe**: select embodiment, normalization, action layout and pretrained config
from the intended baseline rather than assuming its dimensions from the policy name.
No GPU job has been submitted for these copies.

Example native entry (replace data, base checkpoint and baseline options):

```bash
cd OpenWAM
NPROC_PER_NODE=8 bash scripts/train.sh \
  dataloader=robodojo \
  dataloader.dataset_dir=/absolute/path/to/prepared_data \
  training.finetune_ckpt_path=/absolute/path/to/base_checkpoint \
  training.output_path=/absolute/path/to/new_output \
  training.batch_size=8 training.gradient_accumulation_steps=1 \
  training.num_epochs=null training.max_steps=30000
```

The inherited trainer counts `max_steps` as microsteps. The example uses accumulation
1, making them optimizer steps too; do not claim 30,000 optimizer updates when
changing accumulation without adjusting the native training settings.
The XPolicyLab `train.sh` wrapper also accepts quoted Hydra overrides after the
optional explicit GPU count. W&B is disabled by default in this copy.

## Validation boundary

CPU tests exercise analytical targets, mask/gradient behavior, actual small
ActionDiT blocks, joint loss with a mock video backbone, optimizer updates,
checkpoint migration/reload and Hydra config propagation. These checks do not
establish full-size GPU memory use, throughput or real-data training quality.
Tests remain local/ignored following the repository policy.

## Evaluation

`deploy_yam.yml` selects the existing YAM absolute base-frame EEF20 contract.
Supply the newly trained checkpoint through `ckpt_dir`; the loader rejects a
base checkpoint or a checkpoint trained with the other continuation method.
Use the shared XPolicyLab server, as for the original policy. Each copy needs
its own process because its vendored Python package is named `openwam`.

Both policies advertise `default` and `train_time_rtc`. The latter is the
existing **transport contract for a committed action prefix**, not an assertion
that Legato and Training-time RTC are the same algorithm. Metadata records the
actual method and reference. No new ManiMux main-loop or server dispatch is needed.

- `default`: zero prefix, including the Legato all-ones extra input channel.
- `train_time_rtc`: `prefix_steps=d`, physical `action_condition[H,20]`, and
  binary `condition_weights[H]` (ones on the first d rows). For d=0, omit both
  arrays or pass null. The saved checkpoint range bounds d.
- Physical conditions pass through the **action** normalizer/unified mapping;
  the extra Legato weight channel is appended afterward and is never normalized.
- RTC sets prefix token noise-time to zero; Legato retains scalar noise-time
  and appends w=1-omega. Both update from the conditioned action at each step,
  matching the public code. Neither adds a final output clamp. ManiMux retains
  responsibility for actions already committed before the new suffix.
- Legato requires its saved `num_denoise_steps` and uniform action grid.
  Video integration and its first-frame constraint retain the native behavior.
- The adapter supports step schedules only. The optional training linear-ramp
  extension needs full future conditions and is rejected by this hard-prefix adapter.
- `eval_batch=false`, `compile_enabled=false`, `dit_cache_enabled=false`, CFG=1.
  Normal observation batches are served sequentially; conditioned requests need
  one observation. Unsupported options fail explicitly rather than silently using
  a base-model sampler. Specialized inference-time RTC/PAINT/AutoHorizon are not
  advertised by these copies.

The existing ManiMux `algorithm: train_time_rtc` can select either model through
its policy config. Use its generic strategy (no UMI-specific history strategy),
and set `train_time_rtc.fixed_prefix_steps` within the trained range when using
`delay_estimator: fixed`. Existing OpenWAM YAM FK, condition conversion and IK
remain unchanged. Deployment recipes still need the new checkpoint identity;
old baseline expected-artifact hashes cannot be reused.
