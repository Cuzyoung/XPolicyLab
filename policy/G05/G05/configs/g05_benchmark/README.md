# G0.5 benchmark configuration

Each benchmark directory exposes the supported G0.5 action objectives:

| Mode | Discrete | Continuous | CE | FM | BAR |
|---|---:|---:|---:|---:|---:|
| `ar_fm` | on | on | 1 | 1 | off |
| `fm_only` | off | on | 0 | 1 | off |

All launchers must append exactly one of:

```text
+g05_benchmark/<benchmark>=ar_fm
+g05_benchmark/<benchmark>=fm_only
```

`ar_fm` sets `fm.joint_training=false`, so FM gradients train the action
expert but do not flow through the shared VLM KV used by AR. `fm_only` sets
`fm.joint_training=true` so FM can fine-tune the shared VLM.

Run the complete contract check before launching:

```bash
python scripts/validate_g05_benchmark_config.py --all
```

The validator checks both BAR switches, objective heads and weights,
`joint_training`, group-order shuffle, and noop-part dropout. Activation
checkpointing, camera count, action dimension, normalization, and optimizer
budget remain benchmark-specific.

Historical `.hydra/config.yaml` files are immutable records. A source YAML
correction does not change an existing checkpoint. In particular, the existing
VLABench, RoboTwin, and RoboDojo runs record BAR enabled and must be labelled
`legacy_bar` when compared with newly trained BAR-disabled runs.

## Effective epoch calculation

For step-based training:

```text
global_batch = world_size * batch_per_gpu * grad_accumulation_steps
samples_seen = optimizer_steps * global_batch
effective_epochs = samples_seen / training_samples
```

The canonical comparison budgets preserve the global batch of the existing
FM-only controls and use fixed optimizer-step targets:

| Benchmark | Training samples | GPUs | Batch/GPU | Global batch | Approx. steps | Epochs |
|---|---:|---:|---:|---:|---:|---:|
| LIBERO four-suite | 277,713 | 8 | 8 | 128 | 30,000 | ~13.83 |
| VLABench | ~569,350 | 8 | 4 | 32 | 60,000 | ~3.37 |
| RoboTwin clean | 549,787 | 16 | 16 | 256 | 50,000 | ~23.28 |
| RoboTwin mixed | ~6,014,352 | 16 | 16 | 256 | 50,000 | ~2.13 |
| RoboDojo | ~1,841,006 | 16 | 8 | 256 | 60,000 | ~8.34 |
| Humanoid football | ~55,763 | 8 | 2 | 16 | 13,941 | 4 |
| RMBench | ~85,816 | 16 | 16 | 256 | 1,341 | 4 |
| Bridge/Simpler | ~1,899,440 | 8 | 8 | 64 | 118,715 | 4 |
| RLBench | pending final conversion count | 8 | 2 | 16 | computed at startup | 4 |

RoboCasa is intentionally excluded from this four-epoch policy. Its human300
training split contains 28,581,979 samples, which would require about 7.15
million optimizer steps at the currently proven 8-GPU batch.

Approximate counts apply the configured validation fraction. RoboTwin clean
uses the exact 2,500-episode split and exact summed episode lengths.

## Fair comparison requirements

Within one benchmark, AR+FM, AR-only, and FM-only are comparable only when all
of these match:

1. Dataset fingerprint and train/validation episode split.
2. Pretrained checkpoint and tokenizer checkpoint.
3. Global batch, optimizer steps, and therefore total samples seen.
4. Optimizer, learning-rate schedule, augmentation, normalization, and camera
   contract.
5. BAR mode and tokenizer serialization.
6. Evaluation checkpoint step and evaluation protocol.

Matching optimizer steps alone is insufficient when global batch differs.
Matching effective epochs alone is also insufficient for optimizer-schedule
comparisons. The primary controlled budget is total samples seen; steps and
global batch must both be reported.
