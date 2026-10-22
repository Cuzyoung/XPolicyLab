# Pi_05_TrainRTC

Independent native JAX Pi05 continuation policy. This directory contains its own
OpenPI source; the original `Pi_05` policy and ManiMux runtime are unchanged.
It connects directly to XPolicyLab. It does not use a LeRobot Legato/RTC policy.
The existing native LeRobot **dataset** dependency is retained.

## References and adaptation

- Training-time RTC: [official code](https://github.com/Physical-Intelligence/real-time-chunking-kinetix/blob/9296f31d62d5bfeb5779dcb2f9bcf71ca37f448b/src/model.py).
- Legato: [official code](https://github.com/lyfeng001/Legato-kinetix/blob/d302701268aa3a50ec7f07189cc3af3b31014f63/src/model_legato.py).

These reference implementations use a Kinetix MLP Mixer, not this Pi05 model.
This port retains Pi05 SigLIP/Gemma, preprocessing, joint delta targets,
normalization, optimizer and data loading. It translates their flow convention
`t=0 noise -> 1 data` into OpenPI `sigma=1 noise -> 0 data`.

RTC replaces a randomly sized prefix by ground truth, gives those action tokens
`sigma=0` through per-token adaRMS, and trains the remaining suffix. The loss is
normalized by the number of valid suffix tokens. Legato passes the guided action
and one extra weight channel to the input projection; it still predicts D action
coordinates. Its target follows the published implementation's **PLUS** formula:
`(noise - action) * (1 + omega * num_denoise_steps * sigma)`.
`omega=1` in the fixed prefix; the appended channel is `w=1-omega`.

Both inference loops update **from the guided state** each Euler step. There is
no extra final prefix clamp. The existing runtime commits the fixed prefix.
Legato requires the same denoising grid used to construct its training target.
The first/ordinary chunk uses zero prefix and Legato still appends the w channel.

## Configuration

The recipe is `pi05_yam_train_time_rtc`, in `openpi/src/openpi/training/config.py`.
All algorithm settings are under `model.continuation`:

| Field | This run | Meaning |
|---|---|---|
| method | train_time_rtc | Disabled control is `none` |
| prefix_min_steps / prefix_max_steps | 0 / 6 | Inclusive range; RTC upstream `simulated_delay=7` |
| prefix_distribution | exp | Also accepts uniform; bell uses official min-peak Beta(1,4) |
| num_denoise_steps | 10 | Legato target and inference grid |
| time_distribution | native_beta | Retains old Pi05 Beta(1.5,1) scaling; uniform selects upstream time sampling |
| rtc_loss_reduction | native_mean | Retains Pi05 mean over D; official_token_sum reproduces upstream RTC sum over D |

Native beta sampling and native mean loss are explicit Pi05 adaptations, so this
is **not** an exact reproduction of the upstream Kinetix experiment. For its
RTC time/reduction, set `time_distribution=uniform` and
`rtc_loss_reduction=official_token_sum`; all fields support Tyro CLI overrides.
Legato uses the public-code PLUS target, not the alternate paper-v2 MINUS target.

## Train

On QZ use `bash train_yam_latest300.sh`. It reuses the frozen 300 episodes,
170285 frames, six tasks x50, 3 RGB views, global batch64, 30000 optimizer updates,
save5000, workers32, prefetch64, FSDP4 on 8 H200 and the existing Pi05 base weights.
The old remote cross-epoch prefetch implementation is retained. W&B is disabled.
Each independent job has its own source, output and stdout log; it reuses the
installed environment without changing its editable packages.

The Legato input kernel is extended with one **zero** row. All original rows and
biases are preserved; other shape mismatches are errors. `continuation.json` is
stored in the run root and every checkpoint's assets, alongside norm statistics.
Resume checks that the algorithm metadata matches; `train_config.txt` records
the full resolved training configuration.

## Inference

Use this directory's `deploy.yml`, set `model_path` to a completed checkpoint,
and run each policy in a separate server process. The adapter loads algorithm
settings from `assets/continuation.json`; wrong-policy and wrong-grid checkpoints
are rejected. Supported modes are `default` and `train_time_rtc`.
For conditioned sampling send `prefix_steps`, a complete physical 14D action
chunk and its contiguous binary prefix mask. The existing native policy applies
the **same action** delta/normalization/padding transforms used in training.
The extra Legato channel is appended only after those transforms. No new robot
execution or IK logic is introduced.

## Validation

`tests/test_continuation.py` covers official formula oracles, suffix loss gradients,
Euler recurrence, input-kernel migration, per-token adaRMS, configuration parity,
and real small Pi05 attention/FFN training, JIT sampling and parameter reload.
Its vision encoder is replaced and network dimensions reduced for CPU tests;
these tests do not establish full-checkpoint GPU throughput or task performance.

For a fresh deployment environment, run `bash install.sh` first. The QZ training
environment does not include all XPolicyLab server dependencies (its adapter
import probe reported missing `h5py`). That does not affect native training.
No full checkpoint server or physical robot has been run for these new policies.
