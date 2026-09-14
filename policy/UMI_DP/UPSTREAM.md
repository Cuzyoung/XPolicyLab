# Source provenance

- Upstream: https://github.com/real-stanford/universal_manipulation_interface.git
- Upstream revision: `d095ba9590df789df5189eea5ee7e431689038a6`.
- Source adaptation: https://github.com/astral705/CalibWrist.git,
  revision `f17a62a2ab77207de54fdca73dc8ff4d76f87cdb`.
- Imported on 2026-09-13 from the tracked `policies/diffusion_policy` source.
- `upstream/LICENSE` retains the upstream license. This vendored subset contains
  the CLIP/UNet model, normalizer, native LeRobot dataset, and training workspace
  dependency closure, rather than hardware, simulators or unrelated policies.
- `rtc_guidance.py` ports CalibWrist `deploy/tianji/rtc/guidance.py`; preprocessing
  and relative pose transforms reproduce `observation_adapter.py`, `geometry.py`
  and `policy_runner.py` from the same source revision.

Reproduction changes are limited to:

1. Fully qualified imports under `XPolicyLab.policy.UMI_DP.upstream` to avoid
   colliding with XPolicyLab's existing `DP` package. Checkpoint Hydra targets are
   remapped only during loading; serialized state keys and tensors stay intact.
2. Dataset/cache locations come from `UMI_DP_DATASET_DIR` and
   `UMI_DP_IMAGE_CACHE`. No source checkout or developer path is needed.
3. The training task has `env_runner: null`, and the workspace skips simulator
   rollout when no runner is provided. Training and checkpointing remain real.
4. Production loading uses trusted PyTorch checkpoint memory mapping, the embedded
   EMA/model normalizer, and disables pretrained downloads. The known legacy SHA
   receives the same CLIP Normalize correction as the original PolicyRunner.

Upstream random crop, random rotation and color jitter remain active at inference.
Training has not been rerun as part of this integration. Checkpoints and datasets
remain external artifacts and must not be committed.
