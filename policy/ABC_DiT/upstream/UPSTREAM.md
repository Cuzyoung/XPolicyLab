# upstream/ — vendored ABC-DiT inference source

Copied unmodified from `yam-abc-reproduce` @ `e65f7e4`, `third_party/policy/abc/abc_minimal/`,
which vendors [amazon-far/abc](https://github.com/amazon-far/abc) @ `282e153` plus the
yam-abc fork patches (notably `DiTPolicy.sample_actions_pi_rtc`, Physical Intelligence
inference-time RTC). Apache-2.0 (`LICENSE`); CLIP and DINOv3 notices under
`abc_minimal/third_party/`.

| file | used for |
|---|---|
| `abc_minimal/dit.py` | `DiTPolicy` (DINOv3 ViT-B/16 + DiT-XL), CLIP text encoder, samplers, checkpoint loading |
| `abc_minimal/config.py` | `DiTConfig` / `ClipConfig` dataclasses |
| `abc_minimal/preprocess.py` | z-score normalization, ImageNet normalization |
| `abc_minimal/fast_inference.py` | optional bf16 + `torch.compile` + CUDA-graph default sampler |

Left out on purpose: training, data export, and the MuJoCo simulator/eval CLI. To update, copy the
four files again from a newer snapshot and rerun `tests/test_contract.py` plus a GPU forward.
