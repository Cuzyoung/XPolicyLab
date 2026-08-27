# RDT2

**Contributor:** ManiMux | **Paper:** RDT2: Enabling Zero-shot Cross-embodiment Generalization | **arXiv:** [2602.03310](https://arxiv.org/abs/2602.03310) | **Original code:** [thu-ml/RDT2](https://github.com/thu-ml/RDT2)

RDT2 is a bimanual UMI-native VLA: a Qwen2.5-VL backbone with a small flow-matching action expert that predicts a 24-step relative end-effector chunk from **two wrist fisheye views and no third-person camera**. This adapter targets our UMI bench (`env_cfg_type: umi_dual`, `action_type: ee`, 20-D dual-arm EEF state). The upstream source tree is **not vendored** and is **not a submodule** — clone it beside the workspace and point `rdt2_root` at it.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

```bash
cd XPolicyLab/policy/RDT2
bash install.sh [/path/to/RDT2]        # default: <workspace parent>/RDT2
```

`install.sh` clones the upstream tree if it is missing and pins it to
`RDT2_GIT_COMMIT` (`0797b4c`), then builds a uv venv at `$RDT2_ROOT/.venv` and
installs XPolicyLab into it editable. Upstream is **not vendored and not a
submodule** — pinning by commit follows `policy/LingBot_VLA`, which pins lerobot
the same way. The script refuses to run if the upstream tree has local edits to
tracked files: the arm-order and gripper-unit contract in `model.py` cites line
numbers in this exact tree.

Two things upstream's `requirements.txt` does **not** pin, and this script does:

- **torch 2.7.1+cu128.** The default PyPI wheel is cu126 and carries no `sm_120`
  kernels, so it cannot run on Blackwell at all.
- **flash-attn 2.8.3.post1, built from source.** That is the first upstream
  release emitting `arch=compute_120,code=sm_120`; no prebuilt wheel covers
  Blackwell. It is not optional — `rdt/train.py:134` hardcodes
  `attn_implementation="flash_attention_2"` with no fallback, and
  `configs/rdt/post_train.yaml` sets `use_flash_attn: true` for the action expert.
  Budget 30–80 min; `MAX_JOBS` defaults to 4 because each nvcc job on these
  kernels peaks near 4GB. Set `RDT2_FLASH_ATTN_WHEEL` to skip the build.

`vllm` is filtered out of `requirements.txt`: only `deploy/vllm_utils.py` and
`deploy/inference_real_vq.py` import it, lazily, on the VQ inference path.
Nothing under `rdt/` touches it and resolving it drags in a conflicting torch.

Validated on one RTX 5090 (32GB): 1500 steps at batch 8, loss 0.0192 → 0.0057,
23.8/32.6 GiB, 10m05s.

## Model Assets

Three separate downloads. `checkpoint_path`, `normalizer_path` and `vlm_path` in `deploy.yml` are relative to `rdt2_root` (or absolute).

| Asset | HF repo | Size | `deploy.yml` key |
| --- | --- | --- | --- |
| Flow-matching action expert | `robotics-diffusion-transformer/RDT2-FM` | ~0.98 GB | `checkpoint_path` |
| Normalizer `.pt` | `robotics-diffusion-transformer/RVQActionTokenizer` | ~1.75 GB | `normalizer_path` |
| Qwen2.5-VL-7B backbone | `robotics-diffusion-transformer/RDT2-VQ` | ~16.6 GB | `vlm_path` |

```bash
cd ~/Desktop/project/RDT2
hf download robotics-diffusion-transformer/RDT2-FM  --local-dir ckpt/RDT2-FM
hf download robotics-diffusion-transformer/RVQActionTokenizer --local-dir ckpt/RVQ
hf download robotics-diffusion-transformer/RDT2-VQ  --local-dir ckpt/RDT2-VQ
```

**The FM path loads the 7B backbone too.** `RDT2/models/rdt_inferencer.py:115-124` builds a full `Qwen2_5_VLForConditionalGeneration`, and the action expert cross-attends into 14 layers of its KV cache (`rdt_inferencer.py:227-239` with `selected_layers: [0..13]` from `configs/rdt/post_train.yaml:46`). "We only need the 0.98 GB action expert" is true for *training* and false for *inference*: budget ~16.5 GB VRAM in bf16.

The RVQ repo is needed **only** for `umi_normalizer_wo_downsample_indentity_rot.pt` (the misspelling is upstream's). The RVQ tokenizer model itself is not on the FM path, and there is no separate SigLIP/DINO vision encoder — `rdt_inferencer.py:47-48` hard-codes `self.vision_encoder = None`.

The processor is fetched separately by `AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")` (`rdt_inferencer.py:122-123`), independent of `vlm_path`. Populate the HF cache or run online once.

## Data Processing

```bash
bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> [expert_data_num]
```

**Currently a loud stub — it exits 2 rather than pretending.** The LeRobot v3.0 → RDT2 webdataset converter is step 1 of `docs/rdt2-umi-runbook.md` and belongs to the parent repo (`scripts/lerobot_to_rdt2_shards.py`), not to this submodule. Override the path with `RDT2_SHARD_CONVERTER=/path/to/converter.py`.

The converter must reproduce, exactly, the conventions this adapter uses on the way back out (see [Configuration](#configuration)): arm swap, `rot6d_layout`, `relative_translation_frame`, `gripper_mapping`. Nothing detects a mismatch at runtime — the model simply fails to learn.

## Training

```bash
bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id>
```

**Also a thin wrapper that fails loudly.** It resolves the action dim via `utils/get_action_dim.sh` (20 for `umi_dual`), then execs the upstream `scripts/finetune_rdt.sh` from `$RDT2_ROOT`, passing `RDT2_OUTPUT_DIR` / `RDT2_SEED` / `RDT2_DATASET_CONFIG`. It exits 2 when that script is absent. Checkpoints follow the standard `checkpoints/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>-<seed>/` layout.

Fine-tune the **FM action expert only** (~16 GB VRAM). Do not launch RDT2-VQ training on a 30 GB-RAM host. Run it in `tmux`, never `nohup`. Upstream advises ≤ 5 epochs to avoid overfitting.

## Evaluation

```bash
cd XPolicyLab/policy/RDT2
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_conda_env> <eval_env_conda_env>

# Offline wiring check (no simulator, no weights loaded — see below)
EVAL_ENV_TYPE=debug bash eval.sh RoboDojo exchange_ball demo umi_dual ee 0 0 0 rdt2 rdt2
```

Standard 10 positional arguments, no extras. Supported: `action_type: ee` only (the adapter raises on `joint`); `env_cfg_type`: any dual-arm robot whose `arm_dim` + `ee_dim` sum to 20 with `ee_dim: [1, 1]` — in practice `umi_dual`.

### Debug mode does not load the model

Under `EVAL_ENV_TYPE=debug`, `debug_load_model` defaults to **false** and the adapter runs a loudly-announced stub that holds the current observed pose. This is deliberate: a wiring check does not need weights, and loading the 7B backbone on a 30 GB-RAM host risks an OOM that takes the user's input method with it. The stub still routes through the **real** decode path — arm swap, gripper scaling, relative→absolute reconstruction, 6D→quaternion — so the debug loop exercises everything except the network forward. Set `debug_load_model: true` to override.

Outside debug mode a missing checkpoint / normalizer / backbone is a hard `FileNotFoundError`.

## Configuration

Beyond the standard key set, `deploy.yml` carries these. **Every key in the "layout" group fails silently when wrong** — the numbers stay well-formed and the model just does not work.

| Key | Default | Meaning |
| --- | --- | --- |
| `rdt2_root` | `/home/jw/Desktop/project/RDT2` | Upstream tree. `$RDT2_ROOT` overrides; final fallback `<workspace>/../RDT2`. |
| `rdt2_model_config` | `configs/rdt/post_train.yaml` | Upstream FM config. `state_dim` / `action_chunk_size` are read from it and cross-checked. |
| `checkpoint_path` / `normalizer_path` / `vlm_path` | see above | Model assets. `checkpoint_path: null` falls back to `checkpoint_resolver`. |
| `device` / `dtype` | `cuda` / `bfloat16` | |
| `disable_torch_compile` | `true` | `RDTInferencer.reset()` calls `torch.compile` unconditionally (`rdt_inferencer.py:154`); its Inductor workers are host-RAM hungry. |
| `debug_load_model` | `false` | Load real weights under `EVAL_ENV_TYPE=debug`. |
| `action_chunk_size` | `24` | Model's prediction horizon, 0.8 s @ 30 Hz. |
| `exc_action_size` | `12` | Steps executed per network forward (naming follows `policy/Hy_Embodied_05_VLA`). |
| `exc_action_interval` | `1` | Execute every N-th step. `size * interval` must be ≤ `action_chunk_size`. |
| `image_size` | `384` | Per-wrist square resize. |
| `swap_arm_order` | `true` | Our left-first ↔ RDT2's right-first, for the **20-D vectors only**. |
| `rot6d_layout` | `cols` | `cols` = first two columns of R (upstream); `rows` = pytorch3d/GR00T. |
| `relative_translation_frame` | `anchor` | `anchor` = `T_abs = T_anchor @ T_rel` (upstream); `world` = plain coordinate differences. |
| `ee_pose_format` | `quat` | `quat` = 7-D `[x,y,z,qw,qx,qy,qz]`; `rot6d` = 9-D `[x,y,z,r1..r6]`. |
| `gripper_stroke_m` | `0.096` | Our gripper's full stroke in metres (`linear_4310`; `crank_4310` is 0.071). |
| `rdt2_gripper_width_max_m` | `0.088` | RDT2's full-open width. |
| `gripper_mapping` | `full_open_normalized` | How our `[0,1]` fraction maps into RDT2's `[0, 0.088]`. |
| `feed_zero_state` | `true` | Upstream's FM loop sends `np.zeros(20)`. |
| `default_prompt` | `"Exchange the ball."` | Used when the observation carries no `instruction`. |
| `request_timeout_s` | `1200` | One forward is a full 7B prefill; the 120 s client default is not enough. |

### The conventions, with upstream evidence

| Convention | What this adapter does | Evidence |
| --- | --- | --- |
| Arm order (vectors) | ours `[L(10) R(10)]` ↔ RDT2 `[R(10) L(10)]` | `RDT2/README.md:267-276`; `RDT2/configs/robots/eval_bimanual_fr3_config.yaml:18` (`0->right 1->left`) |
| Arm order (images) | **not swapped** — left wrist on the left half | `RDT2/deploy/inference_real_fm.py:390-391` (`left_stereo ← camera1_rgb`); `RDT2/models/rdt_inferencer.py:180-186, 266-268`; `configs/rdt/post_train.yaml:24` |
| Image tensor | two `384×384` uint8 RGB frames; the library concatenates to `(384,768,3)` | `RDT2/models/rdt_inferencer.py:266-268`; `RDT2/README.md:354`; RGB confirmed at `deploy/umi/real_world/camera/mvs_cam.py:308` and `bimanual_umi_env.py:90-97` (`bgr_to_rgb=False`) |
| 6-D rotation | first two **columns** of R; Gram-Schmidt stacks `(b1,b2,b3)` as columns | `RDT2/data/umi/pose_util.py:152-156` (`mat_to_rot6d`), `:98-105` (`rot6d_to_mat`) |
| 6-D "no rotation" | identity `[1,0,0,0,1,0]`, **not** zeros | `pose_util.py:98-105` (a zero 6-D block is degenerate) |
| Relative → absolute | `T_abs = T_anchor @ T_rel`; `abs_pos = anchor_pos + anchor_rotm @ rel_pos` | `RDT2/data/umi/common/pose_repr_util.py:37-38`, driven from `deploy/umi/real_world/real_inference_util.py:165-183` |
| Anchor | the **current observation's** EEF pose | `real_inference_util.py:165-168`; training side `data/umi_video_dataset.py:407-418` (`base_pose_mat=pose_mat[-1]`) |
| Gripper (model space) | width in metres, `[0, 0.088]`, `0.088` = fully open | `RDT2/ckpt/RVQ/README.md:59` and `:62` |
| Gripper (`/0.088*0.10`) | **not applied** | Only in `deploy/inference_real_fm.py:408, :414, :423` (and the same three in `inference_real_vq.py`) — a per-robot calibration for their 0.12 m gripper (`configs/robots/eval_bimanual_fr3_config.yaml:57`), absent from `data/` and `train.py` |
| Gripper (ours) | normalized `[0,1]` stroke fraction | `manimux/src/manimux/kinematics/yam.py:135`; `linear_4310` stroke 0.096 m at `i2rt/i2rt/robots/config/linear_4310.yml:43`; `exchange_ball_v0/meta/stats.json` maxes at exactly 1.0 |
| Proprio | zeros(20) | `RDT2/deploy/inference_real_fm.py:393`; `RDT2/README.md:311-313` |
| Instruction | required, non-empty, "Verb object." | `RDT2/models/rdt_inferencer.py:188-200`; `RDT2/README.md:262-263` |
| Model output | `(24, 20)` CPU float32, **already unnormalized** | `RDT2/models/rdt_inferencer.py:324-330` |

### Two XPolicyLab contracts disagree about `*_ee_pose`

For `umi_dual` (`arm_dim: [9, 9]`):

- `unpack_robot_state` / `pack_robot_state` treat `left_ee_pose` as **9-D** (`xyz + rot6d`), following `arm_dim`. This is what `policy/Pi_05` emits.
- `debug_env_client.validate_robot_state_dict` hard-codes **7-D** for `*_ee_pose` (`debug_env_client.py:258`), matching the integration skill's `[x, y, z, qw, qx, qy, qz]`.

Verified empirically: the shipped env client rejects a 9-D pose. This adapter therefore defaults to `ee_pose_format: quat` (7-D) and offers `rot6d` (9-D) for an environment that wants the `arm_dim` form. **The observation side accepts both widths** so the debug client and the real rig share one code path.

## Notes

- Robot registration: `umi_dual` (`arm_dim: [9, 9]`, `ee_dim: [1, 1]`) is registered in both `utils/robot/_robot_info.json` (here) and `env_cfg/robot/_robot_info.json` (parent workspace). AGENTS.md requires both, or training and evaluation disagree about the action dim.
- `deploy.py`, `__init__.py`, `eval.sh`, `setup_eval_*.sh` are byte-identical to `policy/demo_policy`.
- Unit tests: `tests/unit/test_rdt2_umi_conventions.py` (32 tests) cover the arm-swap round trip, 6D↔matrix↔quaternion, the image concat shape/dtype/halves, the gripper mappings, and the relative↔absolute chunk reconstruction.

## Not yet verified

Be blunt about what this is: **wiring, not results.** Nothing below has been measured.

1. **No real inference has ever run.** The debug loop passed in stub mode only. At the time of writing `ckpt/RDT2-VQ` was 113 MB of 16.6 GB, so the backbone could not load. `_build_upstream_policy` and `_forward`'s real branch are **untested code**.
2. **Training runs and the loss descends; nothing beyond that is measured.** The converter now exists (`manimux/scripts/lerobot_to_rdt2_shards.py`) and a fine-tune of the FM expert has run end to end: 1500 steps at batch 8 on one RTX 5090, loss 0.0192 → 0.0057, head-to-tail drop clearing the head window's own sd. That establishes the plumbing, **not** that the resulting policy is any good — no checkpoint has been evaluated, in sim or on hardware. `process_data.sh` and `train.sh` here remain loud wrappers around the parent repo's converter and the upstream `finetune_rdt.sh`.
3. **`rot6d_layout` is a guess for our own data.** `cols` is verified against *upstream*. Our `exchange_ball_v0` columns `left_tcp.r1..r6` come from a capture pipeline outside this repo and have **not** been checked against either convention. If the converter and this key disagree, nothing raises.
4. **`gripper_mapping` is a modelling choice, not a fact.** Our `linear_4310` opens 0.096 m; RDT2 assumes 0.088 m. `full_open_normalized` compresses one into the other. The converter now defaults to the same mapping, so the two agree — but how much fine-tuning absorbs is still unknown.
5. **`relative_translation_frame: anchor` matches upstream, and now matches the converter too** (`to_relative(anchor, ...)` expresses translation in the anchor frame). Deploy-side and training-side agree; neither has been checked against a real rollout.
6. **The tracker→TCP conjugation is not implemented.** Upstream conjugates the relative pose by `C = inv(T_tracker_to_tcp) @ T_tracker_to_policy` (`real_inference_util.py:191-226`) with hard-coded UMI-gripper geometry. Our TCP definition differs, and fine-tuning on our own data should bake the frame in, so this adapter omits it. If a fine-tune trained in the UMI tool frame, it will be needed.
7. **The normalizer's internals are unverified by this adapter.** `RDTInferencer` unnormalizes internally, so nothing here touches it. It reportedly contains `eef_rot_axis_angle` (axis-angle) entries while the action vector is 6-D; that relationship has not been traced.
8. **Peak host RAM for a real load is unmeasured.** Upstream's own table says inference wants > 32 GB RAM; this host has 30 GB. That was the runbook's step-0 blocking question and it is still open.
9. **`eval_batch: false`.** `get_action_batch` is implemented and smoke-tested directly, but the debug loop only exercised `eval_one_episode`.
10. **No simulator evaluation, no real-robot rollout, no success rates.** Do not quote any number from this adapter.
