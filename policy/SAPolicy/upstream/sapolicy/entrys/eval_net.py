import os
import random
import numpy as np
import torch
from omegaconf import DictConfig
from omegaconf import OmegaConf
from distutils.util import strtobool
from sapolicy.entrys.factory import get_model, find_last_ckpt_path
from sapolicy.entrys.normalizer_utils import (
    normalizer_fingerprint,
    resolve_eval_normalizer,
)
from sapolicy.eval.io import resolve_eval_run_dir, save_eval_metrics
from sapolicy.eval import (
    RobomimicImageRunner,
    SAPolicyEvalAdapter,
    build_observation_transforms,
    build_shape_meta,
    infer_lowdim_obs_shape,
    parse_fovy_overrides,
)


def _resolve_train_dataset_config(cfg: DictConfig):
    """Best-effort peek at the training config's first dataset component yaml.

    `data.train_dataset.dataset_opts` can be a single "CFG:<path>" string, a flat
    list of them (multiple shards of one dataset/embodiment), or a list of lists
    (multiple branches mixing embodiments/dataset types -- cross-embodiment
    training). Only the first two are unambiguous enough to resolve a single
    dataset's config from; the third returns `is_ce=True` with no config, so
    callers know to require an explicit eval-time override instead of guessing.

    Returns (ds_cfg_or_None, is_ce).
    """
    dataset_opts = cfg.get("data", {}).get("train_dataset", {}).get("dataset_opts", [])
    if isinstance(dataset_opts, str):
        dataset_opts = [dataset_opts]
    if len(dataset_opts) == 0:
        return None, False
    first = dataset_opts[0]
    if not isinstance(first, str):
        return None, True
    if not first.startswith("CFG:"):
        return None, False
    try:
        return OmegaConf.load(first[4:]), False
    except Exception:
        return None, False


def _infer_eval_field_from_dataset_path(dataset_path, field_name: str):
    """Infer eval identity fields from the rollout HDF5 when training used mixed CE branches.

    Mirrors scripts/lib/eval_protocols/step60_extra.sh path heuristics so a CE-trained
    checkpoint can still eval on a single concrete dataset (e.g. StackThree_D1) without
    forcing every launcher to repeat eval.embodiment=panda.
    """
    if dataset_path is None:
        return None
    path = os.path.expandvars(os.path.expanduser(str(dataset_path))).lower()
    if field_name == "embodiment":
        for emb in ("iiwa", "ur5e", "panda", "sawyer"):
            if f"/{emb}/" in path:
                return emb
        for task in ("stackthree", "nutassembly", "kitchen", "coffee", "threepiece"):
            if task in path:
                return "panda"
    if field_name == "dataset_type":
        if "/cpgen/" in path or "cpgenfull" in path:
            return "cpgen"
    if field_name == "cpgen_absolute_actions":
        if "/cpgen/" in path or "cpgenfull" in path:
            return True
    return None


def eval_net(cfg: DictConfig) -> None:
    """
    Evaluate a trained SAPolicy model in the CPGen environment using
    RobomimicImageRunner. Reuses the same Hydra config as training to
    guarantee architecture match.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Build eval config (support train-style configs without explicit cfg.eval) ----
    eval_cfg = cfg.get("eval", None)
    if eval_cfg is None:
        base_eval = {}
        cb_eval = cfg.get("callbacks", {}).get("robomimic_eval", {}).get("eval_policy_cfg", None)
        if cb_eval is not None:
            try:
                base_eval.update(OmegaConf.to_container(cb_eval, resolve=True))
            except Exception:
                # yacs CfgNode path
                base_eval.update(dict(cb_eval))

        # Best-effort dataset path fallback from train dataset component config.
        dataset_path = None
        try:
            dataset_opts = cfg.get("data", {}).get("train_dataset", {}).get("dataset_opts", [])
            if len(dataset_opts) > 0 and isinstance(dataset_opts[0], str) and dataset_opts[0].startswith("CFG:"):
                ds_cfg_path = dataset_opts[0][4:]
                ds_cfg = OmegaConf.load(ds_cfg_path)
                dataset_path = ds_cfg.get("hdf5_path", None)
                if dataset_path is not None:
                    dataset_path = os.path.expandvars(os.path.expanduser(str(dataset_path)))
        except Exception:
            dataset_path = None
        if dataset_path is not None:
            base_eval.setdefault("dataset_path", dataset_path)

        eval_cfg = OmegaConf.create(base_eval)

    # ---- Seed + deterministic setup for reproducible eval ----
    eval_seed = eval_cfg.get("seed", 42)
    deterministic = False # eval_cfg.get("deterministic", True) # NOTE: disabled for speed
    if isinstance(deterministic, str):
        deterministic = bool(strtobool(deterministic))
    else:
        deterministic = bool(deterministic)

    # Must be set before CUDA context initializes to fully take effect.
    if deterministic and "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(eval_seed)
    torch.manual_seed(eval_seed)
    torch.cuda.manual_seed_all(eval_seed)
    np.random.seed(eval_seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True, warn_only=False)
        print(
            f"[eval_net] Set eval seed={eval_seed} with deterministic mode enabled "
            "(python/numpy/torch/cudnn/cublas)"
        )
    else:
        print(f"[eval_net] Set eval seed={eval_seed} (deterministic mode disabled)")

    def _as_bool(value, default=False) -> bool:
        if value is None:
            return bool(default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return bool(strtobool(value))
        return bool(value)

    def _as_int(value, default=None):
        if value is None:
            return default
        return int(value)

    # ---- Resolve dataset-identity fields from the training config's dataset
    # component yaml, so eval doesn't silently diverge from what the checkpoint
    # was trained on. A cross-embodiment training run (dataset_opts mixing
    # multiple branches/embodiments) can't be auto-resolved to one value --
    # require an explicit eval.<field> override in that case instead of guessing.
    _ds_cfg, _is_ce = _resolve_train_dataset_config(cfg)

    def _resolve_ds_field(field_name, default, require_on_ce=False):
        explicit = eval_cfg.get(field_name, None)
        if explicit is not None:
            return explicit
        if _is_ce:
            inferred = _infer_eval_field_from_dataset_path(
                eval_cfg.get("dataset_path"), field_name
            )
            if inferred is not None:
                print(
                    f"[eval_net] Inferred eval.{field_name}={inferred!r} from "
                    f"dataset_path={eval_cfg.get('dataset_path')}"
                )
                return inferred
            if require_on_ce:
                raise ValueError(
                    f"eval.{field_name} must be set explicitly: the training config's "
                    f"data.train_dataset.dataset_opts mixes multiple dataset branches "
                    f"(cross-embodiment run), so '{field_name}' can't be auto-resolved "
                    f"from a single dataset component config. Add `eval.{field_name}: <value>`."
                )
            return default
        if _ds_cfg is not None and field_name in _ds_cfg:
            return _ds_cfg.get(field_name)
        return default

    dataset_type = str(_resolve_ds_field("dataset_type", "cpgen", require_on_ce=True)).lower()
    embodiment = _resolve_ds_field("embodiment", None, require_on_ce=True)
    cpgen_absolute_actions = _as_bool(
        _resolve_ds_field("cpgen_absolute_actions", False, require_on_ce=True), default=False
    )
    body_frame_actions = _as_bool(
        _resolve_ds_field("body_frame_actions", True, require_on_ce=True), default=True
    )
    cpgen_action_pos_scale = float(_resolve_ds_field("cpgen_action_pos_scale", 0.05, require_on_ce=False))
    cpgen_action_rot_scale = float(_resolve_ds_field("cpgen_action_rot_scale", 0.5, require_on_ce=False))
    relative_action = _as_bool(
        eval_cfg.get("relative_action", cfg.get("use_relative_actions", False)), default=False
    )

    # ---- 1. Instantiate model from config (same as train_net) ----
    # During evaluation we must NOT delete training outputs. Some training
    # configs enable `model.clear_output_dir`; disable it here defensively.
    if hasattr(cfg, "model") and hasattr(cfg.model, "clear_output_dir"):
        if bool(getattr(cfg.model, "clear_output_dir")):
            print("[eval_net] Disabling cfg.model.clear_output_dir for evaluation.")
            cfg.model.clear_output_dir = False
    model = get_model(cfg)

    # ---- 2. Load checkpoint ----
    ckpt_path = eval_cfg.get("ckpt_path", None)
    if ckpt_path is None:
        ckpt_path = find_last_ckpt_path(cfg.callbacks.model_checkpoint.dirpath)
    else:
        ckpt_path = os.path.expandvars(os.path.expanduser(ckpt_path))
    if ckpt_path is None:
        raise FileNotFoundError(
            f"No checkpoint found in {cfg.callbacks.model_checkpoint.dirpath}"
        )
    ckpt_type = eval_cfg.get("ckpt_type", cfg.get("ckpt_type", None))
    print(f"Loading checkpoint: {ckpt_path}")
    model.load_pretrained_model(ckpt_path, ckpt_type)

    # Joint TCP+action (esp. cross-task) may bake TCP-task stats into the ckpt.
    # Rebind via eval.normalizer_source (default auto):
    #   action_dataset | auto | ckpt | path
    #   eval.normalizer_dataset_opt_index=-1  # action branch
    #   eval.normalizer_path=/path/to/action_normalizer.pt
    source_label, rebound = resolve_eval_normalizer(
        model, cfg, eval_cfg, ckpt_path=ckpt_path
    )
    if rebound is not None:
        print(
            f"[eval_net] Bound pipeline normalizer from {source_label} "
            f"({normalizer_fingerprint(rebound)})"
        )
    else:
        print(
            f"[eval_net] Keeping checkpoint pipeline normalizer "
            f"(source={source_label})"
        )

    # Move pipeline to device and set to eval mode.
    model.pipeline.to(device)
    model.pipeline.eval()

    # ---- Diagnostic: zero out specific camera spatial tokens ----
    zero_cameras = eval_cfg.get("zero_cameras", None)
    if zero_cameras is not None:
        if isinstance(zero_cameras, (int, str)):
            zero_cameras = [int(zero_cameras)]
        else:
            zero_cameras = [int(c) for c in zero_cameras]
        if hasattr(model.pipeline, 'action_head') and model.pipeline.action_head is not None:
            model.pipeline.action_head._zero_camera_indices = set(zero_cameras)
            print(f"[eval_net] VIEW ABLATION: zeroing camera indices {zero_cameras}")
        else:
            print(f"[eval_net] WARNING: no action_head found, cannot zero cameras")

    # ---- Diagnostic: zero out spatial positional encoding ----
    zero_spatial_posenc = _as_bool(eval_cfg.get("zero_spatial_posenc", False), default=False)
    if zero_spatial_posenc:
        if hasattr(model.pipeline, 'action_head') and model.pipeline.action_head is not None:
            ah = model.pipeline.action_head
            if hasattr(ah, 'spatial_pos_enc'):
                with torch.no_grad():
                    ah.spatial_pos_enc.zero_()
                print(f"[eval_net] POSITION SHORTCUT TEST: zeroed spatial_pos_enc")
            else:
                print(f"[eval_net] WARNING: action_head has no spatial_pos_enc")

    # ---- Diagnostic: mask specific spatial token positions ----
    # Usage: eval.token_mask="center" | "edge" | "center_only" | "rows:0,1" | "cols:2,3" | "indices:0,1,5,6"
    token_mask_spec = eval_cfg.get("token_mask", None)
    if token_mask_spec is not None and str(token_mask_spec) != "":
        token_mask_spec = str(token_mask_spec)
        # Determine grid size from crop config (default 5x5 for 70x70 crop)
        grid_h = int(eval_cfg.get("token_grid_h", 5))
        grid_w = int(eval_cfg.get("token_grid_w", 5))
        HW = grid_h * grid_w
        mask = torch.ones(HW, dtype=torch.float32)  # 1=keep, 0=mask

        if token_mask_spec == "center":
            # Mask center tokens (middle 3x3 for 5x5 grid)
            for r in range(1, grid_h - 1):
                for c in range(1, grid_w - 1):
                    mask[r * grid_w + c] = 0.0
        elif token_mask_spec == "edge":
            # Mask edge tokens (border of grid)
            for r in range(grid_h):
                for c in range(grid_w):
                    if r == 0 or r == grid_h - 1 or c == 0 or c == grid_w - 1:
                        mask[r * grid_w + c] = 0.0
        elif token_mask_spec == "center_only":
            # Keep ONLY center tokens, mask everything else
            mask.zero_()
            for r in range(1, grid_h - 1):
                for c in range(1, grid_w - 1):
                    mask[r * grid_w + c] = 1.0
        elif token_mask_spec.startswith("rows:"):
            rows = [int(x) for x in token_mask_spec[5:].split(",")]
            for r in rows:
                for c in range(grid_w):
                    mask[r * grid_w + c] = 0.0
        elif token_mask_spec.startswith("cols:"):
            cols = [int(x) for x in token_mask_spec[5:].split(",")]
            for r in range(grid_h):
                for c in cols:
                    mask[r * grid_w + c] = 0.0
        elif token_mask_spec.startswith("indices:"):
            indices = [int(x) for x in token_mask_spec[8:].split(",")]
            for idx in indices:
                mask[idx] = 0.0
        elif token_mask_spec.startswith("keep_indices:"):
            indices = [int(x) for x in token_mask_spec[13:].split(",")]
            mask.zero_()
            for idx in indices:
                mask[idx] = 1.0
        else:
            print(f"[eval_net] WARNING: unknown token_mask spec: {token_mask_spec}")
            mask = None

        if mask is not None:
            n_kept = int(mask.sum().item())
            n_masked = HW - n_kept
            if hasattr(model.pipeline, 'action_head') and model.pipeline.action_head is not None:
                model.pipeline.action_head._spatial_token_mask = mask
                print(f"[eval_net] TOKEN ABLATION: mask={token_mask_spec}, kept={n_kept}/{HW}, masked={n_masked}")
                # Print visual grid
                for r in range(grid_h):
                    row_str = "  "
                    for c in range(grid_w):
                        row_str += "■ " if mask[r * grid_w + c] > 0 else "□ "
                    print(row_str)

    # ---- Diagnostic: oracle TCP (replace predicted TCP with GT from env state) ----
    use_oracle_tcp = _as_bool(eval_cfg.get("use_oracle_tcp", False), default=False)
    if use_oracle_tcp:
        if hasattr(model.pipeline, 'use_oracle_tcp'):
            model.pipeline.use_oracle_tcp = True
            print(f"[eval_net] ORACLE TCP: replacing predicted TCP with GT from env state + camera extrinsics")
        else:
            print(f"[eval_net] WARNING: pipeline has no use_oracle_tcp attribute")

    # ---- 3. Resolve dataset path from training config ----
    dataset_path = eval_cfg.get("dataset_path", None)
    if dataset_path is None:
        raise ValueError(
            "eval_net requires eval.dataset_path (or callbacks.robomimic_eval.eval_policy_cfg.dataset_path). "
            "No dataset path could be inferred from config."
        )
    dataset_path = os.path.expandvars(os.path.expanduser(str(dataset_path)))

    # ---- 4. Build shape_meta for live env obs/action spaces ----
    camera_height = int(eval_cfg.get("camera_height", 256))
    camera_width = int(eval_cfg.get("camera_width", 256))
    camera_names = list(eval_cfg.get("camera_names", ["agentview", "robot0_eye_in_hand"]))
    # Warn if eval camera count doesn't match model's expected camera count
    model_num_cameras = getattr(getattr(model.pipeline, 'action_head', None), 'num_cameras', None)
    if model_num_cameras is not None and len(camera_names) != model_num_cameras:
        print(f"[eval_net] WARNING: model expects num_cameras={model_num_cameras} but eval.camera_names "
              f"has {len(camera_names)} cameras: {camera_names}. This WILL produce wrong results! "
              f"Set eval.camera_names to match training cameras.")
    camera_depths = _as_bool(eval_cfg.get("camera_depths", True), default=True)

    # Keep shape_meta minimal: only include keys we will query at eval time.
    manual_shape_meta = {
        "obs": {},
        "action": {"shape": [7], "type": "continuous"},
    }
    for cam in camera_names:
        manual_shape_meta["obs"][f"{cam}_image"] = {
            "shape": [camera_height, camera_width, 3],
            "type": "rgb",
        }
        if camera_depths:
            manual_shape_meta["obs"][f"{cam}_depth"] = {
                "shape": [camera_height, camera_width, 1],
                "type": "depth",
            }

    # If state or relative-action conversion is enabled, ensure required keys exist.
    use_state = _as_bool(getattr(model.pipeline, "use_state", False), default=False)
    if use_state or relative_action:
        manual_shape_meta["obs"]["robot0_eef_pos"] = {"shape": [3], "type": "low_dim"}
        manual_shape_meta["obs"]["robot0_eef_quat_site"] = {"shape": [4], "type": "low_dim"}
        # Gripper qpos dim depends on embodiment (2 for Panda/Sawyer, 6 for IIWA/UR5e's
        # Robotiq grippers) -- infer from the dataset actually being evaluated rather than
        # hardcoding, or cross-embodiment eval breaks at env.reset() with a shape mismatch.
        gripper_qpos_shape = list(infer_lowdim_obs_shape(dataset_path, "robot0_gripper_qpos"))
        manual_shape_meta["obs"]["robot0_gripper_qpos"] = {"shape": gripper_qpos_shape, "type": "low_dim"}

    shape_meta = build_shape_meta(
        dataset_path,
        manual_spec=OmegaConf.to_container(OmegaConf.create(manual_shape_meta), resolve=True),
        camera_height=camera_height,
        camera_width=camera_width,
    )
    # Some shape_meta sources may contain numeric strings (e.g., "256").
    # Gym spaces.Box requires all shape elements to be ints.
    def _sanitize_shape_tuple(shape, key_name):
        try:
            return tuple(int(x) for x in shape)
        except Exception as e:
            raise ValueError(f"Invalid shape for '{key_name}': {shape}") from e

    if "obs" in shape_meta:
        for k, v in shape_meta["obs"].items():
            if isinstance(v, dict) and "shape" in v:
                v["shape"] = _sanitize_shape_tuple(v["shape"], k)
    if "action" in shape_meta and isinstance(shape_meta["action"], dict) and "shape" in shape_meta["action"]:
        shape_meta["action"]["shape"] = _sanitize_shape_tuple(shape_meta["action"]["shape"], "action")

    # ---- 5. Build runner ----
    run_dir = resolve_eval_run_dir(cfg, eval_cfg)
    os.makedirs(run_dir, exist_ok=True)
    print(f"[eval_net] eval run dir: {run_dir}")

    transforms_cfg = eval_cfg.get("transforms", "configs/components/transforms/transforms_eval.yaml")
    observation_transforms = build_observation_transforms(transforms_cfg) if transforms_cfg else None

    # Optional: run a local inference smoke test without creating MuJoCo envs.
    if _as_bool(eval_cfg.get("dry_run", False), default=False):
        batch_size = int(eval_cfg.get("dry_run_batch_size", 1))
        # DINOv2 patch size is 14; inputs must be divisible by 14.
        dry_h = (camera_height // 14) * 14
        dry_w = (camera_width // 14) * 14
        images = {}
        depths = {}
        intrinsics = {}
        for cam in camera_names:
            images[cam] = torch.rand(
                batch_size, 1, 3, dry_h, dry_w, device=device, dtype=torch.float32
            )
            if camera_depths:
                depths[cam] = torch.rand(
                    batch_size, 1, 1, dry_h, dry_w, device=device, dtype=torch.float32
                )
            intrinsics[cam] = (
                torch.eye(3, device=device, dtype=torch.float32)
                .reshape(1, 9)
                .repeat(batch_size, 1)
                .reshape(batch_size, 1, 9)
            )

        with torch.no_grad():
            outputs = model.pipeline.infer(
                images,
                depths if camera_depths else None,
                camera_intrinsics=intrinsics,
                state=None,
                camera_names=camera_names,
            )
        actions = outputs.get("actions", outputs.get("action_sequence", None)) if isinstance(outputs, dict) else None
        if actions is None:
            raise RuntimeError(
                f"Dry run: pipeline.infer returned keys {list(outputs.keys()) if isinstance(outputs, dict) else type(outputs)}"
            )
        if isinstance(actions, tuple):
            actions = actions[0]
        print(f"[dry_run] actions shape: {tuple(actions.shape)}")
        return

    runner = RobomimicImageRunner(
        output_dir=run_dir,
        dataset_path=dataset_path,
        shape_meta=shape_meta,
        n_train=_as_int(eval_cfg.get("n_train", 0), 0),
        n_test=_as_int(eval_cfg.get("n_test", 10), 10),
        n_test_vis=_as_int(eval_cfg.get("n_test_vis", 10), 10),
        test_init_from_dataset=_as_bool(eval_cfg.get("test_init_from_dataset", False), default=False),
        test_start_seed=_as_int(eval_cfg.get("test_start_seed", 10000), 10000),
        test_start_idx=_as_int(eval_cfg.get("test_start_idx", None), None),
        panda_reference_hdf5=eval_cfg.get("panda_reference_hdf5", None),
        gripper_types_override=eval_cfg.get("gripper_types_override", None),
        n_envs=_as_int(eval_cfg.get("n_envs", None), None),
        n_obs_steps=_as_int(eval_cfg.get("n_obs_steps", 1), 1),
        n_action_steps=_as_int(eval_cfg.get("n_action_steps", 16), 16),
        max_steps=_as_int(eval_cfg.get("max_steps", 450), 450),
        render_obs_key=eval_cfg.get("render_obs_key", "agentview_image"),
        camera_names=camera_names,
        camera_depths=camera_depths,
        camera_height=camera_height,
        camera_width=camera_width,
        abs_action=_as_bool(eval_cfg.get("abs_action", False), default=False),
        relative_action=relative_action,
        normalize_actions=_as_bool(eval_cfg.get("normalize_actions", True), default=True),
        action_sequence_length=cfg.get("action_sequence_length", 16),
        action_orn_mode=cfg.get("action_orn_mode", "6d"),
        rotation_backend=eval_cfg.get("rotation_backend", "scipy"),
        min_depth=cfg.get("min_depth", 0.1),
        max_depth=cfg.get("max_depth", 5.0),
        observation_transforms=observation_transforms,
        video_render_camera=eval_cfg.get("video_render_camera", None),
        dataset_type=dataset_type,
        embodiment=embodiment,
        cpgen_absolute_actions=cpgen_absolute_actions,
        cpgen_action_pos_scale=eval_cfg.get("cpgen_action_pos_scale", None),
        cpgen_action_rot_scale=eval_cfg.get("cpgen_action_rot_scale", None),
        camera_fovy_overrides=parse_fovy_overrides(eval_cfg.get("camera_fovy_overrides", None)),
        body_frame_actions=body_frame_actions,
        multi_view_config=dict(eval_cfg.multi_view) if eval_cfg.get("multi_view", None) is not None else None,
        determinism_debug=_as_bool(eval_cfg.get("determinism_debug", False), default=False),
    )

    # ---- 6. Wrap model as policy ----
    policy = SAPolicyEvalAdapter(model.pipeline, device)

    # ---- 7. Run evaluation ----
    print("Starting evaluation rollouts...")
    metrics = runner.run(policy)

    meta = {
        "ckpt_path": ckpt_path,
        "cfg_file": getattr(cfg, "cfg_file", None),
        "exp_name": getattr(cfg, "exp_name", None),
        "run_dir": run_dir,
    }
    metrics_path = save_eval_metrics(run_dir, metrics, meta=meta)
    print(f"[eval_net] wrote {metrics_path}")

    # ---- 8. Print results ----
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    for k, v in sorted(metrics.items()):
        if "mean" in k or "score" in k:
            print(f"  {k}: {v}")
        elif "sim_max_reward_" in k or "sim_video_" in k:
            print(f"  {k}: {v}")
    print("=" * 60)
