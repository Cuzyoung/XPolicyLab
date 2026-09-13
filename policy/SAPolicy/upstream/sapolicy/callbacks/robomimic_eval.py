import pytorch_lightning as pl
import torch
import wandb
import os
from os.path import join
from datetime import timedelta
from omegaconf import OmegaConf
import numpy as np
import imageio
import gc
import signal
from distutils.util import strtobool

from sapolicy.logger import Log


def _runner_api():
    """Lazy import so Hydra can construct this callback without loading MuJoCo/EGL."""
    from sapolicy.eval import (
        RobomimicImageRunner,
        SAPolicyEvalAdapter,
        build_observation_transforms,
        build_shape_meta,
        parse_fovy_overrides,
    )

    return (
        RobomimicImageRunner,
        build_shape_meta,
        build_observation_transforms,
        SAPolicyEvalAdapter,
        parse_fovy_overrides,
    )


class _EvalTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise _EvalTimeout("Eval timed out")


    # Cross-view eval configs: 12-point azimuth sweep (every 30°) at el=30°, r=0.85m
    # Matches scripts/eval_ood.sh azimuth-sweep — uniform hemisphere coverage.
    # fovy not specified → uses robosuite agentview default (45°).
AZIMUTH_SWEEP_CONFIGS = [
    {"name": f"az{az:+04d}_el30", "azimuth": float(az), "elevation": 30.0, "radius": 0.85}
    for az in range(-150, 181, 30)  # -150, -120, ..., 150, 180
]

    # CPGen training views (spherical coords relative to target=(0,0,0.8)).
    # fov45 dataset: all cameras use fovy=45° (unified with agentview).
CPGEN_TRAINING_VIEW_CONFIGS = [
    {"name": "tv0_az60_el40",  "azimuth": 60.0,   "elevation": 40.0, "radius": 1.0},
    {"name": "tv1_az120_el40", "azimuth": 120.0,  "elevation": 40.0, "radius": 1.0},
    {"name": "tv2_az-120_el40","azimuth": -120.0, "elevation": 40.0, "radius": 1.0},
]


class RobomimicEvalCallback(pl.Callback):
    """Run robomimic rollout evaluation during training.

    Triggers every `eval_every_n_steps` training steps (aligned with checkpoint saves).
    Only runs on global_rank == 0 to avoid duplicate evals in DDP.
    The runner is lazily constructed once and reused across evals.

    Safety: if eval fails or times out, all future evals are disabled and training
    continues uninterrupted.
    """

    EVAL_TIMEOUT_SECONDS = 60 * 60  # 60 minutes max per eval

    def __init__(self, eval_policy_cfg, eval_every_n_steps=0, eval_every_n_epochs=50):
        super().__init__()
        self._eval_policy_cfg = eval_policy_cfg
        self._eval_every_n_steps = int(eval_every_n_steps)
        self._eval_every_n_epochs = int(eval_every_n_epochs)
        self._runner = None  # lazily built
        self._cross_view_runners = {}  # view_name -> runner
        self._shape_meta = None
        self._last_eval_step = -1
        self._last_eval_epoch = -1
        self._eval_disabled = False  # set True after first failure

    def _get_runner(self, output_dir):
        """Build or reuse the RobomimicImageRunner."""
        if self._runner is not None:
            # Update output dir for this eval
            self._runner.output_dir = output_dir
            return self._runner

        (
            RobomimicImageRunner,
            build_shape_meta,
            build_observation_transforms,
            _policy_cls,
            parse_fovy_overrides,
        ) = _runner_api()
        del _policy_cls  # unused here

        cfg = self._eval_policy_cfg

        manual_shape_meta = None
        if cfg.get("shape_meta", None) is not None:
            meta_path = os.path.expanduser(cfg["shape_meta"])
            if os.path.isfile(meta_path):
                if meta_path.endswith(('.yaml', '.yml')):
                    manual_shape_meta = OmegaConf.to_container(OmegaConf.load(meta_path), resolve=True)
                else:
                    import json
                    with open(meta_path, "r") as f:
                        manual_shape_meta = json.load(f)

        self._shape_meta = build_shape_meta(
            cfg.dataset_path, manual_shape_meta,
            camera_height=cfg.get("camera_height", 256),
            camera_width=cfg.get("camera_width", 256),
            camera_depths=cfg.get("camera_depths", True),
        )
        observation_transforms = cfg.get("transforms", None)
        observation_transforms = build_observation_transforms(observation_transforms)

        relative_action = cfg.get("relative_action", False)
        relative_action = bool(strtobool(str(relative_action))) if isinstance(relative_action, str) else bool(relative_action)
        abs_action = cfg.get("abs_action", False)
        abs_action = bool(strtobool(str(abs_action))) if isinstance(abs_action, str) else bool(abs_action)
        normalize_actions = cfg.get("normalize_actions", True)
        normalize_actions = bool(strtobool(str(normalize_actions))) if isinstance(normalize_actions, str) else bool(normalize_actions)

        camera_fovy_overrides = parse_fovy_overrides(cfg.get("camera_fovy_overrides", None))

        body_frame_actions = cfg.get("body_frame_actions", True)
        body_frame_actions = bool(strtobool(str(body_frame_actions))) if isinstance(body_frame_actions, str) else bool(body_frame_actions)

        self._runner = RobomimicImageRunner(
            output_dir=output_dir,
            dataset_path=cfg.dataset_path,
            shape_meta=self._shape_meta,
            n_train=cfg.get("n_train", 0),
            n_train_vis=cfg.get("n_train_vis", 0),
            n_test=cfg.get("n_test", 20),
            n_test_vis=cfg.get("n_test_vis", 4),
            n_envs=cfg.get("n_envs", 10),
            n_obs_steps=int(cfg.get("n_obs_steps", 1)),
            n_action_steps=int(cfg.get("n_action_steps", 16)),
            max_steps=cfg.get("max_steps", 5000),
            render_obs_key=cfg.get("render_obs_key", "agentview_image"),
            fps=cfg.get("fps", 10),
            save_rollout_states_path=None,
            observation_transforms=observation_transforms,
            camera_height=cfg.get("camera_height", 256),
            camera_width=cfg.get("camera_width", 256),
            action_orn_mode=cfg.get("action_orn_mode", "6d"),
            rotation_backend=cfg.get("rotation_backend", "scipy"),
            abs_action=abs_action,
            relative_action=relative_action,
            normalize_actions=normalize_actions,
            action_sequence_length=cfg.get("action_sequence_length", 16),
            camera_names=cfg.get("camera_names", ["agentview", "robot0_eye_in_hand"]),
            camera_depths=cfg.get("camera_depths", True),
            min_depth=cfg.get("min_depth", 0.1),
            max_depth=cfg.get("max_depth", 5.0),
            video_render_camera=cfg.get("video_render_camera", "agentview"),
            cpgen_action_pos_scale=cfg.get("cpgen_action_pos_scale", None),
            cpgen_action_rot_scale=cfg.get("cpgen_action_rot_scale", None),
            camera_fovy_overrides=camera_fovy_overrides,
            body_frame_actions=body_frame_actions,
            test_init_from_dataset=cfg.get("test_init_from_dataset", True),
            test_start_idx=cfg.get("test_start_idx", 0),
            panda_reference_hdf5=cfg.get("panda_reference_hdf5", None),
            gripper_types_override=cfg.get("gripper_types_override", None),
            multi_view_config=dict(cfg.multi_view) if cfg.get("multi_view", None) is not None else None,
        )
        return self._runner

    def _get_cross_view_runner(self, view_cfg, output_dir):
        """Build or reuse a runner for a specific cross-view eval."""
        name = view_cfg["name"]
        if name in self._cross_view_runners:
            self._cross_view_runners[name].output_dir = output_dir
            return self._cross_view_runners[name]

        (
            RobomimicImageRunner,
            build_shape_meta,
            build_observation_transforms,
            SAPolicyEvalAdapter,
            parse_fovy_overrides,
        ) = _runner_api()
        del SAPolicyEvalAdapter

        cfg = self._eval_policy_cfg
        if self._shape_meta is None:
            self._shape_meta = build_shape_meta(
                cfg.dataset_path, None,
                camera_height=cfg.get("camera_height", 256),
                camera_width=cfg.get("camera_width", 256),
                camera_depths=cfg.get("camera_depths", True),
            )
        observation_transforms = build_observation_transforms(cfg.get("transforms", None))

        relative_action = cfg.get("relative_action", False)
        relative_action = bool(strtobool(str(relative_action))) if isinstance(relative_action, str) else bool(relative_action)
        abs_action = cfg.get("abs_action", False)
        abs_action = bool(strtobool(str(abs_action))) if isinstance(abs_action, str) else bool(abs_action)
        normalize_actions = cfg.get("normalize_actions", True)
        normalize_actions = bool(strtobool(str(normalize_actions))) if isinstance(normalize_actions, str) else bool(normalize_actions)
        body_frame_actions = cfg.get("body_frame_actions", True)
        body_frame_actions = bool(strtobool(str(body_frame_actions))) if isinstance(body_frame_actions, str) else bool(body_frame_actions)
        base_fovy_overrides = parse_fovy_overrides(cfg.get("camera_fovy_overrides", None)) or {}
        # Apply per-view fovy override (e.g. CPGen training views use fovy=60°)
        camera_fovy_overrides = dict(base_fovy_overrides)
        if "fovy" in view_cfg:
            camera_fovy_overrides["agentview"] = view_cfg["fovy"]

        mv_config = {
            "random_view_alias": True,
            "source_pool": "spherical",
            "sphere_fixed_azimuth": view_cfg["azimuth"],
            "sphere_fixed_elevation": view_cfg["elevation"],
            "sphere_fixed_radius": view_cfg["radius"],
            "sphere_look_at": [0.0, 0.0, 0.8],
            "seed": 42,
        }

        cross_n_test = cfg.get("cross_view_n_test", 20)
        runner = RobomimicImageRunner(
            output_dir=output_dir,
            dataset_path=cfg.dataset_path,
            shape_meta=self._shape_meta,
            n_train=0,
            n_train_vis=0,
            n_test=cross_n_test,
            n_test_vis=0,
            n_envs=cfg.get("n_envs", 10),
            n_obs_steps=int(cfg.get("n_obs_steps", 1)),
            n_action_steps=int(cfg.get("n_action_steps", 16)),
            max_steps=cfg.get("max_steps", 5000),
            render_obs_key=cfg.get("render_obs_key", "agentview_image"),
            fps=cfg.get("fps", 10),
            save_rollout_states_path=None,
            observation_transforms=observation_transforms,
            camera_height=cfg.get("camera_height", 256),
            camera_width=cfg.get("camera_width", 256),
            action_orn_mode=cfg.get("action_orn_mode", "6d"),
            rotation_backend=cfg.get("rotation_backend", "scipy"),
            abs_action=abs_action,
            relative_action=relative_action,
            normalize_actions=normalize_actions,
            action_sequence_length=cfg.get("action_sequence_length", 16),
            camera_names=cfg.get("camera_names", ["agentview", "robot0_eye_in_hand"]),
            camera_depths=cfg.get("camera_depths", True),
            min_depth=cfg.get("min_depth", 0.1),
            max_depth=cfg.get("max_depth", 5.0),
            video_render_camera=cfg.get("video_render_camera", "agentview"),
            cpgen_action_pos_scale=cfg.get("cpgen_action_pos_scale", None),
            cpgen_action_rot_scale=cfg.get("cpgen_action_rot_scale", None),
            camera_fovy_overrides=camera_fovy_overrides,
            body_frame_actions=body_frame_actions,
            test_init_from_dataset=cfg.get("test_init_from_dataset", True),
            test_start_idx=cfg.get("test_start_idx", 0),
            panda_reference_hdf5=cfg.get("panda_reference_hdf5", None),
            gripper_types_override=cfg.get("gripper_types_override", None),
            multi_view_config=mv_config,
        )
        self._cross_view_runners[name] = runner
        return runner

    def _run_eval(self, trainer, pl_module, label: str):
        """Shared eval logic for both step-based and epoch-based triggers."""
        is_rank0 = (trainer.global_rank == 0)

        if is_rank0 and not self._eval_disabled:
            Log.info(f"[EvalCallback] Running rollout eval at {label}...")

            # Set alarm-based timeout
            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(self.EVAL_TIMEOUT_SECONDS)

            try:
                output_dir = join(f"{self._eval_policy_cfg.output_dir}/{label}")
                os.makedirs(output_dir, exist_ok=True)

                runner = self._get_runner(output_dir)

                # Use EMA weights for eval: load from latest checkpoint file
                # (in-memory ema_pipeline may have state corruption after many steps)
                eval_pipeline = pl_module.pipeline
                if hasattr(pl_module, 'use_ema') and pl_module.use_ema:
                    try:
                        import copy
                        # Find latest checkpoint
                        ckpt_dir = os.path.join(trainer.default_root_dir, 'checkpoints')
                        if os.path.isdir(ckpt_dir):
                            ckpts = sorted([f for f in os.listdir(ckpt_dir) if f.endswith('.ckpt') and f != 'last.ckpt'])
                            if ckpts:
                                latest_ckpt = os.path.join(ckpt_dir, ckpts[-1])
                                state_dict = torch.load(latest_ckpt, map_location=pl_module.device)['state_dict']
                                # Extract EMA weights and remap to pipeline prefix
                                ema_sd = {}
                                for k, v in state_dict.items():
                                    if k.startswith('ema_pipeline.'):
                                        ema_sd[k.replace('ema_pipeline.', '')] = v
                                if ema_sd:
                                    # Create fresh eval pipeline from EMA weights
                                    eval_pipeline = copy.deepcopy(pl_module.pipeline)
                                    eval_pipeline.load_state_dict(ema_sd, strict=False)
                                    eval_pipeline.eval()
                                    eval_pipeline.requires_grad_(False)
                                    Log.info(f"[EvalCallback] Loaded EMA weights from {os.path.basename(latest_ckpt)}")
                                else:
                                    Log.warn("[EvalCallback] No EMA keys in checkpoint, using regular pipeline")
                            else:
                                Log.warn("[EvalCallback] No checkpoints found, using regular pipeline")
                        else:
                            Log.warn("[EvalCallback] Checkpoint dir missing, using regular pipeline")
                    except Exception as e:
                        Log.warn(f"[EvalCallback] Failed to load EMA from checkpoint: {e}, using regular pipeline")

                _, _, _, SAPolicyEvalAdapter, _ = _runner_api()
                policy = SAPolicyEvalAdapter(eval_pipeline, pl_module.device)

                # Switch to eval mode for inference
                eval_pipeline.eval()
                metrics = runner.run(policy)
                # Restore training mode (only for non-EMA; EMA stays in eval)
                if eval_pipeline is pl_module.pipeline:
                    pl_module.pipeline.train()

                # Cancel alarm — eval finished in time
                signal.alarm(0)

                # Log scalar metrics
                videos = []
                for k, v in metrics.items():
                    if 'mean' in k:
                        val = np.mean(v) if isinstance(v, list) else v
                        pl_module.log(
                            f"eval/{k}",
                            val,
                            on_step=False,
                            on_epoch=True,
                            prog_bar=True,
                            logger=True,
                            sync_dist=False,  # only rank 0 runs this
                        )
                        Log.info(f"[EvalCallback] {label}: {k} = {val}")
                    if 'video' in k:
                        videos.append(v)

                # Log videos
                logger = trainer.logger
                step = trainer.global_step
                for idx, video_path in enumerate(videos):
                    if "TensorBoardLogger" in logger.__class__.__name__:
                        try:
                            writer = logger.experiment
                            video_array = imageio.mimread(video_path)
                            video_array = np.stack(video_array, axis=0)
                            if video_array.dtype != np.uint8:
                                video_array = video_array.astype(np.uint8)
                            video_tensor = torch.from_numpy(video_array).permute(0, 3, 1, 2).unsqueeze(0)
                            writer.add_video(f"eval/video_{idx}", video_tensor, global_step=step, fps=10)
                            writer.flush()
                            del video_array, video_tensor
                        except Exception as e:
                            Log.warn(f"[EvalCallback] Failed to log video: {e}")

                    if "WandbLogger" in logger.__class__.__name__:
                        logger.experiment.log({
                            f"eval/video_{idx}": wandb.Video(video_path, fps=10, format="mp4")
                        })

                Log.info(f"[EvalCallback] Eval at {label} complete.")

                # --- Cross-view evals: azimuth sweep + CPGen training views ---
                cross_view_cfgs = self._eval_policy_cfg.get("cross_view_configs", None)
                if cross_view_cfgs is None and self._eval_policy_cfg.get("enable_cross_view_eval", False):
                    cross_view_cfgs = AZIMUTH_SWEEP_CONFIGS + CPGEN_TRAINING_VIEW_CONFIGS

                if cross_view_cfgs:
                    Log.info(f"[EvalCallback] Running {len(cross_view_cfgs)} cross-view evals...")
                    for view_cfg in cross_view_cfgs:
                        view_name = view_cfg["name"]
                        try:
                            cv_output_dir = join(f"{self._eval_policy_cfg.output_dir}/{label}/cross_view/{view_name}")
                            os.makedirs(cv_output_dir, exist_ok=True)
                            cv_runner = self._get_cross_view_runner(view_cfg, cv_output_dir)
                            cv_metrics = cv_runner.run(policy)
                            for k, v in cv_metrics.items():
                                if 'mean' in k:
                                    val = np.mean(v) if isinstance(v, list) else v
                                    pl_module.log(
                                        f"eval_xview/{view_name}/{k}",
                                        val,
                                        on_step=False,
                                        on_epoch=True,
                                        prog_bar=False,
                                        logger=True,
                                        sync_dist=False,
                                    )
                                    Log.info(f"[EvalCallback] {label} cross-view {view_name}: {k} = {val}")
                        except Exception as e:
                            Log.warn(f"[EvalCallback] Cross-view eval {view_name} failed: {e}")

            except _EvalTimeout:
                signal.alarm(0)
                Log.error(f"[EvalCallback] Eval TIMED OUT at {label} "
                          f"(>{self.EVAL_TIMEOUT_SECONDS}s). Disabling all future evals.")
                self._eval_disabled = True
                pl_module.pipeline.train()

            except Exception as e:
                signal.alarm(0)
                Log.error(f"[EvalCallback] Eval FAILED at {label}: {e}")
                Log.error("[EvalCallback] Disabling all future evals, continuing training.")
                self._eval_disabled = True
                pl_module.pipeline.train()

            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
                gc.collect()
                torch.cuda.empty_cache()

        # DDP barrier: all ranks must wait for rank 0 to finish eval
        # before proceeding to next training batch, otherwise NCCL will timeout
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self._eval_every_n_steps <= 0:
            return
        step = trainer.global_step
        if step == 0 or step % self._eval_every_n_steps != 0:
            return
        if step == self._last_eval_step:
            return
        self._last_eval_step = step
        self._run_eval(trainer, pl_module, f"step_{step:06d}")

    def on_train_epoch_end(self, trainer, pl_module):
        if self._eval_every_n_epochs <= 0:
            return
        epoch = trainer.current_epoch + 1  # 0-indexed → 1-indexed
        if epoch % self._eval_every_n_epochs != 0:
            return
        if epoch == self._last_eval_epoch:
            return
        self._last_eval_epoch = epoch
        self._run_eval(trainer, pl_module, f"epoch_{epoch:04d}")
