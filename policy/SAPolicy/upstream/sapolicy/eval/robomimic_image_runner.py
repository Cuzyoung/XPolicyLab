import argparse
import os
import sys
import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import h5py
import math
import dill
import hashlib
import wandb.sdk.data_types.video as wv
import gc
import copy
import subprocess
import hydra
from typing import Dict, Any, List, Union, Optional
from omegaconf import OmegaConf, DictConfig
from torchvision.transforms import Compose

# Set these before importing cpgen_envs / robosuite so MuJoCo picks the right
# GL backend on import.
from sapolicy.eval.gl_bootstrap import ensure_eval_gl_env

ensure_eval_gl_env()
import glob
from scipy.spatial.transform import Rotation as R
import json
from torch.serialization import add_safe_globals
from omegaconf.listconfig import ListConfig
from omegaconf.base import ContainerMetadata, Metadata
from omegaconf.nodes import AnyNode

add_safe_globals([ListConfig, DictConfig, ContainerMetadata, AnyNode, Metadata, Any, Dict, List, list, collections.defaultdict, dict, int])

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from sapolicy.eval.gym_util.async_vector_env import AsyncVectorEnv
from sapolicy.eval.gym_util.sync_vector_env import SyncVectorEnv
from sapolicy.eval.gym_util.multistep_wrapper import MultiStepWrapper
from sapolicy.eval.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder

from sapolicy.models.utils.rotation import rotation_6d_to_matrix, matrix_to_axis_angle
from sapolicy.dataset.normalizer import LinearNormalizer, dict_apply
from sapolicy.embodiment_transforms import (
    build_data_transform,
    resolve_transform_config,
    _normalise_gripper_family,
)

from sapolicy.eval.gym_util.robomimic_image_wrapper import RobomimicImageWrapper
from sapolicy.eval.dataset_camera_info import enrich_fixed_spherical_config
from sapolicy.eval import env_factory
from sapolicy.eval.env_factory import (
    _apply_robotiq85_osc_kp_default,
    _build_observation_transforms,
    _build_shape_meta,
    _infer_lowdim_obs_shape,
    _load_env_meta_from_dataset,
    _normalize_controller_configs,
    _parse_fovy_overrides,
    create_env,
    get_camera_info,
)
from sapolicy.eval.env_factory import _require_cpgen_envs
from sapolicy.eval.policy_wrapper import (
    SAPolicyEvalAdapter,
    _SAPolicyWrapper,
    _find_latest_checkpoint,
    _instantiate_vla_pipeline,
)

DomainRandomizationWrapper = None
_ROBOSUITE_IMPORT_ERROR = None


def _get_domain_randomization_wrapper():
    global DomainRandomizationWrapper, _ROBOSUITE_IMPORT_ERROR
    if DomainRandomizationWrapper is not None:
        return DomainRandomizationWrapper
    try:
        env_factory._require_robosuite()
        from robosuite.wrappers import DomainRandomizationWrapper as DRW

        DomainRandomizationWrapper = DRW
        return DRW
    except Exception as e:
        _ROBOSUITE_IMPORT_ERROR = e
        raise

class BaseImageRunner:
    def __init__(self, output_dir):
        self.output_dir = output_dir
    def close(self):
        pass

def convert_relative_actions_to_absolute(eef_pos, eef_quat, actions, action_orn_mode='6d', body_frame=True):
    """
    输入：
        actions -> shape (B, T, 10)  (px,py,pz, rot6d, gripper)
    输出：
        actions -> shape (B, T, 7)  (px,py,pz, rx,ry,rz, gripper)
    """
    B, T, D = actions.shape
    action_pos = actions[..., :3]             # (B, T,3)
    
    if action_orn_mode == '6d':
        action_rot6d = actions[..., 3:9]          # (B, T,6)
        action_rot6d = action_rot6d.reshape(B * T, action_rot6d.shape[-1])
        if isinstance(action_rot6d, torch.Tensor):
            invalid_6d = ~torch.isfinite(action_rot6d).all(dim=-1)
            if invalid_6d.any():
                print(
                    f"[eval] Replacing {int(invalid_6d.sum().item())} invalid rot6d action(s) with identity.",
                    flush=True,
                )
                action_rot6d = action_rot6d.clone()
                action_rot6d[invalid_6d] = action_rot6d.new_tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        action_rotmat = rotation_6d_to_matrix(action_rot6d)
        if isinstance(action_rotmat, torch.Tensor):
            action_rotmat = action_rotmat.detach().cpu().numpy()
        invalid_rotmat = ~np.isfinite(action_rotmat).all(axis=(1, 2))
        if invalid_rotmat.any():
            print(
                f"[eval] Replacing {int(invalid_rotmat.sum())} invalid rotation matrix/matrices with identity.",
                flush=True,
            )
            action_rotmat = action_rotmat.copy()
            action_rotmat[invalid_rotmat] = np.eye(3, dtype=action_rotmat.dtype)
        action_rot = R.from_matrix(action_rotmat)
    elif action_orn_mode == 'quat':
        action_quat = actions[..., 3:7]                         # (B,T,4)
        action_quat = action_quat.reshape(B * T, 4)
        action_rot = R.from_quat(action_quat)                  # Rotation(B*T)
    elif action_orn_mode == 'euler':
        action_euler = actions[..., 3:6]          # (B, T,3)
        action_euler = action_euler.reshape(B * T, action_euler.shape[-1])
        action_rot = R.from_euler('xyz', action_euler, degrees=False)  # -> Rotation batch
    else:
        raise ValueError(f"Invalid action orientation mode: {action_orn_mode}")
    
    gripper = actions[..., -1:]                   # (B, T, 1)

    # Build eef_rot from quaternion
    if isinstance(eef_quat, R):
        eef_rot = eef_quat
    else:
        B_, T_, D_ = eef_quat.shape    # (B,T_,4)
        assert T_ == 1, "eef_quat should be a single frame"
        assert B_ == B, "eef_quat should have the same batch size as actions"
        eef_quat = eef_quat.repeat(T, axis=1)
        eef_quat = eef_quat.reshape(B * T, D_)
        eef_rot = R.from_quat(eef_quat)

    if body_frame:
        # Body-frame inversion:
        #   rel_pos = R_cur^T @ (p_target - p_cur)  →  abs_pos = R_cur @ rel_pos + p_cur
        #   rel_rot = R_cur^{-1} * R_target          →  abs_rot = R_cur * rel_rot
        action_pos_flat = action_pos.reshape(B * T, 3)
        world_delta_pos = eef_rot.apply(action_pos_flat)       # R_cur @ rel_pos
        world_delta_pos = world_delta_pos.reshape(B, T, 3)
        abs_pos = world_delta_pos + eef_pos                    # + p_cur
        abs_rot = eef_rot * action_rot                         # R_cur * R_rel = R_target
    else:
        # World-frame inversion:
        #   rel_pos = p_target - p_cur               →  abs_pos = rel_pos + p_cur
        #   rel_rot = R_target * R_cur^{-1}          →  abs_rot = rel_rot * R_cur
        action_pos_np = np.asarray(action_pos)
        abs_pos = action_pos_np + eef_pos                      # rel_pos + p_cur
        abs_rot = action_rot * eef_rot                         # R_rel * R_cur = R_target
    abs_rotvec = abs_rot.as_rotvec()  # (B * T, 3)
    abs_rotvec = abs_rotvec.reshape(B, T, 3)

    # Concatenate (T, 3+3+1) = (T, 7)
    actions = np.concatenate([abs_pos, abs_rotvec, gripper], axis=-1)

    return torch.from_numpy(actions)

class RobomimicImageRunner(BaseImageRunner):
    """
    Robomimic envs already enforces number of steps.
    """

    def __init__(self, 
            output_dir,
            dataset_path,
            shape_meta:dict,
            n_train=10,
            n_train_vis=3,
            train_start_idx=0,
            n_test=22,
            n_test_vis=6,
            test_start_seed=10000,
            test_init_from_dataset: bool = False,
            test_start_idx: int | None = None,
            panda_reference_hdf5: str | None = None,
            gripper_types_override: list[str] | None = None,
            max_steps=450,
            n_obs_steps=1,
            n_action_steps=8,
            render_obs_key='agentview_image',
            camera_names=None,
            fps=10,
            crf=22,
            past_action=False,
            abs_action=False,
            tqdm_interval_sec=5.0,
            n_envs=None,
            save_rollout_states_path: str = None,
            randomize_color=False,
            randomize_camera=False,
            randomize_lighting=False,
            randomize_dynamics=False,
            camera_depths=False,
            camera_segmentations=False,
            observation_transforms=None,
            min_depth=0.1,
            max_depth=5.0,
            camera_height=256,
            camera_width=256,
            action_sequence_length=16,
            action_orn_mode='6d',
            rotation_backend='scipy',
            relative_action=False,
            normalize_actions=False,
            norm_type='percentile_0.02_0.98', # 'mean_std' or 'percentile_0.02_0.98'
            video_render_camera: str | None = None,
            dataset_type: str = "cpgen",
            embodiment: str | None = None,
            cpgen_absolute_actions: bool = False,
            cpgen_action_pos_scale: float | None = None,
            cpgen_action_rot_scale: float | None = None,
            camera_fovy_overrides: Dict[str, float] | None = None,
            body_frame_actions: bool = True,
            multi_view_config: dict | None = None,
            camera_rename: dict | None = None,
            coupled_third_config: dict | None = None,
            determinism_debug: bool = False,
        ):
        super().__init__(output_dir)

        self.randomize_color = randomize_color
        self.randomize_camera = randomize_camera
        self.randomize_lighting = randomize_lighting
        self.randomize_dynamics = randomize_dynamics
        self.determinism_debug = bool(determinism_debug)

        self.render_obs_key = render_obs_key
        self.video_render_camera = video_render_camera
        self.dataset_path = dataset_path
        self.shape_meta = shape_meta

        self.observation_transforms = observation_transforms
        self.camera_names = camera_names
        if self.camera_names is None:
            self.camera_names = ['agentview'] # , 'robot0_eye_in_hand']
        # Rename cameras passed to the model without touching env rendering.
        # Use case: model trained with canonical {view_a, view_b} names, env
        # renders real cameras like third_view_0 / third_view_2. Keep
        # self.camera_names = [third_view_0, third_view_2] so the env loads the
        # correct real cameras; apply camera_rename at the end of
        # _prepare_vla_batch to present {view_a, view_b} to the model.
        self.camera_rename = dict(camera_rename) if camera_rename else None
        if self.camera_rename:
            missing = [c for c in self.camera_rename if c not in self.camera_names]
            if missing:
                raise ValueError(
                    f"camera_rename keys {missing} not found in camera_names {self.camera_names}"
                )
        self.coupled_third_config = dict(coupled_third_config) if coupled_third_config else None
        self.camera_height = camera_height
        self.camera_width = camera_width
        self.camera_depths = camera_depths
        self.camera_segmentations = camera_segmentations
        self.norm_type = norm_type
        assert self.norm_type in ['mean_std', 'percentile_0.02_0.98'], f"Invalid norm type: {self.norm_type}"

        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self._mujoco_depth_near = None
        self._mujoco_depth_far = None

        self.action_orn_mode = action_orn_mode
        self.rotation_backend = rotation_backend
        self.relative_action = relative_action
        self.body_frame_actions = body_frame_actions
        self.dataset_type = str(dataset_type).lower()
        self.embodiment = embodiment
        self.cpgen_absolute_actions = bool(cpgen_absolute_actions)
        self.normalize_actions = normalize_actions
        self.action_sequence_length = action_sequence_length

        if n_envs is None:
            n_envs = n_train + n_test

        # assert n_obs_steps <= n_action_steps
        dataset_path = os.path.expanduser(dataset_path)
        self.dataset_path = dataset_path
        robosuite_fps = 20
        steps_per_render = max(robosuite_fps // fps, 1)

        # read from dataset
        env_meta = _load_env_meta_from_dataset(dataset_path)
        # Must normalize to composite ("body_parts") form before the override
        # merge below reads controller_configs["body_parts"] -- a flat-format
        # shard (sawyer/iiwa/ur5e) has zero body_parts entries to merge into
        # until it's converted, which silently no-ops the override.
        env_meta['env_kwargs']['controller_configs'] = _normalize_controller_configs(
            env_meta['env_kwargs']['controller_configs'], env_meta['env_kwargs'].get('robots')
        )

        # Eval-only gripper swap (e.g. IIWA/UR5e + CalibratedRobotiq85Gripper): everything
        # else (camera_names, controller_configs, robots, init states) still comes from
        # dataset_path's own env_args.
        if gripper_types_override:
            env_meta['env_kwargs']['gripper_types'] = list(gripper_types_override)
            print(f"[RobomimicImageRunner] Overriding gripper_types -> {gripper_types_override} "
                  f"(robots={env_meta['env_kwargs'].get('robots')} still from dataset_path={dataset_path})")

        # Robotiq85 tuned defaults (OSC kp; friction/torsion/grip_kp are applied
        # post-creation in _apply_gripper_physics_overrides). Runs after the
        # gripper_types_override above (so an overridden gripper is detected too)
        # and before the diagnostic controller_configs override below (so that
        # opt-in diagnostic override still wins if a developer sets one).
        _apply_robotiq85_osc_kp_default(env_meta)

        # Non-Panda zero-shot eval: warm up from the native reset pose to the
        # Panda-aligned start for this episode's randomly placed square_nut,
        # instead of letting the policy see the raw cross-embodiment retargeting
        # artifact. No-op for Panda datasets and when no reference is found.
        # panda_reference_hdf5 (constructor kwarg / eval.panda_reference_hdf5
        # config key) is an explicit override; otherwise this is derived
        # automatically from dataset_path for the standard CE dataset layout.
        self._warmup_ref = None
        _robots = env_meta.get('env_kwargs', {}).get('robots') or []
        _env_name = str(env_meta.get('env_name', ''))
        if list(_robots) != ["Panda"]:
            from sapolicy.eval.gym_util.embodiment_warmup_ref import (
                build_panda_reference_table, derive_panda_reference_path,
            )
            from sapolicy.eval.gym_util.embodiment_warmup import _ensure_calibrated_robotiq85_patch

            _ensure_calibrated_robotiq85_patch()
            _panda_reference_hdf5 = panda_reference_hdf5 or derive_panda_reference_path(dataset_path)
            if _panda_reference_hdf5 and _env_name == "NutAssemblySquare":
                self._warmup_ref = build_panda_reference_table(_panda_reference_hdf5)
                print(f"[RobomimicImageRunner] Built Panda alignment reference table "
                      f"({self._warmup_ref['num_demos']} demos) from {_panda_reference_hdf5}")

        # Diagnostic override: source ONLY controller_configs (kp/input_type/
        # input_ref_frame/...) from a DIFFERENT file than dataset_path. env_name,
        # robots, camera_names, gripper_types etc. all still come from dataset_path's
        # own env_args -- only the controller gains/mode are swapped in. This isolates
        # "is the baked controller config wrong" without also silently swapping which
        # robot gets instantiated (a full env_args replacement would e.g. make a
        # sawyer/iiwa/ur5e shard's dataset_path build a Panda sim instead, which is not
        # what this flag is for and would likely crash on a gripper_qpos shape mismatch
        # against dataset_path's own shape_meta).
        _ctrl_override_path = os.environ.get("SAPOLICY_EVAL_ENV_ARGS_PATH")
        if _ctrl_override_path:
            _override_meta = _load_env_meta_from_dataset(_ctrl_override_path)
            _override_ctrl = _override_meta.get("env_kwargs", {}).get("controller_configs")
            if _override_ctrl is not None:
                _override_ctrl = _normalize_controller_configs(
                    _override_ctrl, _override_meta.get("env_kwargs", {}).get("robots")
                )

            _override_keys_raw = os.environ.get("SAPOLICY_EVAL_CONTROLLER_OVERRIDE_KEYS", "").strip()
            if _override_ctrl is not None and _override_keys_raw:
                _override_keys = [k.strip() for k in _override_keys_raw.split(",") if k.strip()]
                print(f"[RobomimicImageRunner] Overriding controller_configs field(s) "
                      f"{_override_keys} from {_ctrl_override_path} (everything else, "
                      f"including kp, stays dataset_path={dataset_path}'s own)")
                for _bp_name, _bp_cfg in env_meta["env_kwargs"]["controller_configs"].get("body_parts", {}).items():
                    _override_bp = _override_ctrl.get("body_parts", {}).get(_bp_name, {})
                    for _k in _override_keys:
                        if _k in _override_bp:
                            _bp_cfg[_k] = _override_bp[_k]
                        else:
                            _bp_cfg.pop(_k, None)  # override file doesn't set it -> fall back to controller default
            elif _override_ctrl is not None:
                print(f"[RobomimicImageRunner] Overriding controller_configs from "
                      f"{_ctrl_override_path} (env_name={env_meta.get('env_name')}, "
                      f"robots={env_meta.get('env_kwargs', {}).get('robots')} still from "
                      f"dataset_path={dataset_path})")
                env_meta["env_kwargs"]["controller_configs"] = _override_ctrl

        # CPGen envs rely on import side effects for registration.
        env_name = str(env_meta.get("env_name", ""))
        if env_name.lower().startswith("kitchen"):
            kitchen_max_steps = 1200
            if int(max_steps) != kitchen_max_steps:
                print(
                    f"[RobomimicImageRunner] Kitchen fixed horizon overrides "
                    f"max_steps={max_steps} -> {kitchen_max_steps}"
                )
            max_steps = kitchen_max_steps
        _robosuite_builtin_envs = {
            "Lift", "Stack", "NutAssembly", "NutAssemblySingle", "NutAssemblySquare",
            "NutAssemblyRound", "PickPlace", "PickPlaceSingle", "PickPlaceMilk",
            "PickPlaceBread", "PickPlaceCereal", "PickPlaceCan", "Door", "Wipe",
            "ToolHang", "TwoArmLift", "TwoArmPegInHole", "TwoArmHandover", "TwoArmTransport",
        }
        if env_factory.cpgen_envs is None and (
            "cpgen" in os.path.normpath(dataset_path).lower().split(os.sep)
            or env_name not in _robosuite_builtin_envs
        ):
            env_factory._require_cpgen_envs()

        # disable object state observation
        env_meta['env_kwargs']['use_object_obs'] = False
        env_meta['env_kwargs']['has_offscreen_renderer'] = False
        env_meta['env_kwargs']['render_gpu_device_id'] = -1
        env_meta['env_kwargs']['renderer'] = 'mujoco'

        print("dataset_path: ", dataset_path)
        print(f"camera_height: {camera_height}, camera_width: {camera_width}")

        # abs_action and relative_action both end up feeding env.step() an ABSOLUTE
        # world-frame pose (relative_action converts via
        # convert_relative_actions_to_absolute below, then skips undo_transform_action's
        # rescaling entirely -- see the `not self.relative_action` guard around the
        # env.step() call). The installed robosuite (1.5.2) uses the composite-controller
        # API, where OSC_POSE reads per-body-part `input_type: "delta"|"absolute"` to
        # decide how to interpret the action -- `control_delta` is a dead legacy key it
        # never looks at. Trusting whatever `input_type` happens to be baked into the
        # --dataset's env_args is wrong whenever that differs from what this runner
        # actually sends: e.g. the cross-embodiment paired199 shards bake
        # input_type="delta", which silently routes an absolute pose through OSC's
        # delta/scale_action path and corrupts every command regardless of how correct
        # the policy is.
        if abs_action or relative_action:
            _ctrl_cfg = env_meta['env_kwargs']['controller_configs']
            _ctrl_cfg['control_delta'] = False
            # create_env() below renames body_parts_controller_configs -> body_parts
            # for legacy env_args; normalize whichever key is currently present so the
            # override survives that rename.
            _bp_key = 'body_parts' if 'body_parts' in _ctrl_cfg else 'body_parts_controller_configs'
            for _bp_cfg in _ctrl_cfg.get(_bp_key, {}).values():
                _bp_cfg['input_type'] = 'absolute'

        # CPGen datasets store controller-input deltas scaled to physical units during
        # training.  At eval we must undo this scaling after converting 6D→AA so the
        # environment receives controller inputs in [-1, 1].
        # IMPORTANT: These scales must match the cpgen_action_pos_scale /
        # cpgen_action_rot_scale used during training (in the dataset config),
        # NOT the controller's output_max from env_meta.
        self._cpgen_action_pos_scale = None
        self._cpgen_action_rot_scale = None
        if not abs_action:
            if cpgen_action_pos_scale is not None:
                self._cpgen_action_pos_scale = float(cpgen_action_pos_scale)
            if cpgen_action_rot_scale is not None:
                self._cpgen_action_rot_scale = float(cpgen_action_rot_scale)
            # Fallback: detect from env_meta controller output_max (may differ
            # from training scales!).
            if self._cpgen_action_pos_scale is None or self._cpgen_action_rot_scale is None:
                ctrl = env_meta.get('env_kwargs', {}).get('controller_configs', {})
                bp = ctrl.get('body_parts', {})
                arm_cfg = bp.get('right', bp.get('left', {}))
                output_max = arm_cfg.get('output_max', None)
                if output_max and isinstance(output_max, (list, tuple)) and len(output_max) >= 6:
                    if self._cpgen_action_pos_scale is None:
                        self._cpgen_action_pos_scale = float(output_max[0])
                    if self._cpgen_action_rot_scale is None:
                        self._cpgen_action_rot_scale = float(output_max[3])

        if self._cpgen_action_pos_scale is not None or self._cpgen_action_rot_scale is not None:
            print(f"[CPGen action scales] pos={self._cpgen_action_pos_scale}, rot={self._cpgen_action_rot_scale}")

        _gripper_types = env_meta.get('env_kwargs', {}).get('gripper_types')
        _gripper_type = _gripper_types[0] if isinstance(_gripper_types, (list, tuple)) else _gripper_types
        _gripper_family = _normalise_gripper_family(_gripper_type)
        transform_config = resolve_transform_config(self.dataset_type, self.embodiment)
        resolved_gripper_family = _gripper_family or transform_config.gripper_family
        resolved_gripper_type = _gripper_type or transform_config.gripper_type
        self.embodiment_transform = build_data_transform(
            transform_config,
            gripper_width={"gripper_family": resolved_gripper_family},
            tcp_alignment={"gripper_type": resolved_gripper_type},
            action_se3=dict(
                action_orn_mode=self.action_orn_mode,
                use_relative_actions=self.relative_action,
                body_frame_actions=self.body_frame_actions,
                dataset_type=self.dataset_type,
                cpgen_action_pos_scale=self._cpgen_action_pos_scale or 0.05,
                cpgen_action_rot_scale=self._cpgen_action_rot_scale or 0.5,
                cpgen_absolute_actions=self.cpgen_absolute_actions,
                rotation_backend=self.rotation_backend,
            ),
        )

        self.save_rollout_states_path = save_rollout_states_path
        self.camera_info = None
        multi_view_config, resolved_camera_pose = enrich_fixed_spherical_config(
            dataset_path, multi_view_config
        )
        camera_fovy_overrides = _parse_fovy_overrides(camera_fovy_overrides) or {}
        if resolved_camera_pose is not None:
            camera_fovy_overrides["agentview"] = resolved_camera_pose.fovy
            print(
                "[camera-info] exact fixed eval camera: "
                f"{resolved_camera_pose.camera_name} "
                f"dataset_az={resolved_camera_pose.dataset_azimuth:g} "
                f"pos={np.array2string(resolved_camera_pose.position, precision=6)} "
                f"fovy={resolved_camera_pose.fovy:.6f} "
                f"source_dataset={multi_view_config.get('fixed_camera_dataset_path')}"
            )
        self.camera_fovy_overrides = camera_fovy_overrides or None
        self.multi_view_config = multi_view_config

        self.env_meta = env_meta

        # Pre-fetch camera_info and MuJoCo depth params from a temporary env
        # so they are available in the main process even when vector envs run
        # env_fn in subprocesses.
        _need_tmp_env = (self.camera_info is None) or (
            camera_depths and self._mujoco_depth_near is None
        )
        if _need_tmp_env:
            _tmp_env = create_env(
                env_meta=env_meta,
                shape_meta=shape_meta,
                camera_depths=camera_depths,
                camera_segmentations=camera_segmentations,
                camera_height=camera_height,
                camera_width=camera_width,
                camera_fovy_overrides=self.camera_fovy_overrides,
                coupled_third_config=self.coupled_third_config,
            )
            if self.camera_info is None:
                self.camera_info = get_camera_info(
                    _tmp_env,
                    camera_names=self.camera_names,
                    camera_height=camera_height,
                    camera_width=camera_width,
                )
            # Extract MuJoCo depth buffer parameters (znear/zfar) so we can
            # convert the [0,1] depth buffer to real metric meters at eval time.
            # Training data stores depth in meters; MuJoCo returns a normalised
            # z-buffer. Without this conversion, the depth encoder receives a
            # completely different input distribution than it was trained on.
            if self._mujoco_depth_near is None:
                _base = getattr(_tmp_env, "base_env", _tmp_env)
                _sim = getattr(_base, "sim", None)
                if _sim is not None:
                    extent = _sim.model.stat.extent
                    self._mujoco_depth_far = _sim.model.vis.map.zfar * extent
                    self._mujoco_depth_near = _sim.model.vis.map.znear * extent
                    print(f"[Depth] MuJoCo znear={self._mujoco_depth_near:.4f}, zfar={self._mujoco_depth_far:.4f}")
            try:
                _tmp_env.close()
            finally:
                del _tmp_env
                gc.collect()

        def env_fn():
            robomimic_env = create_env(
                env_meta=env_meta,
                shape_meta=shape_meta,
                camera_depths=camera_depths,
                camera_segmentations=camera_segmentations,
                camera_height=camera_height,
                camera_width=camera_width,
                camera_fovy_overrides=self.camera_fovy_overrides,
                multi_view_config=self.multi_view_config,
                coupled_third_config=self.coupled_third_config,
            )
            self.camera_info = get_camera_info(
                robomimic_env,
                camera_names=self.camera_names,
                camera_height=camera_height,
                camera_width=camera_width,
            )
            # Robosuite's hard reset causes excessive memory consumption.
            # Disabled to run more envs.
            # https://github.com/ARISE-Initiative/robosuite/blob/92abf5595eddb3a845cd1093703e5a3ccd01e77e/robosuite/environments/base.py#L247-L248
            # Using domain randomization wrapper can lead to failed runs when both training and eval'ing w/ depth
            if (
                self.randomize_color
                or self.randomize_camera
                or self.randomize_dynamics
                or self.randomize_lighting
            ):
                DRW = _get_domain_randomization_wrapper()
                robomimic_env.env = DRW(
                    robomimic_env.env,
                    randomize_color=randomize_color,
                    randomize_camera=randomize_camera,
                    randomize_dynamics=randomize_dynamics,
                    randomize_lighting=randomize_lighting,
                    randomize_on_reset=True,
                    randomize_every_n_steps=-1,
                )
            # Disable hard reset on the underlying robosuite env to reduce memory spikes.
            base_env = getattr(robomimic_env, "base_env", None)
            if base_env is None:
                base_env = getattr(robomimic_env, "env", None)
            if base_env is not None and hasattr(base_env, "hard_reset"):
                base_env.hard_reset = False
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    RobomimicImageWrapper(
                        env=robomimic_env,
                        shape_meta=shape_meta,
                        init_state=None,
                        render_obs_key=render_obs_key,
                        video_render_camera=video_render_camera,
                        warmup_ref=self._warmup_ref,
                    ),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps,
                        codec="h264",
                        input_pix_fmt="rgb24",
                        crf=crf,
                        thread_type="FRAME",
                        thread_count=1,
                    ),
                    file_path=None,
                    steps_per_render=steps_per_render,
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps,
            )
        
        # For each process the OpenGL context can only be initialized once
        # Since AsyncVectorEnv uses fork to create worker process,
        # a separate env_fn that does not create OpenGL context (enable_render=False)
        # is needed to initialize spaces.
        def dummy_env_fn():
            robomimic_env = create_env(
                    env_meta=env_meta,
                    shape_meta=shape_meta,
                    camera_depths=camera_depths,
                    camera_segmentations=camera_segmentations,
                    enable_render=False,
                    camera_height=camera_height,
                    camera_width=camera_width,
                    camera_fovy_overrides=self.camera_fovy_overrides,
                    coupled_third_config=self.coupled_third_config,
                )
            # Using domain randomization wrapper can lead to failed runs when both training and eval'ing w/ depth
            if (self.randomize_color \
                or self.randomize_camera \
                or self.randomize_dynamics \
                or self.randomize_lighting
            ):
                DRW = _get_domain_randomization_wrapper()
                robomimic_env.env = DRW(
                    robomimic_env.env, 
                    randomize_color=randomize_color,
                    randomize_camera=randomize_camera,
                    randomize_dynamics=randomize_dynamics,
                    randomize_lighting=randomize_lighting,
                )
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    RobomimicImageWrapper(
                        env=robomimic_env,
                        shape_meta=shape_meta,
                        init_state=None,
                        render_obs_key=render_obs_key,
                        video_render_camera=video_render_camera,
                        warmup_ref=self._warmup_ref,
                    ),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps,
                        codec='h264',
                        input_pix_fmt='rgb24',
                        crf=crf,
                        thread_type='FRAME',
                        thread_count=1
                    ),
                    file_path=None,
                    steps_per_render=steps_per_render
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps
            )

        env_fns = [env_fn] * n_envs
        env_seeds = list()
        env_prefixs = list()
        env_init_fn_dills = list()

        # train
        with h5py.File(dataset_path, 'r') as f:
            for i in range(n_train):
                train_idx = train_start_idx + i
                enable_render = i < n_train_vis
                # `align_start_offset` (default 0) skips the embodiment-specific
                # "move to Panda-aligned start pose" frames mimicgen bakes into
                # non-Panda demos — see sapolicy/dataset/robomimic_hdf5.py.
                start_offset = int(f[f'data/demo_{train_idx}'].attrs.get('align_start_offset', 0))
                init_state = f[f'data/demo_{train_idx}/states'][start_offset]

                def init_fn(env, init_state=init_state, 
                    enable_render=enable_render):
                    # setup rendering
                    # video_wrapper
                    assert isinstance(env.env, VideoRecordingWrapper)
                    env.env.video_recoder.stop()
                    env.env.file_path = None
                    if enable_render:
                        filename = pathlib.Path(output_dir).joinpath(
                            'media', f"train_{train_idx}_" + wv.util.generate_id() + ".mp4")
                        filename.parent.mkdir(parents=False, exist_ok=True)
                        filename = str(filename)
                        env.env.file_path = filename

                    # switch to init_state reset
                    assert isinstance(env.env.env, RobomimicImageWrapper)
                    env.env.env.init_state = init_state

                env_seeds.append(train_idx)
                env_prefixs.append('train/')
                env_init_fn_dills.append(dill.dumps(init_fn))
        
        # test
        if test_init_from_dataset:
            if test_start_idx is None:
                test_start_idx = train_start_idx + n_train
            with h5py.File(dataset_path, "r") as f:
                for i in range(n_test):
                    demo_idx = test_start_idx + i
                    enable_render = i < n_test_vis
                    try:
                        start_offset = int(f[f"data/demo_{demo_idx}"].attrs.get('align_start_offset', 0))
                        init_state = f[f"data/demo_{demo_idx}/states"][start_offset]
                    except Exception as e:
                        raise KeyError(
                            f"Failed to load init state for demo_{demo_idx} from dataset {dataset_path}. "
                            f"Check eval.test_start_idx / eval.n_test."
                        ) from e

                    def init_fn(env, init_state=init_state, demo_idx=demo_idx, enable_render=enable_render):
                        # setup rendering
                        # video_wrapper
                        assert isinstance(env.env, VideoRecordingWrapper)
                        env.env.video_recoder.stop()
                        env.env.file_path = None
                        if enable_render:
                            filename = pathlib.Path(output_dir).joinpath(
                                "media", f"test_demo_{demo_idx}_" + wv.util.generate_id() + ".mp4"
                            )
                            filename.parent.mkdir(parents=False, exist_ok=True)
                            filename = str(filename)
                            env.env.file_path = filename

                        # switch to init_state reset
                        assert isinstance(env.env.env, RobomimicImageWrapper)
                        env.env.env.init_state = init_state

                    env_seeds.append(demo_idx)
                    env_prefixs.append("test/")
                    env_init_fn_dills.append(dill.dumps(init_fn))
        else:
            for i in range(n_test):
                seed = test_start_seed + i
                enable_render = i < n_test_vis

                def init_fn(env, seed=seed, enable_render=enable_render):
                    # setup rendering
                    # video_wrapper
                    assert isinstance(env.env, VideoRecordingWrapper)
                    env.env.video_recoder.stop()
                    env.env.file_path = None
                    if enable_render:
                        filename = pathlib.Path(output_dir).joinpath(
                            "media", f"test_{seed}_" + wv.util.generate_id() + ".mp4"
                        )
                        filename.parent.mkdir(parents=False, exist_ok=True)
                        filename = str(filename)
                        env.env.file_path = filename

                    # switch to seed reset
                    assert isinstance(env.env.env, RobomimicImageWrapper)
                    env.env.env.init_state = None
                    env.seed(seed)

                env_seeds.append(seed)
                env_prefixs.append("test/")
                env_init_fn_dills.append(dill.dumps(init_fn))

        env = SyncVectorEnv(env_fns)

        self.env = env
        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_prefixs = env_prefixs
        self.env_init_fn_dills = env_init_fn_dills
        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.abs_action = abs_action
        self.tqdm_interval_sec = tqdm_interval_sec

    def _array_digest(self, value) -> str:
        arr = np.asarray(value)
        h = hashlib.sha1()
        h.update(str(arr.shape).encode("utf-8"))
        h.update(str(arr.dtype).encode("utf-8"))
        h.update(np.ascontiguousarray(arr).view(np.uint8))
        return h.hexdigest()[:12]

    def _env_state_digest(self, wrapped_env) -> str:
        obj = wrapped_env
        seen = set()
        for _ in range(8):
            if id(obj) in seen:
                break
            seen.add(id(obj))
            if hasattr(obj, "get_state"):
                state = obj.get_state()
                if isinstance(state, dict) and "states" in state:
                    return self._array_digest(state["states"])
            obj = getattr(obj, "env", None)
            if obj is None:
                break
        return "unavailable"

    def _is_vla_policy(self, policy):
        return hasattr(policy, "pipeline") and hasattr(policy.pipeline, "forward_test")

    def _get_hdf5_array(self, h5_file, path):
        if path is None:
            return None
        obj = h5_file
        for part in path.split('/'):
            if not part:
                continue
            if part not in obj:
                return None
            obj = obj[part]
        if isinstance(obj, h5py.Dataset):
            return obj[()]
        return None

    def _get_batch_size(self, np_obs_dict):
        fallback = None
        # A vectorized image history is unambiguous even when the number of
        # environments equals the history length: (B, T, H, W, C).
        for value in np_obs_dict.values():
            if (
                isinstance(value, np.ndarray)
                and value.ndim >= 5
                and value.shape[1] == self.n_obs_steps
            ):
                return value.shape[0]
        for value in np_obs_dict.values():
            if not isinstance(value, np.ndarray):
                continue
            if value.ndim == 0:
                continue
            first_dim = value.shape[0]
            if value.ndim >= 2 and value.shape[1] == self.n_obs_steps and first_dim != self.n_obs_steps:
                return first_dim
            if fallback is None:
                fallback = first_dim
        if fallback is None:
            return 1
        if fallback == self.n_obs_steps:
            return 1
        return fallback

    def _take_last_step(self, array, batch_size):
        arr = np.asarray(array)
        if arr.ndim == 0:
            return arr
        if arr.shape[0] == batch_size:
            if arr.ndim >= 2 and arr.shape[1] == self.n_obs_steps:
                return arr[:, -1]
            return arr
        if batch_size == 1 and arr.shape[0] == self.n_obs_steps:
            return arr[-1][np.newaxis, ...]
        return arr

    def _ensure_batch_dim(self, array, batch_size):
        arr = np.asarray(array)
        if arr.ndim == 0:
            return np.repeat(arr[np.newaxis], batch_size, axis=0)
        if arr.shape[0] != batch_size:
            if batch_size == 1:
                return arr[np.newaxis, ...]
            raise ValueError(f"Unexpected batch dimension for array with shape {arr.shape}, expected leading dimension {batch_size}")
        return arr

    def _camera_prefix_from_image_key(self, key):
        if key in ('image',):
            return None
        suffixes = ['_image', '_rgb', '_rgb_image']
        for suffix in suffixes:
            if key.endswith(suffix):
                return key[:-len(suffix)]
        return None

    def _resolve_image_key(self, obs_dict, camera_name):
        candidates = []
        for key in obs_dict.keys():
            if key not in candidates and 'image' in key and camera_name in key:
                candidates.append(key)
        for key in candidates:
            if key in obs_dict:
                value = obs_dict[key]
                if isinstance(value, np.ndarray):
                    return key
        return None

    def _resolve_state_key(self, obs_dict):
        candidates = ['robot0_eef_pos', 'robot0_eef_quat_site', 'robot0_gripper_qpos']
        for key in candidates:
            if key not in obs_dict:
                return None
        return candidates

    def _resolve_depth_key(self, obs_dict, image_key, camera_name):
        candidates = ['depth']
        prefix = self._camera_prefix_from_image_key(image_key) if image_key else None
        if prefix:
            candidates.extend([
                f'{prefix}_depth',
                f'{prefix}_depth_image',
                f'{prefix}_depths',
            ])
        for key in candidates:
            if key in obs_dict:
                value = obs_dict[key]
                if isinstance(value, np.ndarray):
                    return key
        for key, value in obs_dict.items():
            if 'depth' in key and isinstance(value, np.ndarray) and camera_name in key:
                return key
        return None

    def _resolve_intrinsics_key(self, obs_dict, image_key):
        candidates = ['camera_intrinsics']
        prefix = self._camera_prefix_from_image_key(image_key) if image_key else None
        if prefix:
            candidates.extend([
                f'{prefix}_camera_intrinsics',
                f'{prefix}_intrinsics',
                f'{prefix}_K',
            ])
        for key in candidates:
            if key in obs_dict:
                value = obs_dict[key]
                if isinstance(value, np.ndarray):
                    return key
        for key, value in obs_dict.items():
            if 'intrinsics' in key and isinstance(value, np.ndarray):
                return key
        return None

    def _prepare_vla_batch(self, np_obs_dict, device):
        batch_size = self._get_batch_size(np_obs_dict)
        debug_depth = os.environ.get("SAPOLICY_DEBUG_DEPTH_NONFINITE", "0") == "1"
        if debug_depth:
            self._debug_depth_call = getattr(self, "_debug_depth_call", 0) + 1
            debug_depth_call = self._debug_depth_call

        def debug_depth_array(stage, camera_name, value):
            if not debug_depth or value is None:
                return
            arr = np.asarray(value)
            finite = np.isfinite(arr)
            if finite.all():
                return
            finite_values = arr[finite]
            finite_min = float(finite_values.min()) if finite_values.size else float("nan")
            finite_max = float(finite_values.max()) if finite_values.size else float("nan")
            frame_bad = []
            if arr.ndim >= 4:
                time_axis = 1 if arr.ndim >= 5 else None
                if time_axis is not None:
                    for batch_idx in range(arr.shape[0]):
                        for time_idx in range(arr.shape[time_axis]):
                            frame = arr[batch_idx, time_idx]
                            frame_bad.append(
                                f"b{batch_idx}t{time_idx}:{int((~np.isfinite(frame)).sum())}"
                            )
            print(
                f"[depth-debug] call={debug_depth_call} stage={stage} camera={camera_name} "
                f"key_shape={arr.shape} bad={int((~finite).sum())} "
                f"finite_min={finite_min:.7g} finite_max={finite_max:.7g} "
                f"frame_bad={','.join(frame_bad) if frame_bad else 'n/a'}",
                flush=True,
            )

        # Runner observations are typically shaped as (B, n_obs_steps, ...).
        # SAPolicy expects an explicit time dimension: (B, T, C, H, W).
        latest_obs = np_obs_dict

        # Cameras added by MultiViewEnvWrapper are already pre-flipped (top-down)
        # at wrapper render time, so skip the runner's secondary flip for them.
        wrapper_flipped_cams = set()
        if self.coupled_third_config is not None:
            n_views = len(self.coupled_third_config.get("fixed_positions", []))
            wrapper_flipped_cams = {f"third_view_{i}" for i in range(n_views)}
        # One-shot image dump for visual debug
        _dump_once = os.environ.get("SAPOLICY_EVAL_DUMP_IMG", "") == "1" and not getattr(self, "_dumped_eval_img", False)
        obs = {}
        for camera_name in self.camera_names:
            image_key = self._resolve_image_key(latest_obs, camera_name)
            if image_key is None:
                raise KeyError("Unable to locate an image observation for VLA policy evaluation.")
            raw_image_batch = self._ensure_batch_dim(latest_obs[image_key], batch_size)

            depth_key = self._resolve_depth_key(latest_obs, image_key, camera_name)
            raw_depth_batch = None
            if depth_key is not None:
                raw_depth_batch = self._ensure_batch_dim(latest_obs[depth_key], batch_size)
                debug_depth_array(f"raw:{depth_key}", camera_name, raw_depth_batch)
                if not np.isfinite(raw_depth_batch).all():
                    bad = int((~np.isfinite(raw_depth_batch)).sum())
                    raise RuntimeError(
                        f"Camera {camera_name!r} returned {bad} non-finite depth values. "
                        "The MuJoCo render context may not be current; aborting to avoid "
                        "silently evaluating on corrupted observations."
                    )

            intrinsics_key = self._resolve_intrinsics_key(latest_obs, image_key)
            if intrinsics_key is not None:
                intrinsics_np = self._ensure_batch_dim(latest_obs[intrinsics_key], batch_size).astype(np.float32)
            else:
                if self.camera_info is None:
                    intrinsics_np = None
                else:
                    intrinsics_np = np.array(self.camera_info[camera_name]["intrinsics"], dtype=np.float32)
                    # Broadcast to batch.
                    intrinsics_np = np.broadcast_to(intrinsics_np, (batch_size, 3, 3)).copy()

            # R57: Load camera extrinsics (cam-to-world transform) for world-frame 3D
            extrinsics_np = None
            if self.camera_info is not None and camera_name in self.camera_info and "extrinsics" in self.camera_info[camera_name]:
                extrinsics_np = np.array(self.camera_info[camera_name]["extrinsics"], dtype=np.float32)
                extrinsics_np = np.broadcast_to(extrinsics_np, (batch_size, 4, 4)).copy()

            if self.observation_transforms is not None:
                processed_images = []
                processed_depths = [] if raw_depth_batch is not None else None
                processed_intrinsics = [] if intrinsics_np is not None else None
                for b in range(batch_size):
                    sample = {}
                    img = raw_image_batch[b]
                    if img.dtype != np.float32:
                        img = img.astype(np.float32)
                    if img.max() > 1.0:
                        img = img / 255.0
                    if img.ndim == 4 and img.shape[1] in (1, 3) and img.shape[-1] not in (1, 3):
                        img = np.transpose(img, (0, 2, 3, 1))
                    elif img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[-1] not in (1, 3):
                        img = np.transpose(img, (1, 2, 0))
                    # MuJoCo/OpenGL renders images bottom-up; training data is top-down.
                    # Vertical flip aligns live env observations with HDF5 training data.
                    # MultiViewEnvWrapper pre-flips wrapper cameras, skip to avoid double flip.
                    if camera_name not in wrapper_flipped_cams:
                        if img.ndim == 4:
                            img = np.ascontiguousarray(img[:, ::-1])
                        else:
                            img = np.ascontiguousarray(img[::-1])
                    sample['image'] = img

                    if raw_depth_batch is not None:
                        depth_val = raw_depth_batch[b].astype(np.float32)
                        if depth_val.ndim == 2:
                            depth_val = depth_val[..., None]
                        elif depth_val.ndim == 4 and depth_val.shape[1] == 1 and depth_val.shape[-1] != 1:
                            depth_val = np.transpose(depth_val, (0, 2, 3, 1))
                        elif depth_val.ndim == 3 and depth_val.shape[0] == 1 and depth_val.shape[-1] != 1:
                            depth_val = np.transpose(depth_val, (1, 2, 0))
                        # Vertical flip to match training data (same as image above).
                        if camera_name not in wrapper_flipped_cams:
                            if depth_val.ndim == 4:
                                depth_val = np.ascontiguousarray(depth_val[:, ::-1])
                            else:
                                depth_val = np.ascontiguousarray(depth_val[::-1])
                        sample['depth'] = depth_val

                    if intrinsics_np is not None:
                        # Each sample should carry its own intrinsics (3x3 or 9,).
                        sample_intr = intrinsics_np
                        if isinstance(intrinsics_np, np.ndarray) and intrinsics_np.ndim >= 2 and intrinsics_np.shape[0] == batch_size:
                            sample_intr = intrinsics_np[b]
                        # Resize mutates K in-place; copy so one sample cannot scale another.
                        if isinstance(sample_intr, np.ndarray) and sample_intr.ndim == 3:
                            # K is constant across the observation history for the eval env.
                            sample_intr = sample_intr[0]
                        if isinstance(sample_intr, np.ndarray):
                            sample_intr = sample_intr.copy()
                        sample['camera_intrinsics'] = sample_intr
                        
                    transformed = self.observation_transforms(sample)
                    
                    image_arr = transformed['image']
                    if isinstance(image_arr, torch.Tensor):
                        image_arr = image_arr.detach().cpu().numpy()
                    if image_arr.ndim == 4 and image_arr.shape[1] not in (1, 3) and image_arr.shape[-1] in (1, 3):
                        image_arr = np.transpose(image_arr, (0, 3, 1, 2))
                    elif image_arr.ndim == 3 and image_arr.shape[0] not in (1, 3) and image_arr.shape[-1] in (1, 3):
                        image_arr = np.transpose(image_arr, (2, 0, 1))
                    if image_arr.ndim == 3:
                        image_arr = image_arr[None]
                    processed_images.append(np.asarray(image_arr))

                    if processed_depths is not None and 'depth' in transformed:
                        depth_arr = transformed['depth']
                        if isinstance(depth_arr, torch.Tensor):
                            depth_arr = depth_arr.detach().cpu().numpy()
                        if depth_arr.ndim == 4 and depth_arr.shape[1] not in (1,) and depth_arr.shape[-1] == 1:
                            depth_arr = np.transpose(depth_arr, (0, 3, 1, 2))
                        elif depth_arr.ndim == 3 and depth_arr.shape[0] not in (1,) and depth_arr.shape[-1] == 1:
                            depth_arr = np.transpose(depth_arr, (2, 0, 1))
                        if depth_arr.ndim == 3:
                            depth_arr = depth_arr[None]
                        processed_depths.append(np.asarray(depth_arr))

                    if processed_intrinsics is not None:
                        processed_intrinsics.append(
                            np.asarray(transformed.get('camera_intrinsics', sample_intr), dtype=np.float32)
                        )

                image_np = np.stack(processed_images, axis=0).astype(np.float32)
                # Per-sample transforms should now yield (T, C, H, W).
                if image_np.ndim != 5:
                    raise ValueError(f"Transformed image has unexpected shape {image_np.shape}")
                if processed_depths is not None and len(processed_depths) == batch_size:
                    depth_np = np.stack(processed_depths, axis=0).astype(np.float32)
                    if depth_np.ndim != 5:
                        raise ValueError(f"Transformed depth has unexpected shape {depth_np.shape}")
                    debug_depth_array("transformed", camera_name, depth_np)
                else:
                    depth_np = None

                if processed_intrinsics is not None and len(processed_intrinsics) == batch_size:
                    intrinsics_np = np.stack(processed_intrinsics, axis=0).astype(np.float32)
            else:
                image_np = raw_image_batch.astype(np.float32)
                # Normalize to (B, T, C, H, W)
                if image_np.ndim == 4:
                    # (B, H, W, C) or (B, C, H, W) -> add T=1
                    image_np = image_np[:, None]
                if image_np.ndim != 5:
                    raise ValueError(f"Image observation has unexpected ndim {image_np.ndim} with shape {image_np.shape}")
                # Now either (B, T, H, W, C) or (B, T, C, H, W)
                if image_np.shape[-1] in (1, 3):
                    image_np = np.transpose(image_np, (0, 1, 4, 2, 3))

                if image_np.size > 0 and image_np.max() > 1.0:
                    image_np = image_np / 255.0

                # Vertical flip: MuJoCo renders bottom-up, training data is top-down.
                # In (B, T, C, H, W) format, H is axis -2.
                # MultiViewEnvWrapper pre-flips wrapper cameras, skip double flip.
                if camera_name not in wrapper_flipped_cams:
                    image_np = np.ascontiguousarray(image_np[..., ::-1, :])

                depth_np = None
                if raw_depth_batch is not None:
                    depth_tmp = raw_depth_batch.astype(np.float32)
                    if depth_tmp.ndim == 4:
                        # (B, H, W, 1) or (B, 1, H, W) -> add T=1
                        depth_tmp = depth_tmp[:, None]
                    if depth_tmp.ndim != 5:
                        raise ValueError(f"Depth observation has unexpected ndim {depth_tmp.ndim} with shape {depth_tmp.shape}")
                    # Now either (B, T, H, W, 1) or (B, T, 1, H, W)
                    if depth_tmp.shape[-1] == 1:
                        depth_tmp = np.transpose(depth_tmp, (0, 1, 4, 2, 3))
                    elif depth_tmp.shape[2] == 1:
                        pass
                    else:
                        raise ValueError(f"Depth observation has unexpected shape {depth_tmp.shape}")
                    # Vertical flip to match training data.
                    if camera_name not in wrapper_flipped_cams:
                        depth_tmp = np.ascontiguousarray(depth_tmp[..., ::-1, :])
                    depth_np = depth_tmp

            if depth_np is not None:
                # MuJoCo returns depth as a [0,1] z-buffer, but the model was
                # trained on metric depth in meters. Convert before normalising.
                # MultiViewEnvWrapper already calls get_real_depth_map() on
                # its cameras, so skip the z-buffer -> metric conversion for them.
                if (camera_name not in wrapper_flipped_cams
                        and self._mujoco_depth_near is not None
                        and self._mujoco_depth_far is not None):
                    near = self._mujoco_depth_near
                    far = self._mujoco_depth_far
                    depth_np = near / (1.0 - depth_np * (1.0 - near / far))
                depth_np = np.clip(depth_np, self.min_depth, self.max_depth)
                depth_np = (depth_np - self.min_depth) / (self.max_depth - self.min_depth + 1e-8)
                depth_np = np.ascontiguousarray(depth_np)
                debug_depth_array("normalized", camera_name, depth_np)

            image_np = np.ascontiguousarray(image_np)
            if "image" not in obs:
                obs["image"] = {}
            obs["image"][camera_name] = torch.from_numpy(image_np).to(device=device, dtype=torch.float32)
            if _dump_once:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                img_vis = image_np[0, 0].transpose(1, 2, 0)
                if img_vis.max() <= 1.0:
                    img_vis = (img_vis * 255).astype(np.uint8)
                path = f"/tmp/eval_dump_{camera_name}.png"
                plt.imsave(path, img_vis)
                print(f"[DUMP] saved {path} shape={img_vis.shape} mean={img_vis.mean():.2f}")

            if depth_np is not None:
                depth_np = np.ascontiguousarray(depth_np)
                if "depth" not in obs:
                    obs["depth"] = {}
                obs['depth'][camera_name] = torch.from_numpy(depth_np).to(device=device, dtype=torch.float32)
            if intrinsics_np is not None:
                intrinsics_np = np.ascontiguousarray(intrinsics_np)
                # Ensure batch dimension is present before reshaping.
                if intrinsics_np.ndim == 2:
                    # Single (3,3) or (1,9) — add batch dim
                    intrinsics_np = np.broadcast_to(intrinsics_np, (batch_size,) + intrinsics_np.shape).copy()
                if "camera_intrinsics" not in obs:
                    obs["camera_intrinsics"] = {}
                # (B, 3, 3) -> (B, 1, 3, 3) for SAPolicy's (B, T, 3, 3) convention.
                if intrinsics_np.ndim == 3:
                    intrinsics_np = intrinsics_np.reshape(batch_size, 1, 3, 3)
                obs['camera_intrinsics'][camera_name] = torch.from_numpy(intrinsics_np).to(device=device, dtype=torch.float32)

            # R57: Add camera extrinsics to observation
            if extrinsics_np is not None:
                extrinsics_np = np.ascontiguousarray(extrinsics_np)
                if extrinsics_np.ndim == 2:
                    extrinsics_np = np.broadcast_to(extrinsics_np, (batch_size,) + extrinsics_np.shape).copy()
                if "camera_extrinsics" not in obs:
                    obs["camera_extrinsics"] = {}
                # Store as [B, 4, 4] — no reshape needed
                obs['camera_extrinsics'][camera_name] = torch.from_numpy(extrinsics_np).to(device=device, dtype=torch.float32)

        if _dump_once:
            self._dumped_eval_img = True

        state_keys = self._resolve_state_key(latest_obs)
        if state_keys is not None:
            eef_pos = self._ensure_batch_dim(latest_obs['robot0_eef_pos'], batch_size).astype(np.float32)
            eef_rot_raw = self._ensure_batch_dim(latest_obs['robot0_eef_quat_site'], batch_size).astype(np.float32)
            gripper_qpos_raw = self._ensure_batch_dim(latest_obs['robot0_gripper_qpos'], batch_size).astype(np.float32)
            # Align eval state with training dataset state construction:
            # [eef_pos(3), eef_rot(6d), gripper_qpos[:1](1)] => 10D per step.
            transformed_obs = self.embodiment_transform.inputs({"observation": {
                "robot0_eef_pos": eef_pos,
                "robot0_eef_quat_site": eef_rot_raw,
                "robot0_gripper_qpos": gripper_qpos_raw,
            }})["observation"]
            eef_pos = transformed_obs["robot0_eef_pos"]
            if eef_rot_raw.shape[-1] == 4:
                eef_rot = transformed_obs["robot0_eef_rot_6d"]
            elif eef_rot_raw.shape[-1] == 6:
                eef_rot = eef_rot_raw
            else:
                raise ValueError(f"Unsupported eef rotation shape for state: {eef_rot_raw.shape}")
            gripper_qpos = transformed_obs["robot0_gripper_qpos"]
            if not (eef_pos.shape[:-1] == eef_rot.shape[:-1] == gripper_qpos.shape[:-1]):
                raise ValueError(
                    f"Eval state shape mismatch before concat: "
                    f"eef_pos={eef_pos.shape}, eef_rot={eef_rot.shape}, gripper={gripper_qpos.shape}"
                )
            state_np = np.concatenate([eef_pos, eef_rot, gripper_qpos], axis=-1).astype(np.float32)
            obs['state'] = torch.from_numpy(np.ascontiguousarray(state_np)).to(device=device, dtype=torch.float32)

        if 'text_embedding' in latest_obs and isinstance(latest_obs['text_embedding'], np.ndarray):
            text_np = self._ensure_batch_dim(latest_obs['text_embedding'], batch_size).astype(np.float32)
            text_np = np.ascontiguousarray(text_np)
            obs['text_embedding'] = torch.from_numpy(text_np).to(device=device, dtype=torch.float32)

        # Camera name remap: env renders real cameras, model was trained with
        # canonical names (e.g. view_a/view_b). Rename only the per-camera dicts;
        # env-side structures (shape_meta, obs_dict) still use real camera names.
        if self.camera_rename:
            for group_key in ('image', 'depth', 'camera_intrinsics', 'camera_extrinsics'):
                if group_key in obs and isinstance(obs[group_key], dict):
                    obs[group_key] = {
                        self.camera_rename.get(cn, cn): v
                        for cn, v in obs[group_key].items()
                    }

        return {'observation': obs}

    def _predict_action_vla(self, policy, np_obs_dict, device, dtype):
        batch = self._prepare_vla_batch(np_obs_dict, device)
        observation = batch['observation']

        # Handle obs_hist_length mismatch: model may expect T>1 but eval
        # provides T=1. Repeat the current frame to fill the history buffer.
        expected_T = getattr(getattr(policy.pipeline, 'action_head', None), 'obs_hist_length',
                            getattr(policy.pipeline, 'obs_hist_length', self.n_obs_steps))
        if expected_T > 1:
            for key in ("image", "depth", "camera_intrinsics"):
                sub = observation.get(key)
                if sub is None or not isinstance(sub, dict):
                    continue
                for cam in list(sub.keys()):
                    t = sub[cam]
                    if isinstance(t, torch.Tensor) and t.ndim >= 2 and t.shape[1] < expected_T:
                        sub[cam] = t.repeat(1, expected_T, *([1] * (t.ndim - 2)))
            # Also repeat state tensor to match expected_T
            state_t = observation.get("state")
            if state_t is not None and isinstance(state_t, torch.Tensor):
                if state_t.ndim == 2:  # [B, state_dim] → [B, T, state_dim]
                    state_t = state_t.unsqueeze(1).repeat(1, expected_T, 1)
                    observation["state"] = state_t
                elif state_t.ndim == 3 and state_t.shape[1] < expected_T:
                    observation["state"] = state_t.repeat(1, expected_T, 1)
        if self.normalize_actions and "state" in observation and hasattr(policy.pipeline, "normalizer") and policy.pipeline.normalizer is not None:
            norm = policy.pipeline.normalizer
            if hasattr(norm, "params_dict") and "state" in norm.params_dict:
                observation["state"] = norm.normalize({"state": observation["state"]})["state"]

        outputs = policy.pipeline.infer(
            observation["image"],
            observation.get("depth", None),
            camera_intrinsics=observation.get("camera_intrinsics", None),
            state=observation.get("state", None),
            camera_names=list(observation["image"].keys()),
            camera_extrinsics=observation.get("camera_extrinsics", None),
        )
        if not isinstance(outputs, dict):
            raise RuntimeError("VLA pipeline forward_test did not return a dict.")

        if 'actions' in outputs:
            actions = outputs['actions']
        elif 'action_sequence' in outputs:
            actions = outputs['action_sequence']
        else:
            available = list(outputs.keys())
            msg = (
                "VLA pipeline output missing 'actions' key for action prediction. "
                f"Available keys: {available}. "
                "This checkpoint may have been trained without an action head. "
                "Please evaluate with a checkpoint that includes action weights "
                "(see --use-action-head) or train the model with action supervision."
            )
            raise RuntimeError(msg)

        if isinstance(actions, tuple):
            actions = actions[0]
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions).to(device=device, dtype=dtype)
        elif isinstance(actions, torch.Tensor):
            actions = actions.to(device=device, dtype=dtype)
        else:
            raise TypeError(f"Unsupported action tensor type {type(actions)} from VLA pipeline.")

        if actions.dim() == 4:
            # Assume (B, num_samples, T, D) -> take first sample
            actions = actions[:, 0]
        if actions.dim() == 2:
            actions = actions.unsqueeze(1)

        if self.normalize_actions:
            actions = policy.pipeline.normalizer.unnormalize({"action": actions})["action"]

        if self.relative_action:
            reference_obs = self.embodiment_transform.inputs({"observation": {
                "robot0_eef_pos": np_obs_dict['robot0_eef_pos'][:, -1:],
                "robot0_eef_quat_site": np_obs_dict['robot0_eef_quat_site'][:, -1:],
            }})["observation"]
            actions = self.embodiment_transform.outputs(actions, context={"observation": reference_obs})

        seq_len = actions.shape[1]
        if seq_len < self.n_action_steps:
            pad = actions[:, -1:].repeat(1, self.n_action_steps - seq_len, 1)
            actions = torch.cat([actions, pad], dim=1)
        elif seq_len > self.n_action_steps:
            actions = actions[:, :self.n_action_steps]

        if isinstance(actions, torch.Tensor):
            if not torch.isfinite(actions).all():
                n_bad = int((~torch.isfinite(actions)).sum().item())
                print(f"[eval] Replacing {n_bad} non-finite action value(s) with zeros.", flush=True)
                actions = torch.nan_to_num(actions, nan=0.0, posinf=0.0, neginf=0.0)

        return {'action': actions}

    def run(self, policy):
        device = policy.device
        dtype = policy.dtype
        env = self.env
        vla_policy = self._is_vla_policy(policy)

        # plan for rollout
        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)

        # allocate data
        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits
        all_cameras = [None] * n_inits  # track selected camera per episode
        if self.save_rollout_states_path is not None:
            all_trajs = []

        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_global_slice = slice(start, end)
            this_n_active_envs = end - start
            this_local_slice = slice(0,this_n_active_envs)
            
            this_init_fns = self.env_init_fn_dills[this_global_slice]
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns.extend([self.env_init_fn_dills[0]]*n_diff)
            assert len(this_init_fns) == n_envs

            # init envs
            env.call_each('run_dill_function', 
                args_list=[(x,) for x in this_init_fns])

            # start rollout
            obs = env.reset()
            if self.determinism_debug:
                state_hashes = [
                    self._env_state_digest(env.envs[i])
                    for i in range(this_n_active_envs)
                ]
                print(
                    f"[determinism_debug] chunk={chunk_idx} reset_state_hashes={state_hashes}",
                    flush=True,
                )
            past_action = None
            if hasattr(policy, "reset") and callable(policy.reset):
                policy.reset()

            env_name = self.env_meta['env_name']
            pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval {env_name}Image {chunk_idx+1}/{n_chunks}", 
                leave=False, mininterval=self.tqdm_interval_sec)
            
            trajs = [[] for _ in range(n_envs)]
            done = False
            first_action_logged = False
            while not done:
                # create obs dict
                np_obs_dict = dict(obs)
                if self.past_action and (past_action is not None):
                    # TODO: not tested
                    np_obs_dict['past_action'] = past_action[
                        :,-(self.n_obs_steps-1):].astype(np.float32)
                
                if vla_policy:
                    with torch.no_grad():
                        action_dict = self._predict_action_vla(policy, np_obs_dict, device, dtype)
                else:
                    raise NotImplementedError("Only VLA policy is supported")
                    # device transfer
                    obs_dict = dict_apply(np_obs_dict, 
                        lambda x: torch.from_numpy(x).to(
                            device=device))

                    # run policy
                    with torch.no_grad():
                        action_dict = policy.predict_action(obs_dict)

                # device_transfer
                np_action_dict = dict_apply(action_dict,
                    lambda x: x.detach().to('cpu').numpy())

                action = np_action_dict['action']
                if self.determinism_debug and not first_action_logged:
                    first_action = action[:this_n_active_envs]
                    print(
                        f"[determinism_debug] chunk={chunk_idx} first_action_hash="
                        f"{self._array_digest(first_action)} shape={tuple(first_action.shape)}",
                        flush=True,
                    )
                    first_action_logged = True
                if not np.all(np.isfinite(action)):
                    n_bad = int((~np.isfinite(action)).sum())
                    print(f"[eval] Replacing {n_bad} non-finite env action value(s) with zeros.", flush=True)
                    action = np.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0)
                
                # step env
                env_action = action
                if not self.relative_action:
                    env_action = self.embodiment_transform.outputs(action)

                obs, reward, done, info = env.step(env_action)
                if self.save_rollout_states_path is not None:
                    for env_idx in range(n_envs):
                        trajs[env_idx].append(info[env_idx])
                done = np.all(done)
                past_action = action

                # update pbar
                pbar.update(action.shape[1])
            pbar.close()
            if self.save_rollout_states_path is not None:
                all_trajs.extend(trajs)

            # collect data for this round
            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            all_rewards[this_global_slice] = env.call('get_attr', 'reward')[this_local_slice]

            # Track selected camera per episode (for multi-view eval analysis).
            if self.multi_view_config is not None:
                try:
                    # SyncVectorEnv → MultiStepWrapper → VideoRecordingWrapper →
                    # RobomimicImageWrapper → RandomViewAliasWrapper
                    for env_idx in range(this_n_active_envs):
                        inner = env.envs[env_idx]  # MultiStepWrapper
                        wrapper = inner.env.env.env  # RandomViewAliasWrapper
                        if hasattr(wrapper, '_selected_camera'):
                            all_cameras[start + env_idx] = wrapper._selected_camera
                except Exception:
                    pass
        # clear out video buffer
        _ = env.reset()
        
        if self.save_rollout_states_path is not None:
            # create h5 file
            with h5py.File(self.save_rollout_states_path, 'w') as f:
                data_grp = f.create_group("data")
                for idx, traj in enumerate(all_trajs):
                    actions = []
                    rewards = []
                    states = []
                    dones = []
                    model = traj[0]['model']
                    for info in traj:
                        actions.append(info['actions'])
                        rewards.append(info['rewards'])
                        states.append(info['states'])
                        dones.append(info['dones'])
                    rewards = np.concatenate(rewards, axis=0)
                    # locate the end of the episode
                    if rewards.max() > 0.5:
                        # successful
                        end_step = np.argmax(rewards) + 1
                    else:
                        end_step = rewards.shape[0]
                    actions = np.concatenate(actions, axis=0)[:end_step]
                    states = np.concatenate(states, axis=0)[:end_step]
                    dones = np.concatenate(dones, axis=0)[:end_step]
                    rewards = rewards[:end_step]
                    ep_data_grp = data_grp.create_group(f"demo_{idx}")
                    ep_data_grp.create_dataset("actions", data=actions)
                    ep_data_grp.create_dataset("states", data=states)
                    ep_data_grp.create_dataset("rewards", data=rewards)
                    ep_data_grp.create_dataset("dones", data=dones)
                    ep_data_grp.attrs["num_samples"] = actions.shape[0] # number of transitions in this episode
                    ep_data_grp.attrs["model_file"] = model
                data_grp.attrs["env_args"] = json.dumps(self.env_meta)

        # log
        max_rewards = collections.defaultdict(list)
        log_data = dict()
        # results reported in the paper are generated using the commented out line below
        # which will only report and average metrics from first n_envs initial condition and seeds
        # fortunately this won't invalidate our conclusion since
        # 1. This bug only affects the variance of metrics, not their mean
        # 2. All baseline methods are evaluated using the same code
        # to completely reproduce reported numbers, uncomment this line:
        # for i in range(len(self.env_fns)):
        # and comment out this line
        for i in range(n_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            max_reward = np.max(all_rewards[i])
            max_rewards[prefix].append(max_reward)
            log_data[prefix+f'sim_max_reward_{seed}'] = max_reward

            # visualize sim
            video_path = all_video_paths[i]
            if video_path is not None:
                sim_video = video_path # wandb.Video(video_path)
                log_data[prefix+f'sim_video_{seed}'] = sim_video
        
        # log aggregate metrics
        for prefix, value in max_rewards.items():
            name = prefix+'mean_score'
            value = np.mean(value)
            log_data[name] = value

        # Per-camera success rate analysis (multi-view eval).
        if any(c is not None for c in all_cameras):
            camera_results = collections.defaultdict(list)
            for i in range(n_inits):
                cam = all_cameras[i]
                if cam is not None:
                    max_reward = np.max(all_rewards[i])
                    camera_results[cam].append(max_reward)
            print("\n" + "=" * 50)
            print("Per-camera success rates:")
            for cam, rewards in sorted(camera_results.items()):
                sr = np.mean(rewards)
                n = len(rewards)
                print(f"  {cam}: {sr:.1%} ({int(sr*n)}/{n})")
                log_data[f'test/camera_{cam}_score'] = sr
                log_data[f'test/camera_{cam}_count'] = n
            print("=" * 50)

        return log_data

    # ============================================================
    #             POLICY-GUIDED PLAYBACK FUNCTION
    # ============================================================

    def playback_with_policy(
        self,
        policy,
        video_path=None,
        first=False,
        n_episodes=None,
    ):
        """
        Replay dataset trajectories but actions come from the policy.
        Uses dataset observations, runs policy, and renders them.

        Args:
            policy: VLA policy wrapper
            video_path: path to save mp4, or None for on-screen render
            first: only visualize first frame of each episode
        """

        import imageio

        print(f"[PLAYBACK] Loading dataset: {self.dataset_path}")
        f = h5py.File(self.dataset_path, "r")
        demos = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[-1]))[:n_episodes]

        # Initialize env (we will load dataset states manually)
        env_meta = self.env_meta
        env = create_env(
            env_meta=env_meta,
            shape_meta={"obs": self.shape_meta["obs"], "action": self.shape_meta["action"]},
            camera_depths=False,
            camera_segmentations=False,
            enable_render=(video_path is not None),
            camera_height=self.camera_height,
            camera_width=self.camera_width,
            coupled_third_config=self.coupled_third_config,
        )
        # Provide intrinsics when dataset doesn't store them.
        self.camera_info = get_camera_info(
            env,
            camera_names=self.camera_names,
            camera_height=self.camera_height,
            camera_width=self.camera_width,
        )

        # Prepare video writer
        writer = None
        if video_path is not None:
            print(f"[PLAYBACK] Writing video to {video_path}")
            writer = imageio.get_writer(video_path, fps=20)

        for ep in demos:
            print(f"[PLAYBACK] Episode: {ep}")
            grp = f[f"data/{ep}"]
            states = grp["states"][:]             # sim states
            obs_grp = grp["obs"]
            actions = grp["actions"][:]

            # reset env to start state
            env.reset_to({"states": states[0]})

            t = 0
            while True:
                if t >= states.shape[0]:
                    break

                # Extract dataset observation dict → runner format
                np_obs_dict = {k: obs_grp[k][t][None, None, ...] for k in obs_grp.keys()}

                # --- Feed observation to policy ---
                with torch.no_grad():
                    action_dict = self._predict_action_vla(policy, np_obs_dict,
                                                           device=policy.device,
                                                           dtype=policy.dtype)
                # device_transfer
                np_action_dict = dict_apply(action_dict,
                    lambda x: x.detach().to('cpu').numpy())

                action = np_action_dict['action'][0] # [n_action_steps, action_dim]
                if not np.all(np.isfinite(action)):
                    raise RuntimeError("Nan or Inf action")

                # step env
                env_action = action
                if not self.relative_action:
                    env_action = self.embodiment_transform.outputs(action)

                # Step the environment using predicted action
                for action_step in env_action:
                    env.step(action_step)
                    t += 1

                    # Render
                    if writer is not None:
                        frame = env.render(
                            mode="rgb_array",
                            height=512,
                            width=512,
                            camera_name=self.camera_names[0],
                        )
                        writer.append_data(frame)

                    if first:
                        break

                if first:
                    break

        if writer is not None:
            writer.close()

        print("[PLAYBACK] Done.")

    def undo_transform_action(self, action):
        raw_shape = action.shape
        if raw_shape[-1] == 20:
            # dual arm
            action = action.reshape(-1,2,10)

        d_rot = action.shape[-1] - 4
        pos = action[...,:3]
        rot = action[...,3:3+d_rot]
        gripper = action[...,[-1]]

        # Convert rotation representation back to axis-angle for robosuite.
        # Use the codebase's own rotation_6d_to_matrix (column-vector convention)
        # which matches the training data encoding, NOT pytorch3d (row-vector).
        if self.action_orn_mode == '6d':
            rot_shape = rot.shape
            rot_tensor = torch.from_numpy(rot).float().reshape(-1, 6)
            rot_mat = rotation_6d_to_matrix(rot_tensor)
            rot_aa = matrix_to_axis_angle(rot_mat)
            rot = rot_aa.numpy().reshape(rot_shape[:-1] + (3,))
        else:
            raise ValueError(f"Unsupported action_orn_mode for undo_transform: {self.action_orn_mode}")

        # CPGen datasets store physical-unit deltas during training (controller
        # inputs * output_max).  Undo the scaling to get back to controller inputs
        # in [-1, 1] for delta-mode environments.
        if self._cpgen_action_pos_scale is not None:
            pos = pos / self._cpgen_action_pos_scale
        if self._cpgen_action_rot_scale is not None:
            rot = rot / self._cpgen_action_rot_scale
        uaction = np.concatenate([
            pos, rot, gripper
        ], axis=-1)

        if raw_shape[-1] == 20:
            # dual arm
            uaction = uaction.reshape(*raw_shape[:-1], 14)

        return uaction

def _parse_cli_args():
    parser = argparse.ArgumentParser(description="Evaluate a SAPolicy VLA checkpoint inside robomimic environment.")
    parser.add_argument("--ckpt", required=True, help="Path to the .ckpt checkpoint or directory containing checkpoints (will auto-select latest).")
    parser.add_argument("--dataset", required=True, help="Path to robomimic-format HDF5 dataset.")
    parser.add_argument(
        "--shape-meta",
        default=None,
        help="Optional path to JSON/YAML describing observation shapes. Overrides automatic dataset inference.",
    )
    parser.add_argument("--action-stats", default=None, help="Optional path to action stats file.")
    parser.add_argument(
        "--transforms",
        default=None,
        help="Optional path to JSON/YAML describing dataset-style observation transforms applied before policy input.",
    )
    parser.add_argument("--output-dir", default="eval_outputs", help="Directory for evaluation artifacts.")
    parser.add_argument("--encoder", default="vits", help="SAPolicy encoder size (vits|vitb|vitl|vitg).")
    parser.add_argument("--backbone-type", default="dinov2", choices=["dinov2", "dinov3", "da3", "vggt"], help="Backbone type.")
    parser.add_argument("--pretrain-backbone", default="", help="Optional pretrained backbone weights path.")
    parser.add_argument("--use-action-head", action="store_true", help="Enable action head when instantiating the policy.")
    parser.add_argument("--n-train", type=int, default=0, help="Number of train trajectories to evaluate.")
    parser.add_argument("--n-train-vis", type=int, default=0, help="Number of train videos to record.")
    parser.add_argument("--n-test", type=int, default=10, help="Number of test trajectories to evaluate.")
    parser.add_argument("--n-test-vis", type=int, default=10, help="Number of test videos to record.")
    parser.add_argument("--n-envs", type=int, default=None, help="Number of vectorized environments to launch.")
    parser.add_argument("--n-obs-steps", type=int, default=1, help="Number of observation history steps.")
    parser.add_argument("--n-action-steps", type=int, default=8, help="Number of action steps per policy call.")
    parser.add_argument("--max-steps", type=int, default=450, help="Maximum episode horizon (Square/Coffee default; Kitchen is fixed to 1200).")
    parser.add_argument("--render-obs-key", default="agentview_image", help="Observation key for rendering/videos.")
    parser.add_argument("--fps", type=int, default=10, help="Recorded video FPS.")
    parser.add_argument("--device", default="cuda", help="Device for policy inference (cuda|cpu).")
    parser.add_argument("--save-rollout-states", default=None, help="Optional path to store rollout states (HDF5).")
    parser.add_argument("--camera-height", type=int, default=256, help="Camera height.")
    parser.add_argument("--camera-width", type=int, default=256, help="Camera width.")
    parser.add_argument("--use-camera-intrinsics", action="store_true", help="Use camera intrinsics.")
    parser.add_argument("--use-state", action="store_true", help="Use state.")
    parser.add_argument("--use-latent-aux-model", action="store_true", help="Use LatentAuxiliaryModel.")
    parser.add_argument("--state-dim", type=int, default=7, help="State dimension.")
    parser.add_argument("--action-orn-mode", type=str, default="6d", help="Action orientation mode.")
    parser.add_argument("--relative-action", action="store_true", help="Use relative action.")
    parser.add_argument("--abs-action", action="store_true", help="Use absolute action.")
    parser.add_argument("--normalize-actions", action="store_true", help="Normalize actions.")
    parser.add_argument("--action-sequence-length", type=int, default=16, help="Action sequence length.")
    parser.add_argument("--camera-names", nargs="+", default=['agentview','robot0_eye_in_hand'], help="List of camera names (space separated).")
    parser.add_argument("--playback-policy", action="store_true", help="Run policy-guided playback on dataset episodes.")
    parser.add_argument("--playback-use-obs", action="store_true", help="Use dataset image observations instead of sim.")
    parser.add_argument("--playback-video", type=str, default=None, help="Optional path to save playback video.")
    parser.add_argument("--playback-first", action="store_true", help="Only visualize the first frame of each episode.")

    return parser.parse_args()


def main():
    args = _parse_cli_args()

    # Handle ckpt path - if directory, find latest checkpoint
    ckpt_path = _find_latest_checkpoint(args.ckpt)

    torch_device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    manual_shape_meta = None
    if args.shape_meta is not None:
        meta_path = os.path.expanduser(args.shape_meta)
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(f"shape meta file {meta_path} not found")
        if meta_path.endswith(('.yaml', '.yml')):
            manual_shape_meta = OmegaConf.to_container(OmegaConf.load(meta_path), resolve=True)
        else:
            import json
            with open(meta_path, "r") as f:
                manual_shape_meta = json.load(f)

    shape_meta = _build_shape_meta(args.dataset, manual_shape_meta, camera_height=args.camera_height, camera_width=args.camera_width)
    print("shape_meta: ", shape_meta)
    observation_transforms = _build_observation_transforms(args.transforms)
    runner = RobomimicImageRunner(
        output_dir=args.output_dir,
        dataset_path=args.dataset,
        shape_meta=shape_meta,
        n_train=args.n_train,
        n_train_vis=args.n_train_vis,
        n_test=args.n_test,
        n_test_vis=args.n_test_vis,
        n_envs=args.n_envs,
        n_obs_steps=args.n_obs_steps,
        n_action_steps=args.n_action_steps,
        max_steps=args.max_steps,
        render_obs_key=args.render_obs_key,
        fps=args.fps,
        save_rollout_states_path=args.save_rollout_states,
        observation_transforms=observation_transforms,
        camera_height=args.camera_height,
        camera_width=args.camera_width,
        action_orn_mode=args.action_orn_mode,
        abs_action=args.abs_action,
        relative_action=args.relative_action,
        normalize_actions=args.normalize_actions,
        action_sequence_length=args.action_sequence_length,
        camera_names=args.camera_names,
    )

    pipeline = _instantiate_vla_pipeline(
        ckpt_path=ckpt_path,
        device=torch_device,
        encoder=args.encoder,
        backbone_type=args.backbone_type,
        load_pretrain_backbone=args.pretrain_backbone,
        use_action_head=args.use_action_head,
        action_length=args.action_sequence_length,
        obs_hist_length=args.n_obs_steps,
        use_camera_intrinsics=args.use_camera_intrinsics,
        use_state=args.use_state,
        state_dim=args.state_dim,
        action_orn_mode=args.action_orn_mode,
        num_cameras=len(args.camera_names),
        use_latent_aux_model=args.use_latent_aux_model,
    )
    if not getattr(pipeline, "use_action_head", False):
        raise RuntimeError(
            "The loaded checkpoint does not include an action head, so the policy cannot produce "
            "environment actions. Please provide a checkpoint trained with action supervision "
            "and run with --use-action-head, or re-export a model that includes action generation."
        )
    policy = _SAPolicyWrapper(pipeline, torch_device)

    # ----------------------------------------------------------
    # PLAYBACK WITH POLICY
    # ----------------------------------------------------------
    if args.playback_policy:
        runner.playback_with_policy(
            policy,
            video_path=args.playback_video,
            first=args.playback_first,
            n_episodes=args.n_test,
        )
        return

    # ----------------------------------------------------------
    # NORMAL EVALUATION
    # ----------------------------------------------------------
    metrics = runner.run(policy)
    print("Evaluation metrics:", metrics)


def eval_only(self, output_dir=None, **kwargs): 
    cfg = copy.deepcopy(self.cfg)
    # configure env
    eval_result_output_dir = self.get_evaluate_only_dir()
    os.makedirs(eval_result_output_dir, exist_ok=True)

    randomize_color = kwargs.get("randomize_color", False)
    randomize_camera = kwargs.get("randomize_camera", False)
    randomize_lighting = kwargs.get("randomize_lighting", False)
    randomize_dynamics = kwargs.get("randomize_dynamics", False)

    env_runner: BaseImageRunner
    env_runner = hydra.utils.instantiate(
        cfg.task.env_runner,
        output_dir=eval_result_output_dir,
        randomize_color=randomize_color,
        randomize_camera=randomize_camera,
        randomize_dynamics=randomize_dynamics,
        randomize_lighting=randomize_lighting
    )
    assert isinstance(env_runner, BaseImageRunner)
    
    wandb_run = wandb.init(
        project="Domain Randomized Evaluation",
        config={
            "name": cfg.name,
            "git_commit": subprocess.getoutput("git rev-parse HEAD"),
        },
        dir=str(eval_result_output_dir)
    )
    wandb.config.update({
        "randomize_color": randomize_color,
        "randomize_camera": randomize_camera,
        "randomize_dynamics": randomize_dynamics,
        "randomize_lighting": randomize_lighting,
    })

    patch_file = pathlib.Path("diff.patch")
    with patch_file.open("w") as f:
        subprocess.run(["git", "diff", "HEAD"], stdout=f, check=True)

    diff_artifact = wandb.Artifact(name="git_diff", type="diff")
    diff_artifact.add_file(str(patch_file))
    wandb.log_artifact(diff_artifact)

    ckpt_paths = os.listdir(os.path.join(output_dir, "checkpoints"))
    for ckpt_path in ckpt_paths: 
        if ckpt_path.endswith(".ckpt") and "latest" not in ckpt_path: 
            pass
        else: 
            print(f"Skipping {ckpt_path}")
            continue
        
        print("Evaluating checkpoint: ", ckpt_path)
        self.load_checkpoint(path=os.path.join(self.output_dir, "checkpoints", ckpt_path))
        self._output_dir = output_dir
        
        # device transfer
        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)

        policy = self.model
        if cfg.training.use_ema: 
            policy = self.ema_model
        policy.eval()

        runner_log = env_runner.run(policy)
        wandb_run.log(runner_log)
        # if cfg.evaluate_mode == "rand_start": 
        #     runner_log = env_runner.run(policy, fix_start=False)
        # elif cfg.evaluate_mode == "fix_start": 
        #     runner_log = env_runner.run(policy, fix_start=True)
        # else: 
        #     raise ValueError("Invalid evaluate_mode")

        print("-------------------")
        if "test/mean_score" in runner_log.keys():
            print("test/mean_score: ", runner_log["test/mean_score"])
        if "train/mean_score" in runner_log.keys():
            print("train/mean_score: ", runner_log["train/mean_score"])
        if "test/avg_step" in runner_log.keys():
            print(runner_log["test/avg_step"])
        print("-------------------")
        
        # write to txt file
        with open(os.path.join(eval_result_output_dir, "success_rate_eval.txt"), "a") as f: 
            if "test/mean_score" in runner_log.keys():
                f.write("{}: {}\n".format(ckpt_path, runner_log["test/mean_score"]))

        if "test/avg_step" in runner_log.keys(): 
            with open(os.path.join(eval_result_output_dir, "avg_step_eval.txt"), "a") as f: 
                f.write("{}: {}\n".format(ckpt_path, runner_log["test/avg_step"]))
    wandb_run.finish()
    env_runner.close()
    gc.collect()


if __name__ == "__main__":
    main()
