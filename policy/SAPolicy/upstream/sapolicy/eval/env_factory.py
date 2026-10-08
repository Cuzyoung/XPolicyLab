"""RoboSuite environment factory and eval observation-shape helpers."""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional, Union

import h5py
import hydra
import numpy as np
from omegaconf import DictConfig, ListConfig, OmegaConf
from torchvision.transforms import Compose

from sapolicy.eval.gl_bootstrap import ensure_eval_gl_env, import_robosuite

ensure_eval_gl_env()

cpgen_envs = None
_CPGEN_IMPORT_ERROR = None

robosuite = None
_ROBOSUITE_IMPORT_ERROR = None

_ROBOTIQ_ANGLE_TO_WIDTH_MM = {
    "robotiq140": (135.5, -196.0),
    "robotiq85": (92.3, -159.0),
}


def _robotiq_gripper_key(gripper_type: Optional[str]) -> Optional[str]:
    # "CalibratedRobotiq85Gripper" / "Robotiq85Gripper" -> "robotiq85".
    if not isinstance(gripper_type, str):
        return None
    return gripper_type.removeprefix("Calibrated").removesuffix("Gripper").lower()


def _adapt_gripper_qpos_for_state(qpos: np.ndarray, gripper_type: Optional[str] = None) -> np.ndarray:
    # Mirrors sapolicy/dataset/robomimic_hdf5.py:_adapt_gripper_qpos — keep in sync.
    # 2-DOF grippers (Panda/Rethink) vs 6-DOF Robotiq (IIWA/UR5e) report gripper_qpos
    # differently; training data is normalized through the same branches.
    if qpos.shape[-1] == 1:
        return qpos
    if qpos.shape[-1] == 2:
        return qpos[..., :1]
    if qpos.shape[-1] == 6:
        angle = -qpos[..., 0:1]
        gripper_key = _robotiq_gripper_key(gripper_type)
        try:
            offset, slope = _ROBOTIQ_ANGLE_TO_WIDTH_MM[gripper_key]
        except KeyError:
            raise ValueError(
                f"Unknown gripper_type {gripper_type!r} for 6-DOF (Robotiq) gripper qpos; "
                f"expected one of {sorted(_ROBOTIQ_ANGLE_TO_WIDTH_MM)}."
            )
        return offset + slope * angle
    raise ValueError(f"Undefined gripper qpos shape: {qpos.shape}")


_ROBOTIQ85_OSC_KP = 500.0


def _apply_robotiq85_osc_kp_default(env_meta: Dict[str, Any]) -> None:
    """Default the OSC arm controller's kp to 500 whenever the resolved gripper
    is Robotiq85 -- see localdocs/mimicgen_gripper_cross_embodiment.md section
    8.5/9.2. Mutates env_meta in place; must run before create_env()/robosuite.make()
    since kp is baked into the controller at construction. Only touches body_parts
    entries that already have a 'kp' key (the OSC arm controller), so a GRIP-type
    gripper body-part config is left untouched.
    """
    if _DISABLE_GRIPPER_TUNING:
        return
    gripper_types = env_meta.get('env_kwargs', {}).get('gripper_types')
    gripper_type = gripper_types[0] if isinstance(gripper_types, (list, tuple)) else gripper_types
    if _robotiq_gripper_key(gripper_type) != "robotiq85":
        return
    body_parts = env_meta.get('env_kwargs', {}).get('controller_configs', {}).get('body_parts', {})
    for bp_cfg in body_parts.values():
        if isinstance(bp_cfg, dict) and 'kp' in bp_cfg:
            bp_cfg['kp'] = _ROBOTIQ85_OSC_KP
    print(f"[gripper-defaults] Robotiq85 detected -> OSC kp={_ROBOTIQ85_OSC_KP}")


def _require_cpgen_envs() -> None:
    # For CPGen datasets, the env registry relies on cpgen_envs import side effects.
    global cpgen_envs, _CPGEN_IMPORT_ERROR
    if cpgen_envs is None:
        _require_robosuite()
        try:
            # CPGen registers robosuite envs / tasks via side effects on import.
            import cpgen_envs as _cpgen_envs  # noqa: F401
        except Exception as e:
            _CPGEN_IMPORT_ERROR = e
            raise ImportError(
                "cpgen_envs failed to import. This is required to create CPGen environments for evaluation. "
                "Common causes: missing mimicgen / robosuite_task_zoo, or a misconfigured MuJoCo GL backend "
                "(try setting MUJOCO_GL=egl or MUJOCO_GL=osmesa explicitly)."
            ) from e
        cpgen_envs = _cpgen_envs
        _CPGEN_IMPORT_ERROR = None


def _require_robosuite():
    global robosuite, _ROBOSUITE_IMPORT_ERROR
    if robosuite is not None:
        return robosuite
    try:
        robosuite = import_robosuite()
        _ROBOSUITE_IMPORT_ERROR = None
        return robosuite
    except Exception as e:
        robosuite = None
        _ROBOSUITE_IMPORT_ERROR = e
        raise ImportError(
            "robosuite could not be imported. It is required to create environments for evaluation."
        ) from e


def _normalize_controller_configs(controller_configs: Dict[str, Any], robots) -> Dict[str, Any]:
    """
    Convert a robosuite controller_configs dict to the composite ("body_parts")
    format robosuite's composite_controller_factory requires. Some
    cross-embodiment shards (sawyer/iiwa/ur5e) still bake the pre-robosuite-1.5
    flat part-controller format (bare "type": "OSC_POSE", no body_parts) or the
    older "body_parts_controller_configs" key name; composite_controller_factory
    does not auto-convert, so do it here. All current cross-embodiment robots
    are single right-arm robots (matches panda's own already-composite baked
    config). Must run before any code reads controller_configs["body_parts"]
    (e.g. the SAPOLICY_EVAL_CONTROLLER_OVERRIDE_KEYS merge below), not just
    before env creation -- a flat config silently has zero body_parts entries
    to merge into.
    """
    if controller_configs.get("body_parts_controller_configs") is not None:
        controller_configs['body_parts'] = controller_configs['body_parts_controller_configs']
    elif "body_parts" not in controller_configs:
        from robosuite.controllers.composite.composite_controller_factory import (
            is_part_controller_config,
            refactor_composite_controller_config,
        )
        if is_part_controller_config(controller_configs):
            robot_type = robots[0] if isinstance(robots, list) else robots
            controller_configs = refactor_composite_controller_config(
                controller_configs, robot_type, ["right"]
            )
    return controller_configs


def _load_env_meta_from_dataset(dataset_path: str) -> Dict[str, Any]:
    """
    Load robosuite / cpgen env metadata from a robomimic-style HDF5 dataset.

    We intentionally avoid importing robomimic.utils.file_utils here, since that
    pulls in robomimic algo modules (and transformers / tensorflow) which can
    conflict with OSMesa / llvmpipe on some machines.
    """
    dataset_path = os.path.expanduser(dataset_path)
    with h5py.File(dataset_path, "r") as f:
        env_args = f["data"].attrs.get("env_args", None)
        if env_args is None:
            raise KeyError(f"Dataset {dataset_path} missing data.attrs['env_args']")
        if isinstance(env_args, bytes):
            env_args = env_args.decode("utf-8")
        if not isinstance(env_args, str):
            raise TypeError(f"Expected env_args to be a JSON string, got {type(env_args)}")
        env_meta = json.loads(env_args)
        # Some datasets may include language fields - ignore for CPGen eval.
        if "env_lang" in env_meta.get("env_kwargs", {}):
            del env_meta["env_kwargs"]["env_lang"]
        return env_meta


# Robotiq85/Robotiq140 (native on UR5e/IIWA, or reachable on any robot via
# gripper_types_override) ship with collision geoms (models/assets/grippers/
# robotiq_gripper_{85,140}.xml) that leave friction and condim at MuJoCo's bare
# defaults -- unlike panda_gripper.xml, which hand-tunes its own finger pads
# (friction "2 0.05 0.0001", condim 4). Robotiq85 additionally gets its finger
# actuators' position-gain bumped -- see localdocs/mimicgen_gripper_cross_embodiment.md
# section 8.5/9.2 for how these values (and the OSC arm kp default in
# _apply_robotiq85_osc_kp_default below) were tuned.
# friction/torsion scale synced to localdocs/mimicgen_gripper_cross_embodiment.md
# section 9 (MG_GRIP_FRICTION_SCALE=2.0 / MG_GRIP_TORSION_SCALE=4.0, the values
# mimicgen's own sweep validated) -- this file's originals (3.0/100.0, commit
# f3de7bf) predate that sweep and were never synced. mimicgen also scales the
# rolling friction component by the same torsion factor (geom_friction[...,2]
# *= _ts alongside [...,1]); this file previously left rolling unscaled.
_ROBOTIQ_GRIPPER_KEYS = {"robotiq85", "robotiq140"}
_GRIPPER_FRICTION_SCALE = 2.0
_GRIPPER_TORSION_SCALE = 4.0
_GRIPPER_CONDIM = 6
_MUJOCO_DEFAULT_GEOM_FRICTION = (1.0, 0.005, 0.0001)  # (sliding, torsional, rolling)
_ROBOTIQ85_GRIP_KP = 45.0

# Diagnostic-only escape hatch: set SAPOLICY_EVAL_DISABLE_GRIPPER_TUNING=1 to skip
# the OSC-kp default and the friction/torsion/grip-kp overrides below entirely,
# leaving whatever gripper is instantiated (native or gripper_types_override) at
# robosuite's own untouched physics. Used to isolate "does the open-cap gripper
# class itself help/hurt" from "does the physics tuning help/hurt", independent
# of each other -- see calib-eval session notes.
_DISABLE_GRIPPER_TUNING = os.environ.get("SAPOLICY_EVAL_DISABLE_GRIPPER_TUNING") == "1"


def _apply_gripper_physics_overrides(base_env):
    """Bump Robotiq gripper pad friction + condim for grip stability, and (Robotiq85
    only) its finger actuator gain.

    Keyed off the actual gripper object's class (via _robotiq_gripper_key), not
    robot.name -- gripper_types_override lets any robot carry a non-native
    gripper, and the tuning is a property of the gripper, not the arm it's
    mounted on. gripper.important_geoms / gripper.actuators are used rather
    than hardcoded XML strings, so this tracks whatever geoms/actuators the
    installed robosuite version actually defines.
    """
    if _DISABLE_GRIPPER_TUNING:
        return
    sim = getattr(base_env, "sim", None)
    robots = getattr(base_env, "robots", None)
    if sim is None or not robots:
        return
    sliding = _MUJOCO_DEFAULT_GEOM_FRICTION[0] * _GRIPPER_FRICTION_SCALE
    torsional = _MUJOCO_DEFAULT_GEOM_FRICTION[1] * _GRIPPER_TORSION_SCALE
    rolling = _MUJOCO_DEFAULT_GEOM_FRICTION[2] * _GRIPPER_TORSION_SCALE
    for robot in robots:
        for gripper in getattr(robot, "gripper", {}).values():
            gripper_key = _robotiq_gripper_key(type(gripper).__name__)
            if gripper_key not in _ROBOTIQ_GRIPPER_KEYS:
                continue
            for geom_names in gripper.important_geoms.values():
                for geom_name in geom_names:
                    gid = sim.model.geom_name2id(geom_name)
                    sim.model.geom_friction[gid] = [sliding, torsional, rolling]
                    sim.model.geom_condim[gid] = _GRIPPER_CONDIM
            print(
                f"[gripper-physics] {robot.name}/{gripper_key}: friction=({sliding}, {torsional}, "
                f"{rolling}) condim={_GRIPPER_CONDIM} on {sum(len(g) for g in gripper.important_geoms.values())} geoms"
            )
            if gripper_key == "robotiq85":
                for actuator_name in gripper.actuators:
                    aid = sim.model.actuator_name2id(actuator_name)
                    sim.model.actuator_gainprm[aid, 0] = _ROBOTIQ85_GRIP_KP
                    sim.model.actuator_biasprm[aid, 1] = -_ROBOTIQ85_GRIP_KP
                print(
                    f"[gripper-physics] {robot.name}/{gripper_key}: grip_kp={_ROBOTIQ85_GRIP_KP} "
                    f"on {list(gripper.actuators)}"
                )


class _RobosuiteEnvAdapter:
    """
    Minimal adapter to match the subset of the robomimic EnvRobosuite API used
    by our evaluation wrappers.
    """

    def __init__(self, env, camera_fovy_overrides=None):
        self.env = env
        # Match robomimic's EnvRobosuite API shape enough for downstream helpers.
        # Some utilities expect `.base_env` to exist.
        self.base_env = env
        self._last_obs = None
        self._camera_fovy_overrides = camera_fovy_overrides

    def _apply_fovy_overrides(self):
        """Re-apply camera fovy overrides after reset (robosuite hard_reset
        rebuilds the MuJoCo model, losing runtime cam_fovy changes)."""
        if not self._camera_fovy_overrides:
            return
        sim = self.env.sim
        for cam_name, fovy_val in self._camera_fovy_overrides.items():
            try:
                cam_id = sim.model.camera_name2id(cam_name)
                sim.model.cam_fovy[cam_id] = float(fovy_val)
            except Exception:
                pass

    def _make_render_context_current(self):
        """Rebind this env's EGL context before robosuite renders observations."""
        sim = getattr(self.env, "sim", None)
        render_context = getattr(sim, "_render_context_offscreen", None)
        gl_context = getattr(render_context, "gl_ctx", None)
        if gl_context is not None:
            gl_context.make_current()

    def reset(self):
        self._make_render_context_current()
        obs = self.env.reset()
        self._apply_fovy_overrides()
        _apply_gripper_physics_overrides(self.env)
        self._last_obs = obs
        return obs

    def step(self, action):
        self._make_render_context_current()
        obs, reward, _done, info = self.env.step(action)
        self._last_obs = obs
        return obs, reward, self.is_done(), info

    def is_done(self):
        """Match robomimic EnvRobosuite fixed-horizon rollout semantics."""
        return False

    def is_success(self):
        """Match robomimic's EnvRobosuite task-success API."""
        success = self.env._check_success()
        if isinstance(success, dict):
            assert "task" in success
            return success
        return {"task": success}

    def reset_to(self, state_dict):
        # robomimic-style API: reset_to({"states": <state>}) -> obs
        self._make_render_context_current()
        if hasattr(self.env, "reset_to"):
            obs = self.env.reset_to(state_dict)
            self._last_obs = obs
            return obs
        if not (isinstance(state_dict, dict) and "states" in state_dict):
            raise TypeError("reset_to expects a dict with key 'states'")
        if not hasattr(self.env, "sim") or not hasattr(self.env.sim, "set_state"):
            raise AttributeError("Underlying env does not expose sim.set_state; cannot reset_to.")
        state_val = state_dict["states"]
        # robomimic / CPGen datasets store states as flattened numpy arrays.
        # robosuite's sim expects either a MuJoCo state object OR uses
        # set_state_from_flattened for numpy arrays.
        if isinstance(state_val, np.ndarray) and hasattr(self.env.sim, "set_state_from_flattened"):
            self.env.sim.set_state_from_flattened(state_val)
        else:
            self.env.sim.set_state(state_val)
        # Ensure derived quantities are consistent.
        if hasattr(self.env.sim, "forward"):
            self.env.sim.forward()
        # Force robosuite to re-render cameras. Without this,
        # _get_observations() returns stale cached images from the
        # previous render (only env.step() normally triggers re-render).
        self._make_render_context_current()
        if hasattr(self.env, "_update_observables"):
            self.env._update_observables(force=True)
        # robosuite uses _get_observations for raw obs dict.
        if hasattr(self.env, "_get_observations"):
            obs = self.env._get_observations()
        else:
            obs = self.env.reset()
        self._last_obs = obs
        return obs

    def get_state(self):
        if hasattr(self.env, "get_state"):
            state = self.env.get_state()
            # Some downstream code expects a "model" field for logging.
            if isinstance(state, dict) and "model" not in state:
                state = dict(state)
                state["model"] = None
            return state
        if not hasattr(self.env, "sim") or not hasattr(self.env.sim, "get_state"):
            raise AttributeError("Underlying env does not expose sim.get_state; cannot get_state.")
        st = self.env.sim.get_state()
        # Prefer flattened numpy for compatibility with robomimic wrappers.
        if hasattr(st, "flatten"):
            try:
                st = st.flatten()
            except Exception:
                pass
        return {"states": st, "model": None}

    def get_observation(self):
        if self._last_obs is None:
            # Fallback for environments that expose _get_observations.
            if hasattr(self.env, "_get_observations"):
                self._last_obs = self.env._get_observations()
            else:
                raise RuntimeError("No cached observation available; call reset() first.")
        return self._last_obs

    def seed(self, seed=None):
        # robosuite uses numpy global RNG for many resets.
        np.random.seed(seed=seed)
        if hasattr(self.env, "seed"):
            return self.env.seed(seed)
        return None

    def render(self, *args, **kwargs):
        self._make_render_context_current()
        return self.env.render(*args, **kwargs)

    def __getattr__(self, name):
        # Pass through robosuite camera helpers, sim, etc.
        return getattr(self.env, name)

def create_env(
    env_meta,
    shape_meta,
    camera_depths=False,
    camera_segmentations=None,
    enable_render=True,
    camera_height=256,
    camera_width=256,
    camera_fovy_overrides: Dict[str, float] | None = None,
    multi_view_config: dict | None = None,
    coupled_third_config: dict | None = None,
):
    _require_robosuite()
    env_meta['env_kwargs']['controller_configs'] = _normalize_controller_configs(
        env_meta['env_kwargs']['controller_configs'], env_meta['env_kwargs']['robots']
    )
    env_meta['camera_depths'] = camera_depths
    env_meta['env_kwargs']['camera_segmentations'] = camera_segmentations
    env_meta['env_kwargs']['renderer'] = "mujoco"
    env_meta['env_kwargs']['camera_heights'] = camera_height
    env_meta['env_kwargs']['camera_widths'] = camera_width

    print("create env with env_meta: ", env_meta)

    env_name = env_meta["env_name"]
    env_kwargs = dict(env_meta.get("env_kwargs", {}))
    # Match robomimic EnvRobosuite constructor overrides.
    env_kwargs["ignore_done"] = True
    env_kwargs["use_object_obs"] = True
    env_kwargs["use_camera_obs"] = bool(enable_render)
    # GUI rendering requires has_renderer=True (robosuite viewer). For image
    # observations (rgb/depth), robosuite still needs an offscreen context.
    # It is safe to enable both at the same time.
    env_kwargs["has_renderer"] = False
    env_kwargs["has_offscreen_renderer"] = bool(enable_render)
    # Depth should be available when requested.
    env_kwargs["camera_depths"] = bool(camera_depths)

    # Inject extra cameras for multi-view eval BEFORE robosuite.make() so
    # robosuite initialises them properly and includes them in observations.
    if multi_view_config is not None:
        extra_cams = multi_view_config.get("extra_cameras",
            ["frontview", "sideview", "birdview", "robot0_robotview"])
        existing_cams = list(env_kwargs.get("camera_names",
            ["agentview", "robot0_eye_in_hand"]))
        for cam in extra_cams:
            if cam and cam not in existing_cams:
                existing_cams.append(cam)
        env_kwargs["camera_names"] = existing_cams
        print(f"[multi-view] Cameras to create: {existing_cams}")

    # Always ensure cameras required by shape_meta are available in the environment.
    # Cameras that will be added post-hoc by MultiViewEnvWrapper (coupled_third_config)
    # MUST be excluded here — they don't exist in the stock robosuite XML yet.
    coupled_third_skip = set()
    if coupled_third_config is not None:
        n_views = len(coupled_third_config.get("fixed_positions", []))
        coupled_third_skip = {f"third_view_{i}" for i in range(n_views)}

    existing_cams = list(env_kwargs.get("camera_names",
        ["agentview", "robot0_eye_in_hand"]))
    for key in shape_meta.get("obs", {}):
        if key.endswith("_image"):
            cam = key[:-6]  # strip "_image"
            if cam in coupled_third_skip:
                print(f"[auto-inject] Skipping '{cam}' (added by MultiViewEnvWrapper post-hoc)")
                continue
            if cam and cam not in existing_cams:
                existing_cams.append(cam)
                print(f"[auto-inject] Adding camera '{cam}' to env (required by shape_meta)")
    env_kwargs["camera_names"] = existing_cams
    print(f"[create_env] final env_kwargs['camera_names']={env_kwargs['camera_names']}  coupled_third_skip={coupled_third_skip}")

    rs = _require_robosuite()
    base_env = rs.make(env_name, **env_kwargs)
    _apply_gripper_physics_overrides(base_env)

    # Override camera fovy to match training dataset if specified.
    if camera_fovy_overrides:
        for cam_name, fovy_val in camera_fovy_overrides.items():
            try:
                cam_id = base_env.sim.model.camera_name2id(cam_name)
                base_env.sim.model.cam_fovy[cam_id] = float(fovy_val)
                print(f"[camera_fovy_override] {cam_name}: fovy={fovy_val}")
            except Exception as e:
                print(f"[camera_fovy_override] WARNING: failed to set fovy for {cam_name}: {e}")

    env = _RobosuiteEnvAdapter(base_env, camera_fovy_overrides=camera_fovy_overrides)

    # Wrap with cpgen's MultiViewEnvWrapper to add third-person cameras at
    # specified spherical positions. These cameras are added via XML patching
    # and rendered manually in step/reset, so they don't need to exist in the
    # stock robosuite env.
    if coupled_third_config is not None:
        try:
            import sys as _sys
            if "/data/yrhuang/cpgen" not in _sys.path:
                _sys.path.insert(0, "/data/yrhuang/cpgen")
            from demo_aug.envs.wrapper.multi_view_wrapper import MultiViewEnvWrapper
            from demo_aug.configs.multi_view_config import (
                MultiViewCameraConfig, ThirdViewCameraConfig,
                EyeInHandCameraConfig, WristCameraPerturbConfig,
            )
            positions = [tuple(p) for p in coupled_third_config["fixed_positions"]]
            third_cfg = ThirdViewCameraConfig(
                width=camera_width,
                height=camera_height,
                fov=float(coupled_third_config.get("fov", 45.0)),
                fixed_positions=positions,
            )
            mv_cfg = MultiViewCameraConfig(
                num_third_views=len(positions),
                num_perturbed_wrist_views=0,
                keep_original_agentview=True,
                keep_wrist_camera=True,
                third_view_config=third_cfg,
                wrist_view_config=EyeInHandCameraConfig(width=camera_width, height=camera_height),
                seed=int(coupled_third_config.get("seed", 42)),
                wrist_perturbation=WristCameraPerturbConfig(enable=False),
            )
            env = MultiViewEnvWrapper(env, mv_cfg)
            print(f"[coupled-third] Wrapped with MultiViewEnvWrapper: {env.camera_names}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Failed to wrap env with MultiViewEnvWrapper: {e}")

    # Wrap with random view aliasing (agentview → random existing camera).
    if multi_view_config is not None and multi_view_config.get("random_view_alias", True):
        mv_seed = multi_view_config.get("seed", None)
        source_pool = multi_view_config.get("source_pool", "all")

        if source_pool == "spherical":
            # Spherical random view: physically move agentview camera to a
            # random position on a hemisphere above the workspace.
            try:
                from sapolicy.eval.gym_util.spherical_random_view_wrapper import SphericalRandomViewWrapper
                # Fixed position overrides (for azimuth sweep experiments)
                fixed_az = multi_view_config.get("sphere_fixed_azimuth", None)
                fixed_el = multi_view_config.get("sphere_fixed_elevation", None)
                fixed_r = multi_view_config.get("sphere_fixed_radius", None)
                fixed_pos = multi_view_config.get("fixed_camera_position", None)
                fixed_quat = multi_view_config.get("fixed_camera_quaternion", None)
                fixed_fovy = multi_view_config.get("fixed_camera_fovy", None)
                env = SphericalRandomViewWrapper(
                    env,
                    target_camera="agentview",
                    wrist_camera="robot0_eye_in_hand",
                    resample_on=multi_view_config.get("resample_on", "episode"),
                    seed=mv_seed,
                    radius_range=tuple(multi_view_config.get("sphere_radius_range", [0.8, 1.5])),
                    elevation_range=tuple(multi_view_config.get("sphere_elevation_range", [15.0, 75.0])),
                    look_at=tuple(multi_view_config.get("sphere_look_at", [0.0, 0.0, 0.85])),
                    wrist_perturb_pos=float(multi_view_config.get("wrist_perturb_pos", 0.0)),
                    wrist_perturb_rot=float(multi_view_config.get("wrist_perturb_rot", 0.0)),
                    fixed_azimuth=float(fixed_az) if fixed_az is not None else None,
                    fixed_elevation=float(fixed_el) if fixed_el is not None else None,
                    fixed_radius=float(fixed_r) if fixed_r is not None else None,
                    fixed_position=tuple(fixed_pos) if fixed_pos is not None else None,
                    fixed_quaternion=tuple(fixed_quat) if fixed_quat is not None else None,
                    fixed_fovy=float(fixed_fovy) if fixed_fovy is not None else None,
                    fixed_source=multi_view_config.get("fixed_camera_source", None),
                )
                if fixed_pos is not None:
                    print(
                        "[spherical-view] agentview uses exact dataset camera_info "
                        f"source={multi_view_config.get('fixed_camera_source')} "
                        f"dataset_az={multi_view_config.get('fixed_camera_dataset_azimuth')}"
                    )
                else:
                    print("[spherical-view] agentview → random hemisphere position each episode")
            except Exception as e:
                raise RuntimeError(f"Failed to initialize spherical evaluation camera: {e}") from e
        else:
            # Discrete random view: alias agentview to a randomly selected
            # existing camera from the observation dict.
            try:
                from sapolicy.eval.gym_util.random_view_alias_wrapper import RandomViewAliasWrapper
                # Use None seed so each env gets a unique RNG from OS entropy.
                # A fixed seed (e.g. 42) makes all parallel envs pick the same camera.
                env = RandomViewAliasWrapper(
                    env,
                    target_camera="agentview",
                    source_pool=source_pool,
                    resample_on=multi_view_config.get("resample_on", "episode"),
                    seed=mv_seed,
                    exclude_original=multi_view_config.get("exclude_original", True),
                )
                print(f"[random-view] Aliasing agentview → random camera each episode")
            except Exception as e:
                print(f"[random-view] WARNING: failed to initialize: {e}")

    return env


def _parse_fovy_overrides(raw):
    """Parse camera_fovy_overrides from CLI string format.

    Accepts: "agentview:60.0,frontview:45.0" or a dict (pass-through).
    Returns: dict like {"agentview": 60.0} or None.
    """
    if raw is None:
        return None
    if isinstance(raw, dict):
        return {k: float(v) for k, v in raw.items()}
    if isinstance(raw, str):
        result = {}
        for pair in raw.split(","):
            pair = pair.strip()
            if ":" in pair:
                cam, fovy = pair.split(":", 1)
                # Strip stray quotes from shell/Hydra quoting
                result[cam.strip().strip('"').strip("'")] = float(fovy.strip().strip('"').strip("'"))
        return result if result else None
    return None


def get_camera_info(
    env,
    camera_names=None, 
    camera_height=256, 
    camera_width=256,
):
    """
    Helper function to get camera intrinsics and extrinsics for cameras being used for observations.
    """

    rs = _require_robosuite()
    is_v15 = (rs.__version__.split(".")[0] == "1") and (rs.__version__.split(".")[1] >= "5")

    if camera_names is None:
        return None

    # robomimic's EnvRobosuite exposes get_camera_* helpers, but raw robosuite
    # envs may not. For evaluation we can compute camera matrices from sim.
    base_env = getattr(env, "base_env", env)
    sim = getattr(base_env, "sim", None)
    if sim is None:
        return None

    camera_info = dict()
    for cam_name in camera_names:
        try:
            from robosuite.utils.camera_utils import (
                get_camera_intrinsic_matrix,
                get_camera_extrinsic_matrix,
            )
            K = get_camera_intrinsic_matrix(
                sim=sim,
                camera_name=cam_name,
                camera_height=camera_height,
                camera_width=camera_width,
            )
            # Camera pose in world frame.
            R = get_camera_extrinsic_matrix(sim=sim, camera_name=cam_name)
        except Exception:
            # Intrinsics are optional for our VLA policy path (we pass None to infer()).
            return None
        if "eye_in_hand" in cam_name:
            # convert extrinsic matrix to be relative to robot eef control frame
            assert cam_name.startswith("robot0") or cam_name.startswith("robot1")
            robot_ind = int(cam_name[5])
            if is_v15:
                eef_site_name = base_env.robots[robot_ind].composite_controller.part_controllers["right"].ref_name
            else:
                eef_site_name = base_env.robots[robot_ind].controller.eef_name
            eef_pos = np.array(sim.data.site_xpos[sim.model.site_name2id(eef_site_name)])
            eef_rot = np.array(sim.data.site_xmat[sim.model.site_name2id(eef_site_name)].reshape([3, 3]))
            eef_pose = np.zeros((4, 4)) # eef pose in world frame
            eef_pose[:3, :3] = eef_rot
            eef_pose[:3, 3] = eef_pos
            eef_pose[3, 3] = 1.0
            # eef_pose_inv = np.zeros((4, 4))
            # eef_pose_inv[:3, :3] = eef_pose[:3, :3].T
            # eef_pose_inv[:3, 3] = -eef_pose_inv[:3, :3].dot(eef_pose[:3, 3])
            # eef_pose_inv[3, 3] = 1.0
            R = np.linalg.inv(R) @ eef_pose
            # R = R.dot(eef_pose_inv) # T_E^W * T_W^C = T_E^C
        camera_info[cam_name] = dict(
            intrinsics=K.tolist(),
            extrinsics=R.tolist(),
        )
    return camera_info


def _build_shape_meta(
    dataset_path: str,
    manual_spec: Dict[str, Any] = None,
    camera_height: int = 256,
    camera_width: int = 256,
    camera_depths: bool = True,
) -> Dict[str, Any]:
    """
    Construct shape metadata for the env runner.

    If ``manual_spec`` is provided it should follow
    ``{"obs": {key: {"shape": [...], "type": "rgb|depth|low_dim"}}}``
    format and will be returned directly (with shapes normalized to tuples).

    Otherwise fall back to inferring a *minimal* shape_meta from the dataset file.
    This is intentionally conservative: CPGen datasets can contain many auxiliary
    keys (e.g. tcp visualization targets) that are not available from the live
    environment at eval time.
    """

    if manual_spec is not None:
        obs_meta = {}
        for key, attr in manual_spec.get("obs", manual_spec).items():
            shape = attr["shape"]
            if "image" in key or "depth" in key:
                shape = (camera_height, camera_width, shape[-1])
            obs_meta[key] = {
                "shape": tuple(int(x) for x in shape),
                "type": attr.get("type", "low_dim"),
            }
        result = {"obs": obs_meta}
        for other_key, other_val in manual_spec.items():
            if other_key == "obs":
                continue
            result[other_key] = other_val
        return result

    dataset_path = os.path.expanduser(dataset_path)
    obs_meta: Dict[str, Dict[str, Any]] = {}
    action_meta: Dict[str, Any] = {}

    # Minimal set of low-dim keys the evaluation pipeline may use.
    # NOTE: robot0_gripper_qpos is required for state assembly (state =
    # [eef_pos, 6Drot, gripper_qpos[:1]] = 10D). Without it, eval state has
    # gripper=0 → catastrophic accuracy drop on grasp tasks.
    keep_lowdim = {
        "robot0_eef_pos",
        "robot0_eef_quat_site",
        "robot0_gripper_qpos",
        "camera_intrinsics",
    }

    with h5py.File(dataset_path, "r") as f:
        data_group = f["data"]
        first_key = next(iter(data_group.keys()))

        # Action shape from dataset (env action space).
        if "actions" in data_group[first_key]:
            action_shape = data_group[first_key]["actions"].shape[1:]
            action_meta = {
                "shape": tuple(int(x) for x in action_shape),
                "type": "continuous",
            }

        obs_group = data_group[first_key]["obs"]
        for key, dataset in obs_group.items():
            raw_shape = dataset.shape[1:]

            is_image = key.endswith("_image") or key == "image"
            is_depth = key.endswith("_depth") or key == "depth"

            if not (is_image or is_depth or key in keep_lowdim):
                continue
            if is_depth and not camera_depths:
                continue

            if is_image:
                obs_type = "rgb"
                channels = raw_shape[-1] if len(raw_shape) == 3 else 3
                shape = (camera_height, camera_width, int(channels))
            elif is_depth:
                obs_type = "depth"
                channels = raw_shape[-1] if len(raw_shape) == 3 else 1
                shape = (camera_height, camera_width, int(channels))
            else:
                obs_type = "low_dim"
                shape = raw_shape

            obs_meta[key] = {
                "shape": tuple(int(x) for x in shape),
                "type": obs_type,
            }

    shape_meta = {"obs": obs_meta}
    if action_meta:
        shape_meta["action"] = action_meta
    return shape_meta


def _infer_lowdim_obs_shape(dataset_path: str, key: str) -> tuple:
    """Peek a low-dim obs key's per-step shape from the first demo of an hdf5 dataset.

    Grippers report different native dims per embodiment (e.g. 2 for Panda/Sawyer's
    parallel-jaw grippers vs 6 for IIWA/UR5e's Robotiq grippers), so callers that build a
    manual shape_meta must not hardcode this — it has to match whatever embodiment's
    dataset/env is actually being evaluated.
    """
    dataset_path = os.path.expanduser(dataset_path)
    with h5py.File(dataset_path, "r") as f:
        data_group = f["data"]
        first_key = next(iter(data_group.keys()))
        return tuple(int(x) for x in data_group[first_key]["obs"][key].shape[1:])


def _build_observation_transforms(config: Union[str, List[Dict[str, Any]]] = None):
    if config is None:
        return None

    if isinstance(config, str):
        path = os.path.expanduser(config)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Transform config {path} not found")

        if path.endswith((".yaml", ".yml")):
            transform_cfg = OmegaConf.load(path)
        else:
            import json

            with open(path, "r") as f:
                transform_cfg = OmegaConf.create(json.load(f))
        cfg_node = transform_cfg

        if isinstance(transform_cfg, DictConfig) and "transforms" in transform_cfg:
            cfg_node = transform_cfg.transforms
        
    elif isinstance(config, ListConfig):
        cfg_node = config

    transforms = []
    for item in cfg_node:
        transforms.append(hydra.utils.instantiate(item))

    if len(transforms) == 0:
        return None
    return Compose(transforms)


# Public API (private names kept as backward-compatible aliases)
parse_fovy_overrides = _parse_fovy_overrides
build_shape_meta = _build_shape_meta
infer_lowdim_obs_shape = _infer_lowdim_obs_shape
build_observation_transforms = _build_observation_transforms
RobosuiteEnvAdapter = _RobosuiteEnvAdapter
load_env_meta_from_dataset = _load_env_meta_from_dataset
apply_robotiq85_osc_kp_default = _apply_robotiq85_osc_kp_default
require_cpgen_envs = _require_cpgen_envs
require_robosuite = _require_robosuite
adapt_gripper_qpos_for_state = _adapt_gripper_qpos_for_state
