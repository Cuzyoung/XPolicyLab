"""
Wrapper that randomises the agentview camera pose by sampling from a
hemisphere above the workspace, and optionally perturbs the wrist camera.

Usage in eval CLI:
    eval.multi_view.random_view_alias=1
    eval.multi_view.source_pool=spherical          # activates this wrapper
    eval.multi_view.sphere_radius_range=[0.8,1.5]  # min/max distance from look_at
    eval.multi_view.sphere_elevation_range=[15,75]  # min/max elevation in degrees
    eval.multi_view.sphere_look_at=[0.0,0.0,0.85]  # workspace centre
    eval.multi_view.wrist_perturb_pos=0.02          # wrist pos noise std (metres)
    eval.multi_view.wrist_perturb_rot=5.0           # wrist rot noise std (degrees)
"""

import numpy as np
from scipy.spatial.transform import Rotation as R


def _look_at_quat(cam_pos: np.ndarray, target: np.ndarray, up: np.ndarray = None) -> np.ndarray:
    """Compute MuJoCo camera quaternion (wxyz) so the camera looks at *target*.

    MuJoCo camera convention: -Z axis points towards the scene (forward),
    +Y axis points up in the image.
    """
    if up is None:
        up = np.array([0.0, 0.0, 1.0])

    forward = target - cam_pos
    forward = forward / (np.linalg.norm(forward) + 1e-8)

    right = np.cross(forward, up)
    norm_right = np.linalg.norm(right)
    if norm_right < 1e-6:
        # Camera is directly above — pick an arbitrary right vector.
        up = np.array([1.0, 0.0, 0.0])
        right = np.cross(forward, up)
        norm_right = np.linalg.norm(right)
    right = right / norm_right

    cam_up = np.cross(right, forward)
    cam_up = cam_up / (np.linalg.norm(cam_up) + 1e-8)

    # MuJoCo convention: columns of rotation matrix are [right, up, -forward]
    rot_mat = np.stack([right, cam_up, -forward], axis=-1)  # 3×3
    quat_xyzw = R.from_matrix(rot_mat).as_quat()  # scipy uses xyzw
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
    return quat_wxyz


class SphericalRandomViewWrapper:
    """
    On each episode reset, sample camera positions on a hemisphere and set
    them via MuJoCo sim.model.cam_pos / cam_quat.

    The wrapper does NOT alias observation keys — it physically moves the
    ``agentview`` camera in the MuJoCo simulation. This is more realistic
    than aliasing a discrete camera, and the observation dict stays unchanged.
    """

    def __init__(
        self,
        env,
        target_camera: str = "agentview",
        wrist_camera: str = "robot0_eye_in_hand",
        resample_on: str = "episode",
        seed: int | None = None,
        # Hemisphere sampling parameters
        radius_range: tuple[float, float] = (0.8, 1.5),
        elevation_range: tuple[float, float] = (15.0, 75.0),
        look_at: tuple[float, float, float] = (0.0, 0.0, 0.85),
        # Wrist perturbation
        wrist_perturb_pos: float = 0.0,
        wrist_perturb_rot: float = 0.0,
        # Fixed position mode (overrides random sampling when set)
        fixed_azimuth: float | None = None,   # degrees
        fixed_elevation: float | None = None,  # degrees
        fixed_radius: float | None = None,     # metres
        fixed_position: tuple[float, float, float] | None = None,
        fixed_quaternion: tuple[float, float, float, float] | None = None,
        fixed_fovy: float | None = None,
        fixed_source: str | None = None,
    ):
        self.env = env
        self.target_camera = target_camera
        self.wrist_camera = wrist_camera
        self.resample_on = resample_on
        self.radius_range = radius_range
        self.elevation_range = tuple(np.deg2rad(e) for e in elevation_range)
        self.look_at = np.array(look_at, dtype=np.float64)
        self.wrist_perturb_pos = wrist_perturb_pos
        self.wrist_perturb_rot_rad = np.deg2rad(wrist_perturb_rot)

        # Fixed position mode
        self.fixed_azimuth = np.deg2rad(fixed_azimuth) if fixed_azimuth is not None else None
        self.fixed_elevation = np.deg2rad(fixed_elevation) if fixed_elevation is not None else None
        self.fixed_radius = fixed_radius
        self.fixed_position = (
            np.asarray(fixed_position, dtype=np.float64) if fixed_position is not None else None
        )
        self.fixed_quaternion = (
            np.asarray(fixed_quaternion, dtype=np.float64) if fixed_quaternion is not None else None
        )
        if (self.fixed_position is None) != (self.fixed_quaternion is None):
            raise ValueError("fixed_position and fixed_quaternion must be provided together")
        if self.fixed_position is not None and self.fixed_position.shape != (3,):
            raise ValueError(f"fixed_position must have shape (3,), got {self.fixed_position.shape}")
        if self.fixed_quaternion is not None and self.fixed_quaternion.shape != (4,):
            raise ValueError(
                f"fixed_quaternion must be MuJoCo wxyz with shape (4,), got {self.fixed_quaternion.shape}"
            )
        self.fixed_fovy = float(fixed_fovy) if fixed_fovy is not None else None
        self.fixed_source = fixed_source or "spherical-parameters"

        self._rng = np.random.default_rng(seed)
        self._last_obs = None
        self._sampled_pos = None  # for logging
        self._original_wrist_pos = None
        self._original_wrist_quat = None

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _sample_hemisphere_position(self) -> np.ndarray:
        """Sample a point on the upper hemisphere, or use fixed position."""
        if self.fixed_azimuth is not None and self.fixed_elevation is not None:
            azimuth = self.fixed_azimuth
            elevation = self.fixed_elevation
            radius = self.fixed_radius if self.fixed_radius is not None else 1.0
        else:
            azimuth = self._rng.uniform(0, 2 * np.pi)
            # Uniform elevation by inverse-CDF of cos (avoids pole bunching)
            el_min, el_max = self.elevation_range
            # sin(elevation) is uniform for uniform area on sphere
            sin_el = self._rng.uniform(np.sin(el_min), np.sin(el_max))
            elevation = np.arcsin(sin_el)
            radius = self._rng.uniform(*self.radius_range)

        # cpgen convention (MultiViewEnvWrapper._spherical_to_pos_quat):
        # az=0 → +Y (front of robot), az=90 → +X (right side).
        # Must match cpgen's training-time convention so eval positions align
        # with the third_view_N images in cpgen-generated datasets.
        x = radius * np.cos(elevation) * np.sin(azimuth)
        y = radius * np.cos(elevation) * np.cos(azimuth)
        z = radius * np.sin(elevation)
        return self.look_at + np.array([x, y, z])

    def _get_sim(self):
        """Navigate the wrapper chain to find the MuJoCo sim."""
        env = self.env
        for _ in range(10):
            if hasattr(env, "sim"):
                return env.sim
            if hasattr(env, "base_env"):
                env = env.base_env
                continue
            if hasattr(env, "env"):
                env = env.env
                continue
            break
        raise AttributeError("Cannot find MuJoCo sim in env wrapper chain")

    def _apply_random_agentview(self):
        """Move the agentview camera to a random hemisphere position."""
        sim = self._get_sim()
        try:
            cam_id = sim.model.camera_name2id(self.target_camera)
        except Exception:
            print(f"[spherical-view] WARNING: camera '{self.target_camera}' not found")
            return

        if self.fixed_position is not None:
            new_pos = self.fixed_position.copy()
            new_quat = self.fixed_quaternion.copy()
        else:
            new_pos = self._sample_hemisphere_position()
            new_quat = _look_at_quat(new_pos, self.look_at)

        sim.model.cam_pos[cam_id] = new_pos
        sim.model.cam_quat[cam_id] = new_quat
        if self.fixed_fovy is not None:
            sim.model.cam_fovy[cam_id] = self.fixed_fovy
        self._sampled_pos = new_pos

    def _apply_wrist_perturbation(self):
        """Add small random perturbation to the wrist camera (relative offset)."""
        if self.wrist_perturb_pos <= 0 and self.wrist_perturb_rot_rad <= 0:
            return

        sim = self._get_sim()
        try:
            cam_id = sim.model.camera_name2id(self.wrist_camera)
        except Exception:
            return

        # Save original on first call
        if self._original_wrist_pos is None:
            self._original_wrist_pos = sim.model.cam_pos[cam_id].copy()
            self._original_wrist_quat = sim.model.cam_quat[cam_id].copy()

        # Position perturbation (local frame)
        if self.wrist_perturb_pos > 0:
            delta_pos = self._rng.normal(0, self.wrist_perturb_pos, size=3)
            sim.model.cam_pos[cam_id] = self._original_wrist_pos + delta_pos

        # Rotation perturbation (small random rotation)
        if self.wrist_perturb_rot_rad > 0:
            axis = self._rng.normal(0, 1, size=3)
            axis = axis / (np.linalg.norm(axis) + 1e-8)
            angle = self._rng.normal(0, self.wrist_perturb_rot_rad)
            delta_rot = R.from_rotvec(axis * angle)
            # Original quat is wxyz, scipy uses xyzw
            orig_wxyz = self._original_wrist_quat
            orig_xyzw = np.array([orig_wxyz[1], orig_wxyz[2], orig_wxyz[3], orig_wxyz[0]])
            orig_rot = R.from_quat(orig_xyzw)
            new_rot = delta_rot * orig_rot
            new_xyzw = new_rot.as_quat()
            sim.model.cam_quat[cam_id] = np.array([new_xyzw[3], new_xyzw[0], new_xyzw[1], new_xyzw[2]])

    def _resample_cameras(self):
        self._apply_random_agentview()
        self._apply_wrist_perturbation()

    def _rerender(self):
        """Force MuJoCo to recompute camera transforms and re-render images.

        After modifying ``sim.model.cam_pos`` / ``cam_quat``, we must call
        ``sim.forward()`` so that MuJoCo recomputes derived quantities
        (``sim.data.cam_xpos`` etc.), then ``_update_observables(force=True)``
        to re-trigger camera rendering, and ``_get_observations()`` to collect
        the updated pixel arrays.
        """
        sim = self._get_sim()
        sim.forward()
        base_env = self._get_base_env()
        if hasattr(base_env, "_update_observables"):
            base_env._update_observables(force=True)
        if hasattr(base_env, "_get_observations"):
            return base_env._get_observations()
        return None

    # ------------------------------------------------------------------
    # Env API
    # ------------------------------------------------------------------

    def seed(self, seed=None):
        self._rng = np.random.default_rng(seed)
        if hasattr(self.env, "seed"):
            return self.env.seed(seed)
        return None

    def reset(self, **kwargs):
        raw_obs = self.env.reset(**kwargs)
        self._resample_cameras()
        new_obs = self._rerender()
        if new_obs is not None:
            raw_obs = new_obs

        pos_str = np.array2string(self._sampled_pos, precision=2, separator=",") if self._sampled_pos is not None else "?"
        print(f"[spherical-view] agentview → pos={pos_str} source={self.fixed_source}")
        self._last_obs = raw_obs
        return raw_obs

    def reset_to(self, state, **kwargs):
        raw_obs = self.env.reset_to(state, **kwargs)
        self._resample_cameras()
        new_obs = self._rerender()
        if new_obs is not None:
            raw_obs = new_obs

        self._last_obs = raw_obs
        return raw_obs

    def step(self, action):
        raw_obs, reward, done, info = self.env.step(action)
        if self.resample_on == "step":
            self._resample_cameras()
            new_obs = self._rerender()
            if new_obs is not None:
                raw_obs = new_obs
        self._last_obs = raw_obs
        if isinstance(info, dict):
            info = dict(info)
            info["eval/sampled_cam_pos"] = self._sampled_pos
        return raw_obs, reward, done, info

    def get_state(self):
        return self.env.get_state()

    def get_observation(self):
        if self._last_obs is not None:
            return self._last_obs
        if hasattr(self.env, "get_observation"):
            return self.env.get_observation()
        raise AttributeError("Wrapped env does not expose get_observation().")

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def _get_base_env(self):
        """Navigate wrappers to find the actual robosuite env."""
        env = self.env
        for _ in range(10):
            if hasattr(env, "_update_observables"):
                return env
            if hasattr(env, "base_env"):
                env = env.base_env
                continue
            if hasattr(env, "env"):
                env = env.env
                continue
            break
        return env

    def __getattr__(self, name):
        return getattr(self.env, name)
