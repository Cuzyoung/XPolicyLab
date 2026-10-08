import numpy as np


class RandomViewAliasWrapper:
    """
    Observation wrapper that aliases a randomly selected camera view into a
    fixed "target" camera name (e.g., agentview).

    This is useful for evaluating a single-camera model on arbitrary camera
    viewpoints, while keeping the model architecture / obs keys unchanged.
    """

    def __init__(
        self,
        env,
        target_camera: str = "agentview",
        source_pool: str = "third",  # "third" | "wrist" | "all"
        resample_on: str = "episode",  # "episode" | "step"
        seed: int | None = None,
        exclude_original: bool = True,
        third_prefix: str = "third_view",
        wrist_prefix: str = "robot0_eye_in_hand_perturbed",
    ):
        self.env = env
        self.target_camera = target_camera
        self.source_pool = source_pool
        self.resample_on = resample_on
        self.exclude_original = exclude_original
        self.third_prefix = third_prefix
        self.wrist_prefix = wrist_prefix

        self._rng = np.random.default_rng(seed)
        self._selected_camera: str | None = None
        self._last_obs = None

    def seed(self, seed=None):
        # Keep parity with other env wrappers.
        self._rng = np.random.default_rng(seed)
        if hasattr(self.env, "seed"):
            return self.env.seed(seed)
        return None

    def _available_cameras(self, obs: dict) -> list[str]:
        cams: list[str] = []
        for k in obs.keys():
            if not k.endswith("_image"):
                continue
            cam = k[: -len("_image")]
            if self.exclude_original and cam == self.target_camera:
                continue
            cams.append(cam)
        return sorted(set(cams))

    def _filter_candidates(self, cams: list[str]) -> list[str]:
        if self.source_pool == "all":
            return cams
        if self.source_pool == "third":
            return [c for c in cams if c.startswith(self.third_prefix)]
        if self.source_pool == "existing_third":
            # Use built-in robosuite third-person cameras (no XML modification).
            _builtin_third = {"frontview", "sideview", "birdview", "robot0_robotview"}
            return [c for c in cams if c in _builtin_third]
        if self.source_pool == "wrist":
            # Allow both original wrist camera and perturbed copies.
            return [c for c in cams if c.startswith(self.wrist_prefix) or c == "robot0_eye_in_hand"]
        # Allow specifying a single camera name directly (e.g. "frontview").
        if self.source_pool in cams:
            return [self.source_pool]
        raise ValueError(f"Unknown source_pool={self.source_pool!r}")

    def _maybe_resample(self, obs: dict) -> None:
        if self.resample_on not in ("episode", "step"):
            raise ValueError(f"Unknown resample_on={self.resample_on!r}")
        if self._selected_camera is not None and self.resample_on == "episode":
            return
        candidates = self._filter_candidates(self._available_cameras(obs))
        if not candidates:
            # Fallback: keep original view (no aliasing).
            self._selected_camera = self.target_camera
            return
        self._selected_camera = candidates[int(self._rng.integers(0, len(candidates)))]

    def _alias_obs(self, obs: dict) -> dict:
        if not isinstance(obs, dict):
            return obs
        if self._selected_camera is None:
            self._maybe_resample(obs)
        src = self._selected_camera
        if src is None or src == self.target_camera:
            return obs

        # Copy so downstream wrappers don't see in-place mutation surprises.
        aliased = dict(obs)
        for suffix in ("image", "depth"):
            src_key = f"{src}_{suffix}"
            tgt_key = f"{self.target_camera}_{suffix}"
            if src_key in obs:
                # Preserve the original target camera obs for video rendering.
                orig_key = f"_orig_{tgt_key}"
                if tgt_key in obs:
                    aliased[orig_key] = obs[tgt_key]
                aliased[tgt_key] = obs[src_key]
        return aliased

    def reset(self, **kwargs):
        raw_obs = self.env.reset(**kwargs)
        # New episode -> resample.
        self._selected_camera = None
        self._maybe_resample(raw_obs)
        aliased = self._alias_obs(raw_obs)
        self._last_obs = aliased
        return aliased

    def reset_to(self, state, **kwargs):
        raw_obs = self.env.reset_to(state, **kwargs)
        self._selected_camera = None
        self._maybe_resample(raw_obs)
        aliased = self._alias_obs(raw_obs)
        self._last_obs = aliased
        return aliased

    def step(self, action):
        raw_obs, reward, done, info = self.env.step(action)
        if self.resample_on == "step":
            self._selected_camera = None
            self._maybe_resample(raw_obs)
        aliased = self._alias_obs(raw_obs)
        self._last_obs = aliased
        if isinstance(info, dict):
            info = dict(info)
            info["eval/selected_camera"] = self._selected_camera
        return aliased, reward, done, info

    def get_state(self):
        return self.env.get_state()

    def get_observation(self):
        if self._last_obs is not None:
            return self._last_obs
        if hasattr(self.env, "get_observation"):
            raw_obs = self.env.get_observation()
            aliased = self._alias_obs(raw_obs)
            self._last_obs = aliased
            return aliased
        raise AttributeError("Wrapped env does not expose get_observation().")

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.env, name)

