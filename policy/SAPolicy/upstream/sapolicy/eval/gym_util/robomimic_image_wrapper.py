from typing import Optional, TYPE_CHECKING

import gym
from gym import spaces
import numpy as np

# Avoid importing robomimic at module import time. robomimic pulls in optional
# language / transformer stacks that can conflict with some MuJoCo GL backends
# (e.g., OSMesa + llvmpipe). This wrapper only needs an env object that follows
# the expected methods (reset / step / get_observation / reset_to / get_state).
if TYPE_CHECKING:
    from robomimic.envs.env_robosuite import EnvRobosuite  # pragma: no cover
else:
    EnvRobosuite = object


def _sync_placement_rng(raw_env, seed) -> None:
    """Make robosuite object placement (e.g. square_nut xy) reproducible across
    embodiments for the same seed.

    Some env classes -- e.g. mimicgen's Square_D0, used for the non-Panda
    cross-embodiment datasets -- build their own UniformRandomSampler(s) without
    passing `rng=`, and do so before `raw_env.rng` even exists (their __init__ runs
    before MujocoEnv.__init__, which is what creates `self.rng`). Those samplers end
    up each with their own unseeded, OS-entropy `numpy.random.Generator`, completely
    disconnected from `np.random.seed(seed)` -- so "the same seed" was silently
    giving every embodiment its own, uncontrollable object placement.

    Fix: rebind every sampler to `raw_env.rng` (the one Generator this env actually
    exposes -- robosuite's own default `NutAssemblySquare._load_model()` already
    does this correctly, so this is a no-op there), then reseed that Generator's
    state *in place*. Must mutate the existing Generator's state rather than
    assigning `raw_env.rng = np.random.default_rng(seed)` -- with `hard_reset=False`
    (used throughout eval for speed) the samplers' `.rng` references are fixed once
    at construction and never rebuilt on reset, so a reassignment wouldn't reach them.
    """
    rng = getattr(raw_env, "rng", None)
    if rng is None:
        return
    samplers = getattr(getattr(raw_env, "placement_initializer", None), "samplers", None)
    if samplers:
        for sampler in samplers.values():
            sampler.rng = rng
    rng.bit_generator.state = np.random.default_rng(seed).bit_generator.state


class RobomimicImageWrapper(gym.Env):
    def __init__(self,
        env: EnvRobosuite,
        shape_meta: dict,
        init_state: Optional[np.ndarray]=None,
        render_obs_key='agentview_image',
        video_render_camera: Optional[str]=None,
        warmup_ref: Optional[dict]=None,
        ):

        self.env = env
        self.render_obs_key = render_obs_key
        self.video_render_camera = video_render_camera
        self.warmup_ref = warmup_ref
        self.init_state = init_state
        self.seed_state_map = dict()
        self._seed = None
        self.shape_meta = shape_meta
        self.render_cache = None
        self.has_reset_before = False

        def _shape_to_int_tuple(shape):
            return tuple(int(x) for x in shape)
        
        # setup spaces
        action_shape = _shape_to_int_tuple(shape_meta['action']['shape'])
        action_space = spaces.Box(
            low=-1,
            high=1,
            shape=action_shape,
            dtype=np.float32
        )
        self.action_space = action_space

        observation_space = spaces.Dict()
        for key, value in shape_meta['obs'].items():
            shape = _shape_to_int_tuple(value['shape'])
            min_value, max_value = -1, 1
            if key.endswith('image'):
                min_value, max_value = 0, 1
            elif key.endswith('quat') or key.endswith('quat_site'):
                min_value, max_value = -1, 1
            elif key.endswith('qpos'):
                min_value, max_value = -1, 1
            elif key.endswith('pos'):
                # better range?
                min_value, max_value = -1, 1
            elif key.endswith('depth'):
                min_value, max_value = 0, 1
            elif key.endswith('segmentation_instance'):
                min_value, max_value = 0, 1
            else:
                raise RuntimeError(f"Unsupported type {key}")
            
            this_space = spaces.Box(
                low=min_value,
                high=max_value,
                shape=shape,
                dtype=np.float32
            )
            observation_space[key] = this_space
        self.observation_space = observation_space


    def get_observation(self, raw_obs=None):
        if raw_obs is None:
            raw_obs = self.env.get_observation()

        self.render_cache = raw_obs[self.render_obs_key]

        obs = dict()
        for key in self.observation_space.keys():
            obs[key] = raw_obs[key]
        return obs

    def seed(self, seed=None):
        np.random.seed(seed=seed)
        self._seed = seed
    
    def reset(self):
        if self.init_state is not None:
            if not self.has_reset_before:
                # the env must be fully reset at least once to ensure correct rendering
                self.env.reset()
                self.has_reset_before = True

            # always reset to the same state
            # to be compatible with gym
            raw_obs = self.env.reset_to({'states': self.init_state})
        elif self._seed is not None:
            # reset to a specific seed
            seed = self._seed
            if seed in self.seed_state_map:
                # env.reset is expensive, use cache
                raw_obs = self.env.reset_to({'states': self.seed_state_map[seed]})
            else:
                # robosuite's initializes all use numpy global random state
                np.random.seed(seed=seed)
                raw_env = getattr(self.env, 'base_env', getattr(self.env, 'env', self.env))
                _sync_placement_rng(raw_env, seed)
                raw_obs = self.env.reset()
                if self.warmup_ref is not None:
                    # non-Panda zero-shot eval: physically move from the native
                    # reset pose to the Panda-aligned start for this episode's
                    # randomly placed nut, before caching/handing off to policy
                    from sapolicy.eval.gym_util.embodiment_warmup import warmup_to_aligned_start
                    warmup_to_aligned_start(self.env, raw_env, self.warmup_ref)
                    raw_obs = self.env.get_observation()
                state = self.env.get_state()['states']
                self.seed_state_map[seed] = state
            self._seed = None
        else:
            # random reset
            raw_obs = self.env.reset()

        # return obs
        obs = self.get_observation(raw_obs)
        return obs
    
    def step(self, action):
        raw_obs, reward, done, info = self.env.step(action)
        obs = self.get_observation(raw_obs)
        return obs, reward, done, info
    
    def render(self, mode='rgb_array'):
        # When a dedicated video camera is configured, render directly from the
        # MuJoCo sim so the video shows a clear view of the robot arm (e.g.
        # frontview / sideview) regardless of which camera the policy uses.
        render_cam = self.video_render_camera
        # When random view aliasing is active, render from the selected camera
        # so the video shows the actual viewpoint the policy is evaluated on.
        if hasattr(self.env, '_selected_camera') and self.env._selected_camera:
            render_cam = self.env._selected_camera
        if render_cam is not None:
            base_env = getattr(self.env, 'base_env', getattr(self.env, 'env', self.env))
            sim = getattr(base_env, 'sim', None)
            if sim is not None:
                h = self.shape_meta['obs'].get(self.render_obs_key, {}).get('shape', [256, 256, 3])[0]
                w = self.shape_meta['obs'].get(self.render_obs_key, {}).get('shape', [256, 256, 3])[1]
                rgb = sim.render(camera_name=render_cam, height=h, width=w, depth=False)
                return rgb[::-1]  # MuJoCo renders upside down
        if self.render_cache is None:
            raise RuntimeError('Must run reset or step before render.')
        img = self.render_cache
        return img


def test():
    import os
    from omegaconf import OmegaConf
    cfg_path = os.path.expanduser('~/dev/diffusion_policy/diffusion_policy/config/task/lift_image.yaml')
    cfg = OmegaConf.load(cfg_path)
    shape_meta = cfg['shape_meta']


    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.env_utils as EnvUtils
    from matplotlib import pyplot as plt

    dataset_path = os.path.expanduser('~/dev/diffusion_policy/data/robomimic/datasets/square/ph/image.hdf5')
    env_meta = FileUtils.get_env_metadata_from_dataset(
        dataset_path)

    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False, 
        render_offscreen=False,
        use_image_obs=True, 
    )

    wrapper = RobomimicImageWrapper(
        env=env,
        shape_meta=shape_meta
    )
    wrapper.seed(0)
    obs = wrapper.reset()
    img = wrapper.render()
    plt.imshow(img)


    # states = list()
    # for _ in range(2):
    #     wrapper.seed(0)
    #     wrapper.reset()
    #     states.append(wrapper.env.get_state()['states'])
    # assert np.allclose(states[0], states[1])

    # img = wrapper.render()
    # plt.imshow(img)
    # wrapper.seed()
    # states.append(wrapper.env.get_state()['states'])
