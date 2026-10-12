import torch
import cv2
import numpy as np
import sys, os

current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)
sys.path.append(parent_dir)

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.process_data import pack_robot_state, unpack_robot_state, get_robot_action_dim_info
from XPolicyLab.utils.checkpoint_resolver import resolve_checkpoint_root

class Model(ModelTemplate):

    def __init__(self, model_cfg):
        from diffusion_policy.env_runner.dp_runner import DPRunner
        self.model_cfg = dict(model_cfg)
        self.action_type = model_cfg['action_type']

        self.model = self.get_model(model_cfg=model_cfg)
        # The saved policy, not the repository training template, owns these lengths.
        self.runner = DPRunner(n_obs_steps=self.model.n_obs_steps, n_action_steps=self.model.n_action_steps)

        self.robot_action_dim_info = get_robot_action_dim_info(model_cfg)
        self._latest_env_idx_list = None

    def get_model(self, model_cfg):
        import hydra
        import dill
        ckpt_dir = resolve_checkpoint_root(
            model_cfg,
            os.path.join(parent_dir, "checkpoints"),
            policy_dir=parent_dir,
            must_exist=False,
        )
        ckpt_file = self._resolve_checkpoint_file(ckpt_dir, model_cfg.get('checkpoint_num', 'latest'))

        # load checkpoint and workspace
        payload = torch.load(open(ckpt_file, "rb"), pickle_module=dill)
        cfg = payload["cfg"]
        cls = hydra.utils.get_class(cfg._target_)
        workspace = cls(cfg, output_dir=None)
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)

        # get policy from workspace
        policy = workspace.model
        if cfg.training.use_ema:
            policy = workspace.ema_model

        device = torch.device("cuda:0")
        policy.to(device)
        policy.eval()
        
        return policy

    def _resolve_checkpoint_file(self, ckpt_dir, checkpoint_num):
        ckpt_dir = os.fspath(ckpt_dir)
        checkpoint_num = "latest" if checkpoint_num is None else str(checkpoint_num)

        if checkpoint_num.lower() not in {"", "latest", "none"}:
            ckpt_file = os.path.join(ckpt_dir, f"{checkpoint_num}.ckpt")
            if not os.path.isfile(ckpt_file):
                raise FileNotFoundError(f"DP checkpoint not found: {ckpt_file}")
            return ckpt_file

        if not os.path.isdir(ckpt_dir):
            raise FileNotFoundError(f"DP checkpoint directory not found: {ckpt_dir}")

        candidates = []
        for name in os.listdir(ckpt_dir):
            if not name.endswith(".ckpt"):
                continue
            stem = name[:-5]
            if stem.isdigit():
                candidates.append((int(stem), os.path.join(ckpt_dir, name)))

        if not candidates:
            raise FileNotFoundError(f"No numeric DP checkpoints found under: {ckpt_dir}")

        return max(candidates, key=lambda item: item[0])[1]

    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        env_idx_list = [obs["env_idx"] for obs in obs_list]
        if self.model_cfg.get("require_observation_history", False):
            for observation, env_idx in zip(obs_list, env_idx_list, strict=True):
                states = observation.get("additional_info", {}).get("dp_history_states")
                if not isinstance(states, list) or len(states) != self.model.n_obs_steps:
                    raise ValueError("DP requires a complete measured observation history")
                self.runner.obs_list[env_idx].clear()
                for index, state in enumerate(states):
                    item = {**observation, "state": state, "vision": {
                        name: observation["vision"][f"{name}_t{index}"]
                        for name in ("cam_head", "cam_left_wrist", "cam_right_wrist")}}
                    encoded = encode_obs(item, self.action_type, self.robot_action_dim_info,
                                         self.model_cfg.get("eef_representation"))
                    self.runner.update_obs([encoded], [env_idx])
            self._latest_env_idx_list = env_idx_list
            return
        obs_list = [encode_obs(obs, self.action_type, self.robot_action_dim_info,
                               self.model_cfg.get("eef_representation")) for obs in obs_list]
        self.runner.update_obs(obs_list, env_idx_list)
        self._latest_env_idx_list = env_idx_list

    def get_action(self):
        if not self._latest_env_idx_list:
            raise RuntimeError("get_action() called before update_obs().")

        action_list = self.get_action_batch(env_idx_list=[self._latest_env_idx_list[0]])
            
        return action_list[0]

    def get_action_batch(self, env_idx_list=None):
        if env_idx_list is None:
            env_idx_list = self._latest_env_idx_list
        if not env_idx_list:
            raise RuntimeError("get_action_batch() called before update_obs_batch().")

        actions = self.runner.get_action(self.model, env_idx_list)
        action_dict_list = []

        for i in range(len(env_idx_list)):
            current_env_action_list = self._decode_actions(actions[i])
            action_dict_list.append(current_env_action_list)
            
        return action_dict_list

    def _decode_actions(self, actions):
        if self.model_cfg.get("eef_representation") != "absolute_xyz_euler":
            return unpack_robot_state(actions, self.action_type, self.robot_action_dim_info, source_type='obs')
        from scipy.spatial.transform import Rotation
        if actions.ndim != 2 or actions.shape[1] != 14 or not np.isfinite(actions).all():
            raise ValueError("DP absolute_xyz_euler requires finite dual-arm 14-D actions")
        result = []
        for row in actions:
            step = {}
            for index, side in enumerate(("left", "right")):
                arm = row[index*7:(index+1)*7]
                quat = Rotation.from_euler("xyz", arm[3:6]).as_quat()
                step[f"{side}_ee_pose"] = np.r_[arm[:3], quat[3], quat[:3]]
                step[f"{side}_ee_joint_state"] = arm[6:7]
            result.append(step)
        return result

    def sampling_modes(self):
        modes = ["default", "aac"]
        import inspect
        if "rtc" in inspect.signature(self.model.predict_action).parameters:
            modes.append("rtc")
        return modes

    def runtime_metadata(self):
        return {"rtc_sampler": "vp_pigdm_capped_v1",
                "eef_representation": self.model_cfg.get("eef_representation"),
                "action_horizon": self.model.n_action_steps,
                "observation_steps": self.model.n_obs_steps}

    def get_action_rtc(self, sampling):
        if not self._latest_env_idx_list or len(self._latest_env_idx_list) != 1:
            raise ValueError("RTC requires exactly one observation")
        if set(sampling) - {"mode", "action_condition", "condition_weights", "beta"}:
            raise ValueError("Unsupported DP RTC sampling options")
        actions = self.runner.get_action(self.model, self._latest_env_idx_list, rtc=sampling)
        return self._decode_actions(actions[0])

    def get_action_aac(self, sampling):
        from XPolicyLab.utils.flow_sampling import validate_samples
        count = validate_samples(sampling)
        if not self._latest_env_idx_list or len(self._latest_env_idx_list) != 1:
            raise ValueError("AAC requires exactly one observation")
        return {"actions": self.get_action_batch(self._latest_env_idx_list * count)}

    def reset(self):
        self.runner.reset_obs()
        self._latest_env_idx_list = None

def encode_obs(observation, action_type, robot_action_dim_info, eef_representation=None):
    head_img = (np.moveaxis(observation["vision"]["cam_head"]["color"], -1, 0) / 255)
    head_img = np.transpose(cv2.resize(np.transpose(head_img, (1, 2, 0)), (320, 240), interpolation=cv2.INTER_AREA), (2, 0, 1))
    left_cam = (np.moveaxis(observation["vision"]["cam_left_wrist"]["color"], -1, 0) / 255)
    left_cam = np.transpose(cv2.resize(np.transpose(left_cam, (1, 2, 0)), (320, 240), interpolation=cv2.INTER_AREA), (2, 0, 1))
    right_cam = (np.moveaxis(observation["vision"]["cam_right_wrist"]["color"], -1, 0) / 255)
    right_cam = np.transpose(cv2.resize(np.transpose(right_cam, (1, 2, 0)), (320, 240), interpolation=cv2.INTER_AREA), (2, 0, 1))
    obs = dict(
        head_cam=head_img,
        left_cam=left_cam,
        right_cam=right_cam,
    )
    if eef_representation == "absolute_xyz_euler":
        from scipy.spatial.transform import Rotation
        parts = []
        for side in ("left", "right"):
            pose = np.asarray(observation["state"][f"{side}_ee_pose"])
            parts.extend([pose[:3], Rotation.from_quat(pose[[4,5,6,3]]).as_euler("xyz"),
                          observation["state"][f"{side}_ee_joint_state"]])
        obs["agent_pos"] = np.concatenate(parts)
    else:
        obs["agent_pos"] = pack_robot_state(observation, action_type, robot_action_dim_info, source_type='obs')
    return obs
