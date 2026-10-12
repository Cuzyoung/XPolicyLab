"""Minimal inference runtime for the released PatchWAM C2R checkpoint."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from PIL import Image

from imagewam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from imagewam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json


ROOT = Path(__file__).resolve().parent


def _resize_rgb(image: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    image = np.asarray(image, dtype=np.uint8)
    return np.asarray(Image.fromarray(image, mode="RGB").resize(size_wh, Image.BILINEAR))


class PatchWAMPolicy:
    def __init__(self, cfg: dict[str, Any]):
        device = str(cfg.get("device") or "cuda")
        dtype = torch.bfloat16 if str(cfg.get("mixed_precision") or "bf16") == "bf16" else torch.float32

        resolved = OmegaConf.load(str(cfg["runtime_config_path"]))
        resolved.model.flux2_model_path = str(cfg["flux2_model_path"])
        resolved.model.ae_model_path = str(cfg["ae_model_path"])
        resolved.model.flux2_src_path = str(ROOT / "vendor" / "flux2")
        resolved.model.qwen3_model_spec = str(cfg.get("qwen3_model_spec") or "Qwen/Qwen3-4B")
        resolved.model.load_text_encoder = True

        self.model = instantiate(resolved.model, model_dtype=dtype, device=device)
        self.model.load_checkpoint(str(cfg["checkpoint_path"]))
        self.model = self.model.to(device).eval()

        self.processor = instantiate(resolved.data.train.processor).eval()
        stats = load_dataset_stats_from_json(str(cfg["dataset_stats_path"]))
        self.processor.set_normalizer_from_stats(stats)

        self.action_horizon = int(cfg.get("action_horizon") or 16)
        self.replan_steps = min(int(cfg.get("replan_steps") or 16), self.action_horizon)
        self.num_inference_steps = int(cfg.get("num_inference_steps") or 10)
        self.sigma_shift = cfg.get("sigma_shift")
        self.seed = int(cfg.get("seed") or 0)
        self.text_cfg_scale = float(cfg.get("text_cfg_scale") or 1.0)
        self.negative_prompt = str(cfg.get("negative_prompt") or "")
        self.rand_device = str(cfg.get("rand_device") or "cpu")
        self.tiled = bool(cfg.get("tiled", False))
        self.num_video_frames = 17

    def _normalize_state(self, state: np.ndarray) -> torch.Tensor:
        state_key = self.processor.shape_meta["state"][0]["key"]
        batch = {"state": {state_key: torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)}}
        batch = self.processor.action_state_transform(batch)
        batch = self.processor.normalizer.forward(batch)
        return batch["state"][state_key]

    def _denormalize_action(self, action: torch.Tensor) -> np.ndarray:
        if action.ndim == 2:
            action = action.unsqueeze(0)
        action_key = self.processor.shape_meta["action"][0]["key"]
        normalizer = self.processor.normalizer.normalizers["action"][action_key]
        return normalizer.backward(action.to(dtype=torch.float32, device="cpu")).numpy()

    def infer(self, observation: dict[str, Any], instruction: str) -> np.ndarray:
        images = observation["observation"]
        head = _resize_rgb(images["head_camera"]["rgb"], (256, 192))
        left = _resize_rgb(images["left_camera"]["rgb"], (128, 96))
        right = _resize_rgb(images["right_camera"]["rgb"], (128, 96))
        image = np.concatenate([head, np.concatenate([left, right], axis=1)], axis=0)
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(
            device=self.model.device, dtype=self.model.torch_dtype
        )
        image_tensor = image_tensor * (2.0 / 255.0) - 1.0
        proprio = self._normalize_state(np.asarray(observation["joint_action"]["vector"], dtype=np.float32))

        kwargs = {
            "prompt": DEFAULT_PROMPT.format(task=instruction),
            "input_image": image_tensor,
            "action_horizon": self.action_horizon,
            "proprio": proprio,
            "negative_prompt": self.negative_prompt,
            "text_cfg_scale": self.text_cfg_scale,
            "num_inference_steps": self.num_inference_steps,
            "sigma_shift": self.sigma_shift,
            "seed": self.seed,
            "rand_device": self.rand_device,
            "tiled": self.tiled,
        }
        if "num_video_frames" in inspect.signature(self.model.infer_action).parameters:
            kwargs["num_video_frames"] = self.num_video_frames
        with torch.no_grad():
            prediction = self.model.infer_action(**kwargs)
        return self._denormalize_action(prediction["action"])[0][: self.replan_steps]
