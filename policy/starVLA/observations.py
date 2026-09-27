"""Convert decoded XPolicyLab observations into StarVLA model inputs."""

from collections.abc import Mapping

import cv2
import numpy as np

from XPolicyLab.utils.process_data import pack_robot_state


class ObservationEncoder:
    """Own camera ordering, resizing and raw-state packing; never decode image bytes."""

    def __init__(
        self,
        camera_names,
        image_size,
        *,
        include_state,
        state_type,
        state_dimensions,
        state_indices,
    ):
        if camera_names is None:
            camera_names = [
                ["cam_head", "head_camera"],
                ["cam_left_wrist", "left_camera"],
                ["cam_right_wrist", "right_camera"],
            ]
        if not isinstance(camera_names, (list, tuple)) or not camera_names:
            raise ValueError("camera_names must be a non-empty ordered list")
        self.camera_names = [[item] if isinstance(item, str) else item for item in camera_names]
        if any(
            not isinstance(names, (list, tuple))
            or not names
            or any(not isinstance(name, str) or not name for name in names)
            for names in self.camera_names
        ):
            raise ValueError("camera_names entries must be names or non-empty lists of aliases")
        if (
            not isinstance(image_size, (list, tuple))
            or len(image_size) != 2
            or any(type(size) is not int or size <= 0 for size in image_size)
        ):
            raise ValueError("image_size must contain positive integer [width, height]")
        self.image_size = tuple(image_size)
        self.include_state = include_state
        self.state_type = state_type
        self.state_dimensions = state_dimensions
        self.state_indices = state_indices

    def _camera(self, vision, aliases):
        name = next((name for name in aliases if name in vision), None)
        if name is None:
            raise KeyError(f"Missing camera from configured aliases: {aliases}")
        camera = vision[name]
        if not isinstance(camera, Mapping) or "color" not in camera:
            raise ValueError(f"vision[{name!r}] must contain a decoded color array")
        image = camera["color"]
        if (
            not isinstance(image, np.ndarray)
            or image.dtype != np.uint8
            or image.ndim != 3
            or image.shape[-1] != 3
            or 0 in image.shape
        ):
            raise ValueError(f"vision[{name!r}].color must be an HWC uint8 RGB array")
        return cv2.resize(image, self.image_size, interpolation=cv2.INTER_AREA)

    def __call__(self, observation):
        instruction = observation["instruction"]
        if not isinstance(instruction, str):
            raise ValueError("observation.instruction must be a string")
        converted = {
            "lang": instruction,
            "image": [self._camera(observation["vision"], names) for names in self.camera_names],
        }
        if self.include_state:
            state = pack_robot_state(
                observation, self.state_type, self.state_dimensions, source_type="obs"
            ).astype(np.float32)
            if state.ndim == 1:
                state = state[None, :]
            width = len(self.state_indices)
            if state.ndim != 2 or state.shape[-1] != width or not np.isfinite(state).all():
                raise ValueError(f"Expected finite state shape (T, {width}), got {state.shape}")
            converted["state"] = state[:, self.state_indices]
        return converted
