"""Policy transforms for bimanual YAM LeRobot datasets.

Dataset layout:
  observation.state              (14,) = [L: 6 joints + gripper, R: 6 joints + gripper]
  action                         (14,) with the same ordering
  observation.images.top_rgb     third-person camera
  observation.images.left_rgb    left-side camera
  observation.images.right_rgb   right-side camera
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

YAM_ACTION_DIM = 14


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class YamInputs(transforms.DataTransformFn):
    """Map YAM observations/actions onto OpenPI's canonical model inputs."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": _parse_image(data["observation/image"]),
                "left_wrist_0_rgb": _parse_image(data["observation/left_wrist"]),
                "right_wrist_0_rgb": _parse_image(data["observation/right_wrist"]),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }
        if "actions" in data:
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class YamOutputs(transforms.DataTransformFn):
    """Remove OpenPI action padding while preserving YAM joint/gripper order."""

    action_dim: int = YAM_ACTION_DIM

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., : self.action_dim])}
