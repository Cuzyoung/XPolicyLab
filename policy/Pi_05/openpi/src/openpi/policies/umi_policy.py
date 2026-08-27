"""Policy transforms for bimanual UMI LeRobot datasets.

Dataset layout:
  observation.state                 (20,) = [L: 9-dim TCP pose + gripper, R: same]
  action                            (20,) with the same ordering
  observation.images.left_wrist     left wrist camera
  observation.images.right_wrist    right wrist camera

The UMI rig has no third-person camera, so ``base_0_rgb`` is padded with a black
image and its mask is cleared, the same convention the other policies use for a
missing camera. The tactile streams stay out of the model: pi05 exposes exactly
the three image slots above.
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

# Per arm: 3 translation + 6 rotation (6D representation) + 1 absolute gripper.
UMI_ACTION_DIM = 20


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class UmiInputs(transforms.DataTransformFn):
    """Map UMI observations/actions onto OpenPI's canonical model inputs."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        left_wrist = _parse_image(data["observation/left_wrist"])
        right_wrist = _parse_image(data["observation/right_wrist"])
        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": np.zeros_like(left_wrist),
                "left_wrist_0_rgb": left_wrist,
                "right_wrist_0_rgb": right_wrist,
            },
            "image_mask": {
                "base_0_rgb": np.False_,
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
class UmiOutputs(transforms.DataTransformFn):
    """Remove OpenPI action padding while preserving UMI TCP/gripper order."""

    action_dim: int = UMI_ACTION_DIM

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., : self.action_dim])}
