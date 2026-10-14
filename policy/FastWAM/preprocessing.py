"""Checkpoint-configured RGB processing for the native three-camera layout."""

import numpy as np
import torch
from torchvision.transforms import functional as TF


def prepare_robotwin_image(observation, processor, video_size):
    """Return [1,3,H,W] float32 using the native training image operations."""
    from fastwam.datasets.dataset_utils import (
        CenterCrop, Normalize, ResizeSmallestSideAspectPreserving,
    )

    camera_names = ("head_camera", "left_camera", "right_camera")
    metas = processor.shape_meta["images"]
    if len(metas) != 3:
        raise ValueError("FastWAM robotwin image layout requires three cameras.")
    images = []
    for name, meta in zip(camera_names, metas, strict=True):
        image = np.asarray(observation["observation"][name]["rgb"])
        if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
            raise ValueError("Expected decoded uint8 HWC RGB camera images.")
        image = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).unsqueeze(0)
        transforms = processor.val_transforms
        transforms = transforms[meta["key"]] if isinstance(transforms, dict) else transforms
        for transform in transforms:
            image = transform(image)
        if tuple(image.shape[1:]) != tuple(meta["shape"]):
            raise ValueError("Camera transforms do not match the saved image shape.")
        images.append(image)
    top, left, right = [
        TF.resize(image, size=size, interpolation=TF.InterpolationMode.BILINEAR, antialias=True)
        for image, size in zip(images, ([256, 320], [128, 160], [128, 160]), strict=True)
    ]
    image = torch.cat((top, torch.cat((left, right), dim=-1)), dim=-2)
    size = {"img_h": int(video_size[0]), "img_w": int(video_size[1])}
    image = ResizeSmallestSideAspectPreserving(size)(image)
    image = CenterCrop(size)(image)
    return Normalize({"mean": 0.5, "std": 0.5})(image)
