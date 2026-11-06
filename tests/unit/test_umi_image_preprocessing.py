"""Preserve legacy RGB inputs and match the training crop pipeline exactly."""

import importlib.util
import math
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from XPolicyLab.policy.UMI_DP.transforms import preprocess_rgb


def test_legacy_center_crop_unchanged():
    rgb = np.random.default_rng(42).integers(0, 256, (481, 641, 3), dtype=np.uint8)
    height, width = 224, 224
    scale = max(width / 641, height / 481)
    rw, rh = math.ceil(641 * scale), math.ceil(481 * scale)
    image = Image.fromarray(rgb).resize((rw, rh), Image.Resampling.BILINEAR)
    left, top = (rw - width) // 2, (rh - height) // 2
    expected = np.asarray(image.crop((left, top, left + width, top + height)), dtype=np.float32)
    np.testing.assert_array_equal(preprocess_rgb(rgb, (3, height, width)), expected.transpose(2, 0, 1) / 255)


@pytest.mark.parametrize("mode", ["center_crop", "letterbox"])
@pytest.mark.parametrize("hw", [(480, 640), (481, 643)])
def test_training_source_pixel_parity(mode, hw):
    root = os.environ.get("CALIBWRIST_REFERENCE_ROOT")
    if not root:
        pytest.skip("Set CALIBWRIST_REFERENCE_ROOT for training-source parity")
    path = Path(root) / "policies/diffusion_policy/diffusion_policy/common/image_preprocessing.py"
    spec = importlib.util.spec_from_file_location("calibwrist_image_reference", path)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    rgb = np.random.default_rng(42).integers(0, 256, (*hw, 3), dtype=np.uint8)
    margins = dict(left=112, right=112, top=0, bottom=0)
    expected = reference.resize_image(reference.crop_image(Image.fromarray(rgb), margins), (256, 224), mode)
    expected = np.asarray(expected, dtype=np.float32).transpose(2, 0, 1) / 255
    actual = preprocess_rgb(rgb, (3, 256, 224), image_crop=margins, image_resize_mode=mode)
    np.testing.assert_array_equal(actual, expected)


def test_letterbox_preserves_color_order_and_black_padding():
    rgb = np.full((480, 640, 3), [255, 127, 31], dtype=np.uint8)
    actual = preprocess_rgb(rgb, (3, 256, 224), image_crop=dict(left=112, right=112), image_resize_mode="letterbox")
    # A 416x480 crop scales to 222x256, with one black column on each side.
    np.testing.assert_array_equal(actual[:, :, [0, -1]], 0)
    np.testing.assert_array_equal(actual[:, 128, 112], np.array([255, 127, 31], dtype=np.float32) / 255)


def test_invalid_crop_rejected():
    with pytest.raises(ValueError, match="entire image"):
        preprocess_rgb(np.zeros((480, 200, 3), dtype=np.uint8), (3, 256, 224), image_crop=dict(left=112, right=112))
