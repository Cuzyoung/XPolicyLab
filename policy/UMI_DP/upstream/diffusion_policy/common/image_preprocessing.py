"""Deterministic RGB preprocessing shared by video caches and direct decoding."""

import math
from PIL import Image


def resize_image(image: Image.Image, output_resolution, mode="center_crop") -> Image.Image:
    """Resize to (height, width), using either center crop or aspect-preserving padding."""
    height, width = output_resolution
    if min(height, width) <= 0:
        raise ValueError("Output dimensions must be positive")
    input_w, input_h = image.size
    if mode == "center_crop":
        scale = max(width / input_w, height / input_h)
        resized_w, resized_h = math.ceil(input_w * scale), math.ceil(input_h * scale)
        image = image.resize((resized_w, resized_h), Image.Resampling.BILINEAR)
        left, top = (resized_w - width) // 2, (resized_h - height) // 2
        return image.crop((left, top, left + width, top + height))
    if mode == "letterbox":
        scale = min(width / input_w, height / input_h)
        resized_w = max(1, min(width, round(input_w * scale)))
        resized_h = max(1, min(height, round(input_h * scale)))
        image = image.resize((resized_w, resized_h), Image.Resampling.BILINEAR)
        result = Image.new("RGB", (width, height), (0, 0, 0))
        result.paste(image, ((width - resized_w) // 2, (height - resized_h) // 2))
        return result
    raise ValueError(f"Unknown image resize mode: {mode}")


def crop_image(image: Image.Image, margins) -> Image.Image:
    left, right, top, bottom = (margins.get(edge, 0) for edge in ("left", "right", "top", "bottom"))
    if min(left, right, top, bottom) < 0:
        raise ValueError("Crop margins must be nonnegative")
    width, height = image.size
    if left + right >= width or top + bottom >= height:
        raise ValueError("Crop margins remove the entire image")
    return image.crop((left, top, width - right, height - bottom))
