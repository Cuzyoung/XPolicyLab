"""Dataset image transforms for RoboMimic / CPGen training and eval.

Active transforms (referenced by configs): Resize, Crop, PrepareForNet.
Legacy depth-simulation transforms were removed in the repo cleanup; only active
transforms remain here.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from sapolicy.logger import Log


def scale_intrinsics_to_resolution(
    K,
    target_height: int,
    target_width: int,
    *,
    orig_height: int | None = None,
    orig_width: int | None = None,
) -> np.ndarray:
    """Scale a 3x3 pinhole K from native to ``(target_height, target_width)``.

    When ``orig_*`` are omitted, infer native resolution from the principal point
    (``cx ≈ w/2``, ``cy ≈ h/2``). That matches ``RobomimicHDF5Dataset`` numpy-cache
    loading, where RGB/depth are pre-resized offline and K is scaled separately.

    When ``orig_*`` are provided, use the actual pre-resize image size — the path
    taken by ``Resize`` during live dataloader transforms.
    """
    K = np.asarray(K, dtype=np.float32).copy()
    if orig_width is None:
        orig_width = int(round(float(K[0, 2]) * 2))
    if orig_height is None:
        orig_height = int(round(float(K[1, 2]) * 2))
    if orig_width <= 0 or orig_height <= 0:
        return K
    ratio_w = target_width / orig_width
    ratio_h = target_height / orig_height
    K[0, 0] *= ratio_w
    K[0, 2] *= ratio_w
    K[1, 1] *= ratio_h
    K[1, 2] *= ratio_h
    return K


def normalize_metric_depth(
    depth,
    min_depth: float = 0.1,
    max_depth: float = 5.0,
) -> np.ndarray:
    """Clip metre-scale depth and map to [0, 1], matching ``RobomimicHDF5Dataset``."""
    depth = np.asarray(depth, dtype=np.float32)
    depth = np.clip(depth, min_depth, max_depth)
    return (depth - min_depth) / (max_depth - min_depth + 1e-8)


class PrepareForNet(object):
    """HWC/THWC numpy sample -> CHW float32 for the network."""

    def __init__(self, image="rgb"):
        self.image = image

    def __str__(self):
        return "PrepareForNet"

    def __repr__(self):
        return "PrepareForNet"

    @staticmethod
    def _hwc_to_chw(image: np.ndarray) -> np.ndarray:
        """Convert HWC or THWC float/uint image stack to CHW / TCHW float32."""
        if image.ndim == 3:
            return np.ascontiguousarray(np.transpose(image, (2, 0, 1))).astype(np.float32)
        if image.ndim == 4:
            res = [
                np.ascontiguousarray(np.transpose(image[i], (2, 0, 1))).astype(np.float32)
                for i in range(image.shape[0])
            ]
            return np.stack(res, axis=0)
        raise ValueError(f"Unsupported image ndim={image.ndim}, shape={image.shape}")

    def __call__(self, sample):
        image = sample["image"]
        if len(image.shape) == 3:
            if self.image == "bgr":
                image = cv2.cvtColor(sample["image"], cv2.COLOR_BGR2RGB)
            image = np.transpose(image, (2, 0, 1))
            sample["image"] = np.ascontiguousarray(image).astype(np.float32)

            if "mask" in sample:
                sample["mask"] = sample["mask"].astype(np.uint8)
                sample["mask"] = np.ascontiguousarray(sample["mask"])[None]

            if "confidence" in sample:
                sample["confidence"] = sample["confidence"].astype(np.uint8)
                sample["confidence"] = np.ascontiguousarray(sample["confidence"])[None]

            if "depth" in sample:
                depth = sample["depth"].astype(np.float32)
                if len(depth.shape) == 2:
                    sample["depth"] = np.ascontiguousarray(depth)[None]
                else:
                    depth = np.transpose(depth, (2, 0, 1))
                    sample["depth"] = np.ascontiguousarray(depth)

        elif len(image.shape) == 4:
            res = []
            for i in range(image.shape[0]):
                image_i = cv2.cvtColor(image[i], cv2.COLOR_BGR2RGB)
                image_i = np.transpose(image[i], (2, 0, 1))
                res.append(np.ascontiguousarray(image_i).astype(np.float32))
            sample["image"] = np.stack(res, axis=0)

            if "mask" in sample:
                res = []
                for i in range(sample["mask"].shape[0]):
                    res.append(sample["mask"][i].astype(np.uint8))
                sample["mask"] = np.stack(res, axis=0)

            if "confidence" in sample:
                res = []
                for i in range(sample["confidence"].shape[0]):
                    res.append(sample["confidence"][i].astype(np.uint8))
                sample["confidence"] = np.stack(res, axis=0)

            if "depth" in sample:
                res = []
                for i in range(sample["depth"].shape[0]):
                    depth = sample["depth"][i].astype(np.float32)
                    if len(depth.shape) == 2:
                        res.append(np.ascontiguousarray(depth)[None])
                    else:
                        depth = np.transpose(depth, (2, 0, 1))
                        res.append(np.ascontiguousarray(depth))
                sample["depth"] = np.stack(res, axis=0)

        if "future_image" in sample and sample["future_image"] is not None:
            sample["future_image"] = self._hwc_to_chw(sample["future_image"])
        return sample


class Crop(object):
    """Random or center crop before PrepareForNet (HWC / THWC)."""

    def __init__(
        self, size, down_scale=7.5, center=False, down_scales=None, down_scale_prob=None
    ):
        self.size = size
        self._center = center
        assert down_scale in [-1, 1, 2, 3.75, 4, 7.5, 8, 14, 15, 28], "Wrong down_scale"
        self._down_scale = down_scale
        local_dict = {
            -1: 1,
            1: 1,
            2: 2,
            3.75: 15,
            4: 4,
            8: 8,
            7.5: 15,
            14: 1,
            28: 1,
            15: 15,
        }
        self._local_dict = local_dict
        self._ensure_round = local_dict[down_scale]
        self._down_scales = down_scales
        self._down_scale_prob = down_scale_prob

    def __str__(self):
        return "RandomCrop: size: {}, down_scale: {}, center: {}".format(
            self.size, self._down_scale, self._center
        )

    def get_bbox(self, sample, h, w, ensure_round):
        if self.size == -1:
            return 0, 0, h, w
        assert h >= self.size and w >= self.size, "Wrong size"
        if self._center:
            h_start = (h - self.size) // 2
            w_start = (w - self.size) // 2
        else:
            h_start = np.random.randint(0, h - self.size + 1)
            w_start = np.random.randint(0, w - self.size + 1)
        if "lowres_depth" in sample:
            h_start = h_start // ensure_round * ensure_round
            w_start = w_start // ensure_round * ensure_round
        h_end = h_start + self.size
        w_end = w_start + self.size
        return h_start, w_start, h_end, w_end

    @staticmethod
    def _crop_spatial(arr, h_start, h_end, w_start, w_end):
        if arr.ndim >= 4:
            return arr[:, h_start:h_end, w_start:w_end]
        return arr[h_start:h_end, w_start:w_end]

    def __call__(self, sample):
        if self._down_scales is not None:
            down_scale = np.random.choice(self._down_scales, p=self._down_scale_prob)
            ensure_round = self._local_dict[down_scale]
        else:
            down_scale = self._down_scale
            ensure_round = self._ensure_round
        if isinstance(self.size, int):
            img = sample["image"]
            h_start, w_start, h_end, w_end = self.get_bbox(
                sample,
                img.shape[-3],
                img.shape[-2],
                ensure_round,
            )
        else:
            h_start, w_start, h_end, w_end = self.size

        sample["image"] = self._crop_spatial(sample["image"], h_start, h_end, w_start, w_end)

        if "depth" in sample:
            sample["depth"] = self._crop_spatial(sample["depth"], h_start, h_end, w_start, w_end)

        if "mask" in sample:
            sample["mask"] = self._crop_spatial(sample["mask"], h_start, h_end, w_start, w_end)

        if "semseg_mask" in sample:
            sample["semseg_mask"] = self._crop_spatial(
                sample["semseg_mask"], h_start, h_end, w_start, w_end
            )

        if "lowres_depth" in sample:
            ds = down_scale
            lh_s, lh_e = int(h_start / ds), int(h_end / ds)
            lw_s, lw_e = int(w_start / ds), int(w_end / ds)
            if isinstance(sample["lowres_depth"], list):
                sample["lowres_depth"] = [
                    self._crop_spatial(img, lh_s, lh_e, lw_s, lw_e)
                    for img in sample["lowres_depth"]
                ]
            else:
                sample["lowres_depth"] = self._crop_spatial(
                    sample["lowres_depth"], lh_s, lh_e, lw_s, lw_e
                )

        if "confidence" in sample:
            ds = down_scale
            sample["confidence"] = self._crop_spatial(
                sample["confidence"],
                int(h_start / ds),
                int(h_end / ds),
                int(w_start / ds),
                int(w_end / ds),
            )

        return sample


class Resize(object):
    """Resize sample to given size (width, height)."""

    def __init__(
        self,
        width=None,
        height=None,
        resize_ratio=None,
        resize_target=False,
        resize_lowres=False,
        keep_aspect_ratio=False,
        ensure_multiple_of=1,
        resize_method="lower_bound",
        image_interpolation_method=cv2.INTER_AREA,
    ):
        self.width = width
        self.height = height
        self.__width = width
        self.__height = height
        self.__resize_ratio = resize_ratio
        assert (width is not None and height is not None) or resize_ratio is not None
        assert (width is None and height is None) or resize_ratio is None

        self.__resize_target = resize_target
        self._resize_lowres = resize_lowres
        self.__keep_aspect_ratio = keep_aspect_ratio
        self.__multiple_of = ensure_multiple_of
        self.__resize_method = resize_method
        self.__image_interpolation_method = image_interpolation_method

    def __str__(self):
        return (
            "Resize: width: {}, height: {}, resize_target: {}, keep_aspect_ratio: {}, "
            "ensure_multiple_of: {}, resize_method: {}"
        ).format(
            self.__width,
            self.__height,
            self.__resize_target,
            self.__keep_aspect_ratio,
            self.__multiple_of,
            self.__resize_method,
        )

    def constrain_to_multiple_of(self, x, min_val=0, max_val=None):
        y = (np.round(x / self.__multiple_of) * self.__multiple_of).astype(int)
        if max_val is not None and y > max_val:
            y = (np.floor(x / self.__multiple_of) * self.__multiple_of).astype(int)

        if y < min_val:
            y = (np.ceil(x / self.__multiple_of) * self.__multiple_of).astype(int)

        return y

    def get_size(self, width, height):
        if self.__resize_ratio is not None:
            __height = int(height * self.__resize_ratio)
            __width = int(width * self.__resize_ratio)
        else:
            __height = self.__height if self.__height is not None else height
            __width = self.__width if self.__width is not None else width
        scale_height = __height / height
        scale_width = __width / width

        if self.__keep_aspect_ratio:
            if self.__resize_method == "lower_bound":
                if scale_width > scale_height:
                    scale_height = scale_width
                else:
                    scale_width = scale_height
            elif self.__resize_method == "upper_bound":
                if scale_width < scale_height:
                    scale_height = scale_width
                else:
                    scale_width = scale_height
            elif self.__resize_method == "minimal":
                if abs(1 - scale_width) < abs(1 - scale_height):
                    scale_height = scale_width
                else:
                    scale_width = scale_height
            else:
                raise ValueError(
                    f"resize_method {self.__resize_method} not implemented"
                )

        if self.__resize_method == "lower_bound":
            new_height = self.constrain_to_multiple_of(
                scale_height * height, min_val=__height
            )
            new_width = self.constrain_to_multiple_of(
                scale_width * width, min_val=__width
            )
        elif self.__resize_method == "upper_bound":
            new_height = self.constrain_to_multiple_of(
                scale_height * height, max_val=__height
            )
            new_width = self.constrain_to_multiple_of(
                scale_width * width, max_val=__width
            )
        elif self.__resize_method == "minimal":
            new_height = self.constrain_to_multiple_of(scale_height * height)
            new_width = self.constrain_to_multiple_of(scale_width * width)
        else:
            raise ValueError(f"resize_method {self.__resize_method} not implemented")

        return (new_width, new_height)

    def __call__(self, sample):
        img = sample["image"]
        if len(img.shape) == 3:
            orig_height, orig_width = img.shape[:2]
        elif len(img.shape) == 4:
            orig_height, orig_width = img.shape[1:3]
        else:
            raise ValueError(f"Unsupported image shape for Resize: {img.shape}")

        width, height = self.get_size(orig_width, orig_height)
        if width == orig_width and height == orig_height:
            return sample
        Log.debug("Resize: {} -> {}".format(sample["image"].shape, (height, width)))

        if len(sample["image"].shape) == 3:
            sample["image"] = cv2.resize(
                sample["image"],
                (width, height),
                interpolation=self.__image_interpolation_method,
            )
        elif len(sample["image"].shape) == 4:
            res = []
            for i in range(sample["image"].shape[0]):
                image_i = cv2.resize(
                    sample["image"][i],
                    (width, height),
                    interpolation=self.__image_interpolation_method,
                )
                res.append(image_i)
            sample["image"] = np.stack(res, axis=0)

        if "future_image" in sample and sample["future_image"] is not None:
            fut = sample["future_image"]
            if fut.ndim == 3:
                sample["future_image"] = cv2.resize(
                    fut, (width, height), interpolation=self.__image_interpolation_method
                )
            elif fut.ndim == 4:
                sample["future_image"] = np.stack(
                    [
                        cv2.resize(
                            fut[i],
                            (width, height),
                            interpolation=self.__image_interpolation_method,
                        )
                        for i in range(fut.shape[0])
                    ],
                    axis=0,
                )
            else:
                raise ValueError(f"Unsupported future_image shape for Resize: {fut.shape}")

        if "camera_intrinsics" in sample:
            sample["camera_intrinsics"] = scale_intrinsics_to_resolution(
                sample["camera_intrinsics"],
                height,
                width,
                orig_height=orig_height,
                orig_width=orig_width,
            )

        if "depth" in sample:
            if len(sample["depth"].shape) == 3:
                sample["depth"] = cv2.resize(
                    sample["depth"],
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )
            elif len(sample["depth"].shape) == 4:
                res = []
                for i in range(sample["depth"].shape[0]):
                    depth_i = cv2.resize(
                        sample["depth"][i],
                        (width, height),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    res.append(depth_i)
                sample["depth"] = np.stack(res, axis=0)

        if self._resize_lowres:
            if "lowres_depth" in sample:
                if isinstance(sample["lowres_depth"], list):
                    sample["lowres_depth"] = [
                        cv2.resize(
                            img,
                            (width, height),
                            interpolation=cv2.INTER_NEAREST,
                        )
                        for img in sample["lowres_depth"]
                    ]
                else:
                    sample["lowres_depth"] = cv2.resize(
                        sample["lowres_depth"],
                        (width, height),
                        interpolation=cv2.INTER_NEAREST,
                    )
            if "lowres_mask" in sample:
                if isinstance(sample["lowres_mask"], list):
                    sample["lowres_mask"] = [
                        cv2.resize(
                            img,
                            (width, height),
                            interpolation=cv2.INTER_NEAREST,
                        )
                        for img in sample["lowres_mask"]
                    ]
                else:
                    sample["lowres_mask"] = cv2.resize(
                        sample["lowres_mask"].astype(np.float32),
                        (width, height),
                        interpolation=cv2.INTER_NEAREST,
                    )

        if self.__resize_target:
            if "disparity" in sample:
                sample["disparity"] = cv2.resize(
                    sample["disparity"],
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )

            if "mesh_depth" in sample:
                sample["mesh_depth"] = cv2.resize(
                    sample["mesh_depth"],
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )

            if "semseg_mask" in sample:
                sample["semseg_mask"] = F.interpolate(
                    torch.from_numpy(sample["semseg_mask"]).float()[None, None, ...],
                    (height, width),
                    mode="nearest",
                ).numpy()[0, 0]

            if "semantic" in sample:
                sample["semantic"] = cv2.resize(
                    sample["semantic"],
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )

            if "mask" in sample:
                sample["mask"] = cv2.resize(
                    sample["mask"].astype(np.float32),
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )

            if "focal" in sample:
                sample["focal"] = sample["focal"] * width / orig_width
        return sample
