"""Resolve evaluation camera poses from a CP-Gen dataset's camera_info.

The fixed-angle evaluation CLI uses CP-Gen's spherical azimuth convention
(``0`` points along +Y), while the dataset metadata labels the ring in the
human-facing convention (``0`` points along +X).  For an angle that exists in
the dataset we return the stored camera pose verbatim; this intentionally
avoids reconstructing a nominally equivalent camera from radius/elevation.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import h5py
import numpy as np
from scipy.spatial.transform import Rotation


_MUJOCO_TO_VISION = np.diag([1.0, -1.0, -1.0])


@dataclass(frozen=True)
class DatasetCameraPose:
    camera_name: str
    dataset_azimuth: float
    position: np.ndarray
    quaternion_wxyz: np.ndarray
    fovy: float
    source: str = "dataset-camera-info-exact"


def _decode_json(value: Any, *, label: str) -> Any:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {label}") from exc
    return value


def _normalise_degrees(angle: float) -> float:
    result = (float(angle) + 180.0) % 360.0 - 180.0
    # Keep +180 as +180 so it matches the canonical CP-Gen metadata label.
    if np.isclose(result, -180.0) and float(angle) > 0:
        return 180.0
    return result


def sphere_to_dataset_azimuth(sphere_azimuth: float) -> float:
    """Convert evaluator / CP-Gen spherical azimuth to dataset ring label."""
    return _normalise_degrees(90.0 - float(sphere_azimuth))


def _camera_azimuths(data_group: h5py.Group, camera_names: list[str]) -> list[float]:
    raw = data_group.attrs.get("multiview_azimuths_deg")
    if raw is not None:
        raw = _decode_json(raw, label="data.attrs['multiview_azimuths_deg']")
        values = np.asarray(raw, dtype=np.float64).reshape(-1).tolist()
        if len(values) == len(camera_names):
            return [float(x) for x in values]

    # Legacy CP-Gen 21/24-view files did not always store the ring attribute.
    # Their stable view-number contract is documented by the generation guide.
    indices = [int(name.rsplit("_", 1)[1]) for name in camera_names]
    canonical = [-150.0 + 15.0 * i for i in range(21)] + [165.0, -165.0, 180.0]
    if indices and max(indices) < len(canonical):
        return [canonical[i] for i in indices]
    raise KeyError(
        "Dataset is missing data.attrs['multiview_azimuths_deg'] and its "
        "third_view indices do not follow the canonical CP-Gen 21/24-view schema"
    )


def _fovy_from_intrinsics(intrinsics: np.ndarray) -> float:
    if intrinsics.shape != (3, 3):
        raise ValueError(f"Expected 3x3 camera intrinsics, got {intrinsics.shape}")
    fy = float(intrinsics[1, 1])
    image_height = 2.0 * float(intrinsics[1, 2])
    if fy <= 0 or image_height <= 0:
        raise ValueError(f"Invalid camera intrinsics fy={fy}, height={image_height}")
    return float(np.degrees(2.0 * np.arctan(image_height / (2.0 * fy))))


def resolve_fixed_camera_from_dataset(
    dataset_path: str,
    sphere_azimuth: float,
    *,
    tolerance_deg: float = 1e-4,
) -> DatasetCameraPose:
    """Return the exact stored pose for a fixed spherical evaluation angle.

    Raises instead of silently falling back to a generic camera when the
    requested angle is absent.  Callers evaluating an interpolation angle must
    explicitly implement and label a dataset-derived interpolation policy.
    """
    requested = sphere_to_dataset_azimuth(sphere_azimuth)
    with h5py.File(dataset_path, "r") as h5:
        data = h5["data"]
        if not data.keys():
            raise ValueError(f"Dataset has no demos: {dataset_path}")
        demo_name = "demo_0" if "demo_0" in data else sorted(data.keys())[0]
        demo = data[demo_name]
        if "camera_info" not in demo.attrs:
            raise KeyError(f"{dataset_path}:{demo_name} missing camera_info")
        camera_info = _decode_json(
            demo.attrs["camera_info"], label=f"{demo_name}.attrs['camera_info']"
        )
        if not isinstance(camera_info, dict):
            raise TypeError(f"Expected camera_info mapping, got {type(camera_info)}")

        camera_names = sorted(
            (name for name in camera_info if name.startswith("third_view_")),
            key=lambda name: int(name.rsplit("_", 1)[1]),
        )
        if not camera_names:
            raise KeyError(f"{dataset_path}:{demo_name} camera_info has no third_view_N entries")
        azimuths = _camera_azimuths(data, camera_names)

        matches = [
            i
            for i, azimuth in enumerate(azimuths)
            if abs(_normalise_degrees(azimuth - requested)) <= tolerance_deg
        ]
        if len(matches) != 1:
            available = ", ".join(f"{x:g}" for x in azimuths)
            raise KeyError(
                f"No unique camera_info view for sphere azimuth {sphere_azimuth:g} "
                f"(dataset azimuth {requested:g}); available dataset azimuths: {available}"
            )

        camera_name = camera_names[matches[0]]
        entry = camera_info[camera_name]
        extrinsics = np.asarray(entry["extrinsics"], dtype=np.float64)
        intrinsics = np.asarray(entry["intrinsics"], dtype=np.float64)
        if extrinsics.shape != (4, 4):
            raise ValueError(f"Expected 4x4 {camera_name} extrinsics, got {extrinsics.shape}")

        # CP-Gen / robosuite camera_info stores camera-to-world in the vision
        # frame. Undo the MuJoCo-to-vision axis correction to recover the exact
        # sim.model.cam_quat convention.
        rotation_mujoco = extrinsics[:3, :3] @ _MUJOCO_TO_VISION
        quat_xyzw = Rotation.from_matrix(rotation_mujoco).as_quat()
        quat_wxyz = quat_xyzw[[3, 0, 1, 2]]
        if quat_wxyz[0] < 0:
            quat_wxyz = -quat_wxyz

        return DatasetCameraPose(
            camera_name=camera_name,
            dataset_azimuth=float(azimuths[matches[0]]),
            position=extrinsics[:3, 3].copy(),
            quaternion_wxyz=quat_wxyz,
            fovy=_fovy_from_intrinsics(intrinsics),
        )


def enrich_fixed_spherical_config(
    dataset_path: str,
    multi_view_config: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, DatasetCameraPose | None]:
    """Add an exact dataset pose to a fixed spherical multi-view config."""
    if not multi_view_config:
        return multi_view_config, None
    config = dict(multi_view_config)
    if config.get("source_pool") != "spherical":
        return config, None
    sphere_azimuth = config.get("sphere_fixed_azimuth")
    if sphere_azimuth is None:
        return config, None

    camera_info_dataset_path = str(
        config.get("camera_info_dataset_path")
        or os.environ.get("SAPOLICY_EVAL_CAMERA_INFO_DATASET")
        or dataset_path
    )
    pose = resolve_fixed_camera_from_dataset(camera_info_dataset_path, float(sphere_azimuth))
    config["fixed_camera_position"] = pose.position.tolist()
    config["fixed_camera_quaternion"] = pose.quaternion_wxyz.tolist()
    config["fixed_camera_fovy"] = pose.fovy
    config["fixed_camera_source"] = f"{pose.source}:{pose.camera_name}"
    config["fixed_camera_dataset_azimuth"] = pose.dataset_azimuth
    config["fixed_camera_dataset_path"] = camera_info_dataset_path
    return config, pose
