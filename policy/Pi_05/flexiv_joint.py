"""Deployment transforms for absolute 7+1, left/right Flexiv joint policies.

This is an explicit profile, not an ALOHA sign/gripper conversion. Training
must use these same image slots, quantile normalization and absolute targets.
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import logging
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
from PIL import Image

PROFILE = "flexiv_joint"
ACTION_DIM = 16
PREPROCESSING_CONTRACT = "stack_cube_joint_3cam_v1"


class SingleFrameCompression:
    """Optional deployment RGB compression before the Flexiv training resize.

    Each call encodes an independent H.264 I-frame, without temporal video state.
    This models collection compression, not camera acquisition or transport.
    """

    @staticmethod
    def parameters(options):
        if not isinstance(options, dict):
            raise ValueError("Pi05 image_processing must be a mapping")
        defaults = {
            "type": "jpeg_h264_single_frame", "jpeg_quality": 80,
            "crf": 20, "fps": 30, "gop_size": 30, "thread_count": 2,
            "preset": "ultrafast", "tune": "zerolatency", "pix_fmt": "yuv420p",
        }
        unknown = options.keys() - defaults.keys()
        if unknown:
            raise ValueError(f"Unknown Pi05 image_processing fields: {sorted(unknown)}")
        values = defaults | options
        for name in ("type", "preset", "tune", "pix_fmt"):
            if values[name] != defaults[name]:
                raise ValueError(f"image_processing.{name} must be {defaults[name]!r}")
        for name, low, high in (
            ("jpeg_quality", 1, 100), ("crf", 0, 51), ("fps", 1, 240),
            ("gop_size", 1, 1000), ("thread_count", 1, 16),
        ):
            if type(values[name]) is not int or not low <= values[name] <= high:
                raise ValueError(f"image_processing.{name} must be an integer in [{low},{high}]")
        return values

    @classmethod
    def from_config(cls, config):
        options = config.get("image_processing")
        return None if options is None else cls(options)

    def __init__(self, options):
        self.options = self.parameters(options)
        try:
            import av
        except ImportError as error:
            raise RuntimeError("Flexiv image compression requires PyAV in the Pi05 model environment") from error
        av.codec.Codec("libx264", "w")
        av.codec.Codec("h264", "r")
        self.av = av
        self.last_timings = {}
        logging.getLogger(__name__).info(
            "flexiv_image_processing_config parameters=%s pyav=%s ffmpeg=%s",
            self.options, av.__version__, av.library_versions,
        )

    def process(self, rgb, *, camera="image"):
        from XPolicyLab.utils.process_data import decode_image_bit, encode_image_bit

        self.last_timings = {}
        rgb = np.asarray(rgb)
        if (rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3
                or min(rgb.shape[:2]) <= 0 or any(n % 2 for n in rgb.shape[:2])):
            raise ValueError(f"{camera}: JPEG/H.264 requires even-sized HxWx3 uint8 RGB")
        start = time.perf_counter_ns()
        opts, av = self.options, self.av
        # Shared helpers own JPEG markers and RGB conventions. Incoming images
        # are already decoded; only this newly encoded temporary buffer is decoded.
        jpeg = encode_image_bit(rgb, quality=opts["jpeg_quality"])
        jpeg_encoded = time.perf_counter_ns()
        decoded = decode_image_bit(jpeg)
        jpeg_decoded = time.perf_counter_ns()
        buffer = io.BytesIO()
        with av.open(buffer, "w", format="mp4") as writer:
            stream = writer.add_stream("libx264", rate=opts["fps"])
            stream.width, stream.height = rgb.shape[1], rgb.shape[0]
            stream.pix_fmt = opts["pix_fmt"]
            stream.options = {"crf": str(opts["crf"]), "preset": opts["preset"], "tune": opts["tune"]}
            stream.codec_context.thread_count = opts["thread_count"]
            stream.codec_context.gop_size = opts["gop_size"]
            # Match the collection encoder's pixel format at the media boundary.
            # Public arrays and XPolicyLab's JPEG helpers remain RGB throughout.
            frame = av.VideoFrame.from_ndarray(decoded, format="rgb24").reformat(format="bgr24")
            frame.pts, frame.time_base = 0, Fraction(1, opts["fps"])
            for packet in stream.encode(frame):
                writer.mux(packet)
            for packet in stream.encode():
                writer.mux(packet)
        h264_encoded = time.perf_counter_ns()
        with av.open(io.BytesIO(buffer.getvalue()), "r") as reader:
            frames = list(reader.decode(video=0))
            if len(frames) != 1:
                raise RuntimeError("Single-frame H.264 round trip did not return exactly one frame")
            output = frames[0].to_ndarray(format="rgb24")
        if output.shape != rgb.shape:
            raise RuntimeError(f"{camera}: compression changed the image shape")
        output.setflags(write=False)
        finished = time.perf_counter_ns()
        self.last_timings = {
            "jpeg_encode_ms": (jpeg_encoded - start) / 1e6,
            "jpeg_decode_ms": (jpeg_decoded - jpeg_encoded) / 1e6,
            "h264_encode_ms": (h264_encoded - jpeg_decoded) / 1e6,
            "h264_decode_ms": (finished - h264_encoded) / 1e6,
            "total_ms": (finished - start) / 1e6,
        }
        logging.getLogger(__name__).info(
            "flexiv_image_processing camera=%s input_sha256=%s timings_ms=%s",
            camera, hashlib.sha256(rgb.tobytes()).hexdigest(), self.last_timings,
        )
        return output

# Submitted stack-cube training source, src/pass_ball/dataset.py (2026-11-02):
# sha256 13db9083da25eacef2fc6d9e28bddc3ca09de0b6ea1ee78fb10f73de84ef0e57.
# Resize before OpenPI's model transforms, exactly as its native dataset does.
def resize_training_rgb(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    ratio = max(width / 224, height / 224)
    resized = Image.fromarray(image).resize(
        (max(1, int(round(width / ratio))), max(1, int(round(height / ratio)))),
        Image.Resampling.BILINEAR,
    )
    result = Image.new("RGB", (224, 224))
    result.paste(resized, ((224 - resized.width) // 2, (224 - resized.height) // 2))
    return np.asarray(result)


def validate_artifacts(config: dict) -> dict:
    if config.get("image_processing") is not None:
        SingleFrameCompression.parameters(config["image_processing"])
    required = {
        "policy_name": "Pi_05",
        "observation_profile": PROFILE,
        "preprocessing_contract": PREPROCESSING_CONTRACT,
        "action_type": "joint",
        "action_semantics": "absolute_joint_position",
        "use_delta_joint_actions": False,
        "robot_action_dim_info": {"arm_dim": [7, 7], "ee_dim": [1, 1]},
    }
    for key, expected in required.items():
        if config.get(key) != expected:
            raise ValueError(f"Flexiv joint profile requires {key}={expected!r}")
    for key in ("action_horizon", "num_steps"):
        value = config.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{key} must be a positive integer")
    checkpoint = Path(config["model_path"]).expanduser().resolve()
    if not (checkpoint / "params/_METADATA").is_file():
        raise FileNotFoundError(f"Orbax Pi05 params metadata missing: {checkpoint}")
    if checkpoint.name != str(config["checkpoint_num"]):
        raise ValueError("checkpoint directory step does not match checkpoint_num")
    stats = Path(config["norm_stats_path"]).expanduser().resolve() / "norm_stats.json"
    digest = hashlib.sha256(stats.read_bytes()).hexdigest()
    if digest != config["norm_stats_sha256"] or config["norm_stats_source"] != "sha256_" + digest:
        raise ValueError("normalization checksum does not match the Flexiv recipe")
    norm = json.loads(stats.read_text())["norm_stats"]
    for name in ("state", "actions"):
        for field in ("mean", "std", "q01", "q99"):
            value = np.asarray(norm[name][field], dtype=float)
            if value.shape != (ACTION_DIM,) or not np.isfinite(value).all():
                raise ValueError(f"normalization {name}.{field} must be finite (16,)")
        if np.any(np.asarray(norm[name]["std"]) <= 0):
            raise ValueError(f"normalization {name}.std must be positive")
        if np.any(np.asarray(norm[name]["q99"]) <= np.asarray(norm[name]["q01"])):
            raise ValueError(f"normalization {name} quantiles must be ordered")
    return {
        "contract_status": "ready",
        "preprocessing_contract": PREPROCESSING_CONTRACT,
        "checkpoint": str(checkpoint),
        "norm_stats": str(stats),
        "norm_stats_sha256": digest,
        "native_action_dim": ACTION_DIM,
        "action_horizon": config["action_horizon"],
        "action_semantics": config["action_semantics"],
        "sampling_modes": ["default", "rtc"],
        "inference_status": "not_run_by_artifact_check",
    }


@dataclasses.dataclass(frozen=True)
class FlexivInputs:
    compression: SingleFrameCompression | None = None

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["observation/state"], dtype=np.float32)
        if state.shape != (ACTION_DIM,) or not np.isfinite(state).all():
            raise ValueError("Flexiv state must be finite (16,) in left/right 7+1 order")
        image = {}
        for target, source in (
            ("base_0_rgb", "observation/image"),
            ("left_wrist_0_rgb", "observation/left_wrist"),
            ("right_wrist_0_rgb", "observation/right_wrist"),
        ):
            rgb = np.asarray(data[source])
            if (
                rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8
                or min(rgb.shape[:2]) < 1
            ):
                raise ValueError(f"{source} must be an HWC uint8 RGB image")
            if self.compression is not None:
                rgb = self.compression.process(rgb, camera=source)
            image[target] = resize_training_rgb(rgb)
        result = {
            "state": state,
            "image": image,
            "image_mask": {name: np.bool_(True) for name in image},
        }
        if "prompt" in data:
            result["prompt"] = str(data["prompt"])
        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
            if actions.ndim != 2 or actions.shape[-1] != ACTION_DIM or not np.isfinite(actions).all():
                raise ValueError("Flexiv actions must be finite (horizon,16)")
            result["actions"] = actions
        return result


@dataclasses.dataclass(frozen=True)
class FlexivOutputs:
    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        if actions.ndim != 2 or actions.shape[-1] < ACTION_DIM or not np.isfinite(actions).all():
            raise ValueError("Flexiv model output must be a finite horizon-by-action matrix")
        return {"actions": actions[..., :ACTION_DIM]}


def make_train_config(config: dict):
    """Rebuild the explicit deployment pipeline without training-site paths."""
    from openpi import transforms
    from openpi.models.pi0_config import Pi0Config
    from openpi.training import config as oc

    compression = SingleFrameCompression.from_config(config)

    @dataclasses.dataclass(frozen=True)
    class FlexivDataConfig(oc.DataConfigFactory):
        def create(self, assets_dirs, model_config):
            return dataclasses.replace(
                self.create_base_config(assets_dirs, model_config),
                data_transforms=transforms.Group(
                    inputs=(FlexivInputs(compression),), outputs=(FlexivOutputs(),)
                ),
                model_transforms=oc.ModelTransformFactory()(model_config),
                action_sequence_keys=("action",),
                prompt_from_task=False,
            )

    stats = Path(config["norm_stats_path"])
    return oc.TrainConfig(
        name=config["train_config_name"],
        model=Pi0Config(pi05=True, action_dim=32, action_horizon=config["action_horizon"]),
        data=FlexivDataConfig(
            repo_id=config["repo_id"],
            assets=oc.AssetsConfig(assets_dir=str(stats.parent), asset_id=stats.name),
        ),
        policy_metadata={
            "action_semantics": "absolute_joint_position", "native_action_dim": ACTION_DIM,
            "image_processing": None if compression is None else dict(config["image_processing"]),
            "image_processing_effective": None if compression is None else compression.options,
        },
    )
