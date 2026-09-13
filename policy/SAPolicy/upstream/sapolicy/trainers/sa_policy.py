import os
from os.path import join
import numpy as np
import imageio
import torch
import cv2
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Dict, List, Optional, Tuple, Any
from omegaconf import OmegaConf
from datetime import datetime
import gc
import copy
from omegaconf.listconfig import ListConfig
from omegaconf.dictconfig import DictConfig
from hydra.utils import instantiate

from sapolicy.trainers.base import BaseModel
from sapolicy.trainers.ema import EMAModel
from sapolicy.logger import Log
from sapolicy.models.utils.rotation import (
    quaternion_to_matrix,
    rotation_6d_to_matrix,
    matrix_to_rotation_6d,
)

matplotlib.use('Agg')

def create_gt_heatmap(gt_coords, h, w, sigma=0.1):
    gt_u, gt_v = gt_coords[0], gt_coords[1]

    y_coords, x_coords = np.meshgrid(
        np.arange(h, dtype=np.float32),
        np.arange(w, dtype=np.float32),
        indexing='ij'
    )

    y_coords /= (h-1)
    x_coords /= (w-1)
    heatmap = np.exp(-((x_coords - gt_u)**2 + (y_coords - gt_v)**2) / (2.0 * sigma**2))

    return heatmap

def scale_intrinsics(K, orig_size, new_size):
    """Scale camera intrinsics matrix when image is resized.

    Args:
        K: [3, 3] - Original intrinsics matrix
        orig_size: (H, W) - Original image dimensions the intrinsics correspond to
        new_size: (H, W) - Current (resized) image dimensions

    Returns:
        K_scaled: [3, 3] - Scaled intrinsics matrix
    """
    K_scaled = K.copy()
    sy = new_size[0] / orig_size[0]
    sx = new_size[1] / orig_size[1]
    K_scaled[0, 0] *= sx  # fx
    K_scaled[0, 2] *= sx  # cx
    K_scaled[1, 1] *= sy  # fy
    K_scaled[1, 2] *= sy  # cy
    return K_scaled


def project_3d_pose_to_2d(tcp_pos, tcp_pixel_coords, tcp_orn, camera_intrinsics, image_shape):
    """
    Project 3D TCP pose to 2D image coordinates for visualization.

    Args:
        tcp_pos: [3] - 3D position (x, y, z) in camera frame (NumPy array)
        tcp_pixel_coords: [3] - 3D pixel coordinates (u, v, depth)
        tcp_orn: [3, 3] or [9] - Rotation matrix (NumPy array)
        camera_intrinsics: [3, 3] - Camera intrinsics matrix (NumPy array)
        image_shape: (H, W) - Image dimensions

    Returns:
        tcp_pixel: [2] - (u, v) pixel coordinates
        axis_endpoints_2d: [3, 2] - 2D projections of X, Y, Z axis endpoints
    """
    assert tcp_pos is not None or tcp_pixel_coords is not None, "tcp_pos or tcp_pixel_coords must be provided"

    H, W = image_shape

    # Ensure tcp_pos is 1D array
    if tcp_pos is None:
        u, v, depth = tcp_pixel_coords
        p = np.array([u, v, 1.0], dtype=np.float32)
        K_inv = np.linalg.inv(camera_intrinsics)
        tcp_pos = depth * (K_inv @ p)

    tcp_pos = np.asarray(tcp_pos).flatten()

    # Ensure tcp_orn is [3, 3]
    tcp_orn = np.asarray(tcp_orn)
    if tcp_orn.shape == (9,):
        tcp_orn = tcp_orn.reshape(3, 3)

    # Ensure camera_intrinsics is [3, 3]
    camera_intrinsics = np.asarray(camera_intrinsics)
    if camera_intrinsics.shape == (9,):
        camera_intrinsics = camera_intrinsics.reshape(3, 3)

    # Project TCP position to 2D
    pixel_coords = None
    if tcp_pos is not None:
        pixel_coords = camera_intrinsics @ tcp_pos.reshape(3, 1)
        pixel_coords = pixel_coords[:2, 0] / pixel_coords[2, 0]
        u, v = pixel_coords[0], pixel_coords[1]

        if not (0 <= u < W and 0 <= v < H):
            Log.warn(f"TCP position is out of image bounds: {u}, {v}, image shape: {H} x {W}")

    # Project orientation axes (visualize as coordinate frame)
    axis_length = 1
    axis_endpoints_2d = []

    for i in range(3):
        # Get axis direction from rotation matrix column
        axis_dir = tcp_orn[:, i] * axis_length
        # Endpoint in 3D
        endpoint_3d = tcp_pos + axis_dir
        if endpoint_3d[2] <= 0:
            if (tcp_pos[2] > 0) and (axis_dir[2] < 0):
                scale = np.abs(tcp_pos[2]) / np.abs(axis_dir[2])
                axis_dir = axis_dir * scale
                endpoint_3d = tcp_pos + axis_dir * 0.99 # make sure in front of the camera
        # Project to 2D
        endpoint_2d = np.array([0.0, 0.0])
        if endpoint_3d[2] > 0:
            endpoint_homogeneous = camera_intrinsics @ endpoint_3d
            endpoint_homogeneous = endpoint_homogeneous / endpoint_homogeneous[2]
            dir_x = endpoint_homogeneous[0] - u
            dir_y = endpoint_homogeneous[1] - v
            dir_vec = np.array([dir_x, dir_y])
            endpoint_2d = np.array([u + (dir_vec[0] / np.linalg.norm(dir_vec)) * 30, v + (dir_vec[1] / np.linalg.norm(dir_vec)) * 30]).astype(np.int32)

        axis_endpoints_2d.append(endpoint_2d)

    axis_endpoints_2d = np.array(axis_endpoints_2d)  # [3, 2]

    return pixel_coords, axis_endpoints_2d

class SAPolicyModel(BaseModel):
    def __init__(
        self,
        pipeline,  # The pipeline is the model itself
        optimizer,  # The optimizer is the optimizer used to train the model
        lr_table,  # The lr_table is the learning rate table
        output_dir: str,
        output_tag: str = "default",
        clear_output_dir: bool = False,
        scheduler_cfg=None,  # The scheduler_cfg is the scheduler configuration
        ignored_weights_prefix=["pipeline.text_encoder", "pipeline.vae"],
        # TCP Prediction specific parameters
        save_tcp_predictions=True,  # Whether to save TCP prediction results
        save_tcp_visualizations=True,  # Whether to save TCP visualization images
        save_tcp_pointclouds=True,  # Whether to save TCP pointcloud visualizations
        save_heatmap_visualizations=True,  # Whether to save attention heatmaps
        save_training_progress=True,  # Whether to save training progress images
        save_training_progress_interval=1000,  # How often to save training progress images
        # Visualization parameters
        tcp_marker_size=10,
        direction_arrow_scale=50.0,
        heatmap_alpha=0.6,
        concat_axis=1,
        # Evaluation parameters
        pixel_error_threshold=5.0,  # pixels
        depth_error_threshold=0.05,  # meters
        direction_error_threshold=0.1,  # radians
        min_depth=0.1,  # meters
        max_depth=10.0,  # meters
        use_ema=False,
        ema=None,
        train_log_sync_dist=False,  # per-step train logs: True = barrier + all-reduce per metric per step (18/step); False = rank-0 values
        train_log_prog_bar=False,  # prog_bar=True makes Lightning .item() every such metric every step (host sync); keep False on the cluster
        # Joint two-loader training: loss_mode for the action dataloader batch.
        # Default "both" matches historical tcpALL+action recipes. Use "action"
        # when TCP supervision must come only from the TCP loader (e.g. cross-task
        # TCP transfer) so the action task does not also get within-task TCP loss.
        joint_action_loss_mode: str = "both",
        **kwargs,
    ):
        super().__init__(
            pipeline,
            optimizer,
            lr_table,
            output_dir,
            output_tag,
            clear_output_dir,
            scheduler_cfg,
            ignored_weights_prefix,
            **kwargs,
        )

        for p in self.pipeline.parameters():
            if not p.is_contiguous():
                p.data = p.data.contiguous()

        # Hydra CLI passes booleans as strings ("false"); convert to proper bool.
        if isinstance(use_ema, str):
            from distutils.util import strtobool
            use_ema = bool(strtobool(use_ema))
        self.use_ema = bool(use_ema)
        self.ema_pipeline = None
        self.ema = ema
        def _as_bool(v):  # Hydra CLI passes 'false' as a str; bool('false') is True
            return v.strip().lower() in ('1', 'true', 'yes', 'y', 't') if isinstance(v, str) else bool(v)
        self.train_log_sync_dist = _as_bool(train_log_sync_dist)
        self.train_log_prog_bar = _as_bool(train_log_prog_bar)
        if joint_action_loss_mode not in ("both", "action"):
            raise ValueError(
                f"joint_action_loss_mode={joint_action_loss_mode!r}; expected 'both' or 'action'."
            )
        self.joint_action_loss_mode = str(joint_action_loss_mode)

        # TCP prediction parameters
        self._save_tcp_predictions = save_tcp_predictions
        self._save_tcp_visualizations = save_tcp_visualizations
        self._save_tcp_pointclouds = save_tcp_pointclouds
        self._save_heatmap_visualizations = save_heatmap_visualizations
        self._save_training_progress = save_training_progress
        self._save_training_progress_interval = save_training_progress_interval

        # Dynamic camera parameters (updated from batch data)
        self._current_focal = {}
        self._current_image_width = {}
        self._current_image_height = {}

        # Visualization parameters
        self._tcp_marker_size = tcp_marker_size
        self._direction_arrow_scale = direction_arrow_scale
        self._heatmap_alpha = heatmap_alpha
        self._concat_axis = concat_axis

        # Evaluation parameters
        self._pixel_error_threshold = pixel_error_threshold
        self._depth_error_threshold = depth_error_threshold
        self._direction_error_threshold = direction_error_threshold
        self._min_depth = float(min_depth)
        self._max_depth = float(max_depth)

        self._default_image_width = 84
        self._default_image_height = 84

        # Initialize training step counter for progress tracking
        self.training_step_count = 0
        self.validation_step_count = 0

        Log.info("TCP VLA Model - Results will be saved to: {}".format(self.output_dir))
        Log.info(f"TCP Visualization Settings: marker_size={tcp_marker_size}, arrow_scale={direction_arrow_scale}")
        Log.info(f"TCP Evaluation Thresholds: pixel={pixel_error_threshold}px, depth={depth_error_threshold}m")

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        """TCP prediction step with comprehensive visualization"""
        # Extract camera parameters from batch data
        self._extract_camera_parameters_from_batch(batch)

        output = self.pipeline.forward_test(batch)

        # # Save TCP predictions if enabled
        # if self._save_tcp_predictions:
        #     self.save_tcp_predictions(output, batch, "tcp_predictions")

        # # Save TCP visualizations on RGB images
        # NOTE: Disabled for now because it requires 2D pixel-frame outputs
        # (output["tcp_pixel_coords"] / output["tcp_heatmap"]).
        # if self._save_tcp_visualizations:
        #     self.save_tcp_visualizations_rgb(output, batch, "tcp_vis_rgb")

        # Add camera parameters to output for reference
        output.update({
            'camera_focal': self._current_focal,
            'camera_width': self._current_image_width,
            'camera_height': self._current_image_height
        })

        return output

    def _extract_camera_parameters_from_batch(self, batch):
        """Extract camera parameters from batch data"""
        try:
            for camera_name in batch["observation"]["image"].keys():
                # Try to extract focal length from camera intrinsics
                if 'camera_intrinsics' in batch and batch['camera_intrinsics'] is not None:
                # Camera intrinsics are typically 3x3 matrix: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
                    intrinsics = batch['camera_intrinsics'][camera_name]
                    if isinstance(intrinsics, (list, tuple)) and len(intrinsics) > 0:
                        # Take the first sample in the batch
                        intrinsics_matrix = intrinsics[0]
                        if hasattr(intrinsics_matrix, 'cpu'):
                            intrinsics_matrix = intrinsics_matrix.cpu().numpy()

                        if intrinsics_matrix.shape == (3, 3):
                            # Extract fx, fy, cx, cy from intrinsics matrix
                            fx, fy = intrinsics_matrix[0, 0], intrinsics_matrix[1, 1]
                            cx, cy = intrinsics_matrix[0, 2], intrinsics_matrix[1, 2]

                            # Use average of fx and fy as focal length
                            self._current_focal[camera_name] = float((fx + fy) / 2.0)

                            # Estimate image dimensions from principal point (cx, cy usually at center)
                            self._current_image_width[camera_name] = int(cx * 2) if cx > 0 else self._default_image_width
                            self._current_image_height[camera_name] = int(cy * 2) if cy > 0 else self._default_image_height

                            Log.debug(f"Extracted camera params: focal={self._current_focal[camera_name]:.3f}, "
                                    f"size=({self._current_image_width[camera_name]}x{self._current_image_height[camera_name]})")
                            return

                # # Try to extract from individual focal length field
                # elif 'focal_length' in batch and batch['focal_length'][camera_name] is not None:
                #     focal_values = batch['focal_length'][camera_name]
                #     if isinstance(focal_values, (list, tuple)) and len(focal_values) > 0:
                #         self._current_focal[camera_name] = float(focal_values[0])
                #     elif hasattr(focal_values, 'item'):
                #         self._current_focal[camera_name] = float(focal_values.item())
                #     else:
                #         self._current_focal[camera_name] = float(focal_values)

                #     Log.debug(f"Extracted focal length from batch: {self._current_focal[camera_name]:.3f}")

                # Try to extract image dimensions from the actual image tensor
                elif 'image' in batch and batch['image'] is not None:
                    image_tensor = batch['image'][camera_name]
                    if len(image_tensor.shape) >= 3:  # [B, C, H, W] or [C, H, W]
                        if len(image_tensor.shape) == 4:  # Batch dimension present
                            _, _, height, width = image_tensor.shape
                        else:  # No batch dimension
                            _, height, width = image_tensor.shape

                        self._current_image_height[camera_name] = int(height)
                        self._current_image_width[camera_name] = int(width)

                        Log.debug(f"Extracted image dimensions: {self._current_image_width}x{self._current_image_height}")

                # Check for explicit focal field in various formats
                # for focal_key in ['focal', 'camera_focal', 'intrinsic_focal']:
                #     if focal_key in batch and batch[focal_key] is not None:
                #         focal_val = batch[focal_key]
                #         if isinstance(focal_val, (list, tuple)) and len(focal_val) > 0:
                #             self._current_focal[camera_name] = float(focal_val[0])
                #         elif hasattr(focal_val, 'item'):
                #             self._current_focal[camera_name] = float(focal_val.item())
                #         else:
                #             self._current_focal[camera_name] = float(focal_val)
                #         break

        except Exception as e:
            Log.warn(f"Failed to extract camera parameters from batch: {e}")
            Log.warn("Using default camera parameters")

    def training_step(self, batch, batch_idx):
        """TCP training step with visualization"""
        self.training_step_count += 1
        if hasattr(self.pipeline, "set_tcp_kv_curriculum_step"):
            self.pipeline.set_tcp_kv_curriculum_step(int(self.global_step))

        # Forward pass and loss computation
        joint_batches = self._split_joint_training_batch(batch)
        if joint_batches is not None:
            tcp_batch, action_batch = joint_batches
            tcp_loss_dict = self.pipeline.forward_train_batch(tcp_batch, loss_mode="tcp")
            action_loss_dict = self.pipeline.forward_train_batch(
                action_batch, loss_mode=self.joint_action_loss_mode
            )
            loss_dict = {"loss": tcp_loss_dict["loss"] + action_loss_dict["loss"]}
            loss_dict.update({f"joint_tcp/{k}": v for k, v in tcp_loss_dict.items()})
            loss_dict.update({f"joint_action/{k}": v for k, v in action_loss_dict.items()})
            batch_size = self._batch_size_from_batch(tcp_batch) + self._batch_size_from_batch(action_batch)
        else:
            if isinstance(batch, list):
                batch = batch[0] # Legacy multiple-loader behavior.
            loss_dict = self.pipeline.forward_train_batch(batch)
            batch_size = self._batch_size_from_batch(batch)

        # Log training losses. Progress bar only shows a short allowlist so
        # aux zeros (tcp/consistency) do not bury action / future-video metrics.
        _prog_bar_keys = {
            "loss",
            "loss_action_flow",
            "loss_future_video",
            "loss_tcp",
            "joint_tcp/loss",
            "joint_action/loss",
            "joint_action/loss_action_flow",
            "joint_action/loss_future_video",
        }
        for key, value in loss_dict.items():
            if isinstance(value, torch.Tensor):
                self.log(
                    f"train/{key}",
                    value.detach(),
                    on_step=True,
                    on_epoch=True,
                    prog_bar=self.train_log_prog_bar and key in _prog_bar_keys,
                    logger=True,
                    batch_size=batch_size,
                    sync_dist=self.train_log_sync_dist,
                )

        # Periodic stdout logging (visible in Slurm .out files)
        if self.training_step_count % 50 == 1 and self.global_rank == 0:
            # Prefer a short readable line; always include video metrics when present.
            prefer = (
                "loss",
                "loss_action_flow",
                "loss_future_video",
                "loss_future_video_baseline_zero",
                "loss_future_video_vs_baseline",
                "loss_future_video_h1",
                "loss_future_video_h2",
                "loss_future_video_h4",
                "loss_tcp",
                "joint_action/loss",
                "joint_action/loss_action_flow",
                "joint_action/loss_future_video",
                "joint_action/loss_future_video_baseline_zero",
                "joint_action/loss_future_video_vs_baseline",
                "joint_tcp/loss",
            )
            parts = []
            for k in prefer:
                v = loss_dict.get(k)
                if isinstance(v, torch.Tensor):
                    parts.append(f"{k}={v.item():.4f}")
            if not parts:
                parts = [
                    f"{k}={v.item():.4f}"
                    for k, v in loss_dict.items()
                    if isinstance(v, torch.Tensor)
                ]
            print(f"[step {self.training_step_count}] {' | '.join(parts)}", flush=True)

        sch_for_lr = self.lr_schedulers()
        if sch_for_lr is not None:
            self.log("train/lr", torch.full((), float(sch_for_lr.get_last_lr()[0]), device=self.device), on_step=True, on_epoch=True, prog_bar=self.train_log_prog_bar, logger=True, batch_size=batch_size, sync_dist=self.train_log_sync_dist)  # device scalar: a Python float would be a pageable H2D copy = stream sync

        sch = self.lr_schedulers()
        if sch is not None:
            sch.step()

        if self.use_ema and isinstance(self.ema, (ListConfig, DictConfig)):
            print("Initializing EMA pipeline")
            # Clear intermediate loss tensors before deepcopy — non-leaf tensors
            # cause RuntimeError during deepcopy
            if hasattr(self.pipeline, 'action_head') and self.pipeline.action_head is not None:
                # Clear per-camera spatial tokens stored during forward (R87+)
                if hasattr(self.pipeline.action_head, '_per_camera_spatial_tokens'):
                    self.pipeline.action_head._per_camera_spatial_tokens = {}
            self.ema_pipeline = copy.deepcopy(self.pipeline)
            self.ema_pipeline.eval()
            self.ema_pipeline.requires_grad_(False)
            self.ema_pipeline.to(self.device)

            self.ema = instantiate(self.ema, model=self.ema_pipeline)

        if self.use_ema:
            self.ema.step(self.pipeline)

        return loss_dict["loss"]

    @staticmethod
    def _batch_size_from_batch(batch):
        return list(batch["observation"]["image"].values())[0].shape[0]

    @staticmethod
    def _split_joint_training_batch(batch):
        """Return (tcp_batch, action_batch) for two-loader joint training, else None.

        Lightning unwraps CombinedLoader's (data, batch_idx, dataloader_idx) before
        calling training_step, so we only see the inner data here. For a 2-loader
        max_size_cycle setup, that data is a length-2 list of per-loader batches.
        """
        if (isinstance(batch, (list, tuple))
                and len(batch) == 2
                and all(isinstance(b, dict) and "observation" in b for b in batch)):
            return batch[0], batch[1]
        return None

    def validation_step(self, batch, batch_idx, dataloader_idx=None) -> None:
        """TCP validation step with metrics computation"""
        self.validation_step_count += 1
        # Avoid mutating the shared batch in-place.
        gt_actions = None
        batch_for_pred = batch
        if "action" in batch:  # remove action from batch to get a sampled action
            gt_actions = batch["action"]
            batch_for_pred = dict(batch)
            batch_for_pred.pop("action", None)

        output = self.predict_step(batch_for_pred, batch_idx, dataloader_idx)
        batch_size = list(batch["observation"]["image"].values())[0].shape[0]

        if gt_actions is not None and "actions" in output:
            valid_action_error = torch.norm(output["actions"] - gt_actions, dim=-1, p=2).mean()
            self.log(
                f"val/valid_action_error",
                valid_action_error.item(),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                batch_size=batch_size,
                sync_dist=True,
            )

        metrics_dict = {}
        if "tcp_orn" in output:
            metrics_dict.update(self._compute_tcp_metrics(output, batch))
        if any(k in output for k in ("tcp_fm_pred_pose_9d", "tcp_fm_clean_estimate")):
            metrics_dict.update(self._compute_fm_metrics(output, batch))

        for k, v in metrics_dict.items():
            self.log(
                f"val/{k}",
                np.mean(v) if isinstance(v, list) else v,
                on_step=False,
                on_epoch=True,
                prog_bar=True if any(x in k for x in ["pixel_error", "depth_error", "accuracy"]) else False,
                logger=True,
                batch_size=batch_size,
                sync_dist=True,
            )

        gc.collect()
        torch.cuda.empty_cache()

        return output

    def _compute_tcp_metrics(self, output, batch):
        """
        Compute metrics for TCP prediction:
            - Orientation error (geodesic in degrees)
        """
        metrics_dict = {}
        for camera_name in batch["observation"]["image"].keys():
            if "tcp_orn" not in batch["observation"]:
                continue
            if isinstance(batch["observation"]["tcp_orn"], dict) and camera_name not in batch["observation"]["tcp_orn"]:
                continue
            device = output['tcp_orn'][camera_name].device

            pred_orn = output.get('tcp_orn')[camera_name]            # [B, 3, 3]

            gt_orn = batch['observation']['tcp_orn'][camera_name].float().to(device)
            if gt_orn.ndim >= 3:
                gt_orn = gt_orn[:, -1]

            if gt_orn.ndim == 2 and gt_orn.shape[1] == 9:
                gt_orn = gt_orn.reshape(-1, 3, 3)

            with torch.no_grad():
                R_rel = torch.bmm(pred_orn.transpose(1, 2), gt_orn)   # R_pred^T * R_gt
                trace = R_rel[:, 0, 0] + R_rel[:, 1, 1] + R_rel[:, 2, 2]
                cos_angle = torch.clamp((trace - 1.0) / 2.0, -0.99999, 0.99999)
                geodesic_rad = torch.acos(cos_angle)                 # in radians
                geodesic_deg = geodesic_rad * 180.0 / torch.pi       # convert to degrees

            metrics_dict[f'{camera_name}_tcp_orientation_error_rad_mean'] = geodesic_rad.mean().item()
            metrics_dict[f'{camera_name}_tcp_orientation_error_deg_mean'] = geodesic_deg.mean().item()

        return metrics_dict

    def _compute_fm_metrics(self, output, batch):
        """Compute metrics for FM TCP pose prediction (9D pose in base frame, when available)."""
        metrics_dict = {}

        pred_pose_by_cam = None
        if "tcp_fm_pred_pose_9d" in output:
            pred_pose_by_cam = output["tcp_fm_pred_pose_9d"]
        elif "tcp_fm_clean_estimate" in output:
            # Training-only path: predicted clean estimate from FM head.
            pred_pose_by_cam = output["tcp_fm_clean_estimate"]
        if pred_pose_by_cam is None:
            return metrics_dict
        if torch.is_tensor(pred_pose_by_cam):
            pred_pose_by_cam = {"all": pred_pose_by_cam}
        if not isinstance(pred_pose_by_cam, dict):
            return metrics_dict

        obs = batch.get("observation", {})
        tcp_pos = obs.get("tcp_pos", None)
        tcp_orn = obs.get("tcp_orn", None)
        eef_pos = obs.get("robot0_eef_pos", None)
        eef_quat = obs.get("robot0_eef_quat_site", None)  # xyzw
        use_camera_gt = (tcp_pos is not None) and (tcp_orn is not None)
        use_base_gt = (eef_pos is not None) and (eef_quat is not None)

        for camera_name, pred_pose_9d in pred_pose_by_cam.items():
            if pred_pose_9d is None or (not torch.is_tensor(pred_pose_9d)):
                continue

            device = pred_pose_9d.device
            dtype = pred_pose_9d.dtype

            gt_pos = None
            gt_R = None

            if use_camera_gt:
                gt_pos = tcp_pos.get(camera_name) if isinstance(tcp_pos, dict) else tcp_pos
                gt_orn = tcp_orn.get(camera_name) if isinstance(tcp_orn, dict) else tcp_orn
                if gt_pos is None or gt_orn is None:
                    continue
                if torch.is_tensor(gt_pos) and gt_pos.ndim >= 3:
                    gt_pos = gt_pos[:, -1]
                if torch.is_tensor(gt_orn) and gt_orn.ndim >= 3:
                    gt_orn = gt_orn[:, -1]
                gt_pos = torch.as_tensor(gt_pos, device=device, dtype=dtype)
                gt_orn = torch.as_tensor(gt_orn, device=device, dtype=dtype)
                if gt_orn.ndim == 2 and gt_orn.shape[-1] == 9:
                    gt_R = gt_orn.reshape(-1, 3, 3)
                else:
                    gt_R = gt_orn
            elif use_base_gt:
                gt_pos = eef_pos
                gt_quat = eef_quat
                if torch.is_tensor(gt_pos) and gt_pos.ndim >= 3:
                    gt_pos = gt_pos[:, -1]
                if torch.is_tensor(gt_quat) and gt_quat.ndim >= 3:
                    gt_quat = gt_quat[:, -1]
                gt_pos = torch.as_tensor(gt_pos, device=device, dtype=dtype)
                gt_quat = torch.as_tensor(gt_quat, device=device, dtype=dtype)
                quat_wxyz = torch.cat([gt_quat[:, 3:4], gt_quat[:, :3]], dim=-1)
                gt_R = quaternion_to_matrix(quat_wxyz)
            else:
                continue

            pred_pos = pred_pose_9d[:, :3]
            pred_R = rotation_6d_to_matrix(pred_pose_9d[:, 3:])

            with torch.no_grad():
                pos_error = torch.norm(pred_pos - gt_pos, dim=-1)
                metrics_dict[f"{camera_name}_tcp_fm_pos_error_mean"] = pos_error.mean().item()
                metrics_dict[f"{camera_name}_tcp_fm_pos_error_median"] = pos_error.median().item()

                R_rel = torch.bmm(pred_R.transpose(1, 2), gt_R)
                trace = R_rel[:, 0, 0] + R_rel[:, 1, 1] + R_rel[:, 2, 2]
                cos_angle = torch.clamp((trace - 1.0) / 2.0, -0.99999, 0.99999)
                geodesic_rad = torch.acos(cos_angle)
                metrics_dict[f"{camera_name}_tcp_fm_orn_error_rad"] = geodesic_rad.mean().item()
                metrics_dict[f"{camera_name}_tcp_fm_orn_error_deg"] = (geodesic_rad * 180.0 / torch.pi).mean().item()

        return metrics_dict

    def update_metrics_dict(self, metrics_dict, metrics_dict_item, prefix):
        for k, v in metrics_dict_item.items():
            if f"{prefix}_{k}" not in metrics_dict:
                metrics_dict[f"{prefix}_{k}"] = []
            metrics_dict[f"{prefix}_{k}"].append(v)
        return metrics_dict

    def create_depth_mask(self, dataset_name, gt_depth):
        # return gt_depth > 1e-3
        return np.logical_and(gt_depth > 1e-3, ~np.isnan(gt_depth)) & (~np.isinf(gt_depth))

    def save_tcp_predictions(self, output, batch, tag):
        """Save raw TCP 3D pose prediction data"""
        pred_data = {}
        for camera_name in batch["observation"]["image"].keys():
            for b in range(len(batch["observation"]["image"][camera_name])):
                pred_data = {
                    'tcp_orn': output["tcp_orn"][camera_name][b].detach().cpu().numpy(),  # [3, 3] - rotation matrix
                }

                pred_data['tcp_pixel_coords'] = output["tcp_pixel_coords"][camera_name][b].detach().cpu().numpy()
                pred_data['tcp_heatmap'] = output["tcp_heatmap"][camera_name][b].detach().cpu().numpy()

                # Also save ground truth if available
                if "tcp_orn" in batch["observation"]:
                    pred_data['gt_tcp_orn'] = batch["observation"]["tcp_orn"][camera_name][b].detach().cpu().numpy()
                if "tcp_pixel_coords" in batch["observation"]:
                    pred_data['gt_coords'] = batch["observation"]["tcp_pixel_coords"][camera_name][b].detach().cpu().numpy()

                # Save camera intrinsics for reference
                if "camera_intrinsics" in batch["observation"]:
                    intrinsics = batch["observation"]["camera_intrinsics"][camera_name][b].detach().cpu().numpy()
                    if len(intrinsics.shape) == 2:  # [3, 3]
                        pred_data['camera_intrinsics'] = intrinsics
                    else:  # [9]
                        pred_data['camera_intrinsics'] = intrinsics.reshape(3, 3)

                if isinstance(batch["episode_id"][b], str):
                    save_name = batch["episode_id"][b] + "_" + str(batch["step_id"][b]) + ".npz"
                else:
                    save_name = batch["episode_id"][b].decode("utf-8") + "_" + str(batch["step_id"][b]) + ".npz"
                save_path = join(self.output_dir, f"{tag}/{camera_name}/{save_name}")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                np.savez_compressed(save_path, **pred_data)

    def save_tcp_visualizations_rgb(self, output, batch, tag):
        """Save TCP 3D pose predictions visualized on RGB images"""
        for camera_name in batch["observation"]["image"].keys():
            for b in range(len(batch["observation"]["image"][camera_name])):
                # Get image
                rgb_image = batch["observation"]["image"][camera_name][b].detach().cpu().numpy().transpose(1, 2, 0)  # [H, W, 3]
                rgb_image = (rgb_image * 255).astype(np.uint8)
                h, w = rgb_image.shape[:2]

                # Get 3D pose predictions
                pred_tcp_orn = output["tcp_orn"][camera_name][b].detach().cpu().numpy()  # [3, 3]
                pred_coords = output["tcp_pixel_coords"][camera_name][b].detach().cpu().numpy() # normalized x, y, z

                # Get ground truth 3D pose
                gt_tcp_pos = batch["observation"]["tcp_pos"][camera_name][b].detach().cpu().numpy()
                gt_tcp_orn = batch["observation"]["tcp_orn"][camera_name][b].detach().cpu().numpy()
                gt_coords = batch["observation"]["tcp_pixel_coords"][camera_name][b].detach().cpu().numpy() # normalized x, y, z
                if len(gt_tcp_orn.shape) == 1 and gt_tcp_orn.shape[0] == 9:
                    gt_tcp_orn = gt_tcp_orn.reshape(3, 3)

                # Get camera intrinsics (scale from original resolution to current image size)
                camera_intrinsics_orig = batch["observation"]["camera_intrinsics"][camera_name][b].detach().cpu().numpy()
                if len(camera_intrinsics_orig.shape) == 1:
                    camera_intrinsics_orig = camera_intrinsics_orig.reshape(3, 3)
                orig_h_k = int(round(camera_intrinsics_orig[1, 2] * 2))
                orig_w_k = int(round(camera_intrinsics_orig[0, 2] * 2))
                camera_intrinsics = scale_intrinsics(camera_intrinsics_orig, (orig_h_k, orig_w_k), (h, w))

                # Project 3D poses to 2D for visualization
                unnormalized_pred_coords = pred_coords.copy()
                unnormalized_pred_coords[0] *= w
                unnormalized_pred_coords[1] *= h
                unnormalized_pred_coords[2] = self._unnormalize_depth(pred_coords[2])
                _, pred_axes_2d = project_3d_pose_to_2d(None, unnormalized_pred_coords, pred_tcp_orn, camera_intrinsics, (h, w))
                _, gt_axes_2d = project_3d_pose_to_2d(gt_tcp_pos, None, gt_tcp_orn, camera_intrinsics, (h, w))

                # Compute real depth values (meters) for display
                pred_depth_m = self._unnormalize_depth(pred_coords[2])
                gt_depth_m = self._unnormalize_depth(gt_coords[2])

                # Get auxiliary 2D outputs if available
                pred_pixel_coords = output["tcp_pixel_coords"][camera_name][b].detach().cpu().numpy()
                if h*w > 1:
                    heatmap = output["tcp_heatmap"][camera_name][b][0].detach().cpu().numpy()
                else:
                    heatmap = None
                gt_pixel_coords = batch["observation"]["tcp_pixel_coords"][camera_name][b].detach().cpu().numpy()
                if h*w > 1:
                    gt_heatmap = create_gt_heatmap(gt_pixel_coords, heatmap.shape[0], heatmap.shape[1])
                else:
                    gt_heatmap = None

                fig, axes = plt.subplots(1, 7, figsize=(35, 5))
                fig.suptitle(f'Evaluation Results', fontsize=14)

                # RGB with prediction
                axes[0].imshow(rgb_image)
                pred_u_px, pred_v_px = pred_coords[0] * (w-1), pred_coords[1] * (h-1)
                axes[0].plot(pred_u_px, pred_v_px, 'ro', markersize=8)
                axes[0].set_title(f'Pred: ({pred_coords[0]:.3f}, {pred_coords[1]:.3f}) d={pred_depth_m:.3f}m')
                axes[0].annotate(f'{pred_depth_m:.3f}m', xy=(pred_u_px, pred_v_px),
                                 xytext=(5, -15), textcoords='offset points',
                                 fontsize=7, color='yellow',
                                 bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.7))

                # RGB with GT
                axes[1].imshow(rgb_image)
                gt_u_px, gt_v_px = gt_coords[0] * (w-1), gt_coords[1] * (h-1)
                axes[1].plot(gt_u_px, gt_v_px, 'ro', markersize=8)
                axes[1].set_title(f'GT: ({gt_coords[0]:.3f}, {gt_coords[1]:.3f}) d={gt_depth_m:.3f}m')
                axes[1].annotate(f'{gt_depth_m:.3f}m', xy=(gt_u_px, gt_v_px),
                                 xytext=(5, -15), textcoords='offset points',
                                 fontsize=7, color='cyan',
                                 bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.7))

                # Heatmap
                if heatmap is not None:
                    axes[2].imshow(heatmap, cmap='jet')
                axes[2].set_title(f'Attention Heatmap: ({int(pred_coords[0]*(w-1))}, {int(pred_coords[1]*(h-1))})')

                # GT Heatmap
                if gt_heatmap is not None:
                    axes[3].imshow(gt_heatmap, cmap='jet')
                axes[3].set_title(f'GT Heatmap: ({int(gt_coords[0]*(w-1))}, {int(gt_coords[1]*(h-1))})')

                # Overlay
                axes[4].imshow(rgb_image)
                if heatmap is not None:
                    heatmap_resized = cv2.resize(heatmap.astype(np.float32), (rgb_image.shape[1], rgb_image.shape[0]))
                    axes[4].imshow(heatmap_resized, alpha=0.6, cmap='jet')
                axes[4].set_title('Heatmap Overlay')

                # TCP Overlay
                result_image = self.draw_tcp_on_image(rgb_image, pred_coords, pred_axes_2d)
                axes[5].imshow(result_image)
                axes[5].set_title('Predict TCP Axis')

                # Ground Truth TCP Overlay
                result_image = self.draw_tcp_on_image(rgb_image, gt_coords, gt_axes_2d)
                axes[6].imshow(result_image)
                axes[6].set_title('Ground Truth TCP Axis')

                plt.tight_layout()

                # Save visualization
                if isinstance(batch["episode_id"][b], str):
                    save_name = batch["episode_id"][b] + "_" + str(batch["step_id"][b].item()) + "_tcp_3d_vis.png"
                else:
                    save_name = batch["episode_id"][b].decode("utf-8") + "_" + str(batch["step_id"][b]) + "_tcp_3d_vis.png"
                save_path = join(self.output_dir, f"{tag}/{camera_name}/{save_name}")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close('all')

    def save_tcp_pointcloud_visualizations(self, output, batch, tag):
        """Save TCP 3D pose predictions on pointclouds"""
        try:
            import open3d as o3d
        except ImportError:
            Log.warn("Open3D not available, skipping pointcloud visualization")
            return
        for camera_name in batch["observation"]["image"].keys():
            for b in range(len(batch["observation"]["image"])):
                if "depth" not in batch["observation"]:
                    continue

                # Get data
                rgb_image = batch["observation"]["image"][camera_name][b].detach().cpu().numpy().transpose(1, 2, 0)
                depth_image = batch["observation"]["depth"][camera_name][b][0].detach().cpu().numpy()

                # Get 3D TCP predictions directly
                pred_tcp_pos = output["tcp_pos"][camera_name][b].detach().cpu().numpy()  # [3] in meters
                pred_tcp_orn = output["tcp_orn"][camera_name][b].detach().cpu().numpy()  # [3, 3] rotation matrix

                # Get ground truth 3D pose
                if "tcp_pos" in batch["observation"]:
                    gt_tcp_pos = batch["observation"]["tcp_pos"][camera_name][b].detach().cpu().numpy()
                else:
                    gt_tcp_pos = None

                if "tcp_orn" in batch["observation"]:
                    gt_tcp_orn = batch["observation"]["tcp_orn"][camera_name][b].detach().cpu().numpy()
                else:
                    gt_tcp_orn = None

                # Get camera intrinsics for pointcloud generation
                if "camera_intrinsics" in batch["observation"]:
                    intrinsics = batch["observation"]["camera_intrinsics"][camera_name][b].detach().cpu().numpy().reshape(3, 3)
                    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
                    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
                else:
                    # Fallback to default intrinsics
                    h, w = depth_image.shape
                    fx = fy = self._current_focal if self._current_focal is not None else 500.0
                    cx, cy = w / 2, h / 2

                # Generate pointcloud from RGB-D
                h, w = depth_image.shape
                points = []
                colors = []

                for v in range(h):
                    for u in range(w):
                        z = depth_image[v, u]
                        if z > 0:  # Valid depth
                            x = (u - cx) * z / fx
                            y = (v - cy) * z / fy
                            points.append([x, y, z])
                            if rgb_image.shape[0] == h and rgb_image.shape[1] == w:
                                colors.append(rgb_image[v, u])

                if len(points) == 0:
                    continue

                # Create Open3D pointcloud
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(np.array(points))
                if len(colors) > 0:
                    pcd.colors = o3d.utility.Vector3dVector(np.array(colors))

                # Create TCP coordinate frame at predicted location
                pred_coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
                # Apply rotation and translation
                transform = np.eye(4)
                transform[:3, :3] = pred_tcp_orn
                transform[:3, 3] = pred_tcp_pos
                pred_coord_frame.transform(transform)

                # Create TCP marker sphere
                pred_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.015)
                pred_sphere.translate(pred_tcp_pos)
                pred_sphere.paint_uniform_color([1.0, 0.0, 0.0])  # Red

                # Combine geometries
                combined_geometry = [pcd, pred_coord_frame, pred_sphere]

                # Add ground truth TCP if available
                if gt_tcp_pos is not None and gt_tcp_orn is not None:
                    # GT coordinate frame
                    gt_coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
                    gt_transform = np.eye(4)
                    gt_transform[:3, :3] = gt_tcp_orn
                    gt_transform[:3, 3] = gt_tcp_pos
                    gt_coord_frame.transform(gt_transform)

                    # GT TCP marker sphere
                    gt_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.015)
                    gt_sphere.translate(gt_tcp_pos)
                    gt_sphere.paint_uniform_color([0.0, 1.0, 0.0])  # Green

                    combined_geometry.extend([gt_coord_frame, gt_sphere])

                    # Draw line between prediction and ground truth
                    line_points = [pred_tcp_pos.tolist(), gt_tcp_pos.tolist()]
                    line_lines = [[0, 1]]
                    line_set = o3d.geometry.LineSet()
                    line_set.points = o3d.utility.Vector3dVector(line_points)
                    line_set.lines = o3d.utility.Vector2iVector(line_lines)
                    line_set.colors = o3d.utility.Vector3dVector([[1.0, 1.0, 0.0]])  # Yellow
                    combined_geometry.append(line_set)

                # Save pointcloud
                if isinstance(batch["episode_id"][b], str):
                    save_name = batch["episode_id"][b] + "_" + str(batch["step_id"][b]) + "_tcp_pointcloud.ply"
                else:
                    save_name = batch["episode_id"][b].decode("utf-8") + "_" + str(batch["step_id"][b]) + "_tcp_pointcloud.ply"
                save_path = join(self.output_dir, f"{tag}/{camera_name}/{save_name}")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                o3d.io.write_point_cloud(save_path, pcd)

                # Save scene metadata
                if isinstance(batch["episode_id"][b], str):
                    scene_name = batch["episode_id"][b] + "_" + str(batch["step_id"][b]) + "_tcp_scene.json"
                else:
                    scene_name = batch["episode_id"][b].decode("utf-8") + "_" + str(batch["step_id"][b]) + "_tcp_scene.json"
                scene_path = join(self.output_dir, f"{tag}/{camera_name}/{scene_name}")
                scene_data = {
                    'pointcloud_path': save_path,
                    'tcp_3d_pred_pos': pred_tcp_pos.tolist(),
                    'tcp_3d_pred_orn': pred_tcp_orn.tolist(),
                }

                if gt_tcp_pos is not None:
                    scene_data['tcp_3d_gt_pos'] = gt_tcp_pos.tolist()
                    scene_data['tcp_3d_gt_orn'] = gt_tcp_orn.tolist()
                    pos_error = np.linalg.norm(pred_tcp_pos - gt_tcp_pos)
                    scene_data['position_error_m'] = float(pos_error)

                import json
                with open(scene_path, 'w') as f:
                    json.dump(scene_data, f, indent=2)

    def save_training_progress_visualization(self, output, batch, step_count):
        """Save training progress visualization for 3D TCP pose"""
        for camera_name in batch["observation"]["image"].keys():
            if len(batch["observation"]["image"][camera_name]) == 0:
                continue

            # Just save the first item in the batch
            b = 0
            # Get image
            rgb_image = batch["observation"]["image"][camera_name][b].detach().cpu().numpy().transpose(1, 2, 0)  # [H, W, 3]
            rgb_image = (rgb_image * 255).astype(np.uint8)
            h, w = rgb_image.shape[:2]

            # Get 3D pose predictions
            pred_tcp_orn = output["tcp_orn"][camera_name][b].detach().cpu().numpy()  # [3, 3]
            pred_coords = output["tcp_pixel_coords"][camera_name][b].detach().cpu().numpy() # normalized x, y, z

            # Get ground truth 3D pose
            gt_tcp_pos = batch["observation"]["tcp_pos"][camera_name][b].detach().cpu().numpy()
            gt_tcp_orn = batch["observation"]["tcp_orn"][camera_name][b].detach().cpu().numpy()
            gt_coords = batch["observation"]["tcp_pixel_coords"][camera_name][b].detach().cpu().numpy() # normalized x, y, z
            if len(gt_tcp_orn.shape) == 1 and gt_tcp_orn.shape[0] == 9:
                gt_tcp_orn = gt_tcp_orn.reshape(3, 3)

            # Get camera intrinsics (scale from original resolution to current image size)
            camera_intrinsics_orig = batch["observation"]["camera_intrinsics"][camera_name][b].detach().cpu().numpy()
            if len(camera_intrinsics_orig.shape) == 1:
                camera_intrinsics_orig = camera_intrinsics_orig.reshape(3, 3)
            orig_h_k = int(round(camera_intrinsics_orig[1, 2] * 2))
            orig_w_k = int(round(camera_intrinsics_orig[0, 2] * 2))
            camera_intrinsics = scale_intrinsics(camera_intrinsics_orig, (orig_h_k, orig_w_k), (h, w))

            # Project 3D poses to 2D for visualization
            unnormalized_pred_coords = pred_coords.copy()
            unnormalized_pred_coords[0] *= w
            unnormalized_pred_coords[1] *= h
            unnormalized_pred_coords[2] = self._unnormalize_depth(pred_coords[2])
            _, pred_axes_2d = project_3d_pose_to_2d(None, unnormalized_pred_coords, pred_tcp_orn, camera_intrinsics, (h, w))
            _, gt_axes_2d = project_3d_pose_to_2d(gt_tcp_pos, None, gt_tcp_orn, camera_intrinsics, (h, w))

            pred_pixel_coords = output["tcp_pixel_coords"][camera_name][b].detach().cpu().numpy()
            heatmap = output["tcp_heatmap"][camera_name][b][0].detach().cpu().numpy()
            gt_pixel_coords = batch["observation"]["tcp_pixel_coords"][camera_name][b].detach().cpu().numpy()
            gt_heatmap = create_gt_heatmap(gt_pixel_coords, heatmap.shape[0], heatmap.shape[1])

            fig, axes = plt.subplots(1, 7, figsize=(35, 5))
            fig.suptitle(f'Training Progress - Step {step_count}', fontsize=14)

            # RGB with prediction
            axes[0].imshow(rgb_image)
            axes[0].plot(pred_coords[0] * (w-1), pred_coords[1] * (h-1), 'ro', markersize=8)
            axes[0].set_title(f'TCP Prediction: ({pred_coords[0]:.3f}, {pred_coords[1]:.3f})')

            # RGB with GT
            axes[1].imshow(rgb_image)
            axes[1].plot(gt_coords[0] * (w-1), gt_coords[1] * (h-1), 'ro', markersize=8)
            axes[1].set_title(f'TCP GT: ({gt_coords[0]:.3f}, {gt_coords[1]:.3f})')

            # Heatmap
            axes[2].imshow(heatmap, cmap='jet')
            axes[2].set_title(f'Attention Heatmap: ({int(pred_coords[0]*(w-1))}, {int(pred_coords[1]*(h-1))})')

            # GT Heatmap
            axes[3].imshow(gt_heatmap, cmap='jet')
            axes[3].set_title(f'GT Heatmap: ({int(gt_coords[0]*(w-1))}, {int(gt_coords[1]*(h-1))})')

            # Overlay
            axes[4].imshow(rgb_image)
            heatmap_resized = cv2.resize(heatmap.astype(np.float32), (rgb_image.shape[1], rgb_image.shape[0]))
            axes[4].imshow(heatmap_resized, alpha=0.6, cmap='jet')
            axes[4].set_title('Heatmap Overlay')

            # TCP Overlay
            result_image = self.draw_tcp_on_image(rgb_image, pred_coords, pred_axes_2d)
            axes[5].imshow(result_image)
            axes[5].set_title('Predict TCP Overlay')

            # Ground Truth TCP Overlay
            result_image = self.draw_tcp_on_image(rgb_image, gt_coords, gt_axes_2d)
            axes[6].imshow(result_image)
            axes[6].set_title('Ground Truth TCP Overlay')

            plt.tight_layout()

            # Save training progress
            save_path = join(self.output_dir, f"training_progress/{camera_name}/step_{step_count:06d}.png")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close('all')

    def save_tcp_training_visualization(self, output, batch, step_count):
        """Visualize TCP head predictions (uv/3d/6d) on training images.

        Uses output keys: tcp_uv [B, num_cam, T, 2], tcp_3d [B, num_cam, T, 3], tcp_6d [B, num_cam, T, 6].
        GT from batch: tcp_pixel_coords / tcp_pos / tcp_orn (all per-camera camera-frame annotations).
        """
        if 'tcp_uv' not in output:
            return

        camera_names = list(batch["observation"]["image"].keys())
        pred_uv = output['tcp_uv'].detach().cpu().numpy()   # [B, num_cam, T, 2]
        pred_3d = output['tcp_3d'].detach().cpu().numpy()   # [B, num_cam, T, 3]
        pred_6d = output['tcp_6d'].detach().cpu().numpy()   # [B, num_cam, T, 6]

        # GT from per-camera TCP annotations (camera frame)
        gt_tcp_pos = batch["observation"].get("tcp_pos", {})
        gt_tcp_orn = batch["observation"].get("tcp_orn", {})
        gt_pos, gt_6d_np = None, None

        # GT UV per camera
        gt_tcp_pixel = batch["observation"].get("tcp_pixel_coords", {})

        b = 0
        for cam_idx, camera_name in enumerate(camera_names):
            img = batch["observation"]["image"][camera_name][b].detach().cpu()
            if img.ndim == 4:
                img = img[-1]
            img = img.numpy().transpose(1, 2, 0)
            img = (img * 255).clip(0, 255).astype(np.uint8)
            h, w = img.shape[:2]

            p_uv = pred_uv[b, cam_idx, -1]    # [2]
            p_3d = pred_3d[b, cam_idx, -1]     # [3]
            p_6d = pred_6d[b, cam_idx, -1]     # [6]

            # GT UV for this camera
            gt_uv = None
            if camera_name in gt_tcp_pixel:
                gt_cam = gt_tcp_pixel[camera_name]
                if isinstance(gt_cam, torch.Tensor):
                    gt_cam = gt_cam.detach().cpu().numpy()
                if gt_cam.ndim == 3:
                    gt_uv = gt_cam[b, -1, :2]
                elif gt_cam.ndim == 2:
                    gt_uv = gt_cam[b, :2]
            if camera_name in gt_tcp_pos:
                gt_cam_pos = gt_tcp_pos[camera_name]
                if isinstance(gt_cam_pos, torch.Tensor):
                    gt_cam_pos = gt_cam_pos.detach().cpu().numpy()
                gt_pos = gt_cam_pos[b, -1] if gt_cam_pos.ndim == 3 else gt_cam_pos[b]
            if camera_name in gt_tcp_orn:
                gt_cam_orn = gt_tcp_orn[camera_name]
                if isinstance(gt_cam_orn, torch.Tensor):
                    gt_cam_orn = gt_cam_orn.detach().cpu()
                else:
                    gt_cam_orn = torch.tensor(gt_cam_orn, dtype=torch.float32)
                gt_last_orn = gt_cam_orn[b, -1] if gt_cam_orn.ndim == 3 else gt_cam_orn[b]
                gt_6d_np = matrix_to_rotation_6d(gt_last_orn.reshape(3, 3).unsqueeze(0))[0].cpu().numpy()

            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            fig.suptitle(f'TCP — Step {step_count} — {camera_name}', fontsize=13)

            # Panel 1: Pred + GT UV on image
            axes[0].imshow(img)
            px, py = p_uv[0] * (w - 1), p_uv[1] * (h - 1)
            axes[0].plot(px, py, 'ro', markersize=10, markeredgecolor='white', markeredgewidth=1.5, label='Pred')
            if gt_uv is not None:
                gx, gy = gt_uv[0] * (w - 1), gt_uv[1] * (h - 1)
                axes[0].plot(gx, gy, 'g^', markersize=10, markeredgecolor='white', markeredgewidth=1.5, label='GT')
            axes[0].legend(fontsize=8)
            axes[0].set_title(f'UV  Pred({p_uv[0]:.3f},{p_uv[1]:.3f})')

            # Panel 2: Pred vs GT 3D position
            x_pos = np.arange(3)
            bar_w = 0.35
            axes[1].bar(x_pos - bar_w / 2, p_3d, bar_w, label='Pred', color='steelblue')
            if gt_pos is not None:
                axes[1].bar(x_pos + bar_w / 2, gt_pos, bar_w, label='GT', color='coral')
            axes[1].set_xticks(x_pos)
            axes[1].set_xticklabels(['X', 'Y', 'Z'])
            axes[1].set_title('3D Position')
            axes[1].legend(fontsize=8)

            # Panel 3: Pred vs GT 6D rotation
            x_pos_6 = np.arange(6)
            axes[2].bar(x_pos_6 - bar_w / 2, p_6d, bar_w, label='Pred', color='steelblue')
            if gt_6d_np is not None:
                axes[2].bar(x_pos_6 + bar_w / 2, gt_6d_np, bar_w, label='GT', color='coral')
            axes[2].set_xticks(x_pos_6)
            axes[2].set_xticklabels([f'r{i}' for i in range(6)])
            axes[2].set_title('6D Rotation')
            axes[2].legend(fontsize=8)

            plt.tight_layout()
            save_path = join(self.output_dir, f"tcp_vis/{camera_name}/step_{step_count:06d}.png")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close('all')

    def draw_tcp_on_image(self, image, tcp_pixel_coords, axis_ends=None, tcp_dir_x=None, tcp_dir_y=None, tcp_dir_z=None):
        """Legacy function for 2D direction visualization (deprecated, use draw_tcp_3d_on_image)"""
        u, v = tcp_pixel_coords[0], tcp_pixel_coords[1]

        # Create visualization
        result_image = image.copy()
        h, w = result_image.shape[:2]

        u = int(u * (w-1))
        v = int(v * (h-1))

        # Draw gripper position if within image bounds
        if 0 <= u < w and 0 <= v < h:
            # Draw gripper center point
            cv2.circle(result_image, (u, v), 5, (0, 255, 0), -1)  # Green circle
            # cv2.circle(result_image, (u, v), 6, (0, 0, 0), 2)    # Black border

            # Extract axes from rotation matrix in camera coordinates
            # The rotation matrix columns represent the gripper's local coordinate axes in camera space
            # Scale each axis by axis_length to make them visible in the visualization
            if axis_ends is None:
                x_axis_end_homogeneous_diff = tcp_dir_x   # x-axis direction
                y_axis_end_homogeneous_diff = tcp_dir_y  # y-axis direction
                z_axis_end_homogeneous_diff = tcp_dir_z  # z-axis direction

                # Project axes endpoints
                for idx, (axis_diff, color) in enumerate([(x_axis_end_homogeneous_diff, (0, 0, 255)),    # Red for X
                                    (y_axis_end_homogeneous_diff, (0, 255, 0)),     # Green for Y
                                    (z_axis_end_homogeneous_diff, (255, 0, 0))]):    # Blue for Z

                    dir_x, dir_y = axis_diff[0], axis_diff[1]
                    length = 30  # Fixed pixel length
                    norm = np.sqrt(dir_x**2 + dir_y**2)
                    if norm > 0:
                        u_end = int(u + (dir_x/norm) * length)
                        v_end = int(v + (dir_y/norm) * length)

                        if 0 <= u_end < w and 0 <= v_end < h:
                            cv2.arrowedLine(result_image, (u, v), (u_end, v_end), color, 3)
            else:
                for idx, axis_end in enumerate(axis_ends):
                    color = (0, 0, 255) if idx == 0 else (0, 255, 0) if idx == 1 else (255, 0, 0)
                    u_end = int(axis_end[0])
                    v_end = int(axis_end[1])
                    if 0 <= u_end < w and 0 <= v_end < h:
                        cv2.arrowedLine(result_image, (u, v), (u_end, v_end), color, 3)

        return result_image

    def _unnormalize_depth(self, d_norm):
        """Convert normalized depth back to real depth in meters.

        Inverse of: d_norm = (d - min_depth) / (max_depth - min_depth)
        """
        return self._min_depth + d_norm * (self._max_depth - self._min_depth)

    def create_tcp_summary_report(self, output_dir):
        """Create a comprehensive summary report of TCP predictions"""
        try:
            import json
            from glob import glob

            # Find all prediction files
            pred_files = glob(join(output_dir, "tcp_predictions/*.npz"))

            if not pred_files:
                Log.warn("No TCP prediction files found for summary report")
                return

            # Collect statistics
            all_pixel_errors = []
            all_depth_errors = []
            all_direction_errors = []

            for pred_file in pred_files:
                try:
                    data = np.load(pred_file)

                    if 'gt_coords' in data:
                        pred_coords = data['tcp_pixel_coords']
                        gt_coords = data['gt_coords']

                        # Pixel error
                        pixel_error = np.sqrt((pred_coords[0] - gt_coords[0])**2 + (pred_coords[1] - gt_coords[1])**2)
                        all_pixel_errors.append(pixel_error)

                        # Depth error
                        depth_error = np.abs(pred_coords[2] - gt_coords[2])
                        all_depth_errors.append(depth_error)

                        # Direction error
                        pred_dir_x, pred_dir_y = data['tcp_dir_x'], data['tcp_dir_y']
                        gt_dir_x, gt_dir_y = data['gt_tcp_dir_x'], data['gt_tcp_dir_y']
                        dir_error = np.sqrt((pred_dir_x - gt_dir_x)**2 + (pred_dir_y - gt_dir_y)**2)
                        all_direction_errors.append(dir_error)

                except Exception as e:
                    Log.warn(f"Error processing {pred_file}: {e}")
                    continue

            if all_pixel_errors:
                # Generate summary statistics
                summary = {
                    'total_predictions': len(all_pixel_errors),
                    'pixel_errors': {
                        'mean': float(np.mean(all_pixel_errors)),
                        'median': float(np.median(all_pixel_errors)),
                        'std': float(np.std(all_pixel_errors)),
                        'min': float(np.min(all_pixel_errors)),
                        'max': float(np.max(all_pixel_errors)),
                        'accuracy_5px': float(np.mean(np.array(all_pixel_errors) < 5.0)),
                        'accuracy_10px': float(np.mean(np.array(all_pixel_errors) < 10.0))
                    },
                    'depth_errors': {
                        'mean': float(np.mean(all_depth_errors)),
                        'median': float(np.median(all_depth_errors)),
                        'std': float(np.std(all_depth_errors)),
                        'accuracy_5cm': float(np.mean(np.array(all_depth_errors) < 0.05)),
                        'accuracy_10cm': float(np.mean(np.array(all_depth_errors) < 0.10))
                    },
                    'direction_errors': {
                        'mean': float(np.mean(all_direction_errors)),
                        'median': float(np.median(all_direction_errors)),
                        'std': float(np.std(all_direction_errors))
                    }
                }

                # Save summary report
                summary_path = join(output_dir, "tcp_summary_report.json")
                with open(summary_path, 'w') as f:
                    json.dump(summary, f, indent=2)

                Log.info(f"TCP Summary Report saved to: {summary_path}")
                Log.info(f"Mean pixel error: {summary['pixel_errors']['mean']:.2f}px")
                Log.info(f"Mean depth error: {summary['depth_errors']['mean']:.4f}m")
                Log.info(f"Pixel accuracy (5px): {summary['pixel_errors']['accuracy_5px']*100:.3f}%")

        except Exception as e:
            Log.error(f"Error creating TCP summary report: {e}")

    def on_validation_epoch_end(self):
        """Generate summary report at the end of validation epoch"""
        print(f"Validation epoch end {self.validation_step_count}")
        super().on_validation_epoch_end()

        # Generate TCP summary report
        if self._compute_tcp_metrics:
            pass
            # self.create_tcp_summary_report(self.output_dir)

    def get_tcp_model_info(self):
        """Get information about the TCP model configuration"""
        return {
            'model_type': 'TCP VLA Model',
            'tcp_marker_size': self._tcp_marker_size,
            'direction_arrow_scale': self._direction_arrow_scale,
            'heatmap_alpha': self._heatmap_alpha,
            'pixel_error_threshold': self._pixel_error_threshold,
            'depth_error_threshold': self._depth_error_threshold,
            'direction_error_threshold': self._direction_error_threshold,
            'camera_parameters': {
                'default_focal': self._default_focal,
                'default_image_size': (self._default_image_width, self._default_image_height),
                'current_focal': self._current_focal,
                'current_image_size': (self._current_image_width, self._current_image_height),
                'dynamic_extraction': True
            },
            'visualizations_enabled': {
                'tcp_predictions': self._save_tcp_predictions,
                'tcp_visualizations': self._save_tcp_visualizations,
                'tcp_pointclouds': self._save_tcp_pointclouds,
                'heatmap_visualizations': self._save_heatmap_visualizations,
                'training_progress': self._save_training_progress
            }
        }

    def on_load_checkpoint(self, checkpoint):
        """Pre-create ema_pipeline before PL loads state_dict so EMA keys are accepted."""
        if self.use_ema and self.ema_pipeline is None:
            state_dict = checkpoint.get("state_dict", {})
            if any(k.startswith("ema_pipeline.") for k in state_dict):
                self.ema_pipeline = copy.deepcopy(self.pipeline)
                self.ema_pipeline.eval()
                self.ema_pipeline.requires_grad_(False)
                self.ema = instantiate(self.ema, model=self.ema_pipeline)

    def load_pretrained_model(self, ckpt_path, ckpt_type=None):
        """Load pretrained Lightning checkpoint into this module."""
        if ckpt_path is None:
            Log.info("No checkpoint path provided; skipping loading pretrained model.")
            return

        Log.info(f"Loading ckpt type `{ckpt_type}': {ckpt_path}")
        state_dict = torch.load(ckpt_path, "cpu")["state_dict"]
        if self.use_ema:
            ema_state_dict = {}
            for k in state_dict.keys():
                if k.startswith("ema_pipeline."):
                    ema_state_dict[k.replace("ema_pipeline.", "pipeline.")] = state_dict[k]
            state_dict = ema_state_dict
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        real_missing = []
        for k in missing:
            miss = True
            for ig_keys in self.ignored_weights_prefix:
                if k.startswith(ig_keys):
                    miss = False
            if miss:
                real_missing.append(k)
        if len(real_missing) > 0:
            Log.warn(f"Missing keys: {real_missing}")
        if len(unexpected) > 0:
            Log.error(f"Unexpected keys: {unexpected}")
