"""Server-side SAPolicy wrapper for RoboTwin evaluation (runs in the `sa` env).

Mirrors the inference contract of sapolicy/eval/robomimic_image_runner.py:

    infer(image, depth, camera_intrinsics, state, camera_names, camera_extrinsics)
      -> outputs['actions']                       normalized, [B, T, 20]
    normalizer.unnormalize({"action": ...})
    convert_relative_actions_to_absolute(...)      body-frame delta -> absolute pose

with two bimanual-specific steps the single-arm runner does not need:

  1. The 20-dim action is [pose_left(9), pose_right(9), grip_left(1), grip_right(1)]
     (see robomimic_hdf5.py: the DiT head only splits into two contiguous parts, so
     the dataset groups poses then grippers). Each arm's delta is resolved against
     that arm's own current pose.
  2. Training targets are the TCP (gripper centre) but take_action(ee) expects the
     `endpose` convention, 12 cm behind it along the gripper's local +x. The offset
     is removed here, once, right before the action leaves this process.
"""
import threading
from collections import deque
from contextlib import nullcontext

import numpy as np
import torch
from scipy.spatial.transform import Rotation as Rot

TCP_FORWARD_OFFSET = 0.12


def _wxyz_to_xyzw(q):
    q = np.asarray(q, dtype=np.float64)
    return np.concatenate([q[..., 1:4], q[..., 0:1]], axis=-1)


def _xyzw_to_wxyz(q):
    q = np.asarray(q, dtype=np.float64)
    return np.concatenate([q[..., 3:4], q[..., 0:3]], axis=-1)


def _endpose_to_tcp(endpose):
    """RoboTwin endpose [x,y,z,qw,qx,qy,qz] -> (tcp_pos, rotation matrix)."""
    pos = np.asarray(endpose[:3], dtype=np.float64)
    rot = Rot.from_quat(_wxyz_to_xyzw(np.asarray(endpose[3:7]))).as_matrix()
    return pos + rot @ np.array([TCP_FORWARD_OFFSET, 0.0, 0.0]), rot


class SAPolicyRoboTwinModel:
    def __init__(self, cfg_file, ckpt_path, workspace=None, n_action_steps=8,
                 device="cuda", use_ema=True, normalizer_path=None,
                 tcp_forward_offset_m=None, warmup_iterations=0,
                 warmup_camera_names=None, resolved_cfg=None):
        import os
        import sys

        if workspace:
            os.environ.setdefault("workspace", str(workspace))
        if tcp_forward_offset_m is not None:
            global TCP_FORWARD_OFFSET
            TCP_FORWARD_OFFSET = float(tcp_forward_offset_m)

        cfg_opts = []
        if resolved_cfg is not None:
            cfg = resolved_cfg
        else:
            # Two things fight the fact that this server is launched from the RoboTwin
            # tree rather than from SpatialAlignVLA:
            #   * sapolicy/config/config.py runs parser.parse_args() at import time, so
            #     the server's own CLI would be consumed (and rejected) by it.
            #   * `CFG:configs/...` references inside the exp yaml resolve against CWD.
            # Feed the parser what it expects and resolve configs from the SAPolicy root.
            cfg_file = os.path.abspath(str(cfg_file))
            root = cfg_file
            while root != "/" and not os.path.isdir(os.path.join(root, "configs")):
                root = os.path.dirname(root)
            if root == "/":
                raise FileNotFoundError(f"could not locate a configs/ root above {cfg_file}")
            rel_cfg = os.path.relpath(cfg_file, root)

            # Runtime-only cfg overrides, e.g. the torch.compile switches. They change
            # no weights, so an already-trained checkpoint can be evaluated with them
            # without editing the training yaml. Space-separated key=value; dotted keys
            # nest (config.merge_from_opts -> yacs.CfgNode.__setitem__).
            cfg_opts = os.environ.get("SAPOLICY_CFG_OPTS", "").split()
            saved_argv, saved_cwd = sys.argv, os.getcwd()
            sys.argv = [saved_argv[0], "--cfg_file", rel_cfg,
                        "--entry", "eval_net"] + cfg_opts
            try:
                os.chdir(root)
                from sapolicy.config.config import make_cfg, args as cfg_args
                cfg = make_cfg(cfg_args)
            finally:
                sys.argv, _ = saved_argv, os.chdir(saved_cwd)

        from sapolicy.entrys.factory import get_model
        from sapolicy.entrys.normalizer_utils import (
            normalizer_fingerprint,
            resolve_eval_normalizer,
        )

        if hasattr(cfg, "model") and hasattr(cfg.model, "clear_output_dir"):
            cfg.model.clear_output_dir = False

        self.cfg = cfg
        self.device = torch.device(device)
        self.dtype = torch.float32

        model = get_model(cfg)
        # EMA weights live under ema_pipeline.* in the Lightning ckpt. Our EMAModel
        # has no copy_to(); load_pretrained_model(use_ema=True) remaps those keys
        # onto pipeline.* — do that here and do not call ema.copy_to on the still-
        # uninstantiated hydra DictConfig (that produced a false "EMA copy skipped").
        if hasattr(model, "use_ema"):
            model.use_ema = bool(use_ema)
        model.load_pretrained_model(ckpt_path, cfg.get("ckpt_type", None))
        if use_ema and getattr(model, "use_ema", False):
            print(
                "[SAPolicy] using EMA weights (ema_pipeline.* -> pipeline via load)",
                flush=True,
            )
        else:
            print("[SAPolicy] using non-EMA pipeline weights", flush=True)
        # Match eval_net: resolve normalizer (default auto: sidecar, then action
        # branch for joint runs, else ckpt).
        eval_cfg = dict(cfg.get("eval", None) or {})
        if normalizer_path:
            eval_cfg["normalizer_path"] = str(normalizer_path)
            eval_cfg.setdefault("normalizer_source", "path")
        source_label, rebound = resolve_eval_normalizer(
            model, cfg, eval_cfg, ckpt_path=ckpt_path
        )
        if rebound is not None:
            print(
                f"[SAPolicy] normalizer from {source_label} "
                f"({normalizer_fingerprint(rebound)})",
                flush=True,
            )
        else:
            print(f"[SAPolicy] keeping ckpt normalizer (source={source_label})", flush=True)
        model.pipeline.to(self.device).eval()
        self.policy = model

        self.obs_hist = int(cfg.get("obs_hist_length", 3))
        self.n_action_steps = int(n_action_steps)
        self.camera_name = str(cfg.get("eval_camera_name", "agentview"))
        self.normalize_actions = True
        self.min_depth = float(cfg.get("min_depth", getattr(cfg.model, "min_depth", 0.1)))
        self.max_depth = float(cfg.get("max_depth", getattr(cfg.model, "max_depth", 5.0)))

        self.transforms = self._build_transforms(cfg)
        # serve.py is a ThreadingTCPServer: one thread per client for the whole
        # connection lifetime. Keeping the observation window in thread-local storage
        # therefore isolates concurrent evaluation shards from each other -- a single
        # shared deque would interleave frames from different scenes and silently feed
        # the policy a mixed history. Model weights stay shared (one 1.7 GB copy).
        self._local = threading.local()
        # Forward passes are serialised: several shards issue get_action concurrently,
        # and a single nn.Module is not safe to run in parallel threads.
        self._infer_lock = threading.Lock()
        pipe = model.pipeline
        print(
            f"[SAPolicy] compile: backbone={bool(getattr(pipe, 'compile_backbone', False))} "
            f"action_head={bool(getattr(getattr(pipe, 'action_head', None), 'compile_step', False))} "
            f"mode={getattr(pipe, 'compile_mode', None)} "
            f"cuda_graph_sampler={bool(getattr(getattr(pipe, 'action_head', None), 'cuda_graph_sampler', False))}",
            flush=True,
        )
        if cfg_opts:
            print(f"[SAPolicy] cfg opts: {' '.join(cfg_opts)}", flush=True)
        print(f"[SAPolicy] ready: obs_hist={self.obs_hist}, "
              f"n_action_steps={self.n_action_steps}, camera={self.camera_name}, "
              f"depth_norm=[{self.min_depth},{self.max_depth}]", flush=True)
        if int(warmup_iterations) > 0:
            self._warmup(int(warmup_iterations), warmup_camera_names)

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _build_transforms(cfg):
        """Reuse the training transform stack so eval preprocessing cannot drift."""
        import hydra
        from torchvision.transforms import Compose  # same one robomimic_hdf5.py uses
        from sapolicy.entrys.normalizer_utils import is_joint_train_cfg, resolve_dataset_opt

        # Joint TCP+action: dataset_opts[0] is the TCP branch (single agentview);
        # closed-loop eval executes the action branch (third + wrists) -> use [-1].
        if is_joint_train_cfg(cfg):
            node = resolve_dataset_opt(cfg, -1)
        else:
            opts = cfg.data.train_dataset.dataset_opts
            node = opts[0] if isinstance(opts, (list, tuple)) else opts
        tf_cfg = node.get("transforms", None)
        if not tf_cfg:
            return None
        return Compose([hydra.utils.instantiate(t) for t in tf_cfg])

    def _resize_target_hw(self, orig_width, orig_height):
        """Training Resize output (h, w) for this input, via ``Resize.get_size``.

        Do not read ``width``/``height`` off the yaml object: ``get_size`` also
        applies ``keep_aspect_ratio``, ``ensure_multiple_of``, and ``resize_method``.
        """
        for transform in getattr(self.transforms, "transforms", None) or ():
            get_size = getattr(transform, "get_size", None)
            if not callable(get_size):
                continue
            new_width, new_height = get_size(int(orig_width), int(orig_height))
            return (int(new_height), int(new_width))
        return None

    @staticmethod
    def _as_uint8_rgb(img_hwc):
        image = np.asarray(img_hwc)
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"RGB must have shape [H,W,3], got {image.shape}")
        if image.dtype == np.uint8:
            return np.ascontiguousarray(image)
        if np.issubdtype(image.dtype, np.floating):
            scale = 255.0 if float(np.nanmax(image)) <= 1.0 + 1e-3 else 1.0
            image = np.clip(np.asarray(image, dtype=np.float32) * scale, 0, 255)
        return np.ascontiguousarray(image.astype(np.uint8))

    def _prep_frame(self, img_hwc, K=None, depth=None, *, native_hw=None):
        """Apply the training Resize/PrepareForNet stack to image (+ depth).

        RGB is AREA-resized as uint8 first (Pi-style: shrink then /255), then the
        remaining Compose runs on the small [0, 1] float. K is scaled from the
        *native* capture size, not guessed as ``(2*cx, 2*cy)``.
        """
        import cv2
        from sapolicy.dataset.transform import (
            normalize_metric_depth,
            scale_intrinsics_to_resolution,
        )

        image_u8 = self._as_uint8_rgb(img_hwc)
        orig_h, orig_w = int(image_u8.shape[0]), int(image_u8.shape[1])
        if native_hw is None:
            native_h, native_w = orig_h, orig_w
        else:
            native_h, native_w = int(native_hw[0]), int(native_hw[1])
            if native_h <= 0 or native_w <= 0:
                raise ValueError(f"native_hw must be positive, got {native_hw}")

        target = self._resize_target_hw(orig_w, orig_h)
        if target is not None and (orig_h, orig_w) != target:
            image_u8 = cv2.resize(
                image_u8, (target[1], target[0]), interpolation=cv2.INTER_AREA
            )

        K_native = (
            np.asarray(K, dtype=np.float32).copy() if K is not None else None
        )
        sample = {"image": image_u8.astype(np.float32) / 255.0}
        if depth is not None:
            d = np.asarray(depth, dtype=np.float32)
            if d.ndim == 3:
                d = d[..., 0]
            # RoboTwin depth is millimetres; training uses metres.
            if float(d.max()) > 100.0:
                d = d / 1000.0
            if target is not None and d.shape[:2] != target:
                d = cv2.resize(d, (target[1], target[0]), interpolation=cv2.INTER_NEAREST)
            sample["depth"] = d
        if self.transforms is not None:
            sample = self.transforms(sample)
        img = sample["image"]
        if img.ndim == 3 and img.shape[0] not in (1, 3):
            img = np.transpose(img, (2, 0, 1))
        img = np.ascontiguousarray(img, dtype=np.float32)
        if target is not None:
            got = (int(img.shape[-2]), int(img.shape[-1]))
            if got != target:
                raise RuntimeError(
                    f"prep produced {got}, expected {target} from "
                    f"Resize.get_size({orig_w}, {orig_h})"
                )
        K_out = None
        if K_native is not None:
            out_h, out_w = int(img.shape[-2]), int(img.shape[-1])
            K_out = scale_intrinsics_to_resolution(
                K_native,
                out_h,
                out_w,
                orig_height=native_h,
                orig_width=native_w,
            )
            K_out = np.ascontiguousarray(K_out, dtype=np.float32)
        d_out = sample.get("depth")
        if d_out is not None:
            d_out = np.ascontiguousarray(d_out, dtype=np.float32)
            d_out = np.squeeze(d_out)
            if d_out.ndim != 2:
                raise ValueError(f"depth must be HW after prep, got {d_out.shape}")
            # Match dataloader: resize first (above), then clip/normalize to [0, 1].
            d_out = normalize_metric_depth(d_out, self.min_depth, self.max_depth)
        return img, K_out, d_out

    def _prep_image(self, img_hwc):
        """uint8 HWC -> float CHW through the training transforms."""
        img, _, _ = self._prep_frame(img_hwc)
        return img

    def _state_vector(self, obs):
        """[pos(3), rot6d(6), grip(1)] per arm -> 20, matching the dataset layout."""
        blocks = []
        for side in ("left", "right"):
            tcp_pos, rot = _endpose_to_tcp(obs[f"{side}_endpose"])
            rot6d = rot[:, :2].T.reshape(6)  # column-vector convention, as in training
            blocks += [tcp_pos, rot6d, np.array([obs[f"{side}_gripper"]], dtype=np.float64)]
        return np.concatenate(blocks).astype(np.float32)

    def _warmup(self, iterations, camera_names=None):
        """Pay the torch.compile cost before the first real chunk.

        Compilation happens on the first forward and ``dynamic=False`` specialises
        on shape, so a dummy pass at the config's own ``input_image_size`` compiles
        exactly the graph the episode will use. Without it the first ``get_action``
        blocks for seconds (measured: ~7 s for the DiT step alone).

        Runs in the caller's (main) thread, whose ``obs_cache`` is separate from the
        per-connection ones, and clears it afterwards. Never fatal: a failed warmup
        costs latency on the first chunk, not the eval run.
        """
        import os
        import time

        pipe_cfg = self.cfg.model.get("pipeline", {}) if hasattr(self.cfg, "model") else {}
        size = pipe_cfg.get("input_image_size", None)
        if not size or len(size) != 2:
            print("[SAPolicy] warmup skipped: model.pipeline.input_image_size missing",
                  flush=True)
            return
        h, w = int(size[0]), int(size[1])

        if camera_names is None:
            # Mirror the client: SAPOLICY_CAMERA_MAP present -> multi-camera obs.
            spec = os.environ.get("SAPOLICY_CAMERA_MAP", "").strip()
            camera_names = [item.split("=", 1)[0].strip()
                            for item in spec.split(",") if item.strip()] or None
        elif isinstance(camera_names, str):
            camera_names = [c.strip() for c in camera_names.split(",") if c.strip()]
        multi = bool(camera_names)
        cams = list(camera_names) if multi else [self.camera_name]

        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        # Centred principal point: _prep_frame gets native_hw from the array anyway,
        # and the dummy is already at the training resolution, so K passes through.
        K = np.array([[w / 2.0, 0.0, w / 2.0],
                      [0.0, h / 2.0, h / 2.0],
                      [0.0, 0.0, 1.0]], dtype=np.float64)
        depth = (np.ones((h, w), dtype=np.float32)
                 if bool(pipe_cfg.get("use_depth", False)) else None)

        identity_pose = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        obs = {
            "left_endpose": identity_pose.copy(),
            "right_endpose": identity_pose.copy(),
            "left_gripper": 0.0,
            "right_gripper": 0.0,
            "image": rgb,
            "intrinsic_cv": K,
        }
        if depth is not None:
            obs["depth"] = depth
        if multi:
            obs["camera_names"] = cams
            obs["images"] = {c: rgb for c in cams}
            obs["intrinsics"] = {c: K for c in cams}
            if depth is not None:
                obs["depths"] = {c: depth for c in cams}

        try:
            self.reset_model()
            t0 = time.perf_counter()
            for i in range(iterations):
                self.update_obs(obs)
                self.get_action()
                if i == 0:
                    print(f"[SAPolicy] warmup 1/{iterations} "
                          f"({time.perf_counter() - t0:.1f}s, compile included)",
                          flush=True)
            print(f"[SAPolicy] warmup done: {iterations} iters, "
                  f"{time.perf_counter() - t0:.1f}s, {h}x{w}, cameras={cams}",
                  flush=True)
        except Exception as exc:  # never block startup on a dummy forward
            print(f"[SAPolicy] warmup failed ({type(exc).__name__}: {exc}); "
                  f"the first real chunk will pay the compile cost", flush=True)
        finally:
            self.reset_model()

    # ------------------------------------------------------------- rpc surface
    @property
    def obs_cache(self):
        """Per-connection observation window (see _local in __init__)."""
        cache = getattr(self._local, "obs_cache", None)
        if cache is None:
            cache = deque(maxlen=self.obs_hist)
            self._local.obs_cache = cache
        return cache

    def reset_model(self):
        self.obs_cache.clear()
        return True

    def update_obs(self, obs):
        obs = {k: (np.asarray(v) if isinstance(v, (list, np.ndarray)) else v)
               for k, v in obs.items()}
        cache = self.obs_cache
        if not cache:  # first frame of an episode fills the whole window
            for _ in range(self.obs_hist):
                cache.append(obs)
        else:
            cache.append(obs)
        return True

    @torch.no_grad()
    def get_action(self, rtc_sampling=None):
        if not self.obs_cache:
            raise RuntimeError("get_action called before any update_obs")
        frames = list(self.obs_cache)

        # Camera set for this episode. Multi-camera policies (anchor view + both
        # wrists) get a per-camera dict from encode_obs; single-camera checkpoints
        # keep the flat "image" key and the historical single-name behaviour.
        # update_obs turns any list into an ndarray, so `or` on it would raise
        # "truth value of an array is ambiguous" -- test for None/emptiness explicitly.
        _cn = frames[-1].get("camera_names")
        cam_names = [str(c) for c in _cn] if _cn is not None and len(_cn) else [self.camera_name]
        multi = "images" in frames[-1]

        def _raw_K(frame, cam):
            if multi:
                return (frame.get("intrinsics") or {}).get(cam)
            return frame.get("intrinsic_cv")

        def _raw_depth(frame, cam):
            if multi:
                return (frame.get("depths") or {}).get(cam)
            return frame.get("depth")

        def _native_hw(frame, cam, img_raw):
            raw = frame.get("image_native_hw")
            if isinstance(raw, dict):
                raw = raw.get(cam)
            if raw is not None and len(raw) == 2:
                return (int(raw[0]), int(raw[1]))
            return (int(img_raw.shape[0]), int(img_raw.shape[1]))

        def _window(cam):
            """Stack obs-history for one camera; scale K with the same Resize."""
            imgs, Ks, depths = [], [], []
            last_id = None
            last_prep = None
            for f in frames:
                img_raw = f["images"][cam] if multi else f["image"]
                # First-obs padding stores the same array several times.
                key = id(img_raw)
                if key == last_id and last_prep is not None:
                    img, K, d = last_prep
                else:
                    img, K, d = self._prep_frame(
                        img_raw,
                        K=_raw_K(f, cam),
                        depth=_raw_depth(f, cam),
                        native_hw=_native_hw(f, cam, img_raw),
                    )
                    last_id = key
                    last_prep = (img, K, d)
                imgs.append(img)
                Ks.append(K)
                depths.append(d)
            img_t = torch.from_numpy(np.stack(imgs)).unsqueeze(0).to(
                self.device, self.dtype)
            missing_k = [i for i, k in enumerate(Ks) if k is None]
            if missing_k:
                raise ValueError(
                    f"Missing camera intrinsics for {cam!r} in obs-history "
                    f"frames {missing_k} (need K for every camera; do not "
                    f"assume cameras share intrinsics or silently drop to None)"
                )
            # Latest-frame K (static cameras); shape [1,3,3]
            K_t = torch.as_tensor(Ks[-1], dtype=self.dtype, device=self.device).unsqueeze(0)
            if any(d is None for d in depths):
                d_t = None
            else:
                d_t = torch.from_numpy(np.stack(depths)[:, None]).unsqueeze(0).to(
                    self.device, self.dtype)
            return img_t, K_t, d_t

        # Pass per-camera K for every stream. Wrist / third-person intrinsics can
        # differ; silently keeping only the anchor (or setting the whole dict to
        # None when any cam is missing) hides train/eval mismatches.
        images_arg, intrinsics_arg, depths_arg = {}, {}, {}
        for cam in cam_names:
            img_t, K_t, d_t = _window(cam)
            images_arg[cam] = img_t
            intrinsics_arg[cam] = K_t
            if d_t is not None:
                depths_arg[cam] = d_t
        if not depths_arg:
            depths_arg = None
        states = np.stack([self._state_vector(f) for f in frames])         # [T,20]
        state_t = torch.from_numpy(states).unsqueeze(0).to(self.device, self.dtype)

        norm = getattr(self.policy.pipeline, "normalizer", None)
        if norm is not None and "state" in getattr(norm, "params_dict", {}):
            state_t = norm.normalize({"state": state_t})["state"]

        with self._infer_lock:
            context = nullcontext()
            if rtc_sampling is not None:
                head = self.policy.pipeline.action_head
                if not callable(getattr(head, "rtc_condition", None)):
                    raise NotImplementedError("RTC requires the SAPolicy DiT action head")
                relative = self._rtc_relative_actions(rtc_sampling["action_condition"], frames[-1])
                condition = torch.as_tensor(relative, device=self.device, dtype=self.dtype)[None]
                if self.normalize_actions and norm is not None:
                    condition = norm.normalize({"action": condition})["action"]
                weights = torch.as_tensor(
                    rtc_sampling["condition_weights"], device=self.device, dtype=self.dtype
                )[None, :, None]
                context = head.rtc_condition(condition, weights, rtc_sampling["beta"])
            with context:
                outputs = self.policy.pipeline.infer(
                    images_arg,
                    depths_arg,
                    camera_intrinsics=intrinsics_arg,
                    state=state_t,
                    camera_names=cam_names,
                    camera_extrinsics=None,
                )
        actions = outputs.get("actions", outputs.get("action_sequence"))
        if isinstance(actions, tuple):
            actions = actions[0]
        # infer() may hand back numpy; the normalizer needs a tensor on-device.
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions)
        actions = actions.to(device=self.device, dtype=self.dtype)
        if actions.dim() == 2:
            actions = actions.unsqueeze(0)
        if actions.dim() == 4:
            actions = actions[:, 0]
        if self.normalize_actions and norm is not None:
            actions = norm.unnormalize({"action": actions})["action"]
        actions = actions[0].detach().cpu().numpy().astype(np.float64)  # [T, 20]

        return self._to_robotwin_ee(actions, frames[-1])

    # --------------------------------------------------------------- action io
    def _rtc_relative_actions(self, condition, cur_obs):
        """Absolute model-frame WXYZ EE16 -> current-observation-relative pose18/grip2.

        Inverse of _to_robotwin_ee, before applying the checkpoint normalizer.
        FK belongs to the embodiment adapter, so this method has no robot dependency.
        """
        condition = np.asarray(condition, dtype=np.float64)
        relative = np.empty((len(condition), 20), dtype=np.float64)
        for arm, side in enumerate(("left", "right")):
            row = condition[:, arm * 8 : arm * 8 + 8]
            current_position, current_rotation = _endpose_to_tcp(cur_obs[f"{side}_endpose"])
            rotation = Rot.from_quat(_wxyz_to_xyzw(row[:, 3:7])).as_matrix()
            position = row[:, :3] + rotation @ np.array([TCP_FORWARD_OFFSET, 0.0, 0.0])
            relative[:, arm * 9 : arm * 9 + 3] = (position - current_position) @ current_rotation
            relative_rotation = current_rotation.T @ rotation
            relative[:, arm * 9 + 3 : arm * 9 + 9] = np.concatenate(
                [relative_rotation[:, :, 0], relative_rotation[:, :, 1]], axis=-1
            )
            relative[:, 18 + arm] = row[:, 7]
        return relative

    def _to_robotwin_ee(self, actions, cur_obs):
        """[T,20] body-frame relative -> [n_action_steps,16] absolute ee commands."""
        T = actions.shape[0]
        pose_l, pose_r = actions[:, 0:9], actions[:, 9:18]
        grip_l, grip_r = actions[:, 18:19], actions[:, 19:20]

        cmds = []
        for pose, grip, side in ((pose_l, grip_l, "left"), (pose_r, grip_r, "right")):
            cur_pos, cur_rot = _endpose_to_tcp(cur_obs[f"{side}_endpose"])
            rel_pos, rel_6d = pose[:, :3], pose[:, 3:9]

            # 6D -> matrix, same column-vector convention the dataset used.
            a = rel_6d[:, :3]
            b = rel_6d[:, 3:]
            a = a / np.clip(np.linalg.norm(a, axis=-1, keepdims=True), 1e-8, None)
            b = b - (a * b).sum(-1, keepdims=True) * a
            b = b / np.clip(np.linalg.norm(b, axis=-1, keepdims=True), 1e-8, None)
            rel_rot = np.stack([a, b, np.cross(a, b)], axis=-1)  # [T,3,3]

            abs_pos = cur_pos + (cur_rot @ rel_pos.T).T          # p_cur + R_cur @ rel
            abs_rot = cur_rot @ rel_rot                          # R_cur * rel_rot

            # TCP -> endpose: take_action(ee) wants the pose 12 cm behind the TCP.
            ep_pos = abs_pos - np.einsum(
                "tij,j->ti", abs_rot, np.array([TCP_FORWARD_OFFSET, 0.0, 0.0]))
            quat_wxyz = _xyzw_to_wxyz(Rot.from_matrix(abs_rot).as_quat())
            cmds.append(np.concatenate([ep_pos, quat_wxyz, grip], axis=-1))  # [T,8]

        out = np.concatenate(cmds, axis=-1)  # [T,16]
        if T < self.n_action_steps:
            out = np.concatenate([out, np.repeat(out[-1:], self.n_action_steps - T, 0)])
        return out[: self.n_action_steps]
