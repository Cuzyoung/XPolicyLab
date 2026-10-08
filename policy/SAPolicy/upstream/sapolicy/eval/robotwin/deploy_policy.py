"""RoboTwin eval adapter for SAPolicy.

This module is imported in BOTH processes, so it must stay importable under the
RoboTwin conda env (sapien/curobo, no lightning/hydra/timm):

  * client  (RoboTwin env) -- runs `encode_obs`, `eval`, `reset_model`; drives the
    simulator and talks to the policy over RoboTwin's socket protocol.
  * server  (sa env)       -- runs `get_model`; everything torch/SAPolicy is
    imported lazily inside it, never at module scope.

Launch:
    # policy side, sa env
    workspace=... python script/policy_model_server.py --config policy/SAPolicy/deploy_policy.yml
    # sim side, RoboTwin env
    python script/eval_policy_client.py --config policy/SAPolicy/deploy_policy.yml
"""
import numpy as np

# envs/robot/robot.py::_trans_endpose subtracts exactly this along the gripper's
# local +x when producing what RoboTwin stores as `endpose`. The converter adds it
# back to recover the TCP, so the two conventions differ by 12 cm and the eval path
# must undo it again before handing poses to take_action(). Verified empirically:
# feeding `endpose` to _trans_from_gripper_to_endlink reproduces the true end link
# to 0.000 mm, feeding the TCP is off by 120.000 mm.
TCP_FORWARD_OFFSET = 0.12


def _camera_map():
    """canonical model-facing name -> RoboTwin camera name, from the environment.

    Multi-camera policies (e.g. anchor view + both wrists) need every camera the
    training config declared under camera_pair_choices.canonical_names, and the
    RoboTwin side names them differently. encode_obs only receives the observation
    dict, so the mapping arrives out-of-band:

        SAPOLICY_CAMERA_MAP="agentview=observer_camera,\
    robot0_eye_in_hand=left_camera,robot1_eye_in_hand=right_camera"

    Unset -> single-camera behaviour, unchanged.
    """
    import os
    spec = os.environ.get("SAPOLICY_CAMERA_MAP", "").strip()
    if not spec:
        return None
    pairs = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        canonical, _, robotwin = item.partition("=")
        if not robotwin:
            raise ValueError(f"SAPOLICY_CAMERA_MAP entry {item!r} is not canonical=robotwin_name")
        pairs.append((canonical.strip(), robotwin.strip()))
    return pairs or None


def _pack_camera(cams, name):
    """One RoboTwin camera -> (rgb, depth, K), with depth/K omitted when absent."""
    if name not in cams:
        raise KeyError(f"camera {name!r} not in observation (have: {sorted(cams)})")
    cam = cams[name]
    rgb = np.asarray(cam["rgb"], dtype=np.uint8)
    depth = None if cam.get("depth") is None else np.asarray(cam["depth"], dtype=np.float32)
    K = None if cam.get("intrinsic_cv") is None else np.asarray(cam["intrinsic_cv"], dtype=np.float64)
    return rgb, depth, K


def encode_obs(observation):
    """RoboTwin's nested observation dict -> flat, JSON/base64-friendly arrays.

    Runs on the client. Keep it to plain numpy so the socket layer can encode it.
    """
    cams = observation["observation"]
    ep = observation["endpose"]
    out = {
        "left_endpose": np.asarray(ep["left_endpose"], dtype=np.float64),
        "right_endpose": np.asarray(ep["right_endpose"], dtype=np.float64),
        "left_gripper": float(ep["left_gripper"]),
        "right_gripper": float(ep["right_gripper"]),
    }

    mapping = _camera_map()
    if mapping is not None:
        # Multi-camera: send every canonical stream. The flat single-camera keys
        # stay populated from the first entry so the recorder and any
        # single-camera checkpoint keep working unchanged.
        out["camera_names"] = [c for c, _ in mapping]
        images, depths, intrinsics = {}, {}, {}
        for canonical, robotwin in mapping:
            rgb, depth, K = _pack_camera(cams, robotwin)
            images[canonical] = rgb
            if depth is not None:
                depths[canonical] = depth
            if K is not None:
                intrinsics[canonical] = K
        out["images"] = images
        if depths:
            out["depths"] = depths
        if intrinsics:
            out["intrinsics"] = intrinsics
        first = mapping[0][0]
        out["image"] = images[first]
        if first in depths:
            out["depth"] = depths[first]
        if first in intrinsics:
            out["intrinsic_cv"] = intrinsics[first]
        return out

    cam_name = "agentview" if "agentview" in cams else "head_camera"
    rgb, depth, K = _pack_camera(cams, cam_name)
    out["image"] = rgb
    # Depth + intrinsics ride along when the env provides them. lowtcp models
    # REQUIRE both (the latent-aux trunk sits in the action path and its
    # ray_depth geometry needs real depth); notcp models use the intrinsics
    # embedding too (use_camera_intrinsics=true), so omitting it is a silent
    # train/eval mismatch.
    if depth is not None:
        out["depth"] = depth
    if K is not None:
        out["intrinsic_cv"] = K
    return out


def get_model(usr_args):
    """Server side only. Heavy imports live in the lazily-imported module."""
    from .sa_policy_server import SAPolicyRoboTwinModel

    return SAPolicyRoboTwinModel(
        cfg_file=usr_args["sapolicy_cfg"],
        resolved_cfg=usr_args.get("resolved_cfg"),
        ckpt_path=usr_args["ckpt_path"],
        workspace=usr_args.get("workspace"),
        n_action_steps=int(usr_args.get("n_action_steps", 8)),
        device=usr_args.get("device", "cuda"),
        use_ema=bool(usr_args.get("use_ema", True)),
        normalizer_path=usr_args.get("normalizer_path"),
        tcp_forward_offset_m=usr_args.get("tcp_forward_offset_m"),
        warmup_iterations=int(usr_args.get("warmup_iterations", 0)),
        warmup_camera_names=usr_args.get("warmup_camera_names"),
    )


# Policy-view recording: the built-in eval video hardcodes head_camera, which is
# NOT what the policy sees and has repeatedly misled failure analysis. When
# POLICY_VIEW_DIR is set, every frame handed to the model is also buffered here
# and flushed to <dir>/episode<N>_<succ|fail>.mp4 at the next reset.
_pv_frames = []
_pv_episode = [0]


def _pv_record(obs_dict):
    """Buffer exactly what the policy sees. Multi-camera runs tile the streams
    left-to-right (anchor, then each wrist) so one video shows the whole input."""
    import os
    if not os.environ.get("POLICY_VIEW_DIR"):
        return
    imgs = obs_dict.get("images")
    if imgs:
        order = list(obs_dict.get("camera_names") or imgs.keys())
        tiles = [np.asarray(imgs[c], dtype=np.uint8) for c in order if c in imgs]
        h = max(t.shape[0] for t in tiles)
        tiles = [t if t.shape[0] == h else
                 np.pad(t, ((0, h - t.shape[0]), (0, 0), (0, 0))) for t in tiles]
        _pv_frames.append(np.concatenate(tiles, axis=1).copy())
    else:
        _pv_frames.append(np.asarray(obs_dict["image"], dtype=np.uint8).copy())


def _pv_flush(success=None, task_env=None):
    import os
    out_dir = os.environ.get("POLICY_VIEW_DIR")
    if not out_dir or not _pv_frames:
        _pv_frames.clear()
        return
    # take_action(ee) runs a whole planned segment (many scene.step()s) and the
    # success check fires *inside* it, so the decisive motion -- the lift itself --
    # happens after the last observation the policy ever saw. Grab one final frame
    # so the video shows the outcome, mirroring what RoboTwin does for its own
    # recording (_base_task.py: writes a frame when check_success() flips).
    if task_env is not None:
        try:
            _pv_record(encode_obs(task_env.get_obs()))
        except Exception:
            pass
    os.makedirs(out_dir, exist_ok=True)
    import cv2
    h, w = _pv_frames[0].shape[:2]
    tag = "" if success is None else ("_succ" if success else "_fail")
    # Eval runs 4 shards in parallel, each a separate process with its own episode
    # counter; without the seed prefix they overwrite each other's files.
    shard = os.environ.get("POLICY_VIEW_TAG", "")
    path = os.path.join(out_dir, f"{shard}episode{_pv_episode[0]}{tag}.mp4")
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 15, (w, h))
    for f in _pv_frames:
        vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    vw.release()
    _pv_frames.clear()
    _pv_episode[0] += 1


def eval(TASK_ENV, model, observation):
    """One policy step: observe -> action chunk -> execute the chunk.

    `model` is a ModelClient proxy when running split; `.call(name, obs)` maps to
    the method of that name on the server-side model object.
    """
    obs = encode_obs(observation)
    _pv_record(obs)
    model.call("update_obs", obs)

    actions = model.call("get_action")  # (n_action_steps, 16) in RoboTwin ee layout
    actions = np.asarray(actions, dtype=np.float64)

    for action in actions:
        # ee layout: [left_pos(3) left_quat_wxyz(4) left_grip(1)
        #             right_pos(3) right_quat_wxyz(4) right_grip(1)]
        TASK_ENV.take_action(action, action_type="ee")
        if TASK_ENV.eval_success or TASK_ENV.take_action_cnt >= TASK_ENV.step_lim:
            _pv_flush(success=bool(TASK_ENV.eval_success), task_env=TASK_ENV)
            return
        obs = encode_obs(TASK_ENV.get_obs())
        _pv_record(obs)
        model.call("update_obs", obs)


def reset_model(model):
    """Clear the observation window between evaluation episodes."""
    _pv_flush()  # episode ended without hitting the in-chunk terminal branch
    model.call("reset_model")
