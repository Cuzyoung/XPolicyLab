#!/usr/bin/env python3
"""Capture/verify local Pi05 model evidence without a server or robot.

Run --help in the lightweight environment. Actual capture/verify requires the
installed Pi05 model environment, local weights and already cached tokenizer.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import importlib.metadata
import importlib.util
import inspect
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = 1


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def content_manifest(path):
    """Hash actual bytes, including checkpoint shards and normalization assets."""
    path = Path(path).resolve(strict=True)
    if path.is_dir() and any(p.is_symlink() and (p.is_dir() or not p.exists()) for p in path.rglob("*")):
        raise ValueError("Directory/broken symlinks are not supported in evidence manifests")
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    if not files:
        raise ValueError(f"No files found to identify {path}")
    records = {}
    for file in files:
        before = file.stat()
        value = digest_file(file)
        after = file.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(f"File changed while hashing: {file}")
        records[file.name if path.is_file() else str(file.relative_to(path))] = value
    encoded = json.dumps(records, sort_keys=True).encode()
    return {"path": str(path), "sha256": hashlib.sha256(encoded).hexdigest(), "files": records}


def source_identity(checkout, expected):
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=checkout, text=True).strip()
    revision = git("rev-parse", "HEAD")
    if len(expected) != 40 or revision != expected:
        raise ValueError(f"Expected full checkout revision {expected}, found {revision}")
    # Untracked files also matter: a locally added import can change the model.
    if git("status", "--porcelain", "--untracked-files=all"):
        raise ValueError("Use a clean committed checkout for model evidence")
    return {"revision": revision, "clean": True}


def read_inputs(observation_path, arrays_path):
    """JSON contains only an observation tree with explicit NPZ array leaves."""
    document = json.loads(observation_path.read_text())
    if type(document.get("schema_version")) is not int or document["schema_version"] != SCHEMA:
        raise ValueError("Observation schema_version must be integer 1")
    if not isinstance(document.get("source"), str) or not document["source"].strip():
        raise ValueError("Observation source must describe recorded/live/synthetic provenance")
    with np.load(arrays_path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    for name, value in arrays.items():
        if value.dtype.kind not in "buif" or not np.isfinite(value).all():
            raise ValueError(f"{name} must contain finite numeric data")
    used = {"noise"}

    def expand(value):
        if isinstance(value, dict):
            if "array" in value:
                if set(value) != {"array"} or not isinstance(value["array"], str):
                    raise ValueError("Array references must be exactly {\"array\": \"name\"}")
                name = value["array"]
                if name == "noise" or name not in arrays:
                    raise ValueError(f"Invalid observation array reference: {name}")
                used.add(name)
                return arrays[name].copy()
            return {key: expand(item) for key, item in value.items()}
        if isinstance(value, list):
            return [expand(item) for item in value]
        if value is None or type(value) in (str, int, float, bool):
            return value
        raise ValueError("Unsupported observation value")

    observation = expand(document.get("observation"))
    noise = arrays.get("noise")
    if not isinstance(observation, dict) or noise is None or noise.dtype != np.float32 or noise.ndim != 2:
        raise ValueError("Provide an observation mapping and float32 noise[H,Dmodel]")
    if used != set(arrays):
        raise ValueError(f"Unused NPZ arrays: {sorted(set(arrays) - used)}")
    identity = {"observation_json_sha256": digest_file(observation_path),
                "arrays_npz_sha256": digest_file(arrays_path), "source": document["source"]}
    return observation, noise, identity


def check_baseline(report, identity, expected_revision, current_revision):
    if (type(report.get("schema_version")) is not int or report["schema_version"] != SCHEMA
            or report.get("mode") != "capture" or report.get("passed") is not True):
        raise ValueError("Baseline must be a successful capture report from this probe")
    if report.get("probe_sha256") != digest_file(Path(__file__)):
        raise ValueError("Capture and verify must use the same probe source")
    source = report.get("source", {})
    if source.get("revision") != expected_revision or source.get("clean") is not True:
        raise ValueError("Baseline does not identify the requested clean old revision")
    if expected_revision == current_revision:
        raise ValueError("The modified checkout cannot serve as its own old-backend baseline")
    if report.get("identity") != identity:
        mismatches = [key for key in set(identity) | set(report.get("identity", {}))
                      if report.get("identity", {}).get(key) != identity.get(key)]
        raise ValueError(f"Baseline identity mismatch: {sorted(mismatches)}")


def compare(actual, expected, *, rtol, atol):
    if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError("Compared actions must have identical shapes and finite values")
    error = float(np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64))))
    passed = bool(np.allclose(actual, expected, rtol=rtol, atol=atol, equal_nan=False))
    return {"passed": passed, "max_abs_error": error, "rtol": rtol, "atol": atol}


def load_backend(checkout):
    """Use the selected real checkout, even when its directory is not XPolicyLab."""
    paths = [checkout, checkout / "policy/Pi_05/openpi/src",
             checkout / "policy/Pi_05/openpi/packages/openpi-client/src"]
    sys.path[:0] = [str(path) for path in paths]
    spec = importlib.util.spec_from_file_location("XPolicyLab", checkout / "__init__.py",
                                                submodule_search_locations=[str(checkout)])
    module = importlib.util.module_from_spec(spec)
    sys.modules["XPolicyLab"] = module
    spec.loader.exec_module(module)
    return importlib.import_module("XPolicyLab.policy.Pi_05.model")


def run(args, report, arrays_out):
    import yaml
    config = yaml.safe_load(args.config.read_text())
    if not isinstance(config, dict) or config.get("policy_name") != "Pi_05":
        raise ValueError("--config must be a resolved standalone Pi_05 deployment YAML")
    if "${" in str(config):
        raise ValueError("Resolve all station/config interpolation before running the probe")
    observation, noise, input_identity = read_inputs(args.observation, args.arrays)
    report["source"] = source_identity(args.checkout, args.expected_revision)
    backend = load_backend(args.checkout)
    from openpi.policies.policy import Policy
    if not Path(inspect.getfile(Policy)).resolve().is_relative_to(args.checkout):
        raise ValueError("OpenPI was imported from a different checkout")
    supports_trace = "return_denoise_trace" in inspect.signature(Policy.infer).parameters
    if supports_trace != (args.mode == "verify"):
        raise ValueError("capture requires the old no-trace Policy; verify requires the trace patch")
    train_config = backend._resolve_train_config(config)
    if (not getattr(train_config.model, "pi05", False)
            or type(train_config.data).__name__ not in {"LeRobotAlohaDataConfig", "LeRobotYamDataConfig"}
            or config.get("action_type", "joint") != "joint"
            or config.get("robot_action_dim_info") != {"arm_dim": [6, 6], "ee_dim": [1, 1]}
            or config.get("num_envs", 1) != 1):
        raise ValueError("Use Pi05 native ALOHA/YAM joint data, explicit dual 6+1 layout and num_envs=1")
    checkpoint = backend._resolve_pi05_model_root(config).resolve(strict=True)
    if not (checkpoint / "params").is_dir() or (checkpoint / "model.safetensors").exists():
        raise ValueError("Provide an existing local JAX checkpoint with params/")
    checkpoint_manifest = content_manifest(checkpoint)
    normalized_config = {key: value for key, value in config.items() if key != "return_denoise_trace"}
    from openpi.shared import download
    assets = {}
    original_download = download.maybe_download

    def local_asset(url, **_kwargs):
        # Deny downloads, including tokenizer refresh. Use only existing bytes.
        parsed = urlparse(str(url))
        if parsed.scheme:
            cache = Path(os.getenv("OPENPI_DATA_HOME", "~/.cache/openpi")).expanduser().resolve()
            path = (cache / parsed.netloc / parsed.path.lstrip("/")).resolve()
            if not path.is_relative_to(cache):
                raise ValueError("Cached asset escapes OPENPI_DATA_HOME")
        else:
            path = Path(url).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Provision asset before this offline probe: {url} -> {path}")
        # The checkpoint is already fully hashed above, avoid hashing it twice.
        assets[str(url)] = checkpoint_manifest if path == checkpoint else content_manifest(path)
        return path

    download.maybe_download = local_asset
    try:
        configured = dict(config, return_denoise_trace=args.mode == "verify")
        report["model_execution_attempted"] = True
        model = backend.Model(configured)
        report["model_loaded"] = True
        if model.policy._is_pytorch_model or not model._train_config.model.pi05:
            raise ValueError("Only real JAX Pi05 is supported")
        if model.observation_profile not in {"aloha", "yam_native"} or model.robot_action_dim != 14:
            raise ValueError("Use the native dual-arm joint14 profile")
        if noise.shape != (model.action_horizon, model.action_dim):
            raise ValueError(f"noise must have shape {(model.action_horizon, model.action_dim)}")
        import jax
        versions = {}
        for name in ("jax", "jaxlib", "flax", "numpy", "torch", "openpi"):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = None
        identity = {
            "inputs": input_identity, "config": normalized_config,
            "checkpoint": checkpoint_manifest,
            "normalization": content_manifest(model.norm_stats_path or checkpoint / "assets"),
            "assets": assets, "versions": versions, "python": platform.python_version(),
            "devices": [{"platform": d.platform, "kind": d.device_kind, "id": d.id} for d in jax.devices()],
            "model": {"horizon": model.action_horizon, "dim": model.action_dim,
                      "profile": model.observation_profile, "train_config": model._train_config.name},
            "num_steps": args.num_steps, "repeats": args.repeats,
        }
        report["identity"] = identity
        reference = None
        if args.mode == "verify":
            baseline = json.loads((args.baseline / "report.json").read_text())
            check_baseline(baseline, identity, args.baseline_revision, report["source"]["revision"])
            path = args.baseline / "arrays.npz"
            if digest_file(path) != baseline.get("arrays_sha256"):
                raise ValueError("Baseline output array hash mismatch")
            with np.load(path, allow_pickle=False) as archive:
                reference = {key: archive[key].copy() for key in archive.files}
        model.update_obs(copy.deepcopy(observation))
        policy_observation = backend.slice_stacked_obs(model.observation_window, 0)
        report["checks"], report["timings"] = {}, {}

        def infer(steps, trace, explicit_noise):
            kwargs = {"num_steps": steps}
            if trace:
                kwargs["return_denoise_trace"] = True
            if explicit_noise:
                kwargs["noise"] = noise.copy()
            started = time.perf_counter()
            result = model.policy.infer(copy.deepcopy(policy_observation), **kwargs)
            actions = np.asarray(result["actions"])
            if actions.shape != (model.action_horizon, 14) or not np.isfinite(actions).all():
                raise ValueError("Expected finite transformed actions[H,14]")
            velocity = None
            if trace:
                velocity = np.asarray(result["denoise_trace"]["v"])
                if (velocity.dtype != np.float32 or not np.isfinite(velocity).all()
                        or velocity.shape != (steps, model.action_horizon, model.action_dim)):
                    raise ValueError("Expected finite float32 trace[T,H,Dmodel]")
            elapsed = (time.perf_counter() - started) * 1000
            if trace and explicit_noise:
                reconstructed = noise.copy()
                for row in velocity:
                    reconstructed += np.float32(-1 / steps) * row
                inputs = model.policy._input_transform(copy.deepcopy(policy_observation))
                decoded = model.policy._output_transform({"state": inputs["state"], "actions": reconstructed})
                report["checks"][f"trace_reconstruction_{len(report['checks'])}"] = compare(
                    np.asarray(decoded["actions"]), actions, rtol=args.rtol, atol=args.atol)
            return actions.copy(), velocity, elapsed

        for steps in args.num_steps:
            for trace in (False, True) if args.mode == "verify" else (False,):
                label = f"steps{steps}_{'on' if trace else 'off'}"
                model.reset()
                timings = []
                for repeat in range(args.repeats):
                    actions, velocity, elapsed = infer(steps, trace, True)
                    timings.append(elapsed)
                    key = f"{label}_explicit_{repeat}"
                    arrays_out[key] = actions
                    if trace:
                        arrays_out[f"{label}_trace_{repeat}"] = velocity
                    if reference is not None:
                        old = reference[f"steps{steps}_off_explicit_{repeat}"]
                        report["checks"][key + "_old"] = compare(
                            actions, old, rtol=args.rtol if trace else 0, atol=args.atol if trace else 0)
                    if trace:
                        off = arrays_out[f"steps{steps}_off_explicit_{repeat}"]
                        report["checks"][key + "_off"] = compare(actions, off, rtol=args.rtol, atol=args.atol)
                report["timings"][label] = {"first_call_ms": timings[0], "subsequent_ms": timings[1:]}
                # No explicit noise: verify real policy RNG consumption and reset.
                for reset_pass in range(2):
                    model.reset()
                    for index in range(args.repeats):
                        actions, _, _ = infer(steps, trace, False)
                        key = f"{label}_sequence_{index}"
                        if reset_pass:
                            report["checks"][key + "_reset"] = compare(actions, arrays_out[key], rtol=0, atol=0)
                        else:
                            arrays_out[key] = actions
                            if reference is not None:
                                report["checks"][key + "_old"] = compare(
                                    actions, reference[f"steps{steps}_off_sequence_{index}"],
                                    rtol=args.rtol if trace else 0, atol=args.atol if trace else 0)
                            if trace:
                                report["checks"][key + "_off"] = compare(
                                    actions, arrays_out[f"steps{steps}_off_sequence_{index}"], rtol=args.rtol, atol=args.atol)
                rng = np.asarray(jax.random.key_data(model.policy._rng))
                arrays_out[label + "_rng"] = rng
                if reference is not None:
                    report["checks"][label + "_rng_old"] = {"passed": bool(np.array_equal(rng, reference[f"steps{steps}_off_rng"]))}
                if trace:
                    report["checks"][label + "_rng"] = {"passed": bool(np.array_equal(rng, arrays_out[f"steps{steps}_off_rng"]))}
            if args.mode == "verify":
                model.reset()
                model.update_obs(copy.deepcopy(observation))
                reply = model.get_action(noise=noise.copy(), num_steps=steps)
                if not isinstance(reply, dict) or "actions" not in reply:
                    raise ValueError("Model must return actions and denoise_trace together")
                report["checks"][f"steps{steps}_model_trace"] = compare(
                    np.asarray(reply["denoise_trace"]["v"]), arrays_out[f"steps{steps}_on_trace_0"], rtol=0, atol=0)
                packed = np.stack([backend.pack_robot_state(
                    {"state": row}, model.action_type, model.robot_action_dim_info, source_type="obs"
                ) for row in reply["actions"]])
                report["checks"][f"steps{steps}_model_actions"] = compare(
                    packed, arrays_out[f"steps{steps}_on_explicit_0"], rtol=0, atol=0)
        report["passed"] = all(check["passed"] for check in report["checks"].values())
    finally:
        download.maybe_download = original_download


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("capture", "verify"))
    parser.add_argument("--checkout", type=Path, default=ROOT)
    parser.add_argument("--expected-revision", required=True, help="Full clean source SHA to run")
    parser.add_argument("--config", type=Path, required=True, help="Resolved native Pi_05 YAML")
    parser.add_argument("--observation", type=Path, required=True, help="JSON observation tree and provenance")
    parser.add_argument("--arrays", type=Path, required=True, help="Non-pickle NPZ, including float32 noise[H,D]")
    parser.add_argument("--output", type=Path, required=True, help="New artifact directory outside the checkout")
    parser.add_argument("--baseline", type=Path, help="Old capture artifact directory, required for verify")
    parser.add_argument("--baseline-revision", help="Expected old full SHA, required for verify")
    parser.add_argument("--num-steps", type=int, nargs="+", default=[10])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-6)
    args = parser.parse_args(argv)
    if args.mode == "verify" and (args.baseline is None or args.baseline_revision is None):
        parser.error("verify requires --baseline and --baseline-revision")
    if args.mode == "capture" and (args.baseline is not None or args.baseline_revision is not None):
        parser.error("capture does not use a baseline")
    if args.repeats < 2 or any(step < 1 for step in args.num_steps) or len(set(args.num_steps)) != len(args.num_steps):
        parser.error("use at least two repeats and distinct positive num-steps")
    if any(not np.isfinite(value) or value < 0 for value in (args.rtol, args.atol)):
        parser.error("tolerances must be finite and nonnegative")
    args.checkout, args.output = args.checkout.resolve(), args.output.resolve()
    if args.output.is_relative_to(args.checkout):
        parser.error("write outputs outside the source checkout")
    if args.output.exists():
        parser.error("refusing to overwrite an existing artifact directory")
    args.output.mkdir(parents=True)
    report = {"schema_version": SCHEMA, "mode": args.mode, "passed": False,
              "scope": "real local model/Policy probe; no WebSocket, ManiMux runtime, robot or task-success validation",
              "probe_sha256": digest_file(Path(__file__)), "model_execution_attempted": False,
              "model_loaded": False}
    arrays_out = {}
    try:
        run(args, report, arrays_out)
    except Exception as error:  # noqa: BLE001 - preserve native failures in the report and exit nonzero.
        report["error"] = f"{type(error).__name__}: {error}"
    if arrays_out:
        np.savez_compressed(args.output / "arrays.npz", **arrays_out)
        report["arrays_sha256"] = digest_file(args.output / "arrays.npz")
        report["array_bytes"] = sum(value.nbytes for value in arrays_out.values())
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"], "report": str(args.output / "report.json"), "error": report.get("error")}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
