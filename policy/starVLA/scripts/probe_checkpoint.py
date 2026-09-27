"""Validate real StarVLA inference without reinterpreting its native action space."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base-vlm", type=Path, required=True)
    parser.add_argument("--unnorm-key", required=True)
    parser.add_argument("--camera-count", type=int, required=True)
    parser.add_argument("--include-state", choices=("auto", "true", "false"), default="auto")
    parser.add_argument("--fast-tokenizer", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--sampling-modes",
        nargs="*",
        choices=("rtc", "paint", "aac", "autohorizon", "dvac"),
        default=[],
    )
    args = parser.parse_args()
    if args.camera_count <= 0:
        parser.error("--camera-count must be positive")
    policy_dir = Path(__file__).resolve().parents[1]
    source = policy_dir / "source_starvla"
    sys.path.insert(0, str(source))
    sys.path.insert(0, str(policy_dir))
    os.environ["STARVLA_BASE_VLM"] = str(args.base_vlm.absolute())
    if args.fast_tokenizer is not None:
        os.environ["STARVLA_FAST_TOKENIZER"] = str(args.fast_tokenizer.absolute())

    report = {
        "status": "running",
        "validation_stage": "native_model_forward",
        "checkpoint": str(args.checkpoint.absolute()),
        "base_vlm": str(args.base_vlm.absolute()),
        "camera_count": args.camera_count,
        "input": "synthetic 224x224 RGB gradients and optional zero state",
        "hardware_used": False,
        "robot_task_success_tested": False,
        "calls": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    outputs = []
    try:
        import torch
        from deployment.model_server.policy_wrapper import PolicyServerWrapper
        from runtime_config import resolve_include_state

        started = time.perf_counter()
        policy = PolicyServerWrapper(
            ckpt_path=str(args.checkpoint.absolute()),
            device=args.device,
            use_bf16=True,
            unnorm_key=args.unnorm_key,
        )
        if args.device.startswith("cuda"):
            torch.cuda.synchronize(args.device)
            torch.cuda.reset_peak_memory_stats(args.device)
        report["load_seconds"] = time.perf_counter() - started
        metadata = policy.metadata
        report["metadata"] = metadata
        framework = policy._framework
        module_path = Path(sys.modules[type(framework).__module__].__file__).resolve()
        if not module_path.is_relative_to(source.resolve()):
            raise RuntimeError(f"Model was imported outside the vendored runtime: {module_path}")
        report["framework_module"] = str(module_path)
        report["parameter_count"] = sum(p.numel() for p in framework.parameters())
        report["parameter_devices"] = sorted({str(p.device) for p in framework.parameters()})
        include_state = resolve_include_state(args.include_state, args.checkpoint)
        report["include_state"] = include_state
        contract = metadata["runtime_contract"]
        expected = (1, metadata["action_chunk_size"], contract["action_dim"])
        y, x = np.indices((224, 224))
        images = [
            np.stack((x, y, (x + y + i * 31) % 256), axis=-1).astype(np.uint8)
            for i in range(args.camera_count)
        ]
        for index, seed in enumerate((42, 43, 42, 42)):
            torch.manual_seed(seed)
            np.random.seed(seed)
            example = {
                "image": images if index < 3 else [255 - image for image in images],
                "lang": "Pick up the red block.",
            }
            if include_state:
                example["state"] = np.zeros((1, contract["state_dim"]), dtype=np.float32)
            started = time.perf_counter()
            actions = np.asarray(
                policy.predict_action(
                    examples=[example],
                    unnorm_key=args.unnorm_key,
                    do_sample=False,
                    use_ddim=True,
                    num_ddim_steps=10,
                )["actions"]
            )
            if args.device.startswith("cuda"):
                torch.cuda.synchronize(args.device)
            elapsed = time.perf_counter() - started
            if actions.shape != expected or not np.isfinite(actions).all():
                raise ValueError(f"Expected finite {expected} actions, got {actions.shape}")
            outputs.append(actions.copy())
            row = {
                "index": index,
                "seed": seed,
                "seconds": elapsed,
                "shape": list(actions.shape),
                "finite": True,
                "minimum": float(actions.min()),
                "maximum": float(actions.max()),
            }
            report["calls"].append(row)
            print(json.dumps(row), flush=True)
        report["same_seed_max_abs_diff"] = float(np.max(np.abs(outputs[0] - outputs[2])))
        report["different_seed_max_abs_diff"] = float(np.max(np.abs(outputs[0] - outputs[1])))
        report["changed_images_max_abs_diff"] = float(np.max(np.abs(outputs[0] - outputs[3])))
        np.testing.assert_allclose(outputs[0], outputs[2], atol=1e-5, rtol=1e-5)
        report["sampler_calls"] = []
        for mode in args.sampling_modes:
            sampling = {"mode": mode}
            if mode in {"rtc", "paint"}:
                sampling["action_condition"] = outputs[0][0]
            if mode == "rtc":
                sampling.update(
                    condition_weights=np.linspace(1, 0, expected[1], dtype=np.float32), beta=1.0
                )
            if mode == "paint":
                sampling["delay_steps"] = 2
            if mode == "aac":
                sampling["num_samples"] = 4
            if mode == "dvac":
                sampling["tail_steps"] = min(4, metadata["num_inference_timesteps"])
            started = time.perf_counter()
            result = policy.predict_action(
                examples=[example], unnorm_key=args.unnorm_key, sampling=sampling
            )
            actions = np.asarray(result["actions"])
            shape = (sampling.get("num_samples", 1), *expected[1:])
            if actions.shape != shape or not np.isfinite(actions).all():
                raise ValueError(f"Invalid {mode} actions: {actions.shape}, expected {shape}")
            row = {
                "mode": mode,
                "shape": list(actions.shape),
                "seconds": time.perf_counter() - started,
                "metadata": {key: value for key, value in result.items() if key != "actions"},
            }
            report["sampler_calls"].append(row)
            print(
                json.dumps({key: value for key, value in row.items() if key != "metadata"}),
                flush=True,
            )
        if args.device.startswith("cuda"):
            report["peak_allocated_mib"] = torch.cuda.max_memory_allocated(args.device) / 2**20
        report["status"] = "passed"
    except Exception as exc:
        report.update(
            status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc()
        )
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        if outputs:
            np.savez(args.output.with_suffix(".npz"), actions=np.stack(outputs))
        print(
            json.dumps({key: report[key] for key in ("status", "validation_stage", "checkpoint")}),
            flush=True,
        )


if __name__ == "__main__":
    main()
