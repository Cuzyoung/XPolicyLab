"""GPU-only offline contract validation; never connects to a robot or camera."""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np

from XPolicyLab.policy.SAPolicy.model import Model, _wxyz_wire_to_xyzw
from XPolicyLab.utils.process_data import pack_robot_state


def observation():
    image = np.zeros((168, 224, 3), dtype=np.uint8)
    image[:, :75, 0] = 200
    image[:, 75:150, 1] = 180
    image[:, 150:, 2] = 220
    return {
        "vision": {
            name: {
                "color": image.copy(),
                "shape": [168, 224],
                "intrinsic_matrix": [[138, 0, 112], [0, 138, 84], [0, 0, 1]],
            }
            for name in ("top", "left", "right")
        },
        "state": {
            f"{side}_{key}": value
            for side, y in (("left", 0.31), ("right", -0.31))
            for key, value in (("ee_pose", [0.35, y, 0.9, 1, 0, 0, 0]), ("ee_joint_state", [0.5]))
        },
        "env_idx": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--legacy-wrapper", type=Path, help="Optional archived wrapper for numerical comparison"
    )
    parser.add_argument(
        "--legacy-config", type=Path, help="Independently load the archived resolved YAML"
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--rtc", action="store_true", help="Validate real RTC guidance and reset")
    args = parser.parse_args()
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Checkpoint validation requires a CUDA GPU")
    cfg = {
        "model_path": args.checkpoint,
        "env_cfg_type": "yam_dual",
        "action_type": "ee",
        "action_horizon": 50,
        "camera_names": ["top", "left", "right"],
        "use_ema": False,
    }
    model = Model(cfg)
    obs = observation()
    model.update_obs(obs)
    torch.manual_seed(123)
    standard = model.get_action()
    wxyz = np.stack(
        [pack_robot_state({"state": row}, "ee", model._ee_dimensions) for row in standard]
    )
    assert wxyz.shape == (50, 16) and np.isfinite(wxyz).all()
    model._output_format = "packed_ee_wire"
    torch.manual_seed(123)
    wire = model.get_action()
    np.testing.assert_array_equal(wire, _wxyz_wire_to_xyzw(wxyz))
    result = {
        "checkpoint": args.checkpoint,
        "action_shape": [50, 16],
        "standard_vs_compatibility_max_abs": 0.0,
    }
    if args.rtc:
        assert model.sampling_modes() == ["default", "rtc"]
        sampling = {"action_condition": wxyz, "condition_weights": np.zeros(50), "beta": 5.0}
        torch.manual_seed(123)
        np.testing.assert_array_equal(model.get_action_rtc(sampling), wire)
        result["rtc_zero_mask_max_abs"] = 0.0
        # A separate noise draw supplies a valid in-distribution target. Compare
        # guidance against an ordinary forward from exactly the same initial noise.
        sampling["condition_weights"] = np.r_[np.ones(4), np.linspace(1, 0, 21), np.zeros(25)]
        head = model._backend.policy.pipeline.action_head
        norm = model._backend.policy.pipeline.normalizer
        spatial = model._to_spatial_obs(obs)

        def normalized(absolute):
            relative = model._backend._rtc_relative_actions(absolute, spatial)
            tensor = torch.as_tensor(
                relative, device=model._backend.device, dtype=model._backend.dtype
            )
            return norm.normalize({"action": tensor})["action"].cpu().numpy()

        target = normalized(wxyz)
        # packed_ee_wire uses XYZW, whereas RTC input always uses standard WXYZ.
        model._output_format = "action_dict"

        def packed(actions):
            return np.stack(
                [pack_robot_state({"state": row}, "ee", model._ee_dimensions) for row in actions]
            )

        errors, timings = [], []
        for seed in (321, 456, 789):
            torch.manual_seed(seed)
            baseline = packed(model.get_action())
            torch.manual_seed(seed)
            started = time.perf_counter()
            guided = packed(model.get_action_rtc(sampling))
            timings.append((time.perf_counter() - started) * 1000)
            weights = sampling["condition_weights"][:, None]
            before = float(np.mean((normalized(baseline) - target) ** 2 * weights))
            after = float(np.mean((normalized(guided) - target) ** 2 * weights))
            assert np.isfinite(guided).all() and after < before, (seed, before, after)
            errors.append({"seed": seed, "before": before, "after": after})
            assert head._rtc_sampling is None
            assert all(parameter.grad is None for parameter in model._backend.policy.parameters())
        result["rtc_weighted_normalized_mse"] = errors
        result["rtc_forward_ms"] = timings
        model.reset()
        try:
            model.get_action_rtc(sampling)
        except RuntimeError:
            result["rtc_reset_clears_observation"] = True
        else:
            raise AssertionError("RTC reset did not clear observation")
        model.update_obs(obs)
        model._output_format = "packed_ee_wire"
        torch.manual_seed(123)
        np.testing.assert_array_equal(model.get_action(), wire)
        result["rtc_does_not_leak_to_default"] = True
    if args.legacy_wrapper:
        spec = importlib.util.spec_from_file_location(
            "sapolicy_reference_wrapper", args.legacy_wrapper
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.TCP_FORWARD_OFFSET = 0.0
        if args.legacy_config:
            from XPolicyLab.policy.SAPolicy.assets import resolve_assets

            assets = resolve_assets(cfg)
            reference = module.SAPolicyRoboTwinModel(
                cfg_file=str(args.legacy_config.resolve()),
                ckpt_path=assets["model_path"],
                n_action_steps=50,
                use_ema=False,
                normalizer_path=assets["normalizer_path"],
                tcp_forward_offset_m=0.0,
            )
            result["independent_legacy_load"] = True
        else:
            reference = object.__new__(module.SAPolicyRoboTwinModel)
            reference.__dict__.update(model._backend.__dict__)
        reference.reset_model()
        reference.update_obs(model._to_spatial_obs(obs))
        torch.manual_seed(123)
        original = _wxyz_wire_to_xyzw(reference.get_action())
        np.testing.assert_allclose(wire, original, atol=1e-7, rtol=0)
        result["archived_wrapper_max_abs"] = float(np.max(np.abs(wire - original)))
    model.reset()
    model._output_format = "action_dict"
    model.update_obs_batch([obs, {**observation(), "env_idx": 7}])
    batch = model.get_action_batch([7, 0])
    assert len(batch) == 2 and all(len(chunk) == 50 for chunk in batch)
    assert all(np.isfinite(v).all() for chunk in batch for row in chunk for v in row.values())
    model.reset()
    try:
        model.get_action_batch([7])
    except RuntimeError:
        result["reset_clears_batch"] = True
    else:
        raise AssertionError("Reset did not clear batch")
    result["real_batch_chunks"] = 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
