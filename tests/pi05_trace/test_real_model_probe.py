"""CPU tests of evidence/input guards, never replacements for model validation."""

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

PATH = Path(__file__).with_name("real_model_probe.py")
SPEC = importlib.util.spec_from_file_location("real_model_probe", PATH)
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def inputs(tmp_path, **arrays):
    meta = tmp_path / "observation.json"
    meta.write_text(json.dumps({"schema_version": 1, "source": "synthetic CPU fixture",
                               "observation": {"state": {"array": "state"}, "instruction": "test"}}))
    data = tmp_path / "arrays.npz"
    np.savez(data, **arrays)
    return meta, data


def test_nonpickle_input_and_owned_arrays(tmp_path):
    meta, data = inputs(tmp_path, state=np.zeros(14, np.float32), noise=np.zeros((50, 32), np.float32))
    observation, noise, identity = probe.read_inputs(meta, data)
    assert observation["state"].shape == (14,) and noise.shape == (50, 32)
    assert identity["arrays_npz_sha256"] == probe.digest_file(data)
    observation["state"][:] = 3
    assert np.load(data, allow_pickle=False)["state"].sum() == 0


@pytest.mark.parametrize("noise", [np.zeros((50, 32), np.float64), np.zeros(32, np.float32),
                                   np.full((50, 32), np.nan, np.float32), np.array([object()])])
def test_reject_bad_or_pickled_noise(tmp_path, noise):
    with pytest.raises(ValueError):
        probe.read_inputs(*inputs(tmp_path, state=np.zeros(14), noise=noise))


def test_reject_missing_and_unused_arrays(tmp_path):
    with pytest.raises(ValueError, match="Unused NPZ"):
        probe.read_inputs(*inputs(tmp_path, state=np.zeros(14), noise=np.zeros((50, 32), np.float32), unused=np.ones(1)))
    with pytest.raises(ValueError, match="Invalid observation"):
        probe.read_inputs(*inputs(tmp_path, noise=np.zeros((50, 32), np.float32)))


def baseline(identity):
    return {"schema_version": 1, "mode": "capture", "passed": True,
            "probe_sha256": probe.digest_file(PATH),
            "source": {"revision": "a" * 40, "clean": True}, "identity": copy.deepcopy(identity)}


def test_baseline_requires_old_clean_capture_and_exact_identity():
    identity = {"checkpoint": {"sha256": "weights"}, "normalization": {"sha256": "norm"},
                "inputs": "input", "versions": {"jax": "installed"}, "model": {"dim": 32}}
    probe.check_baseline(baseline(identity), identity, "a" * 40, "b" * 40)
    for field in identity:
        changed = baseline(identity)
        changed["identity"][field] = "wrong"
        with pytest.raises(ValueError, match="identity mismatch"):
            probe.check_baseline(changed, identity, "a" * 40, "b" * 40)
    with pytest.raises(ValueError, match="own old-backend"):
        probe.check_baseline(baseline(identity), identity, "a" * 40, "a" * 40)
    for field, value in (("mode", "verify"), ("passed", False), ("schema_version", True), ("probe_sha256", "wrong")):
        changed = baseline(identity)
        changed[field] = value
        with pytest.raises(ValueError):
            probe.check_baseline(changed, identity, "a" * 40, "b" * 40)
    changed = baseline(identity)
    changed["source"]["clean"] = False
    with pytest.raises(ValueError, match="clean old revision"):
        probe.check_baseline(changed, identity, "a" * 40, "b" * 40)


def test_file_manifest_hashes_bytes_not_just_labels_or_size(tmp_path):
    shard = tmp_path / "weight.bin"
    shard.write_bytes(b"abcd")
    before = probe.content_manifest(tmp_path)
    shard.write_bytes(b"abce")
    after = probe.content_manifest(tmp_path)
    assert before["sha256"] != after["sha256"]
    assert before["files"]["weight.bin"] != after["files"]["weight.bin"]


def test_comparison_refuses_nonfinite_and_shape_changes():
    assert probe.compare(np.ones(2), np.ones(2), rtol=0, atol=0)["passed"]
    assert not probe.compare(np.ones(2), np.zeros(2), rtol=1e-5, atol=1e-6)["passed"]
    with pytest.raises(ValueError):
        probe.compare(np.array([np.nan]), np.array([np.nan]), rtol=1e-5, atol=1e-6)
    with pytest.raises(ValueError):
        probe.compare(np.ones((1, 2)), np.ones(2), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("option,value", [("--rtol", "inf"), ("--atol", "nan"),
                                         ("--atol", "-1"), ("--num-steps", "0"),
                                         ("--repeats", "1")])
def test_cli_rejects_invalid_acceptance_parameters_before_loading(tmp_path, option, value):
    output = tmp_path / "report"
    args = ["capture", "--expected-revision", "a" * 40, "--config", "missing.yml",
            "--observation", "missing.json", "--arrays", "missing.npz", "--output", str(output), option, value]
    with pytest.raises(SystemExit) as error:
        probe.main(args)
    assert error.value.code == 2
    assert not output.exists()


def test_missing_inputs_dont_claim_model_execution(tmp_path):
    output = tmp_path / "report"
    code = probe.main(["capture", "--expected-revision", "a" * 40, "--config", "missing.yml",
                       "--observation", "missing.json", "--arrays", "missing.npz", "--output", str(output)])
    report = json.loads((output / "report.json").read_text())
    assert code == 1 and not report["passed"]
    assert not report["model_execution_attempted"] and not report["model_loaded"]
