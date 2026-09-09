"""Tiny artifact fixtures only; no neural network imports or GPU allocation."""

import hashlib

from XPolicyLab.policy.OpenWAM.artifact_identity import artifact_identity


def test_identity_selects_numeric_latest_and_tracks_stats(tmp_path):
    (tmp_path / "checkpoint_step_2.safetensors").write_bytes(b"older")
    (tmp_path / "checkpoint_step_10.safetensors").write_bytes(b"newer")
    (tmp_path / "checkpoint_step_bad.safetensors").write_bytes(b"ignored")
    (tmp_path / "config.yaml").write_text("test: true\n")
    stats = tmp_path / "normalization_stats.npy"
    stats.write_bytes(b"stats-a")
    before = artifact_identity(tmp_path)
    assert before["checkpoint_file"] == "checkpoint_step_10.safetensors"
    assert before["checkpoint_sha256"] == hashlib.sha256(b"newer").hexdigest()
    stats.write_bytes(b"stats-b")
    after = artifact_identity(tmp_path)
    assert before["norm_stats_sha256"] != after["norm_stats_sha256"]
    assert before["checkpoint_sha256"] == after["checkpoint_sha256"]
