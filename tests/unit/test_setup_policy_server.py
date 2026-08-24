from __future__ import annotations

from setup_policy_server import _deployment_model_metadata


def test_deployment_metadata_keeps_identity_and_excludes_transport_fields() -> None:
    metadata = _deployment_model_metadata(
        {
            "policy_name": "Pi_05",
            "checkpoint_variant": "pi05-step-1000",
            "checkpoint_source": "local-finetune",
            "norm_stats_source": "checkpoint-matched",
            "host": "127.0.0.1",
            "port": 8500,
            "ws_ping_interval_s": 20.0,
        }
    )

    assert metadata == {
        "policy_name": "Pi_05",
        "checkpoint_variant": "pi05-step-1000",
        "checkpoint_source": "local-finetune",
        "norm_stats_source": "checkpoint-matched",
    }
