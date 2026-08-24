from __future__ import annotations

import torch

from XPolicyLab.policy.Pi_05.openpi.src.openpi.models_pytorch.autohorizon_official import (
    bidir_soft_pointer,
)


def test_autohorizon_bidirectional_pointer_keeps_identity_horizon() -> None:
    execution_steps, diagnostics = bidir_soft_pointer(torch.eye(4))

    assert execution_steps.item() == 4
    assert diagnostics["method"] == "bidir_soft_pointer"


def test_autohorizon_pointer_stops_when_all_rows_hold_first_action() -> None:
    attention = torch.zeros((4, 4))
    attention[:, 0] = 1.0

    execution_steps, diagnostics = bidir_soft_pointer(attention)

    assert execution_steps.item() == 1
    assert diagnostics["N_forward"] == 1
