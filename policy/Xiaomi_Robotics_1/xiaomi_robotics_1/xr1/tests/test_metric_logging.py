# Copyright (C) 2026 Xiaomi Corporation.
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from mibot.models.runner.base_runner import (
    BaseRunner,
    distributed_mean_metrics,
    mean_detached_metrics,
)


def test_micro_batch_metric_mean_is_detached_and_does_not_change_gradients():
    parameter = torch.tensor(2.0, requires_grad=True)
    first = parameter.square()
    second = 3.0 * parameter

    averaged = mean_detached_metrics({"loss": [first, second]})["loss"]

    assert averaged.item() == 5.0
    assert not averaged.requires_grad
    (first + second).backward()
    assert parameter.grad.item() == 7.0


def test_distributed_metric_mean_is_identity_without_distributed_runtime():
    value = torch.tensor(3.5, requires_grad=True)

    result = distributed_mean_metrics({"loss": value})["loss"]

    assert result.item() == 3.5
    assert not result.requires_grad


def test_optimizer_hook_logs_accumulated_mean_and_clears_buffer():
    runner = BaseRunner({"model": {}, "optimizer": {}, "scheduler": None})
    runner.log = Mock()
    first = torch.tensor(1.0, requires_grad=True)
    second = torch.tensor(3.0, requires_grad=True)
    runner._pending_train_metrics["loss"].extend([first, second])
    optimizer = SimpleNamespace(param_groups=[{"lr": 1.0e-5}])

    runner.on_before_optimizer_step(optimizer)

    logged = {call.args[0]: call.args[1] for call in runner.log.call_args_list}
    assert logged["train/loss"].item() == 2.0
    assert logged["lr"] == 1.0e-5
    assert not runner._pending_train_metrics
    assert first.grad is None and second.grad is None
