# Copyright (C) 2026 Xiaomi Corporation.
from collections import defaultdict
from importlib import import_module
from typing import Any, DefaultDict, Dict, Iterator, List, Optional, Tuple

import torch
import torch.distributed as dist
from lightning import LightningModule
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler

from transformers.utils import logging

from mibot.models import MIMODEL

logger = logging.get_logger(__name__)


def mean_detached_metrics(
    metrics: Dict[str, List[torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """Average detached micro-batch scalars without touching the loss graph."""
    return {
        name: torch.stack([value.detach().float() for value in values]).mean()
        for name, values in metrics.items()
        if values
    }


def distributed_mean_metrics(
    metrics: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Return logging-only data-parallel means with one small collective."""
    if not metrics:
        return {}
    names = list(metrics)
    values = torch.stack([metrics[name].detach().float() for name in names])
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        values /= dist.get_world_size()
    return dict(zip(names, values.unbind()))


@MIMODEL.register_module()
class BaseRunner(LightningModule):
    def __init__(self, params: Dict[str, Any]):
        super().__init__()
        self._pretrained: Optional[str] = params.get("pretrained")
        self._model: Dict[str, Any] = params.get("model")
        self._optimizer: Dict[str, Any] = params.get("optimizer")
        self._scheduler: Optional[Dict[str, Any]] = params.get("scheduler")
        self._pending_train_metrics: DefaultDict[str, List[torch.Tensor]] = defaultdict(list)

    def configure_model(self) -> None:
        self.model = MIMODEL.build(self._model)

        if self._pretrained:
            ckpt = torch.load(
                self._pretrained,
                map_location="cpu",
                mmap=True,
                weights_only=False,
            )
            info = self.load_state_dict(ckpt["module"], strict=False)
            if info.missing_keys or info.unexpected_keys:
                raise RuntimeError(f"Checkpoint does not match the configured model: {info}")
            logger.info(f"Loaded pretrained model from {self._pretrained}: {info}")

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.build_optimizer(self._optimizer, self.named_parameters())
        scheduler = self.build_scheduler(self._scheduler, optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    @staticmethod
    def build_optimizer(cfg: Dict[str, Any], parameters: Iterator[Tuple[str, torch.Tensor]]) -> Optimizer:
        module_params = cfg.get("params", {})
        parameters = list(parameters)

        no_decay = [
            "bias",
            "norm",
            "Norm",
            "ln",
            "Ln",
            "rotary_emb",
            "adaln",
        ]

        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in parameters if not any(nd in n.lower() for nd in no_decay) and p.requires_grad],
                "weight_decay": module_params.get("weight_decay", 0.1),
            },
            {
                "params": [p for n, p in parameters if any(nd in n.lower() for nd in no_decay) and p.requires_grad],
                "weight_decay": 0.0,
            },
        ]

        module_type = cfg["type"]
        module_name, class_name = module_type.rsplit(".", 1)
        module = import_module(module_name)
        optimizer_class = getattr(module, class_name)

        return optimizer_class(optimizer_grouped_parameters, **module_params)

    @staticmethod
    def build_scheduler(cfg: Optional[Dict[str, Any]], optimizer: Optimizer) -> Optional[LRScheduler]:
        if cfg is None:
            return None

        module_type = cfg["type"]
        module_params = cfg.get("params", {})
        module_name, class_name = module_type.rsplit(".", 1)
        module = import_module(module_name)
        scheduler_class = getattr(module, class_name)

        return scheduler_class(optimizer, **module_params)

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, torch.Tensor]:
        loss_dict = self.model(batch, return_loss=True)
        metric_device = next(iter(loss_dict.values())).device
        self._pending_train_metrics["token"].append(
            torch.tensor(float(batch["input_ids"].shape[1]), device=metric_device)
        )
        for loss_name, loss_value in loss_dict.items():
            self._pending_train_metrics[loss_name].append(loss_value.detach())
        return loss_dict

    def on_before_optimizer_step(self, optimizer: Optimizer) -> None:
        """Log the accumulated, cross-rank mean immediately before each update.

        Lightning calls this hook only after gradient accumulation is complete.
        Every value here is detached, so logging cannot change backward,
        clipping, optimizer state, or the learning-rate schedule.
        """
        local_means = mean_detached_metrics(self._pending_train_metrics)
        self._pending_train_metrics.clear()
        global_means = distributed_mean_metrics(local_means)
        for name, global_value in global_means.items():
            self.log(
                f"train/{name}",
                global_value,
                on_step=True,
                on_epoch=False,
                logger=True,
                sync_dist=False,
            )
        self.log(
            "lr",
            optimizer.param_groups[0]["lr"],
            on_step=True,
            on_epoch=False,
            logger=True,
            sync_dist=False,
        )
