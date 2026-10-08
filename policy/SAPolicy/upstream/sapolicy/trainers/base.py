import collections
import math
import os
from os.path import join
from typing import Any, Dict, List, Optional, Union

import pytorch_lightning as pl
import torch
from diffusers.optimization import (
    TYPE_TO_SCHEDULER_FUNCTION,
    Optimizer,
    SchedulerType,
)
from hydra.utils import instantiate
from omegaconf.base import ContainerMetadata, Metadata
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from omegaconf.nodes import AnyNode

from sapolicy.logger import Log


if torch.__version__ >= "2.6.0":
    from torch.serialization import add_safe_globals

    add_safe_globals([
        ListConfig, DictConfig, ContainerMetadata, AnyNode, Metadata, Any, Dict,
        List, list, collections.defaultdict, dict, int,
    ])


def get_scheduler(
    name: Union[str, SchedulerType],
    optimizer: Optimizer,
    num_warmup_steps: Optional[int] = None,
    num_training_steps: Optional[int] = None,
    **kwargs,
):
    """Unified API to get any Diffusers scheduler from its name."""
    name = SchedulerType(name)
    schedule_func = TYPE_TO_SCHEDULER_FUNCTION[name]
    if name == SchedulerType.CONSTANT:
        return schedule_func(optimizer, **kwargs)

    if num_warmup_steps is None:
        raise ValueError(
            f"{name} requires `num_warmup_steps`, please provide that argument."
        )

    if name == SchedulerType.CONSTANT_WITH_WARMUP:
        return schedule_func(optimizer, num_warmup_steps=num_warmup_steps, **kwargs)

    if num_training_steps is None:
        raise ValueError(
            f"{name} requires `num_training_steps`, please provide that argument."
        )

    return schedule_func(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        **kwargs,
    )


def _lrs_at_step(sched, step: int) -> List[float]:
    """Evaluate scheduler LRs at `step` under the *current* schedule params."""
    if isinstance(sched, torch.optim.lr_scheduler.CosineAnnealingLR):
        t_max = max(int(sched.T_max), 1)
        return [
            sched.eta_min
            + (base_lr - sched.eta_min)
            * (1.0 + math.cos(math.pi * step / t_max))
            / 2.0
            for base_lr in sched.base_lrs
        ]
    if isinstance(sched, torch.optim.lr_scheduler.LambdaLR):
        return [
            base_lr * float(lmbda(step))
            for lmbda, base_lr in zip(sched.lr_lambdas, sched.base_lrs)
        ]
    prev = sched.last_epoch
    sched.last_epoch = step
    try:
        return [float(lr) for lr in sched.get_lr()]
    finally:
        sched.last_epoch = prev


class BaseModel(pl.LightningModule):
    """Shared Lightning setup for models backed by a Hydra-instantiated pipeline."""

    def __init__(
        self,
        pipeline,
        optimizer,
        lr_table,
        output_dir: str,
        output_tag: str = "default",
        clear_output_dir: bool = False,
        scheduler_cfg=None,
        ignored_weights_prefix=None,
        resume_reset_scheduler: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.pipeline = instantiate(pipeline, _recursive_=False)
        self.optimizer = instantiate(optimizer)
        self.lr_table = instantiate(lr_table)
        self.scheduler_cfg = scheduler_cfg
        self.resume_reset_scheduler = bool(resume_reset_scheduler)
        self._resume_scheduler_synced = False
        self._scheduler_num_training_steps: Optional[int] = None
        self.ignored_weights_prefix = (
            ["pipeline.text_encoder", "pipeline.vae"]
            if ignored_weights_prefix is None
            else ignored_weights_prefix
        )

        if clear_output_dir:
            Log.warn(f"Clear output dir: {join(output_dir, output_tag)}")
            os.system(f"rm -rf {join(output_dir, output_tag)}")
        self.output_dir = join(output_dir, output_tag)
        self.metrics_dict = {}

        self.test_step = self.validation_step

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        raise NotImplementedError

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        raise NotImplementedError

    def configure_optimizers(self):
        group_table = {}
        params = []
        for name, parameter in self.pipeline.named_parameters():
            if not parameter.requires_grad:
                continue

            group, lr = self.lr_table.get_lr(name)
            weight_decay = self.lr_table.get_weight_decay(name)
            if lr == 0:
                parameter.requires_grad = False
            if group not in group_table:
                group_table[group] = len(group_table)
                params.append({
                    "params": [parameter],
                    "lr": lr,
                    "name": group,
                    "weight_decay": weight_decay,
                })
            else:
                params[group_table[group]]["params"].append(parameter)

        optimizer = self.optimizer(params=params)
        if self.scheduler_cfg is None:
            return optimizer

        scheduler_cfg = self.scheduler_cfg
        kwargs = dict(scheduler_cfg["kwargs"])
        total_steps = self.trainer.estimated_stepping_batches
        steps_per_epoch = total_steps // self.trainer.max_epochs
        Log.info(
            f"[Scheduler] trainer reports {total_steps} total steps, "
            f"{steps_per_epoch} steps/epoch, {self.trainer.max_epochs} epochs"
        )

        if "num_warmup_steps" in kwargs:
            warmup_steps = int(kwargs["num_warmup_steps"])
            kwargs.pop("num_warmup_epochs", None)
            kwargs["num_warmup_steps"] = warmup_steps
            Log.info(
                f"[Scheduler] using explicit num_warmup_steps={warmup_steps}"
            )
        elif "num_warmup_epochs" in kwargs:
            warmup_epochs = kwargs.pop("num_warmup_epochs")
            kwargs["num_warmup_steps"] = int(warmup_epochs * steps_per_epoch)
            Log.info(
                f"[Scheduler] num_warmup_epochs={warmup_epochs} -> "
                f"num_warmup_steps={kwargs['num_warmup_steps']}"
            )
        if "num_training_epochs" in kwargs:
            training_epochs = kwargs.pop("num_training_epochs")
            kwargs["num_training_steps"] = int(training_epochs * steps_per_epoch)
            Log.info(
                f"[Scheduler] num_training_epochs={training_epochs} -> "
                f"num_training_steps={kwargs['num_training_steps']}"
            )
        elif "num_training_steps" not in kwargs:
            kwargs["num_training_steps"] = total_steps
            Log.info(f"[Scheduler] auto num_training_steps={total_steps}")

        self._scheduler_num_training_steps = int(kwargs["num_training_steps"])
        scheduler = get_scheduler(
            scheduler_cfg["name"],
            optimizer,
            **kwargs,
            last_epoch=self.global_step - 1,
        )
        return [optimizer], [scheduler]

    def _iter_lr_schedulers(self):
        sch = self.lr_schedulers()
        if sch is None:
            return []
        if isinstance(sch, (list, tuple)):
            return list(sch)
        return [sch]

    def _sync_resume_scheduler(self) -> None:
        """Realign LR under the *new* total length after Lightning restores ckpt state.

        Mirror of dreamer4 `resume_reset_scheduler`: checkpoint restore keeps
        `last_epoch` / optimizer LR from the prior run, while
        `configure_optimizers` already built the schedule for the new
        `max_epochs` / `num_training_steps`. Re-seat `last_epoch` at the resumed
        `global_step` and rewrite param-group LRs immediately.
        """
        if not self.resume_reset_scheduler or self._resume_scheduler_synced:
            return
        step = int(self.trainer.global_step)
        if step <= 0:
            return

        schedulers = self._iter_lr_schedulers()
        if not schedulers:
            return

        total_steps = self._scheduler_num_training_steps
        if total_steps is None:
            total_steps = int(self.trainer.estimated_stepping_batches)

        for sched in schedulers:
            if isinstance(sched, torch.optim.lr_scheduler.CosineAnnealingLR):
                sched.T_max = total_steps
            sched.last_epoch = step
            lrs = _lrs_at_step(sched, step)
            for pg, lr in zip(sched.optimizer.param_groups, lrs):
                pg["lr"] = lr
            sched._last_lr = lrs

        if self.trainer.is_global_zero:
            lr = schedulers[0].optimizer.param_groups[0]["lr"]
            Log.info(
                f"resume_reset_scheduler: global_step={step} "
                f"num_training_steps={total_steps} lr={lr:.6g}"
            )
        self._resume_scheduler_synced = True

    def on_train_start(self) -> None:
        self._sync_resume_scheduler()

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        for prefix in self.ignored_weights_prefix:
            Log.debug(f"Remove key `{prefix}' from checkpoint.")
            for key in list(checkpoint["state_dict"]):
                if key.startswith(prefix):
                    checkpoint["state_dict"].pop(key)
        super().on_save_checkpoint(checkpoint)
