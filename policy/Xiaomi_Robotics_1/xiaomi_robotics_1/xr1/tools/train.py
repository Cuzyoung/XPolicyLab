# Copyright (C) 2026 Xiaomi Corporation.
import logging
import os
import warnings
from typing import Any, Dict, List, Optional, Tuple

import hydra
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, ModelSummary
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

from mmengine import Config, DATASETS
from omegaconf import DictConfig
from mibot.models import MIMODEL

from mibot.utils.cfg_utils import helper

import mibot.data

warnings.filterwarnings("ignore")


def prepare(cfg: Dict[str, Any]) -> Tuple[Config, LightningDataModule, LightningModule, List[Any]]:
    cfg: Config = helper(cfg)
    datamodule: LightningDataModule = DATASETS.build(cfg.data)
    model: LightningModule = MIMODEL.build(cfg.model)
    cfg.trainer["callbacks"] = [
        ModelSummary(max_depth=2),
        ModelCheckpoint(
            save_top_k=-1,
            save_last=True,
            every_n_train_steps=cfg.trainer.pop("save_interval", 10000),
            dirpath=cfg.trainer.default_root_dir,
            enable_version_counter=False,
        ),
    ]

    project = cfg.trainer.pop("project")
    exp_name = cfg.trainer.pop("exp_name")
    logger_backend = os.environ.get("XR1_LOGGER", "tensorboard").lower()
    if logger_backend == "tensorboard":
        logger: List[Any] = [
            TensorBoardLogger(
                save_dir=cfg.trainer.default_root_dir,
                name="tensorboard",
                version=exp_name,
            )
        ]
    elif logger_backend == "csv":
        logger = [
            CSVLogger(
                save_dir=cfg.trainer.default_root_dir,
                name="csv",
                version=exp_name,
            )
        ]
    else:
        raise ValueError(
            f"Unsupported XR1_LOGGER={logger_backend!r}; expected 'tensorboard' or 'csv'. "
            "Network loggers are intentionally disabled for YAM training."
        )
    logging.info("XR-1 local logger=%s project_label=%s", logger_backend, project)

    logging.getLogger("lightning.pytorch").setLevel(logging.INFO)

    return (cfg, datamodule, model, logger)


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    cfg, datamodule, model, logger = prepare(cfg)
    ckpt_path: Optional[str] = cfg.trainer.pop("ckpt_path", None)
    trainer = Trainer(**cfg.trainer, logger=logger)
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
