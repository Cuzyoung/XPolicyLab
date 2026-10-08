import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig

from sapolicy.entrys.factory import get_callbacks, get_data, get_model, print_cfg, find_last_ckpt_path
from sapolicy.entrys.normalizer_utils import (
    normalizer_fingerprint,
    resolve_eval_normalizer,
)
from sapolicy.logger import Log


def setup_trainer(cfg: DictConfig):
    """Build trainer/model/datamodule for standalone validate/test runs."""
    if cfg.print_cfg:
        print_cfg(cfg, use_rich=True)
    pl.seed_everything(cfg.seed)

    datamodule = get_data(cfg, wo_train=True)
    model = get_model(cfg)
    ckpt_path = find_last_ckpt_path(cfg.callbacks.model_checkpoint.dirpath)
    ckpt_type = getattr(cfg, "ckpt_type", None)
    model.load_pretrained_model(ckpt_path, ckpt_type)
    source_label, rebound = resolve_eval_normalizer(
        model, cfg, cfg.get("eval", None), ckpt_path=ckpt_path
    )
    if rebound is not None:
        Log.info(
            f"[val] Bound pipeline normalizer from {source_label} "
            f"({normalizer_fingerprint(rebound)})"
        )

    callbacks = get_callbacks(cfg)
    logger = hydra.utils.instantiate(cfg.logger, _recursive_=False)
    trainer = pl.Trainer(
        logger=logger if logger is not None else False,
        callbacks=callbacks,
        **cfg.pl_trainer,
    )
    return trainer, model, datamodule


def val(cfg: DictConfig) -> None:
    """Run Lightning validation on val_dataloader (val/* metrics, TensorBoard)."""
    trainer, model, datamodule = setup_trainer(cfg)
    trainer.validate(model, datamodule.val_dataloader())
