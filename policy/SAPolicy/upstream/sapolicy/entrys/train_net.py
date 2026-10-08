import copy
import os
import shutil

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf

from sapolicy.entrys.factory import get_callbacks, get_data, get_model, print_cfg, find_last_ckpt_path
from sapolicy.entrys.normalizer_utils import (
    find_train_normalizer,
    fit_combined_normalizer,
    supports_combined_normalizer,
    n_train_dataset_opts,
    normalizer_fingerprint,
    save_action_normalizer,
)
from sapolicy.logger import Log

torch.set_float32_matmul_precision("high")

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(True)
    if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
        torch.backends.cuda.enable_mem_efficient_sdp(True)


def _is_local_rank_zero() -> bool:
    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"]) == 0
    if "RANK" in os.environ:
        return int(os.environ["RANK"]) == 0
    return True


def _barrier_if_distributed() -> None:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _delete_output_dir(resume_training, output_dir, confirm_delete_previous_dir):
    """Delete output_dir when starting a fresh run (rank0 only)."""
    if not _is_local_rank_zero():
        _barrier_if_distributed()
        return

    if not resume_training and os.path.exists(output_dir):
        Log.warn("Not resume_training, training from scratch.")
        Log.warn(f"Deleting the output path: {output_dir}")
        if confirm_delete_previous_dir:
            shutil.rmtree(output_dir)
            Log.info(f"Deleted the output path: {output_dir}")
        else:
            while True:
                user_input = input(f"Delete the output path: {output_dir}? (y/n)")
                if user_input == "y":
                    shutil.rmtree(output_dir)
                    break
                if user_input == "n":
                    Log.warn("Not deleting the output path.")
                    break
                Log.warn("Invalid input, please input again.")

    _barrier_if_distributed()


def _save_resolved_config(cfg: DictConfig) -> None:
    os.makedirs(cfg.output_dir, exist_ok=True)
    output_path = os.path.join(cfg.output_dir, "resolved_config.yaml")
    if hasattr(cfg, "dump"):
        text = cfg.dump()
    else:
        text = OmegaConf.to_yaml(cfg, resolve=True)
    with open(output_path, "w") as f:
        f.write(text)
    Log.info(f"Saved resolved config to {output_path}")


def train_net(cfg: DictConfig) -> None:
    """Instantiate the trainer, and then train the model."""
    if cfg.print_cfg:
        print_cfg(cfg, use_rich=True)
    callbacks = get_callbacks(cfg)
    logger = hydra.utils.instantiate(cfg.logger, _recursive_=False)
    trainer = pl.Trainer(
        logger=logger if logger is not None else False,
        callbacks=callbacks,
        **cfg.pl_trainer,
    )
    pl.seed_everything(cfg.seed)
    datamodule: pl.LightningDataModule = get_data(cfg)
    model: pl.LightningModule = get_model(cfg)
    # Top-level flag (dreamer4-style); also accept model.resume_reset_scheduler.
    if getattr(cfg, "resume_reset_scheduler", False):
        model.resume_reset_scheduler = True

    train_loader = datamodule.train_dataloader()

    if supports_combined_normalizer(train_loader):
        normalizer = fit_combined_normalizer(train_loader)
    else:
        # Datasets without embodiment_transform (AbcEpisodeDataset) keep the
        # pre-2026-09 binding: the loader's own fitted normalizer.
        normalizer = find_train_normalizer(train_loader, cfg=cfg)
    if normalizer is not None:
        # Deep copy: model gets its own normalizer (moves to GPU with model),
        # dataset keeps its CPU normalizer (safe for DataLoader workers with num_workers>0)
        model.pipeline.normalizer = copy.deepcopy(normalizer)
        Log.info(
            "Bound pipeline normalizer "
            f"({normalizer_fingerprint(normalizer)}; "
            f"n_dataset_opts={n_train_dataset_opts(cfg)})"
        )

    if not cfg.get("preserve_output_dir", False):
        _delete_output_dir(
            cfg.resume_training, cfg.output_dir, cfg.confirm_delete_previous_dir
        )
    if getattr(cfg, "local_rank", 0) == 0:
        _save_resolved_config(cfg)
        # Sidecar so eval can rebind without re-fitting / guessing CombinedLoader order.
        if normalizer is not None:
            save_action_normalizer(
                normalizer,
                cfg.output_dir,
                dataset_opt_index=-1,
                n_opts=n_train_dataset_opts(cfg),
            )
    ckpt_path = cfg.get("resume_checkpoint") or find_last_ckpt_path(cfg.callbacks.model_checkpoint.dirpath)

    # Pass pre-created train_loader to trainer.fit() to avoid double dataset
    # instantiation (datamodule.train_dataloader() would create a second copy).
    # This is critical when load_to_memory=true on memory-constrained machines.
    if cfg.resume_training:
        print(f"Resuming training from {ckpt_path}")
        trainer.fit(model, train_dataloaders=train_loader, ckpt_path=ckpt_path)
    else:
        warm_start = getattr(cfg, "warm_start_ckpt", None)
        ckpt_type = getattr(cfg, "ckpt_type", None)
        if warm_start:
            model.load_pretrained_model(warm_start, ckpt_type)
            print(f"Warm-starting from: {warm_start}")
        elif ckpt_path:
            model.load_pretrained_model(ckpt_path, ckpt_type)
            print(f"Loading pretrained from: {ckpt_path}")
        else:
            print("Starting new training from scratch")
        trainer.fit(model, train_dataloaders=train_loader, ckpt_path=None)
