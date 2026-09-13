import hydra
import pytorch_lightning as pl
import rich
import rich.syntax
import rich.tree

from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from pytorch_lightning.utilities import rank_zero_only

from sapolicy.logger import Log, monitor_process_wrapper


@monitor_process_wrapper
def get_data(cfg: DictConfig, wo_train: bool = False) -> pl.LightningDataModule:
    return hydra.utils.instantiate(cfg.data, wo_train=wo_train, _recursive_=False)


# SAPolicy kwargs removed upstream (67ace8d, 6bd0e6b); old resolved_config.yaml files still carry them.
_REMOVED_PIPELINE_KEYS = ("unfreeze_rgb_layers", "use_bn", "use_clstoken", "orientation_head", "use_tcp_head", "loss_cfg")


@monitor_process_wrapper
def get_model(cfg: DictConfig) -> pl.LightningModule:
    pipeline = cfg.model.get("pipeline")
    if pipeline is not None:
        for key in _REMOVED_PIPELINE_KEYS:
            pipeline.pop(key, None)
    model = hydra.utils.instantiate(cfg.model, _recursive_=False)
    if hasattr(cfg, "exp_name"):
        model.exp_name = cfg.exp_name
    return model


@monitor_process_wrapper
def get_callbacks(cfg: DictConfig) -> list:
    if not hasattr(cfg, "callbacks"):
        return None
    callbacks = []
    for callback in cfg.callbacks.values():
        if callback is not None:
            callbacks.append(hydra.utils.instantiate(callback, _recursive_=False))
    return callbacks


def find_last_ckpt_path(dirpath):
    """Assume ckpt is named as e{}* or last*, following pytorch-lightning convention."""
    dirpath = Path(dirpath)
    model_paths = []
    for p in sorted(dirpath.glob("*.ckpt")):
        if "last" in p.name:
            continue
        model_paths.append(p)
    if model_paths:
        return model_paths[-1]
    Log.info("No checkpoint found, set model_path to None")
    return None


@rank_zero_only
def print_cfg(cfg: DictConfig, use_rich: bool = False):
    if not use_rich:
        Log.info(OmegaConf.to_yaml(cfg, resolve=False))
        return

    print_order = ("data", "model", "callbacks", "logger", "pl_trainer", "exp")
    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    queue = []
    for field in print_order:
        if field in cfg:
            queue.append(field)
        else:
            Log.warn(f"Field '{field}' not found in config. Skipping.")
    for field in cfg:
        if field not in queue:
            queue.append(field)

    for field in queue:
        branch = tree.add(field, style=style, guide_style=style)
        config_group = cfg[field]
        if isinstance(config_group, DictConfig):
            branch_content = OmegaConf.to_yaml(config_group, resolve=False)
        else:
            branch_content = str(config_group)
        branch.add(rich.syntax.Syntax(branch_content, "yaml"))
    rich.print(tree)
