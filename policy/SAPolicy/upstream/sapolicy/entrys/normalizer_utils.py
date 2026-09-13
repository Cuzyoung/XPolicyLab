"""Helpers for selecting the action-branch normalizer in joint TCP+action runs.

Joint configs use CombinedLoader([tcp_loader, action_loader]). A naive
first-match (or an older train that bound the TCP branch) bakes TCP-task
action stats into the checkpoint. Cross-task setups (e.g. Nut TCP + Coffee
action) then unnormalize task actions with Nut stats and collapse to zero
success.

Contract:
  - Train binds + persists the action-branch normalizer (last CombinedLoader /
    last dataset_opts by convention).
  - Eval resolves via ``normalizer_source`` (default ``auto``: sidecar, then
    action-branch dataset for joint runs, else ckpt).
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import torch
from hydra.utils import instantiate
from numpy.lib.stride_tricks import sliding_window_view
from omegaconf import DictConfig, ListConfig, OmegaConf

from sapolicy.dataset.normalizer import LinearNormalizer
from sapolicy.dataset.robomimic_hdf5 import _assemble_state
from sapolicy.embodiment_transforms import resolve_transform_config
from sapolicy.logger import Log

ACTION_NORMALIZER_FILENAME = "action_normalizer.pt"
ACTION_NORMALIZER_META_FILENAME = "action_normalizer_meta.json"

ASSETS_ROOT = Path(__file__).resolve().parents[2] / "assets" / "normalizer_stats"


def _find_first_normalizer(loader_or_dataset: Any) -> Any:
    """Recursively find the first action normalizer on a DataLoader/dataset tree."""
    if loader_or_dataset is None:
        return None

    normalizer = getattr(loader_or_dataset, "normalizer", None)
    if normalizer is not None:
        return normalizer

    dataset = getattr(loader_or_dataset, "dataset", None)
    if dataset is not None:
        normalizer = _find_first_normalizer(dataset)
        if normalizer is not None:
            return normalizer

    datasets = getattr(loader_or_dataset, "datasets", None)
    if datasets is not None:
        normalizer = _find_first_normalizer(datasets)
        if normalizer is not None:
            return normalizer

    if isinstance(loader_or_dataset, dict):
        children = list(loader_or_dataset.values())
    elif isinstance(loader_or_dataset, (list, tuple)):
        children = list(loader_or_dataset)
    else:
        children = []

    for child in children:
        normalizer = _find_first_normalizer(child)
        if normalizer is not None:
            return normalizer

    iterables = getattr(loader_or_dataset, "iterables", None)
    if iterables is not None:
        return _find_first_normalizer(iterables)

    return None


def _find_last_normalizer(loader_or_dataset: Any) -> Any:
    """Like first-match, but prefer the last child (action branch by convention)."""
    if loader_or_dataset is None:
        return None

    dataset = getattr(loader_or_dataset, "dataset", None)
    if dataset is not None:
        normalizer = _find_last_normalizer(dataset)
        if normalizer is not None:
            return normalizer

    datasets = getattr(loader_or_dataset, "datasets", None)
    if datasets is not None:
        normalizer = _find_last_normalizer(datasets)
        if normalizer is not None:
            return normalizer

    if isinstance(loader_or_dataset, dict):
        children = list(loader_or_dataset.values())
    elif isinstance(loader_or_dataset, (list, tuple)):
        children = list(loader_or_dataset)
    else:
        children = []

    if children:
        for child in reversed(children):
            normalizer = _find_last_normalizer(child)
            if normalizer is not None:
                return normalizer

    iterables = getattr(loader_or_dataset, "iterables", None)
    if iterables is not None:
        return _find_last_normalizer(iterables)

    return getattr(loader_or_dataset, "normalizer", None)


def _combined_loader_branches(train_loader: Any) -> list:
    iterables = getattr(train_loader, "iterables", None)
    if iterables is None:
        return []
    if isinstance(iterables, dict):
        # Prefer numeric key order when CombinedLoader converted a list to {0:..,1:..}.
        try:
            keys = sorted(iterables.keys(), key=lambda k: int(k) if str(k).isdigit() else str(k))
            return [iterables[k] for k in keys]
        except Exception:
            return list(iterables.values())
    if isinstance(iterables, (list, tuple)):
        return list(iterables)
    return []


def _underlying_datasets(loader_or_dataset: Any) -> list:
    """Flatten a DataLoader/ConcatDataset(/nested ConcatDataset) into leaf per-shard datasets."""
    if loader_or_dataset is None:
        return []

    dataset = getattr(loader_or_dataset, "dataset", None)
    if dataset is not None:
        return _underlying_datasets(dataset)

    datasets = getattr(loader_or_dataset, "datasets", None)
    if datasets is not None:
        out = []
        for d in datasets:
            out.extend(_underlying_datasets(d))
        return out

    return [loader_or_dataset]


def _pooled_raw_array(datasets: list, attr: str) -> Optional[np.ndarray]:
    """Concatenate a per-demo {demo_key: array} store (e.g. .actions) across shards. Raw,
    unconverted -- callers run it through the dataset's own embodiment_transform."""
    arrays = []
    for ds in datasets:
        store = getattr(ds, attr, None)
        if not store:
            continue
        arrays.extend(store.values())
    if not arrays:
        return None
    return np.concatenate(arrays, axis=0).astype(np.float32)


def _pooled_raw_proprio(datasets: list, attr: str = "states") -> Optional[dict]:
    """Pool a per-demo {field_name: array} store (e.g. .states) into one merged
    {field_name: concatenated_array} dict of raw, unconverted proprio."""
    merged: dict = {}
    for ds in datasets:
        store = getattr(ds, attr, None)
        if not store:
            continue
        for demo_fields in store.values():
            for field, arr in demo_fields.items():
                merged.setdefault(field, []).append(arr)
    if not merged:
        return None
    return {field: np.concatenate(arrs, axis=0) for field, arrs in merged.items()}


def _pooled_windowed_actions(datasets: list, action_sequence_length: int):
    """Pool raw per-demo actions into a windowed (sum_T, action_sequence_length, D)
    array, plus the matching per-step reference eef pose/quat (broadcast-shaped against
    the window dim) from each dataset's .states -- mirrors the windowing
    RobomimicHDF5Dataset.__getitem__ does per-item for use_relative_actions=True."""
    all_windows = []
    ref_fields: dict = {}
    for ds in datasets:
        action_store = getattr(ds, "actions", None)
        state_store = getattr(ds, "states", None) or {}
        if not action_store:
            continue
        for demo_key, raw in action_store.items():
            proprio = state_store.get(demo_key)
            if proprio is None:
                continue
            pad_len = action_sequence_length - 1
            padded = (
                np.concatenate([raw, np.repeat(raw[-1:], pad_len, axis=0)], axis=0)
                if pad_len > 0 else raw
            )
            windows = np.ascontiguousarray(
                np.moveaxis(sliding_window_view(padded, action_sequence_length, axis=0), -1, 1)
            )
            all_windows.append(windows)
            for field, arr in proprio.items():
                if field.endswith("_eef_pos") or field.endswith("_eef_quat_site"):
                    ref_fields.setdefault(field, []).append(arr[:, None, :])
    if not all_windows:
        return None, None
    pooled_windows = np.concatenate(all_windows, axis=0)
    pooled_ref = {k: np.concatenate(v, axis=0) for k, v in ref_fields.items()} if ref_fields else None
    return pooled_windows, pooled_ref


def _dataset_stats_path(ds: Any) -> Path:
    embodiment = getattr(ds, "embodiment", None) or "none"
    gripper_family = resolve_transform_config(ds.dataset_type, ds.embodiment).gripper_family or "none"
    return ASSETS_ROOT / ds.dataset_name / f"{embodiment}_{gripper_family}" / "stats.json"


def _field_stats(pooled: Optional[np.ndarray]) -> Optional[dict]:
    if pooled is None:
        return None
    return {
        "n": int(pooled.shape[0]),
        "min": pooled.min(axis=0).tolist(),
        "max": pooled.max(axis=0).tolist(),
        "mean": pooled.mean(axis=0).tolist(),
        "std": pooled.std(axis=0, ddof=1).tolist(),
    }


def _compute_dataset_stats(ds: Any) -> dict:
    transform = ds.embodiment_transform
    if bool(getattr(ds, "use_relative_actions", False)):
        windows, ref = _pooled_windowed_actions([ds], int(ds.action_sequence_length))
        pooled_actions = transform.inputs({"observation": ref, "action": windows})["action"] if windows is not None else None
    else:
        raw = _pooled_raw_array([ds], "actions")
        pooled_actions = transform.inputs({"observation": None, "action": raw})["action"] if raw is not None else None

    raw_proprio = _pooled_raw_proprio([ds], "states")
    pooled_states = None
    if raw_proprio is not None:
        transformed_obs = transform.inputs({"observation": dict(raw_proprio), "action": None})["observation"]
        pooled_states = _assemble_state(transformed_obs, getattr(ds, "arm_obs_prefixes", ["robot0"]))

    return {"action": _field_stats(pooled_actions), "state": _field_stats(pooled_states)}


def dataset_action_state_stats(ds: Any) -> dict:
    """This dataset's own (embodiment_transform-converted) action/state stats, loaded
    from assets/normalizer_stats/ if cached, else computed once and cached."""
    path = _dataset_stats_path(ds)
    if path.is_file():
        with open(path) as f:
            return json.load(f)
    stats = _compute_dataset_stats(ds)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(stats, f)
    return stats


def train_dataset_opts(cfg: DictConfig) -> Optional[Any]:
    """Return ``data.train_dataset.dataset_opts`` or None."""
    try:
        data = cfg.get("data", None)
        if data is None:
            return None
        train_ds = data.get("train_dataset", None)
        if train_ds is None:
            return None
        return train_ds.get("dataset_opts", None)
    except Exception:
        return None


def n_train_dataset_opts(cfg: DictConfig) -> int:
    opts = train_dataset_opts(cfg)
    if opts is None:
        return 0
    if isinstance(opts, (list, ListConfig)):
        return len(opts)
    return 1


def is_joint_train_cfg(cfg: DictConfig) -> bool:
    """True when train uses >=2 dataset_opts (TCP + action CombinedLoader)."""
    return n_train_dataset_opts(cfg) >= 2


def resolve_dataset_opt(cfg: DictConfig, index: int = -1) -> Any:
    """Pick one entry from ``data.train_dataset.dataset_opts`` (supports negative index)."""
    opts = train_dataset_opts(cfg)
    if opts is None:
        raise ValueError("cfg.data.train_dataset.dataset_opts is missing")
    if not isinstance(opts, (list, ListConfig)):
        if index not in (0, -1):
            raise IndexError(f"single dataset_opts entry; cannot index {index}")
        return opts
    if len(opts) < 1:
        raise ValueError("cfg.data.train_dataset.dataset_opts is empty")
    return opts[index]


def _prepare_stats_only_opt(opt: Any) -> Any:
    """Clone a dataset opt and strip RGB-cache work (actions-only normalizer fit)."""
    try:
        opt_cfg = OmegaConf.create(OmegaConf.to_container(opt, resolve=True))
    except Exception:
        opt_cfg = copy.deepcopy(opt)

    if OmegaConf.is_config(opt_cfg):
        if "load_to_memory" in opt_cfg:
            opt_cfg.load_to_memory = False
        # Skip numpy RGB/depth mmap; action stats only need HDF5 actions (+ state).
        if "cache_dir" in opt_cfg:
            opt_cfg.cache_dir = None
        try:
            opt_cfg.cache_mmap = True
        except Exception:
            pass
    elif isinstance(opt_cfg, dict):
        opt_cfg["load_to_memory"] = False
        opt_cfg["cache_dir"] = None
        opt_cfg["cache_mmap"] = True
    return opt_cfg


def normalizer_from_dataset_opt(opt: Any) -> Any:
    """Instantiate one dataset opt (stats-only) and return its ``.normalizer``."""
    opt_cfg = _prepare_stats_only_opt(opt)
    try:
        dataset = instantiate(opt_cfg)
    except Exception as e:
        Log.warn(f"Failed to instantiate dataset for normalizer: {e}")
        return None

    normalizer = getattr(dataset, "normalizer", None)
    name = getattr(dataset, "dataset_name", None) or type(dataset).__name__
    if normalizer is None:
        Log.warn(f"Dataset {name!r} has no normalizer.")
        return None
    Log.info(f"Using dataset normalizer from cfg ({name}).")
    return normalizer


def normalizer_from_action_dataset_cfg(
    cfg: DictConfig, index: int = -1
) -> Optional[Any]:
    """Instantiate ``dataset_opts[index]`` (default: last = action branch)."""
    try:
        opt = resolve_dataset_opt(cfg, index=index)
    except Exception as e:
        Log.warn(f"Cannot resolve dataset_opts[{index}]: {e}")
        return None
    return normalizer_from_dataset_opt(opt)


def load_normalizer_state_dict(path: str) -> LinearNormalizer:
    path = os.path.expandvars(os.path.expanduser(path))
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state and not any(
        k.startswith("params_dict") for k in state
    ):
        state = state["state_dict"]
    normalizer = LinearNormalizer()
    normalizer.load_state_dict(state)
    return normalizer


def save_action_normalizer(
    normalizer: Any,
    output_dir: str,
    *,
    dataset_name: Optional[str] = None,
    dataset_opt_index: int = -1,
    n_opts: Optional[int] = None,
) -> str:
    """Persist action-branch normalizer next to ``resolved_config.yaml``."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, ACTION_NORMALIZER_FILENAME)
    torch.save(normalizer.state_dict(), path)
    meta = {
        "filename": ACTION_NORMALIZER_FILENAME,
        "dataset_name": dataset_name,
        "dataset_opt_index": dataset_opt_index,
        "n_train_dataset_opts": n_opts,
    }
    meta_path = os.path.join(output_dir, ACTION_NORMALIZER_META_FILENAME)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")
    Log.info(f"Saved action-branch normalizer to {path}")
    return path


def find_action_normalizer_sidecar(
    cfg: DictConfig, ckpt_path: Optional[str] = None
) -> Optional[str]:
    """Locate ``action_normalizer.pt`` beside the run dir / checkpoint."""
    candidates = []
    output_dir = getattr(cfg, "output_dir", None)
    if output_dir:
        candidates.append(os.path.join(str(output_dir), ACTION_NORMALIZER_FILENAME))
    if ckpt_path:
        ckpt = Path(os.path.expandvars(os.path.expanduser(str(ckpt_path))))
        # .../run/checkpoints/last.ckpt → .../run/action_normalizer.pt
        candidates.append(str(ckpt.parent.parent / ACTION_NORMALIZER_FILENAME))
        candidates.append(str(ckpt.parent / ACTION_NORMALIZER_FILENAME))
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    return None


def normalizer_fingerprint(normalizer: Any) -> str:
    """Short fingerprint for log lines (detect Nut-vs-task misbind)."""
    try:
        stats = normalizer.get_input_stats()
        action = stats.get("action", stats)
        amin = action["min"].detach().float().flatten()[:4].tolist()
        amax = action["max"].detach().float().flatten()[:4].tolist()
        return f"action.min[:4]={amin} max[:4]={amax}"
    except Exception as e:
        return f"<fingerprint unavailable: {e}>"


def _set_pipeline_normalizer(model: Any, normalizer: Any) -> None:
    model.pipeline.normalizer = copy.deepcopy(normalizer)
    ema = getattr(model, "ema_pipeline", None)
    if ema is not None:
        ema.normalizer = copy.deepcopy(normalizer)


def supports_combined_normalizer(train_loader: Any) -> bool:
    """True when every leaf dataset exposes the embodiment_transform/.actions store that
    ``fit_combined_normalizer`` pools. Datasets that fit their own normalizer (e.g.
    AbcEpisodeDataset direct-read of ABC-130k) fall back to ``find_train_normalizer``."""
    branches = _combined_loader_branches(train_loader) or [train_loader]
    leaves = [d for b in branches for d in _underlying_datasets(b)]
    return bool(leaves) and all(hasattr(d, "embodiment_transform") for d in leaves)


def fit_combined_normalizer(train_loader: Any) -> Any:
    """Fit one normalizer over the action-branch dataset(s), combining each dataset's own
    cached (embodiment_transform-converted) stats rather than re-pooling raw data."""
    branches = _combined_loader_branches(train_loader)
    tcp_source = branches[0] if branches else train_loader
    action_source = branches[-1] if branches else train_loader

    tcp_datasets = _underlying_datasets(tcp_source)
    datasets = _underlying_datasets(action_source)
    if not datasets:
        Log.warn("fit_combined_normalizer: no underlying action-branch dataset found.")
        return None
    if not tcp_datasets:
        Log.warn("fit_combined_normalizer: no underlying tcp-branch dataset found.")
        return None

    action_stats = [s for ds in datasets if (s := dataset_action_state_stats(ds)["action"]) is not None]
    if not action_stats:
        Log.warn("fit_combined_normalizer: action-branch dataset(s) have no .actions.")
        return None

    full_datasets = tcp_datasets + datasets if len(branches) >= 2 else datasets
    state_stats = [s for ds in full_datasets if (s := dataset_action_state_stats(ds)["state"]) is not None]

    normalizer_data = {"action": action_stats}
    if state_stats:
        normalizer_data["state"] = state_stats

    normalizer = LinearNormalizer()
    normalizer.fit_from_stats(normalizer_data, mode="limits")
    Log.info(
        f"Fit combined normalizer over {len(datasets)} action-branch dataset(s) "
        f"({sum(s['n'] for s in action_stats)} pooled action rows"
        + (f", {sum(s['n'] for s in state_stats)} pooled state rows)" if state_stats else ")")
    )
    return normalizer

def find_train_normalizer(train_loader: Any, cfg: Optional[DictConfig] = None) -> Any:
    """Pick the normalizer used for eval unnormalize / state norm.

    Preference order:
      1) last CombinedLoader branch (action)
      2) last train dataset_opts via cfg (action)
      3) last-match walk of the loader tree
      4) first-match fallback
    """
    branches = _combined_loader_branches(train_loader)
    if len(branches) >= 2:
        normalizer = _find_first_normalizer(branches[-1])
        if normalizer is not None:
            Log.info(
                "Using action-branch (last CombinedLoader) normalizer for pipeline "
                f"({len(branches)} loaders)."
            )
            return normalizer
        Log.warn(
            "CombinedLoader has >=2 branches but last branch has no normalizer; "
            "trying cfg action dataset / last-match."
        )
    elif len(branches) == 1:
        normalizer = _find_first_normalizer(branches[0])
        if normalizer is not None:
            return normalizer

    if cfg is not None:
        normalizer = normalizer_from_action_dataset_cfg(cfg, index=-1)
        if normalizer is not None:
            return normalizer

    normalizer = _find_last_normalizer(train_loader)
    if normalizer is not None:
        Log.info("Using last-match normalizer walk for pipeline.")
        return normalizer

    return _find_first_normalizer(train_loader)


def apply_action_branch_normalizer(
    model: Any, cfg: DictConfig, index: int = -1
) -> bool:
    """Overwrite pipeline(+ema) normalizer from ``dataset_opts[index]``."""
    normalizer = normalizer_from_action_dataset_cfg(cfg, index=index)
    if normalizer is None:
        return False
    _set_pipeline_normalizer(model, normalizer)
    return True


def resolve_eval_normalizer(
    model: Any,
    cfg: DictConfig,
    eval_cfg: Optional[DictConfig] = None,
    *,
    ckpt_path: Optional[str] = None,
) -> Tuple[str, Optional[Any]]:
    """Bind the eval normalizer according to ``eval.normalizer_*`` knobs.

    Config (under ``cfg.eval`` or the passed ``eval_cfg``):
      normalizer_source: action_dataset | auto | ckpt | path
        (default: auto — prefer train-time sidecar, then action-branch
         dataset_opts for joint runs, else ckpt)
      normalizer_dataset_opt_index: -1   # into data.train_dataset.dataset_opts
      normalizer_path: null             # .pt state_dict (also used by auto)

    ``auto`` resolution:
      1. ``eval.normalizer_path`` if set
      2. run-dir ``action_normalizer.pt`` sidecar (written at train)
      3. if joint (``len(dataset_opts) >= 2``): rebind from
         ``dataset_opts[normalizer_dataset_opt_index]`` (default -1 = action)
      4. else keep checkpoint normalizer

    Returns ``(source_label, normalizer_or_None)``. Raises if a required
    rebind fails (never silently keep TCP-task stats when source demands it).
    """
    if eval_cfg is None:
        eval_cfg = cfg.get("eval", None) or {}

    source = str(
        eval_cfg.get("normalizer_source", "auto") or "auto"
    ).lower()
    index = int(eval_cfg.get("normalizer_dataset_opt_index", -1))
    explicit_path = eval_cfg.get("normalizer_path", None)
    if explicit_path is not None:
        explicit_path = str(explicit_path).strip() or None

    joint = is_joint_train_cfg(cfg)

    def _require(normalizer: Optional[Any], label: str) -> Any:
        if normalizer is None:
            raise RuntimeError(
                f"[eval_net] Failed to load normalizer via source={label!r} "
                f"(joint={joint}, dataset_opt_index={index}). "
                "Set eval.normalizer_source=action_dataset|path or provide "
                "eval.normalizer_path / action_normalizer.pt."
            )
        return normalizer

    normalizer: Optional[Any] = None
    label = source

    if source == "ckpt":
        return "ckpt", None

    if source == "path":
        if not explicit_path:
            raise RuntimeError(
                "eval.normalizer_source=path requires eval.normalizer_path"
            )
        normalizer = _require(load_normalizer_state_dict(explicit_path), "path")
        label = f"path:{explicit_path}"

    elif source == "action_dataset":
        normalizer = _require(
            normalizer_from_action_dataset_cfg(cfg, index=index),
            "action_dataset",
        )
        label = f"action_dataset:dataset_opts[{index}]"

    elif source == "auto":
        if explicit_path:
            normalizer = _require(load_normalizer_state_dict(explicit_path), "path")
            label = f"auto:path:{explicit_path}"
        else:
            sidecar = find_action_normalizer_sidecar(cfg, ckpt_path=ckpt_path)
            if sidecar is not None:
                normalizer = _require(load_normalizer_state_dict(sidecar), "sidecar")
                label = f"auto:sidecar:{sidecar}"
            elif joint:
                normalizer = _require(
                    normalizer_from_action_dataset_cfg(cfg, index=index),
                    "action_dataset",
                )
                label = f"auto:action_dataset:dataset_opts[{index}]"
            else:
                return "auto:ckpt", None
    else:
        raise ValueError(
            f"Unknown eval.normalizer_source={source!r}; "
            "expected auto|ckpt|action_dataset|path"
        )

    _set_pipeline_normalizer(model, normalizer)
    return label, normalizer
