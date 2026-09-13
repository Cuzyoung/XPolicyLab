import pytorch_lightning as pl
from pytorch_lightning.utilities.combined_loader import CombinedLoader
from hydra.utils import instantiate
from torch.utils.data import DataLoader, ConcatDataset, IterableDataset, WeightedRandomSampler
from omegaconf import ListConfig, DictConfig
from numpy.random import choice
from copy import deepcopy
import importlib


def get_func_from_path(func_path: str):
    """
    func_path: e.g. "sapolicy.dataset.maniskill_tfds.collate_fn"
    returns: function object
    """
    module_path, func_name = func_path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    func = getattr(module, func_name)
    return func


def _balanced_sample_weights(datasets, group_attr="embodiment"):
    """Per-sample weight = 1 / (total steps of its group). Also returns each
    group's total step count."""
    group_lengths = {}
    keys = []
    for i, ds in enumerate(datasets):
        key = getattr(ds, group_attr, None) or f"__solo_{i}"
        keys.append(key)
        group_lengths[key] = group_lengths.get(key, 0) + len(ds)
    weights = []
    for ds, key in zip(datasets, keys):
        weights.extend([1.0 / group_lengths[key]] * len(ds))
    return weights, group_lengths


def _resolve_num_samples(flag, dataset_len, group_lengths):
    """
    `balanced_sampling` entries are `True` (draw len(dataset) samples/epoch,.
    """
    if flag == "min_group":
        return min(group_lengths.values())
    return dataset_len


def _sanitize_loader_opts(opts):
    """torch.utils.data.DataLoader rejects prefetch_factor / persistent_workers when
    num_workers == 0 (single-process smokes, pods with a tiny /dev/shm); drop them."""
    opts = dict(opts)
    if int(opts.get("num_workers", 0) or 0) == 0:
        opts.pop("prefetch_factor", None)
        opts["persistent_workers"] = False
    return opts


class GeneralDataModule(pl.LightningDataModule):
    default_train_loader_opts = DictConfig(
        {
            "batch_size": 1,
            "num_workers": 0,
            "shuffle": False,
            "pin_memory": True,
            "drop_last": True,
            "persistent_workers": True,
        }
    )
    default_val_loader_opts = DictConfig(
        {
            "batch_size": 1,
            "num_workers": 0,
            "shuffle": False,
            "pin_memory": False,
            "drop_last": False,
            "persistent_workers": True,
        }
    )

    def __init__(
        self,
        train_dataset: DictConfig = None,
        val_dataset: DictConfig = None,
        test_dataset: DictConfig = None,
        train_loader_opts: DictConfig = None,
        val_loader_opts: DictConfig = None,
        check_mask: bool = True,
        **kwargs
    ):
        """This is a general datamodule that can be used for any dataset.
        Train uses ConcatDataset
        Val and Test use CombinedLoader, sequential, completely consumes ecah iterable sequentially, and returns a triplet (data, idx, iterable_idx)
        Args:
            name: used by other module
            dataset_opts: the target of the dataset. e.g. dataset_opts.train = {_target_: ..., limit_size: None}
            loader_opts: the options for the dataset
            limit_each_trainset: limit the size of each dataset, None means no limit, useful for debugging
        """
        super().__init__()
        self.check_mask = check_mask
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.train_loader_opts = self.default_train_loader_opts
        self.val_loader_opts = self.default_val_loader_opts

        if train_loader_opts is not None:
            self.train_loader_opts.update(train_loader_opts)
        if val_loader_opts is not None:
            self.val_loader_opts.update(val_loader_opts)
        self.train_loader_opts.persistent_workers = (
            True if self.train_loader_opts.num_workers > 0 else False
        )
        self.val_loader_opts.persistent_workers = (
            True if self.val_loader_opts.num_workers > 0 else False
        )

        if not isinstance(self.val_dataset.dataset_opts, ListConfig):
            self.val_dataset.dataset_opts.update(
                {"check_mask": self.check_mask}
            )
        else:
            for idx, dataset_opt in enumerate(self.val_dataset.dataset_opts):
                if isinstance(dataset_opt, ListConfig):
                    for opt in dataset_opt:
                        opt.update({"check_mask": self.check_mask})
                else:
                    dataset_opt.update({"check_mask": self.check_mask})

    def val_dataloader(self):
        loaders = GeneralDataModule._parse_loaders(
            self.val_dataset, self.val_loader_opts
        )
        if isinstance(loaders, list):
            return CombinedLoader(loaders, mode="sequential")
        else:
            return loaders

    def test_dataloader(self):
        loaders = GeneralDataModule._parse_loaders(
            self.test_dataset, self.val_loader_opts
        )
        if isinstance(loaders, list):
            return CombinedLoader(loaders, mode="sequential")
        else:
            return loaders

    def train_dataloader(self):
        return GeneralDataModule._parse_train_dataloader(
            self.train_dataset, self.train_loader_opts
        )

    @staticmethod
    def _parse_train_dataloader(config, loader_opts):
        if (
            isinstance(config.dataset_opts, ListConfig)
            and "combined_loader_opts" in config
        ):
            dataloaders = GeneralDataModule._parse_loaders(config, loader_opts)
            return CombinedLoader(dataloaders, **config.combined_loader_opts)

        elif isinstance(config.dataset_opts, ListConfig):
            datasets = GeneralDataModule._parse_datasets(config)

            # ✅ 判断是否含 IterableDataset
            if any(isinstance(ds, IterableDataset) for ds in datasets):
                # 不能用 ConcatDataset，改成多个 DataLoader + CombinedLoader
                loaders = []
                for idx, ds in enumerate(datasets):
                    local_loader_opts = deepcopy(loader_opts)
                    if "loader_opts" in config:
                        if isinstance(config.loader_opts, ListConfig):
                            local_loader_opts.update(config.loader_opts[idx])
                        else:
                            local_loader_opts.update(config.loader_opts)

                    # 解析 collate_fn
                    collate_fn = None
                    if "collate_fn" in local_loader_opts:
                        collate_fn = get_func_from_path(local_loader_opts.pop("collate_fn"))

                    loaders.append(DataLoader(ds, collate_fn=collate_fn, **_sanitize_loader_opts(local_loader_opts)))

                return CombinedLoader(loaders, mode="max_size_cycle")
            else:
                # 全是 map-style dataset，正常 Concat
                dataset = ConcatDataset(datasets)
                loader_opts = deepcopy(loader_opts)
                if "loader_opts" in config:
                    loader_opts.update(config.loader_opts)

                # 解析 collate_fn
                collate_fn = None
                if "collate_fn" in loader_opts:
                    collate_fn = get_func_from_path(loader_opts.pop("collate_fn"))

                sampler = None
                flag = config.get("balanced_sampling", False)
                if flag:
                    weights, group_lengths = _balanced_sample_weights(datasets)
                    num_samples = _resolve_num_samples(flag, len(dataset), group_lengths)
                    sampler = WeightedRandomSampler(weights, num_samples=num_samples, replacement=True)
                    loader_opts.pop("shuffle", None)

                return DataLoader(dataset, sampler=sampler, collate_fn=collate_fn, **_sanitize_loader_opts(loader_opts))

        else:
            return GeneralDataModule._parse_loaders(config, loader_opts)

    @staticmethod
    def _parse_datasets(config):
        datasets = []
        for idx, dataset_opt in enumerate(config.dataset_opts):
            dataset = instantiate(dataset_opt)
            datasets.append(dataset)
        return datasets

    @staticmethod
    def _parse_loaders(config, loader_opts):
        if not isinstance(config.dataset_opts, ListConfig):
            dataset = instantiate(config.dataset_opts)
            local_loader_opts = deepcopy(loader_opts)
            if "loader_opts" in config:
                local_loader_opts.update(config.loader_opts)

            # 只在这里解析函数，不存回 loader_opts
            collate_fn = None
            if "collate_fn" in local_loader_opts:
                collate_fn = get_func_from_path(local_loader_opts.pop("collate_fn"))

            return DataLoader(dataset, collate_fn=collate_fn, **_sanitize_loader_opts(local_loader_opts))

        else:
            balanced_flags = config.get("balanced_sampling", None)
            dataloaders = []
            for idx, dataset_opt in enumerate(config.dataset_opts):
                sampler = None
                if isinstance(dataset_opt, ListConfig):
                    datasets = [instantiate(opt) for opt in dataset_opt]
                    dataset = ConcatDataset(datasets)
                    flag = balanced_flags[idx] if balanced_flags is not None and idx < len(balanced_flags) else False
                    if flag:
                        weights, group_lengths = _balanced_sample_weights(datasets)
                        num_samples = _resolve_num_samples(flag, len(dataset), group_lengths)
                        sampler = WeightedRandomSampler(weights, num_samples=num_samples, replacement=True)
                else:
                    dataset = instantiate(dataset_opt)

                local_loader_opts = deepcopy(loader_opts)
                if "loader_opts" in config:
                    if isinstance(config.loader_opts, ListConfig):
                        local_loader_opts.update(config.loader_opts[idx])
                    else:
                        local_loader_opts.update(config.loader_opts)

                # 解析 collate_fn
                collate_fn = None
                if "collate_fn" in local_loader_opts:
                    collate_fn = get_func_from_path(local_loader_opts.pop("collate_fn"))

                if sampler is not None:
                    local_loader_opts.pop("shuffle", None)

                print(f"local_loader_opts: {local_loader_opts}")
                dataloaders.append(DataLoader(dataset, sampler=sampler, collate_fn=collate_fn, **_sanitize_loader_opts(local_loader_opts)))
            return dataloaders
