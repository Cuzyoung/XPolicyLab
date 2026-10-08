from typing import Union, Dict
from collections import OrderedDict

import zarr
import numpy as np
import torch
import torch.nn as nn


def dict_apply(x, func):
    if not isinstance(x, (dict, OrderedDict)):
        return func(x)
    dict_type = type(x)

    result = dict_type()
    for key, value in x.items():
        if isinstance(value, (str, list)):
            result[key] = value
        elif isinstance(value, (dict_type, dict, OrderedDict)):
            result[key] = dict_apply(value, func)
        else:
            result[key] = func(value)
    return result


class DictOfTensorMixin(nn.Module):
    def __init__(self, params_dict=None):
        super().__init__()
        if params_dict is None:
            params_dict = nn.ParameterDict()
        self.params_dict = params_dict

    @property
    def device(self):
        return next(iter(self.parameters())).device

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        def dfs_add(dest, keys, value: torch.Tensor):
            if len(keys) == 1:
                dest[keys[0]] = value
                return

            if keys[0] not in dest:
                dest[keys[0]] = nn.ParameterDict()
            dfs_add(dest[keys[0]], keys[1:], value)

        def load_dict(state_dict, prefix):
            out_dict = nn.ParameterDict()
            for key, value in state_dict.items():
                value: torch.Tensor
                if key.startswith(prefix):
                    param_keys = key[len(prefix) :].split(".")[1:]
                    # if len(param_keys) == 0:
                    #     import pdb; pdb.set_trace()
                    dfs_add(out_dict, param_keys, value.clone())
            return out_dict

        self.params_dict = load_dict(state_dict, prefix + "params_dict")
        self.params_dict.requires_grad_(False)
        # Keep on CPU for DataLoader worker compatibility (fork safety)
        for p in self.params_dict.parameters():
            p.data = p.data.cpu()
        return
        

class LinearNormalizer(DictOfTensorMixin):
    avaliable_modes = ["limits", "gaussian"]

    @torch.no_grad()
    def fit(
        self,
        data: Union[Dict, torch.Tensor, np.ndarray, zarr.Array],
        last_n_dims=1,
        dtype=torch.float32,
        mode="limits",
        output_max=1.0,
        output_min=-1.0,
        range_eps=1e-4,
        fit_offset=True,
        horizon=1,
        horizon_stats=False,
    ):
        if isinstance(data, dict):
            for key, value in data.items():
                expand_scale = 1 if "action" not in key else horizon
                self.params_dict[key] = _fit(
                    value,
                    last_n_dims=last_n_dims,
                    dtype=dtype,
                    mode=mode,
                    output_max=output_max,
                    output_min=output_min,
                    range_eps=range_eps,
                    fit_offset=fit_offset,
                    horizon=expand_scale,
                    key=key,
                    horizon_stats=horizon_stats,
                )
        else:
            self.params_dict["_default"] = _fit(
                data,
                last_n_dims=last_n_dims,
                dtype=dtype,
                mode=mode,
                output_max=output_max,
                output_min=output_min,
                range_eps=range_eps,
                fit_offset=fit_offset,
                horizon=horizon,
                horizon_stats=horizon_stats,
            )

    @torch.no_grad()
    def fit_from_stats(self, data: Dict[str, list], mode="limits", output_max=1.0,
                        output_min=-1.0, range_eps=1e-4, fit_offset=True):
        for key, per_dataset_stats in data.items():
            self.params_dict[key] = _fit_from_stats(
                per_dataset_stats, mode=mode, output_max=output_max, output_min=output_min,
                range_eps=range_eps, fit_offset=fit_offset,
            )

    def __call__(self, x: Union[Dict, torch.Tensor, np.ndarray]) -> torch.Tensor:
        return self.normalize(x)

    def __getitem__(self, key: str):
        return SingleFieldLinearNormalizer(self.params_dict[key])

    def __setitem__(self, key: str, value: "SingleFieldLinearNormalizer"):
        self.params_dict[key] = value.params_dict

    def _normalize_impl(self, x, forward=True):
        if isinstance(x, dict):
            result = dict()
            for key, value in x.items():
                if key in self.params_dict:
                    params = self.params_dict[key]
                    result[key] = _normalize(value, params, forward=forward)
                else:
                    result[key] = value
            return result
        else:
            if "_default" not in self.params_dict:
                raise RuntimeError("Not initialized")
            params = self.params_dict["_default"]
            return _normalize(x, params, forward=forward)

    def normalize(self, x: Union[Dict, torch.Tensor, np.ndarray]) -> Union[Dict, torch.Tensor]:
        return self._normalize_impl(x, forward=True)

    def unnormalize(self, x: Union[Dict, torch.Tensor, np.ndarray]) -> Union[Dict, torch.Tensor]:
        return self._normalize_impl(x, forward=False)

    def get_input_stats(self) -> Dict:
        if len(self.params_dict) == 0:
            raise RuntimeError("Not initialized")
        if len(self.params_dict) == 1 and "_default" in self.params_dict:
            return self.params_dict["_default"]["input_stats"]

        result = dict()
        for key, value in self.params_dict.items():
            if key != "_default":
                result[key] = value["input_stats"]
        return result

    def get_output_stats(self, key="_default"):
        input_stats = self.get_input_stats()
        if "min" in input_stats:
            # no dict
            return dict_apply(input_stats, self.normalize)

        result = dict()
        for key, group in input_stats.items():
            this_dict = dict()
            for name, value in group.items():
                this_dict[name] = self.normalize({key: value})[key]
            result[key] = this_dict
        return result


class SingleFieldLinearNormalizer(DictOfTensorMixin):
    avaliable_modes = ["limits", "gaussian"]

    @torch.no_grad()
    def fit(
        self,
        data: Union[torch.Tensor, np.ndarray, zarr.Array],
        last_n_dims=1,
        dtype=torch.float32,
        mode="limits",
        output_max=1.0,
        output_min=-1.0,
        range_eps=1e-4,
        fit_offset=True,
        horizon=1,
        horizon_stats=False,
    ):
        self.params_dict = _fit(
            data,
            last_n_dims=last_n_dims,
            dtype=dtype,
            mode=mode,
            output_max=output_max,
            output_min=output_min,
            range_eps=range_eps,
            fit_offset=fit_offset,
            horizon=horizon,
            horizon_stats=horizon_stats,
        )

    @torch.no_grad()
    def fit_from_stats(self, per_dataset_stats: list, **kwargs):
        self.params_dict = _fit_from_stats(per_dataset_stats, **kwargs)

    @classmethod
    def create_fit(cls, data: Union[torch.Tensor, np.ndarray, zarr.Array], **kwargs):
        obj = cls()
        obj.fit(data, **kwargs)
        return obj

    @classmethod
    def create_manual(
        cls,
        scale: Union[torch.Tensor, np.ndarray],
        offset: Union[torch.Tensor, np.ndarray],
        input_stats_dict: Dict[str, Union[torch.Tensor, np.ndarray]],
    ):
        def to_tensor(x):
            if not isinstance(x, torch.Tensor):
                x = torch.from_numpy(x)
            x = x.flatten()
            return x

        # check
        for x in [offset] + list(input_stats_dict.values()):
            assert x.shape == scale.shape
            assert x.dtype == scale.dtype

        params_dict = nn.ParameterDict(
            {
                "scale": to_tensor(scale),
                "offset": to_tensor(offset),
                "input_stats": nn.ParameterDict(
                    dict_apply(input_stats_dict, to_tensor)
                ),
            }
        )
        for p in params_dict.parameters():
            p.data = p.data.cpu()  # Keep on CPU for DataLoader worker compatibility
        return cls(params_dict)

    @classmethod
    def create_identity(cls, dtype=torch.float32):
        scale = torch.tensor([1], dtype=dtype)
        offset = torch.tensor([0], dtype=dtype)
        input_stats_dict = {
            "min": torch.tensor([-1], dtype=dtype),
            "max": torch.tensor([1], dtype=dtype),
            "mean": torch.tensor([0], dtype=dtype),
            "std": torch.tensor([1], dtype=dtype),
        }
        return cls.create_manual(scale, offset, input_stats_dict)

    def normalize(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        return _normalize(x, self.params_dict, forward=True)

    def unnormalize(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        return _normalize(x, self.params_dict, forward=False)

    def get_input_stats(self):
        return self.params_dict["input_stats"]

    def get_output_stats(self):
        return dict_apply(self.params_dict["input_stats"], self.normalize)

    def __call__(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        return self.normalize(x)


def _scale_offset_from_stats(input_min, input_max, input_mean, input_std, mode="limits",
                              output_max=1.0, output_min=-1.0, range_eps=1e-4, fit_offset=True):
    if mode == "limits":
        if fit_offset:
            input_range = input_max - input_min
            ignore_dim = input_range < range_eps
            input_range[ignore_dim] = output_max - output_min
            scale = (output_max - output_min) / input_range
            offset = output_min - scale * input_min
            offset[ignore_dim] = (output_max + output_min) / 2 - input_min[ignore_dim]
        else:
            output_abs = min(abs(output_min), abs(output_max))
            input_abs = torch.maximum(torch.abs(input_min), torch.abs(input_max))
            ignore_dim = input_abs < range_eps
            input_abs[ignore_dim] = output_abs
            scale = output_abs / input_abs
            offset = torch.zeros_like(input_mean)
    else:
        ignore_dim = input_std < range_eps
        scale = input_std.clone()
        scale[ignore_dim] = 1
        scale = 1 / scale
        offset = -input_mean * scale if fit_offset else torch.zeros_like(input_mean)
    return scale, offset


def _pack_params(scale, offset, input_min, input_max, input_mean, input_std) -> nn.ParameterDict:
    params = nn.ParameterDict({
        "scale": scale,
        "offset": offset,
        "input_stats": nn.ParameterDict(
            {"min": input_min, "max": input_max, "mean": input_mean, "std": input_std}
        ),
    })
    for p in params.parameters():
        p.requires_grad_(False)
        p.data = p.data.cpu()  # Keep on CPU for DataLoader worker compatibility
    return params


def _fit(
    data: Union[torch.Tensor, np.ndarray, zarr.Array],
    last_n_dims=1,
    dtype=torch.float32,
    mode="limits",
    output_max=1.0,
    output_min=-1.0,
    range_eps=1e-4,
    fit_offset=True,
    horizon=1,
    horizon_stats=False,
    key="action",
):
    assert mode in ["limits", "gaussian"]
    assert last_n_dims >= 0
    assert output_max > output_min

    # convert data to torch and type
    if isinstance(data, zarr.Array):
        data = data[:]
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    if dtype is not None:
        data = data.type(dtype)

    # convert shape
    dim = 1
    if last_n_dims > 0:
        dim = np.prod(data.shape[-last_n_dims:])
    if horizon_stats and len(data.shape) > 2:
        assert len(data.shape) == 3
    else:
        data = data.reshape(-1, dim)

    # hack around integrate actions
    if horizon > 1 and key == "action" and data.shape[1] == 6:
        data = data.clone() * horizon
        data[3:] = torch.clamp(data[3:], -np.pi, np.pi)

    # compute input stats min max mean std
    input_min, _ = data.min(axis=0)
    input_max, _ = data.max(axis=0)
    input_mean = data.mean(axis=0)
    input_std = data.std(axis=0)

    scale, offset = _scale_offset_from_stats(
        input_min, input_max, input_mean, input_std,
        mode=mode, output_max=output_max, output_min=output_min,
        range_eps=range_eps, fit_offset=fit_offset,
    )
    return _pack_params(scale, offset, input_min, input_max, input_mean, input_std)


def _combine_stats(per_dataset_stats: list) -> tuple:
    # std must be unbiased (ddof=1) per dataset, matching _fit's data.std(axis=0) default.
    # Pooled variance is NOT a weighted average of per-dataset variances -- it also needs
    # the between-group term below, or it undercounts spread when per-dataset means differ.
    ns = [float(s["n"]) for s in per_dataset_stats]
    mins = torch.stack([torch.as_tensor(s["min"], dtype=torch.float64) for s in per_dataset_stats])
    maxs = torch.stack([torch.as_tensor(s["max"], dtype=torch.float64) for s in per_dataset_stats])
    means = [torch.as_tensor(s["mean"], dtype=torch.float64) for s in per_dataset_stats]
    stds = [torch.as_tensor(s["std"], dtype=torch.float64) for s in per_dataset_stats]

    n_total = sum(ns)
    combined_mean = sum(n_i * mean_i for n_i, mean_i in zip(ns, means)) / n_total
    m2 = sum(
        std_i**2 * (n_i - 1) + n_i * (mean_i - combined_mean) ** 2
        for n_i, mean_i, std_i in zip(ns, means, stds)
    )
    combined_std = torch.sqrt(m2 / (n_total - 1))
    combined_min, _ = mins.min(dim=0)
    combined_max, _ = maxs.max(dim=0)
    return combined_min.float(), combined_max.float(), combined_mean.float(), combined_std.float()


def _fit_from_stats(per_dataset_stats: list, mode="limits", output_max=1.0, output_min=-1.0,
                     range_eps=1e-4, fit_offset=True) -> nn.ParameterDict:
    combined_min, combined_max, combined_mean, combined_std = _combine_stats(per_dataset_stats)
    scale, offset = _scale_offset_from_stats(
        combined_min, combined_max, combined_mean, combined_std,
        mode=mode, output_max=output_max, output_min=output_min,
        range_eps=range_eps, fit_offset=fit_offset,
    )
    return _pack_params(scale, offset, combined_min, combined_max, combined_mean, combined_std)


def _normalize(x, params, forward=True):
    assert "scale" in params
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    scale = params["scale"]
    offset = params["offset"]
    # Adapt to input device (supports DataLoader workers on CPU)
    if scale.device != x.device:
        scale = scale.to(device=x.device)
        offset = offset.to(device=x.device)
    x = x.to(dtype=scale.dtype)
    src_shape = x.shape
    x = x.reshape(-1, *scale.shape)
    if forward:
        x = x * scale + offset
    else:
        x = (x - offset) / scale
    x = x.reshape(src_shape)
    return x


def test():
    data = torch.zeros((100, 10, 9, 2)).uniform_()
    data[..., 0, 0] = 0

    normalizer = SingleFieldLinearNormalizer()
    normalizer.fit(data, mode="limits", last_n_dims=2)
    datan = normalizer.normalize(data)
    assert datan.shape == data.shape
    assert np.allclose(datan.max(), 1.0)
    assert np.allclose(datan.min(), -1.0)
    dataun = normalizer.unnormalize(datan)
    assert torch.allclose(data, dataun, atol=1e-7)

    input_stats = normalizer.get_input_stats()
    output_stats = normalizer.get_output_stats()

    normalizer = SingleFieldLinearNormalizer()
    normalizer.fit(data, mode="limits", last_n_dims=1, fit_offset=False)
    datan = normalizer.normalize(data)
    assert datan.shape == data.shape
    assert np.allclose(datan.max(), 1.0, atol=1e-3)
    assert np.allclose(datan.min(), 0.0, atol=1e-3)
    dataun = normalizer.unnormalize(datan)
    assert torch.allclose(data, dataun, atol=1e-7)

    data = torch.zeros((100, 10, 9, 2)).uniform_()
    normalizer = SingleFieldLinearNormalizer()
    normalizer.fit(data, mode="gaussian", last_n_dims=0)
    datan = normalizer.normalize(data)
    assert datan.shape == data.shape
    assert np.allclose(datan.mean(), 0.0, atol=1e-3)
    assert np.allclose(datan.std(), 1.0, atol=1e-3)
    dataun = normalizer.unnormalize(datan)
    assert torch.allclose(data, dataun, atol=1e-7)

    # dict
    data = torch.zeros((100, 10, 9, 2)).uniform_()
    data[..., 0, 0] = 0

    normalizer = LinearNormalizer()
    normalizer.fit(data, mode="limits", last_n_dims=2)
    datan = normalizer.normalize(data)
    assert datan.shape == data.shape
    assert np.allclose(datan.max(), 1.0)
    assert np.allclose(datan.min(), -1.0)
    dataun = normalizer.unnormalize(datan)
    assert torch.allclose(data, dataun, atol=1e-7)

    input_stats = normalizer.get_input_stats()
    output_stats = normalizer.get_output_stats()

    data = {
        "obs": torch.zeros((1000, 128, 9, 2)).uniform_() * 512,
        "action": torch.zeros((1000, 128, 2)).uniform_() * 512,
    }
    normalizer = LinearNormalizer()
    normalizer.fit(data)
    datan = normalizer.normalize(data)
    dataun = normalizer.unnormalize(datan)
    for key in data:
        assert torch.allclose(data[key], dataun[key], atol=1e-4)

    input_stats = normalizer.get_input_stats()
    output_stats = normalizer.get_output_stats()

    state_dict = normalizer.state_dict()
    n = LinearNormalizer()
    n.load_state_dict(state_dict)
    datan = n.normalize(data)
    dataun = n.unnormalize(datan)
    for key in data:
        assert torch.allclose(data[key], dataun[key], atol=1e-4)