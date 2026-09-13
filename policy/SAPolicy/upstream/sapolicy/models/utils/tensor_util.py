"""Minimal nested tensor helpers used by CropRandomizer."""
import collections

import numpy as np
import torch


def recursive_dict_list_tuple_apply(x, type_func_dict):
    """Recursively apply functions keyed by data type over nested structures."""
    assert list not in type_func_dict
    assert tuple not in type_func_dict
    assert dict not in type_func_dict

    if isinstance(x, (dict, collections.OrderedDict)):
        new_x = collections.OrderedDict() if isinstance(x, collections.OrderedDict) else dict()
        for k, v in x.items():
            new_x[k] = recursive_dict_list_tuple_apply(v, type_func_dict)
        return new_x
    if isinstance(x, (list, tuple)):
        ret = [recursive_dict_list_tuple_apply(v, type_func_dict) for v in x]
        return tuple(ret) if isinstance(x, tuple) else ret

    for t, f in type_func_dict.items():
        if isinstance(x, t):
            return f(x)
    raise NotImplementedError(f"Cannot handle data type {type(x)}")


def map_tensor(x, func):
    return recursive_dict_list_tuple_apply(
        x,
        {
            torch.Tensor: func,
            type(None): lambda y: y,
        },
    )


def unsqueeze(x, dim):
    return recursive_dict_list_tuple_apply(
        x,
        {
            torch.Tensor: lambda t: t.unsqueeze(dim=dim),
            np.ndarray: lambda a: np.expand_dims(a, axis=dim),
            type(None): lambda y: y,
        },
    )


def flatten_single(x, begin_axis=1):
    fixed_size = x.size()[:begin_axis]
    return x.reshape(*list(fixed_size), -1)


def flatten(x, begin_axis=1):
    return recursive_dict_list_tuple_apply(
        x,
        {
            torch.Tensor: lambda t, b=begin_axis: flatten_single(t, begin_axis=b),
        },
    )


def reshape_dimensions_single(x, begin_axis, end_axis, target_dims):
    assert begin_axis <= end_axis
    assert begin_axis >= 0
    assert end_axis < len(x.shape)
    assert isinstance(target_dims, (tuple, list))
    s = x.shape
    final_s = []
    for i in range(len(s)):
        if i == begin_axis:
            final_s.extend(target_dims)
        elif i < begin_axis or i > end_axis:
            final_s.append(s[i])
    return x.reshape(*final_s)


def reshape_dimensions(x, begin_axis, end_axis, target_dims):
    return recursive_dict_list_tuple_apply(
        x,
        {
            torch.Tensor: lambda t, b=begin_axis, e=end_axis, td=target_dims: reshape_dimensions_single(
                t, begin_axis=b, end_axis=e, target_dims=td
            ),
            np.ndarray: lambda a, b=begin_axis, e=end_axis, td=target_dims: reshape_dimensions_single(
                a, begin_axis=b, end_axis=e, target_dims=td
            ),
            type(None): lambda y: y,
        },
    )


def join_dimensions(x, begin_axis, end_axis):
    return recursive_dict_list_tuple_apply(
        x,
        {
            torch.Tensor: lambda t, b=begin_axis, e=end_axis: reshape_dimensions_single(
                t, begin_axis=b, end_axis=e, target_dims=[-1]
            ),
            np.ndarray: lambda a, b=begin_axis, e=end_axis: reshape_dimensions_single(
                a, begin_axis=b, end_axis=e, target_dims=[-1]
            ),
            type(None): lambda y: y,
        },
    )


def expand_at_single(x, size, dim):
    assert dim < x.ndimension()
    assert x.shape[dim] == 1
    expand_dims = [-1] * x.ndimension()
    expand_dims[dim] = size
    return x.expand(*expand_dims)


def expand_at(x, size, dim):
    return map_tensor(x, lambda t, s=size, d=dim: expand_at_single(t, s, d))


def unsqueeze_expand_at(x, size, dim):
    x = unsqueeze(x, dim)
    return expand_at(x, size, dim)
