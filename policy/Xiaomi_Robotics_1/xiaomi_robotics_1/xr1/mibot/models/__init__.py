# Copyright (C) 2026 Xiaomi Corporation.
from mmengine import Registry

MIMODEL = Registry("MIMODEL")

try:
    from mibot.models.runner.base_runner import BaseRunner
except ModuleNotFoundError as exc:
    if exc.name != "lightning":
        raise
    BaseRunner = None

from mibot.models.VLA.xr1 import xr1  # noqa: E402

__all__ = ["MIMODEL", "BaseRunner", "xr1"]
