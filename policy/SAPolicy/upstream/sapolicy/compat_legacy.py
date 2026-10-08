"""Backward-compat imports: ``savla.*`` → ``sapolicy.*`` (pre-rename configs / ckpts).

Old ``resolved_config.yaml`` and Hydra ``_target_`` strings still say
``savla.trainers.vla.TCPVLAModel`` / ``savla.models.savla.SAVLA``. Install this
finder early (see ``main.py``) so eval/train can resolve those paths.
"""
from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import sys
from types import ModuleType

# Modules whose leaf name changed in the rename.
_MODULE_MAP = {
    "savla.trainers.vla": "sapolicy.trainers.sa_policy",
    "savla.models.savla": "sapolicy.models.sa_policy",
}
# Extra attribute aliases on remapped modules.
_ATTR_MAP = {
    "savla.trainers.vla": {
        "TCPVLAModel": "SAPolicyModel",
        "SAPolicyModel": "SAPolicyModel",
    },
    "savla.models.savla": {
        "SAVLA": "SAPolicy",
        "SAPolicy": "SAPolicy",
    },
}


class _AliasLoader(importlib.abc.Loader):
    def __init__(self, real_name: str, attr_aliases: dict[str, str] | None = None):
        self.real_name = real_name
        self.attr_aliases = attr_aliases or {}

    def create_module(self, spec):
        return None

    def exec_module(self, module: ModuleType) -> None:
        real = importlib.import_module(self.real_name)
        module.__dict__.update(
            {k: v for k, v in real.__dict__.items() if k not in ("__name__", "__file__", "__package__", "__loader__", "__spec__", "__path__")}
        )
        module.__file__ = getattr(real, "__file__", None)
        if hasattr(real, "__path__"):
            module.__path__ = list(real.__path__)
        for old, new in self.attr_aliases.items():
            setattr(module, old, getattr(real, new))


class _SavlaFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):  # noqa: ARG002
        if fullname != "savla" and not fullname.startswith("savla."):
            return None
        if fullname in sys.modules:
            return None

        if fullname in _MODULE_MAP:
            real_name = _MODULE_MAP[fullname]
            loader = _AliasLoader(real_name, _ATTR_MAP.get(fullname))
            return importlib.machinery.ModuleSpec(fullname, loader, origin=real_name, is_package=False)

        real_name = "sapolicy" + fullname[len("savla") :]
        try:
            real_spec = importlib.util.find_spec(real_name)
        except (ModuleNotFoundError, ValueError):
            return None
        if real_spec is None:
            return None

        is_pkg = real_spec.submodule_search_locations is not None
        loader = _AliasLoader(real_name)
        spec = importlib.machinery.ModuleSpec(
            fullname, loader, origin=real_spec.origin, is_package=is_pkg
        )
        if is_pkg:
            # Keep child imports as ``savla.*`` so they re-enter this finder.
            spec.submodule_search_locations = []
        return spec


def install() -> None:
    if any(type(f).__name__ == "_SavlaFinder" for f in sys.meta_path):
        return
    sys.meta_path.insert(0, _SavlaFinder())
