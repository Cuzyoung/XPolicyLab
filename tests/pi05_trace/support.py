"""Import real adapter/transport code with scoped, unused heavyweight dependencies.

The fake policy supplies model outputs, not adapter or protocol implementations.
The actual shared action pack/unpack and plain-RGB decode helpers remain active.
"""

from contextlib import contextmanager
import importlib
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
_PREFIXES = ("XPolicyLab", "openpi", "client_server", "openpi_client", "jax", "flax", "torch")


def _module(name, path=None):
    result = ModuleType(name)
    if path is not None:
        result.__path__ = [str(path)]
    sys.modules[name] = result
    return result


def _unused_dependency(name):
    result = ModuleType(name)

    def unavailable(attribute):
        if attribute.startswith("__"):
            raise AttributeError(attribute)
        raise AssertionError(f"{name}.{attribute} is outside the lightweight test scope")

    result.__getattr__ = unavailable
    sys.modules[name] = result


@contextmanager
def isolated_adapter_modules():
    """Restore preexisting modules/path, including when another test imported them."""
    def owned(name):
        return name in {"cv2", "h5py"} or any(
            name == prefix or name.startswith(prefix + ".") for prefix in _PREFIXES
        )

    saved_modules = {key: value for key, value in sys.modules.items() if owned(key)}
    saved_path = sys.path[:]
    try:
        for key in list(sys.modules):
            if owned(key):
                del sys.modules[key]
        sys.path[:0] = [str(ROOT), str(ROOT.parent)]
        # Pi_05/__init__.py eagerly imports deploy.py. Import the real model module
        # through a package path without loading unrelated deployment entrypoints.
        _module("XPolicyLab", ROOT)
        _module("XPolicyLab.policy", ROOT / "policy")
        _module("XPolicyLab.policy.Pi_05", ROOT / "policy/Pi_05")
        _module("XPolicyLab.utils", ROOT / "utils")
        _module("openpi", ROOT / "policy/Pi_05/openpi/src/openpi")
        _module("openpi.policies", ROOT / "policy/Pi_05/openpi/src/openpi/policies")
        _module("openpi.shared", ROOT / "policy/Pi_05/openpi/src/openpi/shared")
        _module("openpi.training", ROOT / "policy/Pi_05/openpi/src/openpi/training")
        _module("openpi.policies.policy_config")
        _module("openpi.shared.normalize")
        config = _module("openpi.training.config")
        for name in (
            "LeRobotAlohaDataConfig", "LeRobotYamDataConfig",
            "LeRobotYamJointEeDataConfig", "SimpleDataConfig",
        ):
            setattr(config, name, type(name, (), {}))
        _unused_dependency("cv2")
        _unused_dependency("h5py")
        model = importlib.import_module("XPolicyLab.policy.Pi_05.model")
        server = importlib.import_module("client_server.ws.model_server")
        codec = importlib.import_module("client_server.ws.protocol.codec")
        schemas = importlib.import_module("client_server.ws.protocol.schemas")
        messages = importlib.import_module("client_server.ws.protocol.messages")
        exceptions = importlib.import_module("client_server.ws.protocol.exceptions")
        yield SimpleNamespace(
            model=model, config=config, server=server, codec=codec, schemas=schemas,
            messages=messages, exceptions=exceptions,
        )
    finally:
        for key in list(sys.modules):
            if owned(key):
                del sys.modules[key]
        sys.modules.update(saved_modules)
        sys.path[:] = saved_path


def load_policy_with_numpy_backend():
    """Load actual Policy and transforms; replace only unavailable backend imports.

    These NumPy calls do not emulate JIT, differentiation, device execution, or
    the native sampler. Tests use them only to verify Policy's Python boundary.
    The caller must be inside isolated_adapter_modules() so all stubs are scoped.
    """
    state = SimpleNamespace(jit_calls=[], random_splits=[])
    jax = _module("jax")
    jax.Array = np.ndarray

    def tree_map(function, value):
        if isinstance(value, dict):
            return {key: tree_map(function, item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(tree_map(function, item) for item in value)
        return function(value)

    def split(key):
        state.random_splits.append(key)
        return key + 1, key + 2

    jax.tree = SimpleNamespace(map=tree_map)
    jax.random = SimpleNamespace(key=lambda seed: seed, split=split)
    jnp = _module("jax.numpy")
    jnp.__getattr__ = lambda name: getattr(np, name)
    flax = _module("flax", ROOT / "tests/pi05_trace")
    flax.traverse_util = _module("flax.traverse_util")
    torch = _module("torch")
    torch.Tensor = type("Tensor", (), {})
    _module("openpi_client", ROOT / "policy/Pi_05/openpi/packages/openpi-client/src/openpi_client")
    _module("openpi.models", ROOT / "policy/Pi_05/openpi/src/openpi/models")
    model = _module("openpi.models.model")
    model.BaseModel = type("BaseModel", (), {})
    model.Observation = SimpleNamespace(from_dict=lambda value: SimpleNamespace(**value))
    tokenizer = _module("openpi.models.tokenizer")
    tokenizer.PaligemmaTokenizer = type("PaligemmaTokenizer", (), {})
    tokenizer.FASTTokenizer = type("FASTTokenizer", (), {})
    _module("openpi_client.image_tools")
    typing = _module("openpi.shared.array_typing")

    class PyTree:
        def __class_getitem__(cls, _item):
            return cls

    typing.PyTree = PyTree
    typing.KeyArrayLike = int
    sys.modules["openpi.shared.normalize"].NormStats = type("NormStats", (), {})
    nnx_utils = _module("openpi.shared.nnx_utils")

    def module_jit(method, **kwargs):
        state.jit_calls.append(kwargs)
        return method

    nnx_utils.module_jit = module_jit
    _module("openpi.models_pytorch", ROOT / "tests/pi05_trace")
    attention = _module("openpi.models_pytorch.autohorizon_official")
    attention.UPSTREAM_COMMIT = "unused-in-default-trace-tests"
    attention.bidir_soft_pointer = lambda *_args: (_ for _ in ()).throw(
        AssertionError("AutoHorizon is outside trace-test scope")
    )
    module = importlib.import_module("openpi.policies.policy")
    return module, state


class FakePolicy:
    """Deterministic normalized trace and visibly postprocessed action targets."""

    def __init__(self, *, horizon=5, model_dim=32, is_pytorch=False):
        self._is_pytorch_model = is_pytorch
        self._model = SimpleNamespace(pi05=True)
        self.calls = []
        self.resets = []
        self.horizon = horizon
        self.model_dim = model_dim
        self.actions = np.arange(horizon * 14, dtype=np.float32).reshape(horizon, 14) + 100
        self.trace_override = None
        self.include_trace = True

    def reset_rng(self, seed):
        self.resets.append(seed)

    def velocity(self, steps):
        return np.arange(steps * self.horizon * self.model_dim, dtype=np.float32).reshape(
            steps, self.horizon, self.model_dim
        ) / 1000

    def infer(self, observation, **kwargs):
        self.calls.append(dict(kwargs))
        result = {"actions": self.actions.copy()}
        if kwargs.get("return_denoise_trace", False) and self.include_trace:
            velocity = (
                self.velocity(kwargs.get("num_steps", 10))
                if self.trace_override is None else self.trace_override
            )
            result["denoise_trace"] = {"v": velocity}
        return result


def observation(*, index=0, yam=False):
    if yam:
        return {
            "env_idx": index, "observation/state": np.arange(14, dtype=np.float32),
            "observation/image": np.full((4, 5, 3), 31, dtype=np.uint8),
            "observation/left_wrist": np.full((4, 5, 3), 63, dtype=np.uint8),
            "observation/right_wrist": np.full((4, 5, 3), 95, dtype=np.uint8),
            "instruction": "synthetic trace check",
        }
    return {
        "env_idx": index, "state": np.arange(14, dtype=np.float32),
        "images": {
            name: np.full((3, 4, 5), value, dtype=np.uint8)
            for name, value in (("cam_high", 31), ("cam_left_wrist", 63), ("cam_right_wrist", 95))
        },
        "instruction": "synthetic trace check",
    }
