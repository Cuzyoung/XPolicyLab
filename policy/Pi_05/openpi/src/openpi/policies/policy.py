from collections.abc import Sequence
import logging
import math
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)

    @override
    def infer(
        self,
        obs: dict,
        *,
        noise: np.ndarray | None = None,
        num_steps: int | None = None,
        action_condition: np.ndarray | None = None,
        condition_weights: np.ndarray | None = None,
        rtc_beta: float = 5.0,
    ) -> dict:  # type: ignore[misc]
        if num_steps is not None and num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {num_steps}")
        if not math.isfinite(rtc_beta) or rtc_beta <= 0:
            raise ValueError(f"rtc_beta must be finite and positive, got {rtc_beta}")
        if (action_condition is None) != (condition_weights is None):
            raise ValueError("action_condition and condition_weights must be provided together")
        if action_condition is not None and self._is_pytorch_model:
            raise NotImplementedError("RTC action conditioning is only implemented for JAX Pi0")

        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        if action_condition is not None:
            inputs["actions"] = np.asarray(action_condition, dtype=np.float32)
        inputs = self._input_transform(inputs)
        transformed_condition = inputs.pop("actions", None) if action_condition is not None else None

        is_batched = False
        if "state" in inputs:
            is_batched = np.asarray(inputs["state"]).ndim > 1
        elif "image" in inputs and inputs["image"]:
            first_image = next(iter(inputs["image"].values()))
            is_batched = np.asarray(first_image).ndim > 3

        if not self._is_pytorch_model:
            if not is_batched:
                inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            else:
                inputs = jax.tree.map(jnp.asarray, inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            if not is_batched:
                inputs = jax.tree.map(
                    lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs
                )
            else:
                inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device), inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if num_steps is not None:
            sample_kwargs["num_steps"] = int(num_steps)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if not is_batched and noise.ndim == 2:
                noise = noise[None, ...]
            sample_kwargs["noise"] = noise

        if transformed_condition is not None:
            target = jnp.asarray(transformed_condition)
            if not is_batched and target.ndim == 2:
                target = target[None, ...]
            weights = jnp.asarray(condition_weights, dtype=target.dtype)
            if not is_batched and weights.ndim == 1:
                weights = weights[None, ...]
            if weights.ndim == 2:
                weights = weights[..., None]
            expected_target = (
                inputs["state"].shape[0],
                self._model.action_horizon,
                self._model.action_dim,
            )
            expected_weights = (*expected_target[:2], 1)
            if target.shape != expected_target:
                raise ValueError(
                    "transformed action_condition must have shape "
                    f"{expected_target}, got {target.shape}"
                )
            if weights.shape != expected_weights:
                raise ValueError(
                    f"condition_weights must have shape {expected_weights}, got {weights.shape}"
                )
            sample_kwargs.update(
                action_condition=target,
                condition_weights=weights,
                rtc_beta=float(rtc_beta),
            )

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
        }
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            if not is_batched:
                outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
            else:
                outputs = jax.tree.map(lambda x: np.asarray(x.detach().cpu()), outputs)
        else:
            if not is_batched:
                outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
            else:
                outputs = jax.tree.map(np.asarray, outputs)

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
