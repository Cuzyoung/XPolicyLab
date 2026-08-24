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
from openpi.models_pytorch.autohorizon_official import (
    UPSTREAM_COMMIT as AUTOHORIZON_UPSTREAM_COMMIT,
)
from openpi.models_pytorch.autohorizon_official import bidir_soft_pointer

BasePolicy: TypeAlias = _base_policy.BasePolicy


def compute_dvac_variance(clean_tail: np.ndarray, action_dim: int) -> np.ndarray:
    tail = np.asarray(clean_tail, dtype=np.float32)
    if tail.ndim != 4:
        raise ValueError(f"DVAC clean estimates must be [L, B, H, D], got {tail.shape}")
    if isinstance(action_dim, bool) or not isinstance(action_dim, int):
        raise ValueError("DVAC action_dim must be an integer")
    if not 0 < action_dim <= tail.shape[-1]:
        raise ValueError(
            f"DVAC action_dim must satisfy 1 <= D <= {tail.shape[-1]}, got {action_dim}"
        )
    valid_tail = tail[..., :action_dim]
    return np.var(valid_tail, axis=0, ddof=0).sum(axis=-1)


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
            self._sample_actions = nnx_utils.module_jit(
                model.sample_actions,
                static_argnames=(
                    "return_attention",
                    "return_denoising_variance",
                    "dvac_tail_steps",
                ),
            )
            self._sample_actions_multi = nnx_utils.module_jit(
                model.sample_actions,
                static_argnames=(
                    "num_samples",
                    "return_attention",
                    "return_denoising_variance",
                    "dvac_tail_steps",
                ),
            )
            self._rng = rng or jax.random.key(0)

    @override
    def infer(
        self,
        obs: dict,
        *,
        noise: np.ndarray | None = None,
        num_steps: int | None = None,
        num_samples: int = 1,
        action_condition: np.ndarray | None = None,
        condition_weights: np.ndarray | None = None,
        rtc_beta: float = 5.0,
        paint_action_condition: np.ndarray | None = None,
        paint_delay_steps: int | None = None,
        autohorizon: bool = False,
        dvac: bool = False,
        dvac_tail_steps: int = 5,
        dvac_action_dim: int | None = None,
    ) -> dict:  # type: ignore[misc]
        if num_steps is not None and num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {num_steps}")
        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}")
        if not math.isfinite(rtc_beta) or rtc_beta <= 0:
            raise ValueError(f"rtc_beta must be finite and positive, got {rtc_beta}")
        if (action_condition is None) != (condition_weights is None):
            raise ValueError("action_condition and condition_weights must be provided together")
        if (paint_action_condition is None) != (paint_delay_steps is None):
            raise ValueError(
                "paint_action_condition and paint_delay_steps must be provided together"
            )
        if action_condition is not None and self._is_pytorch_model:
            raise NotImplementedError("RTC action conditioning is only implemented for JAX Pi0")
        if paint_action_condition is not None and self._is_pytorch_model:
            raise NotImplementedError("PAINT is only implemented for JAX Pi0")
        if num_samples > 1 and self._is_pytorch_model:
            raise NotImplementedError("multi-sample inference is only implemented for JAX Pi0")
        if num_samples > 1 and action_condition is not None:
            raise ValueError("multi-sample inference cannot be combined with RTC conditioning")
        if num_samples > 1 and paint_action_condition is not None:
            raise ValueError("multi-sample inference cannot be combined with PAINT")
        if action_condition is not None and paint_action_condition is not None:
            raise ValueError("RTC conditioning cannot be combined with PAINT")
        if autohorizon and self._is_pytorch_model:
            raise NotImplementedError("the XPolicy AutoHorizon hook is implemented for JAX Pi0")
        if autohorizon and num_samples > 1:
            raise ValueError("AutoHorizon cannot be combined with multi-sample inference")
        if autohorizon and action_condition is not None:
            raise ValueError("AutoHorizon cannot be combined with RTC conditioning")
        if autohorizon and paint_action_condition is not None:
            raise ValueError("AutoHorizon cannot be combined with PAINT")
        if dvac and self._is_pytorch_model:
            raise NotImplementedError("the XPolicy DVAC hook is implemented for JAX Pi0")
        if dvac and num_samples > 1:
            raise ValueError("DVAC cannot be combined with multi-sample inference")
        if dvac and action_condition is not None:
            raise ValueError("DVAC cannot be combined with RTC conditioning")
        if dvac and paint_action_condition is not None:
            raise ValueError("DVAC cannot be combined with PAINT")
        if dvac and autohorizon:
            raise ValueError("DVAC cannot be combined with AutoHorizon")
        if dvac:
            if isinstance(dvac_tail_steps, bool) or not isinstance(dvac_tail_steps, int):
                raise ValueError("dvac_tail_steps must be an integer")
            if dvac_tail_steps <= 0:
                raise ValueError("dvac_tail_steps must be positive")
            effective_num_steps = int(num_steps or self._sample_kwargs.get("num_steps", 10))
            if dvac_tail_steps > effective_num_steps:
                raise ValueError(
                    f"dvac_tail_steps must be <= num_steps ({effective_num_steps})"
                )
            if dvac_action_dim is None:
                dvac_action_dim = int(self._model.action_dim)
            if (
                isinstance(dvac_action_dim, bool)
                or not isinstance(dvac_action_dim, int)
                or not 0 < dvac_action_dim <= self._model.action_dim
            ):
                raise ValueError(
                    "dvac_action_dim must satisfy "
                    f"1 <= D <= {self._model.action_dim}, got {dvac_action_dim!r}"
                )
        if paint_delay_steps is not None:
            if isinstance(paint_delay_steps, bool) or not isinstance(paint_delay_steps, int):
                raise ValueError("paint_delay_steps must be an integer")
            if not 0 < paint_delay_steps < self._model.action_horizon:
                raise ValueError(
                    "paint_delay_steps must satisfy "
                    f"0 < d < {self._model.action_horizon}, got {paint_delay_steps}"
                )

        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        raw_condition = (
            action_condition
            if action_condition is not None
            else paint_action_condition
        )
        if raw_condition is not None:
            inputs["actions"] = np.asarray(raw_condition, dtype=np.float32)
        inputs = self._input_transform(inputs)
        transformed_condition = inputs.pop("actions", None) if raw_condition is not None else None

        is_batched = False
        if "state" in inputs:
            is_batched = np.asarray(inputs["state"]).ndim > 1
        elif inputs.get("image"):
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
        if num_samples > 1:
            sample_kwargs["num_samples"] = int(num_samples)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if not is_batched and noise.ndim == 2:
                noise = noise[None, ...]
            sample_kwargs["noise"] = noise

        if transformed_condition is not None:
            target = jnp.asarray(transformed_condition)
            if not is_batched and target.ndim == 2:
                target = target[None, ...]
            expected_target = (
                inputs["state"].shape[0],
                self._model.action_horizon,
                self._model.action_dim,
            )
            if target.shape != expected_target:
                raise ValueError(
                    "transformed action condition must have shape "
                    f"{expected_target}, got {target.shape}"
                )
            if paint_action_condition is not None:
                sample_kwargs.update(
                    paint_action_condition=target,
                    paint_delay_steps=int(paint_delay_steps),
                )
            else:
                weights = jnp.asarray(condition_weights, dtype=target.dtype)
                if not is_batched and weights.ndim == 1:
                    weights = weights[None, ...]
                if weights.ndim == 2:
                    weights = weights[..., None]
                expected_weights = (*expected_target[:2], 1)
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
        sample_actions = (
            self._sample_actions_multi if num_samples > 1 else self._sample_actions
        )
        output_state = inputs["state"]
        if num_samples > 1:
            output_state = jnp.repeat(output_state, num_samples, axis=0)
        autohorizon_metadata = None
        dvac_metadata = None
        if autohorizon:
            actions, attention = sample_actions(
                sample_rng_or_pytorch_device,
                observation,
                return_attention=True,
                **sample_kwargs,
            )
            attention_tensor = torch.from_numpy(np.asarray(attention, dtype=np.float32))
            execution_steps, diagnostics = bidir_soft_pointer(attention_tensor)
            autohorizon_metadata = {
                "execution_steps": int(execution_steps.item()),
                "attention_step": 3,
                "hold_threshold": 0.3,
                "entropy_quantile": 0.9,
                "run_length": 1,
                "method": diagnostics["method"],
                "forward_horizon": int(diagnostics["N_forward"]),
                "backward_horizon": int(diagnostics["N_backward"]),
                "join_row": diagnostics["join_row"],
                "framework": "jax_attention_port",
                "upstream_commit": AUTOHORIZON_UPSTREAM_COMMIT,
            }
        elif dvac:
            actions, clean_tail = sample_actions(
                sample_rng_or_pytorch_device,
                observation,
                return_denoising_variance=True,
                dvac_tail_steps=dvac_tail_steps,
                **sample_kwargs,
            )
            clean_tail_array = np.asarray(clean_tail, dtype=np.float32)
            if clean_tail_array.shape != (
                dvac_tail_steps,
                inputs["state"].shape[0],
                self._model.action_horizon,
                self._model.action_dim,
            ):
                raise ValueError(
                    "DVAC clean estimates have unexpected shape: "
                    f"{clean_tail_array.shape}"
                )
            variance = compute_dvac_variance(clean_tail_array, dvac_action_dim)
            if variance.shape[0] != 1:
                raise ValueError("DVAC requires one observation")
            dvac_metadata = {
                "variance": variance[0].astype(np.float64).tolist(),
                "total_variance": float(np.sum(variance[0], dtype=np.float64)),
                "tail_steps": dvac_tail_steps,
                "action_dim": dvac_action_dim,
                "variance_space": "normalized_valid_action",
            }
        else:
            actions = sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs)
        outputs = {
            "state": output_state,
            "actions": actions,
        }
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            if not is_batched:
                outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
            else:
                outputs = jax.tree.map(lambda x: np.asarray(x.detach().cpu()), outputs)
        elif not is_batched and num_samples == 1:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        else:
            outputs = jax.tree.map(np.asarray, outputs)

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        if autohorizon_metadata is not None:
            outputs["autohorizon"] = autohorizon_metadata
        if dvac_metadata is not None:
            outputs["dvac"] = dvac_metadata
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
