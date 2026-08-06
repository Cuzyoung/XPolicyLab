"""Synchronous environment-side adapter for the policy websocket protocol."""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any, cast

from client_server.ws.protocol.client import PolicyEvalClient, PolicyEvalClientConfig


# Keep the wire protocol backward compatible: a batch is carried as a regular
# INFER observation with a namespaced marker.  Old single-observation calls are
# unchanged, while batch-aware policy adapters can execute one real model batch.
BATCH_OBSERVATIONS_KEY = "__xpolicylab_batch_observations__"
BATCH_ENV_INDICES_KEY = "__xpolicylab_batch_env_indices__"


class WsModelClient:
    def __init__(
        self,
        *,
        url: str,
        evaluation_id: str,
        trial_id: str,
        action_case_id: str | None = None,
        repeat_index: int | None = None,
        ws_ping_interval_s: float | None = 20.0,
        ws_ping_timeout_s: float | None = 20.0,
        client: Any | None = None,
    ):
        self.action_case_id = action_case_id
        self.trial_id = trial_id
        self.repeat_index = repeat_index
        self._step = 0
        self._latest_obs: Any | None = None
        self._latest_obs_batch: list[Any] | None = None
        self._batch_calls = 0
        self._batch_metrics_every = max(
            0, int(os.environ.get("ROBODOJO_BATCH_METRICS_EVERY", "10"))
        )
        self._loop = asyncio.new_event_loop()
        self._client = client or PolicyEvalClient(
            PolicyEvalClientConfig(
                url=url,
                evaluation_id=evaluation_id,
                ws_ping_interval_s=ws_ping_interval_s,
                ws_ping_timeout_s=ws_ping_timeout_s,
            )
        )
        self._loop.run_until_complete(self._client.connect(handshake=True))

    def call(self, func_name: str | None = None, obs: Any = None, **kwargs: Any) -> Any:
        if func_name == "prepare_case":
            if self.action_case_id is None:
                raise ValueError("prepare_case requires action_case_id")
            response = self._loop.run_until_complete(
                self._client.prepare_case(
                    self.action_case_id,
                    case_meta=obs if isinstance(obs, dict) else None,
                )
            )
            return response.payload.get("result")

        if func_name == "reset":
            self._step = 0
            self._latest_obs = None
            self._latest_obs_batch = None
            response = self._loop.run_until_complete(
                self._client.reset(
                    trial_id=self.trial_id,
                    action_case_id=self.action_case_id,
                    repeat_index=self.repeat_index,
                    payload=obs if isinstance(obs, dict) else None,
                )
            )
            return response.payload.get("result")

        if func_name == "update_obs":
            self._latest_obs = obs
            return None

        if func_name == "get_action":
            observation = obs if obs is not None else self._latest_obs
            if observation is None:
                raise ValueError(
                    "get_action requires obs or a previous update_obs call"
                )
            response = self._loop.run_until_complete(
                self._client.infer(
                    cast(dict[str, Any], observation),
                    trial_id=self.trial_id,
                    action_case_id=self.action_case_id,
                    step=self._step,
                )
            )
            self._step += 1
            return response.payload.get("actions")

        if func_name == "update_obs_batch":
            self._latest_obs_batch = list(obs) if obs is not None else []
            return None

        if func_name == "get_action_batch":
            observations = self._latest_obs_batch
            if observations is None:
                raise ValueError(
                    "get_action_batch requires a previous update_obs_batch call"
                )
            if not observations:
                return []
            env_idx_list = list(obs) if obs is not None else None
            payload: dict[str, Any] = {BATCH_OBSERVATIONS_KEY: observations}
            if env_idx_list is not None:
                payload[BATCH_ENV_INDICES_KEY] = env_idx_list
            started = time.perf_counter()
            response = self._loop.run_until_complete(
                self._client.infer(
                    payload,
                    trial_id=self.trial_id,
                    action_case_id=self.action_case_id,
                    step=self._step,
                )
            )
            e2e_ms = (time.perf_counter() - started) * 1000.0
            self._step += 1
            self._batch_calls = getattr(self, "_batch_calls", 0) + 1
            actions = response.payload.get("actions")
            if not isinstance(actions, list) or len(actions) != len(observations):
                raise RuntimeError(
                    "batch policy response size mismatch: "
                    f"expected {len(observations)}, got "
                    f"{len(actions) if isinstance(actions, list) else type(actions).__name__}"
                )
            metrics_every = getattr(self, "_batch_metrics_every", 10)
            if metrics_every and self._batch_calls % metrics_every == 0:
                server_ms = float(response.payload.get("latency_ms", 0.0))
                print(
                    "[BatchV2] "
                    f"calls={self._batch_calls} size={len(observations)} "
                    f"server_ms={server_ms:.1f} e2e_ms={e2e_ms:.1f} "
                    f"transport_ms={max(0.0, e2e_ms - server_ms):.1f}"
                )
            return actions

        if func_name == "trial_end":
            response = self._loop.run_until_complete(
                self._client.trial_end(
                    trial_id=self.trial_id,
                    action_case_id=self.action_case_id,
                    result=obs if isinstance(obs, dict) else None,
                )
            )
            return response.payload.get("result")

        raise NotImplementedError(f"unsupported websocket model call: {func_name}")

    def close(self) -> None:
        if self._loop.is_closed():
            return
        try:
            self._loop.run_until_complete(self._client.close())
        finally:
            self._loop.close()

    def __enter__(self) -> WsModelClient:
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close()
