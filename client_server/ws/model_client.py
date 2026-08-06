"""Synchronous environment-side adapter for the policy websocket protocol."""

from __future__ import annotations

import asyncio
from typing import Any

from client_server.ws.protocol.client import PolicyEvalClient, PolicyEvalClientConfig


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
            response = self._loop.run_until_complete(
                self._client.reset(
                    trial_id=self.trial_id,
                    action_case_id=self.action_case_id,
                    repeat_index=self.repeat_index,
                    payload=obs if isinstance(obs, dict) else None,
                )
            )
            return response.payload.get("result")

        if func_name in {
            "update_obs",
            "update_obs_batch",
            "get_action",
            "get_action_batch",
        }:
            # Same semantics as the legacy TCP client: obs is shipped with
            # update_obs/update_obs_batch (obs list) and get_action_batch
            # (env_idx_list); get_action sends no obs.
            response = self._loop.run_until_complete(
                self._client.call(
                    func_name,
                    obs,
                    trial_id=self.trial_id,
                    action_case_id=self.action_case_id,
                    step=self._step,
                )
            )
            if func_name in {"get_action", "get_action_batch"}:
                self._step += 1
            return response.payload.get("result")

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
