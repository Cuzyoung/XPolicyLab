"""Synchronous environment-side adapter for the policy websocket protocol."""

from __future__ import annotations

import asyncio
import warnings
from typing import Any

from client_server.ws.protocol.client import PolicyEvalClient, PolicyEvalClientConfig


class WsModelClient:
    """Synchronous env-side client for the websocket policy protocol.

    Not thread-safe: create and use it from a single thread (the bundled
    event loop is driven with run_until_complete on that thread).

    The `step` sent on each frame counts INFERENCE calls (get_action /
    get_action_batch), not environment control steps — one inference call
    yields a whole action chunk. The server currently only echoes it back.
    """

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
        connect_timeout_s: float | None = None,
        handshake_timeout_s: float | None = None,
        request_timeout_s: float | None = None,
        max_connect_attempts: int | None = None,
        connect_retry_delay_s: float | None = None,
        client: Any | None = None,
    ):
        self.action_case_id = action_case_id
        self.trial_id = trial_id
        self.repeat_index = repeat_index
        self._step = 0
        self._loop = asyncio.new_event_loop()
        # None means "keep the PolicyEvalClientConfig default", so the defaults
        # stay defined in exactly one place.
        tuning = {
            "connect_timeout_s": connect_timeout_s,
            "handshake_timeout_s": handshake_timeout_s,
            "request_timeout_s": request_timeout_s,
            "max_connect_attempts": max_connect_attempts,
            "connect_retry_delay_s": connect_retry_delay_s,
        }
        self._client = client or PolicyEvalClient(
            PolicyEvalClientConfig(
                url=url,
                evaluation_id=evaluation_id,
                ws_ping_interval_s=ws_ping_interval_s,
                ws_ping_timeout_s=ws_ping_timeout_s,
                **{k: v for k, v in tuning.items() if v is not None},
            )
        )
        try:
            self._loop.run_until_complete(self._client.connect(handshake=True))
        except Exception:
            # Don't leak the event loop when the server is unreachable.
            self._loop.close()
            raise

    def call(self, func_name: str | None = None, obs: Any = None, **kwargs: Any) -> Any:
        # Payload channel is only `obs` (legacy TCP compatibility). Extra kwargs
        # such as env_idx_list=... are not forwarded — reject them loudly.
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            message = (
                f"WsModelClient.call() only accepts func_name and obs; "
                f"unsupported keyword argument(s): {unexpected}. "
                f"For get_action_batch, pass env indices as obs=env_idx_list."
            )
            warnings.warn(message, UserWarning, stacklevel=2)
            raise TypeError(message)

        if func_name == "prepare_case":
            if self.action_case_id is None:
                raise ValueError("prepare_case requires action_case_id")
            response = self._loop.run_until_complete(
                self._client.prepare_case(
                    self.action_case_id,
                    case_meta=self._dict_payload(func_name, obs),
                    repeat_index=self.repeat_index,
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
                    payload=self._dict_payload(func_name, obs),
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
                    repeat_index=self.repeat_index,
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
                    result=self._dict_payload(func_name, obs),
                    repeat_index=self.repeat_index,
                )
            )
            return response.payload.get("result")

        raise NotImplementedError(f"unsupported websocket model call: {func_name}")

    @staticmethod
    def _dict_payload(func_name: str, obs: Any) -> dict | None:
        # Reject loudly instead of silently dropping a mistyped payload.
        if obs is not None and not isinstance(obs, dict):
            raise TypeError(
                f"{func_name} payload must be a dict or None, "
                f"got {type(obs).__name__}"
            )
        return obs

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
