from __future__ import annotations

import asyncio
from typing import Any

import pytest
from client_server.ws.model_server import PolicyServer
from client_server.ws.protocol.exceptions import ErrorCode, WsError
from client_server.ws.protocol.messages import MessageType
from client_server.ws.protocol.schemas import Frame


@pytest.fixture(autouse=True)
def run_thread_calls_inline(monkeypatch: pytest.MonkeyPatch) -> None:
    async def to_thread(function, /, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", to_thread)


class FakeModel:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    def update_obs(self, observation: dict[str, Any]) -> None:
        self.calls.append(("update_obs", observation))

    def get_action(self) -> list[int]:
        self.calls.append(("get_action", None))
        return [1]

    def get_action_rtc(self, sampling: dict[str, Any]) -> list[int]:
        self.calls.append(("get_action_rtc", sampling))
        return [2]


def _frame(sampling: dict[str, Any] | None = None) -> Frame:
    payload: dict[str, Any] = {"observation": {"state": [1, 2, 3]}}
    if sampling is not None:
        payload["sampling"] = sampling
    return Frame(
        message_type=MessageType.INFER,
        request_id="request-1",
        evaluation_id="evaluation-1",
        payload=payload,
    )


def test_default_infer_updates_observation_then_gets_action() -> None:
    model = FakeModel()
    reply = asyncio.run(PolicyServer(model)._handle_infer(_frame()))

    assert reply.payload["actions"] == [1]
    assert [name for name, _ in model.calls] == ["update_obs", "get_action"]


def test_rtc_infer_updates_observation_then_uses_rtc_action() -> None:
    model = FakeModel()
    sampling = {
        "mode": "rtc",
        "action_condition": [[0.0]],
        "condition_weights": [1.0],
        "beta": 5.0,
    }
    reply = asyncio.run(PolicyServer(model)._handle_infer(_frame(sampling)))

    assert reply.payload["actions"] == [2]
    assert [name for name, _ in model.calls] == ["update_obs", "get_action_rtc"]
    assert model.calls[-1][1] == sampling


def test_rtc_infer_fails_when_model_has_no_rtc_method() -> None:
    model = FakeModel()
    model.get_action_rtc = None  # type: ignore[method-assign]

    with pytest.raises(WsError) as exc_info:
        asyncio.run(PolicyServer(model)._handle_infer(_frame({"mode": "rtc"})))

    assert exc_info.value.code == ErrorCode.INFER_FAILED
    assert "does not support RTC" in exc_info.value.message
