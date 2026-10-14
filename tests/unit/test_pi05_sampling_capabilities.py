from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from client_server.ws.model_server import PolicyServer
from client_server.ws.protocol.messages import MessageType
from client_server.ws.protocol.schemas import Frame
from XPolicyLab.policy.Pi_05.model import Model


@pytest.mark.parametrize(
    ("is_pytorch", "num_steps", "expected"),
    [
        (True, 10, ["default"]),
        (False, 1, ["default", "rtc", "aac", "paint"]),
        (False, 2, ["default", "rtc", "aac", "paint"]),
        (False, 3, ["default", "rtc", "aac", "paint", "autohorizon"]),
        (False, 10, ["default", "rtc", "aac", "paint", "autohorizon"]),
    ],
)
def test_hello_reports_pi05_backend_and_step_capabilities(
    is_pytorch, num_steps, expected, monkeypatch,
):
    # Exercise the real server handshake without loading model weights.
    model = Model.__new__(Model)
    model.policy = SimpleNamespace(_is_pytorch_model=is_pytorch)
    model.num_steps = num_steps
    monkeypatch.setattr(model, "runtime_metadata", lambda: {})

    async def to_thread(function, /, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", to_thread)
    frame = Frame(
        message_type=MessageType.HELLO,
        request_id="pi05-capabilities",
        evaluation_id="test",
        payload={},
    )
    reply = asyncio.run(PolicyServer(model)._dispatch_frame(frame))
    assert reply.payload["capabilities"]["sampling_modes"] == expected
