from __future__ import annotations

import asyncio
from contextlib import suppress

import client_server.ws.model_server as server_module
import pytest
from client_server.ws.protocol.exceptions import ErrorCode
from client_server.ws.protocol.messages import MessageType
from client_server.ws.protocol.schemas import Frame


def frame(kind, request_id, *, trial="trial", observation=None):
    return Frame(
        message_type=kind,
        request_id=request_id,
        evaluation_id="evaluation",
        trial_id=trial,
        payload={} if observation is None else {"observation": observation},
    )


class Model:
    def __init__(self, *, reset_fails=False):
        self.calls = []
        self.reset_fails = reset_fails

    def reset(self):
        self.calls.append(("reset", None))
        if self.reset_fails:
            raise ValueError("partial reset failure")

    def update_obs(self, observation):
        self.calls.append(("update_obs", observation))

    def get_action(self):
        self.calls.append(("get_action", None))
        return [1]


@pytest.mark.parametrize("disconnect_old_waiter", [False, True])
@pytest.mark.parametrize("reset_fails", [False, True])
def test_old_decode_cannot_mutate_model_after_reset(
    monkeypatch, disconnect_old_waiter, reset_fails
):
    async def scenario():
        decode_started, release_decode = asyncio.Event(), asyncio.Event()
        model = Model(reset_fails=reset_fails)
        server = server_module.PolicyServer(model)

        async def to_thread(function, /, *args, **kwargs):
            if function is server_module.decode_obs_images:
                if args[0] == {"which": "old"}:
                    decode_started.set()
                    await release_decode.wait()
                return args[0]
            return function(*args, **kwargs)

        monkeypatch.setattr(asyncio, "to_thread", to_thread)
        old_frame = frame(
            MessageType.INFER, "old-infer", trial="old", observation={"which": "old"}
        )
        waiter = asyncio.create_task(server.process_frame(old_frame))
        await decode_started.wait()
        execution = server._inflight["old-infer"][1]
        if disconnect_old_waiter:
            waiter.cancel()
            with suppress(asyncio.CancelledError):
                await waiter
        reset_reply = await server.process_frame(frame(MessageType.RESET, "reset", trial="new"))
        expected_reset_type = MessageType.ERROR if reset_fails else MessageType.RESET_RESULT
        assert reset_reply.message_type == expected_reset_type
        assert model.calls == [("reset", None)]
        release_decode.set()
        old_reply = await execution
        if not disconnect_old_waiter:
            assert await waiter is old_reply
        assert old_reply.message_type == MessageType.ERROR
        assert old_reply.payload["code"] == ErrorCode.INFER_FAILED.value
        assert "newer model reset" in old_reply.payload["message"]
        assert model.calls == [("reset", None)]
        # Duplicate retry preserves exactly-once replay of the stale error.
        assert await server.process_frame(old_frame) is old_reply
        if not reset_fails:
            new_reply = await server.process_frame(frame(
                MessageType.INFER, "new-infer", trial="new", observation={"which": "new"}
            ))
            assert new_reply.message_type == MessageType.INFER_RESULT
            assert model.calls == [
                ("reset", None), ("update_obs", {"which": "new"}), ("get_action", None)
            ]

    asyncio.run(scenario())


def test_connection_captures_generation_before_response_task_runs(monkeypatch):
    async def scenario():
        model = Model()
        server = server_module.PolicyServer(model)
        release_infer = asyncio.Event()
        original_process = server.process_frame
        old_frame = frame(MessageType.INFER, "old-infer", observation={"which": "old"})
        replies = []

        async def delayed_process(request, **kwargs):
            if request.message_type == MessageType.INFER:
                await release_infer.wait()
            return await original_process(request, **kwargs)

        async def to_thread(function, /, *args, **kwargs):
            if function is server_module.decode_obs_images:
                return args[0]
            return function(*args, **kwargs)

        monkeypatch.setattr(server, "process_frame", delayed_process)
        monkeypatch.setattr(asyncio, "to_thread", to_thread)
        monkeypatch.setattr(server_module, "decode_envelope", lambda raw: old_frame)
        monkeypatch.setattr(server_module, "encode_frame", lambda reply: reply)

        class Connection:
            async def __aiter__(self):
                yield b"old-infer"
                # The old respond task has been queued, but is held before
                # process_frame. A RESET from a new connection overtakes it.
                await original_process(frame(MessageType.RESET, "new-reset", trial="new"))
                release_infer.set()

            async def send(self, reply):
                replies.append(reply)

        await server._handle_connection(Connection())
        assert len(replies) == 1
        assert replies[0].message_type == MessageType.ERROR
        assert "newer model reset" in replies[0].payload["message"]
        assert model.calls == [("reset", None)]

    asyncio.run(scenario())


def test_reset_waits_for_inference_already_using_the_model(monkeypatch):
    async def scenario():
        action_started, release_action = asyncio.Event(), asyncio.Event()
        model = Model()

        async def get_action():
            action_started.set()
            await release_action.wait()
            model.calls.append(("get_action", None))
            return [1]

        async def to_thread(function, /, *args, **kwargs):
            if function is server_module.decode_obs_images:
                return args[0]
            return function(*args, **kwargs)

        model.get_action = get_action
        monkeypatch.setattr(asyncio, "to_thread", to_thread)
        server = server_module.PolicyServer(model)
        infer = asyncio.create_task(server.process_frame(frame(
            MessageType.INFER, "infer", observation={"which": "old"}
        )))
        await action_started.wait()
        reset = asyncio.create_task(server.process_frame(frame(MessageType.RESET, "reset")))
        await asyncio.sleep(0)
        assert ("reset", None) not in model.calls
        release_action.set()
        assert (await infer).message_type == MessageType.INFER_RESULT
        assert (await reset).message_type == MessageType.RESET_RESULT
        assert model.calls == [
            ("update_obs", {"which": "old"}), ("get_action", None), ("reset", None)
        ]

    asyncio.run(scenario())


@pytest.mark.parametrize("opt_in", [False, True])
def test_disconnect_invalidates_queued_decode_only_for_opted_in_connections(monkeypatch, opt_in):
    async def scenario():
        decode_started, release_decode, closed = asyncio.Event(), asyncio.Event(), asyncio.Event()
        model = Model()
        server = server_module.PolicyServer(model)
        hello = frame(MessageType.HELLO, "hello")
        hello.payload["cancel_pending_on_disconnect"] = opt_in
        infer = frame(MessageType.INFER, "infer", observation={"which": "old"})
        replies = []

        async def to_thread(function, /, *args, **kwargs):
            if function is server_module.decode_obs_images:
                decode_started.set()
                await release_decode.wait()
                return args[0]
            return function(*args, **kwargs)

        monkeypatch.setattr(asyncio, "to_thread", to_thread)
        monkeypatch.setattr(server_module, "_git_revision", lambda: None)
        monkeypatch.setattr(server_module, "_MAX_INFLIGHT_RESPONSES", 0)
        monkeypatch.setattr(
            server_module, "decode_envelope", lambda raw: hello if raw == b"h" else infer
        )
        monkeypatch.setattr(server_module, "encode_frame", lambda reply: reply)

        class Connection:
            async def __aiter__(self):
                yield b"h"
                yield b"i"

            async def send(self, reply):
                replies.append(reply)

            async def wait_closed(self):
                await closed.wait()

        handler = asyncio.create_task(server._handle_connection(Connection()))
        await decode_started.wait()
        closed.set()
        await asyncio.sleep(0)
        release_decode.set()
        await handler
        infer_reply = next(reply for reply in replies if reply.request_id == "infer")
        if opt_in:
            assert infer_reply.message_type == MessageType.ERROR
            assert "client disconnect" in infer_reply.payload["message"]
            assert model.calls == []
        else:
            assert infer_reply.message_type == MessageType.INFER_RESULT
            assert model.calls == [("update_obs", {"which": "old"}), ("get_action", None)]
        # Cancelled request IDs replay an error, never a successful late action.
        assert await server.process_frame(infer) is infer_reply

    asyncio.run(scenario())


def test_disconnect_does_not_interrupt_running_model_but_discards_its_output(monkeypatch):
    async def scenario():
        started, release = asyncio.Event(), asyncio.Event()
        model = Model()
        server = server_module.PolicyServer(model)
        connection = server_module._InferenceConnection(cancel_pending_on_disconnect=True)

        async def get_action():
            started.set()
            await release.wait()
            model.calls.append(("get_action", None))
            return [1]

        async def to_thread(function, /, *args, **kwargs):
            if function is server_module.decode_obs_images:
                return args[0]
            return function(*args, **kwargs)

        model.get_action = get_action
        monkeypatch.setattr(asyncio, "to_thread", to_thread)
        request = frame(MessageType.INFER, "infer", observation={"which": "old"})
        pending = asyncio.create_task(server.process_frame(
            request, inference_connection=connection,
        ))
        await started.wait()
        connection.disconnected = True
        assert not pending.done()
        release.set()
        reply = await pending
        assert model.calls == [("update_obs", {"which": "old"}), ("get_action", None)]
        assert reply.message_type == MessageType.ERROR
        assert "client disconnect" in reply.payload["message"]
        assert await server.process_frame(request) is reply

    asyncio.run(scenario())
