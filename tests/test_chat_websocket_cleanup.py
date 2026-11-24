"""Browser disconnects must close idle backend requests without another chat event."""

import asyncio
import json
from unittest.mock import AsyncMock

import anyio
import httpx
import pytest
from fastapi import FastAPI

from xpeech.subapp.channel import helper
from xpeech.subapp.channel.web_client.dao import User
from xpeech.subapp.channel.web_client.history import JsonlEventHistoryRepository
from xpeech.subapp.channel.web_client.routes import proxy

SESSION_ID = "018f0000-0000-7000-8000-000000000001"


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["opening", "idle", "keepalive", "partial", "complete", "queued"])
async def test_websocket_disconnect_closes_upstream_and_saves_partial_history(tmp_path, monkeypatch, phase):
    ready = asyncio.Event()
    closed = asyncio.Event()
    client_closed = asyncio.Event()
    messages = asyncio.Queue()
    sent = []
    history = JsonlEventHistoryRepository(tmp_path)
    maintain_history = AsyncMock(wraps=history._maintain_history_files)
    monkeypatch.setattr(history, "_maintain_history_files", maintain_history)
    user = User(user_id="alice", username="Alice", password_hash="unused")
    user_event = {"event": "user", "context": "hi", "turn_id": SESSION_ID}
    assistant_event = {"event": "assistant", "context": "hello", "turn_id": SESSION_ID}
    ack = {"event": "command", "context": "这条消息将用于引导后续回复", "turn_id": SESSION_ID}

    def sse(event):
        return f"data: {json.dumps(event)}\n\n".encode()

    class Upstream(httpx.AsyncByteStream):
        async def __aiter__(self):
            if phase != "idle":
                yield sse(user_event)
            if phase == "keepalive":
                yield b": ping\n\n"
            elif phase in {"partial", "complete"}:
                yield sse(assistant_event)
            elif phase == "queued":
                yield sse(ack)
            ready.set()
            if phase not in {"complete", "queued"}:
                await asyncio.Event().wait()

        async def aclose(self):
            # A cancelled AnyIO scope must not interrupt socket cleanup.
            await anyio.lowlevel.checkpoint()
            closed.set()

    class BackendTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            if phase == "opening":
                ready.set()
                await asyncio.Event().wait()
            return httpx.Response(200, headers={"Content-Type": "text/event-stream"}, stream=Upstream())

        async def aclose(self):
            await anyio.lowlevel.checkpoint()
            client_closed.set()

    client_class = httpx.AsyncClient
    monkeypatch.setattr(helper.httpx, "AsyncClient", lambda **kwargs: client_class(
        transport=BackendTransport(), **kwargs,
    ))

    async def current_user(**kwargs):
        return user

    app = FastAPI()
    app.include_router(proxy.create_proxy_router(
        "http://backend.test", current_user, current_user, history, "cookie",
    ))
    await messages.put({"type": "websocket.connect"})
    await messages.put({"type": "websocket.receive", "text": json.dumps({
        "type": "start", "session_id": SESSION_ID, "content": '[{"text":"hi"}]',
    })})
    scope = {
        "type": "websocket", "asgi": {"version": "3.0"}, "scheme": "ws",
        "path": "/api/chat/ws", "raw_path": b"/api/chat/ws", "query_string": b"",
        "root_path": "", "headers": [], "subprotocols": [],
        "client": ("127.0.0.1", 12345), "server": ("test", 80),
    }

    async def send(message):
        sent.append(message)

    expected = [] if phase in {"opening", "idle"} else [user_event]
    if phase in {"partial", "complete"}:
        expected.append(assistant_event)
    elif phase == "queued":
        expected.append(ack)

    task = asyncio.create_task(app(scope, messages.get, send))
    try:
        await asyncio.wait_for(ready.wait(), timeout=1)
        if phase in {"partial", "idle", "keepalive"}:
            assert await history.load("alice", SESSION_ID, channel="web_client") == expected
        if phase not in {"complete", "queued"}:
            await messages.put({"type": "websocket.disconnect", "code": 1000})
        await asyncio.wait_for(asyncio.shield(task), timeout=1)
        assert client_closed.is_set()
        if phase == "complete":
            maintain_history.assert_awaited_once_with(user_id="alice", protected_session_id=SESSION_ID)
        if phase == "opening":
            assert not closed.is_set()
        else:
            assert closed.is_set()
            saved = await history.load("alice", SESSION_ID, channel="web_client")
            assert saved == expected
        replies = [json.loads(message["text"]) for message in sent if message["type"] == "websocket.send"]
        assert any(reply["type"] == "done" for reply in replies) is (phase in {"complete", "queued"})
        assert [reply["event"] for reply in replies if reply["type"] == "event"] == expected
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
