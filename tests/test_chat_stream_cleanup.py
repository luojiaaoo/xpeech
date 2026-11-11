"""通过实际的路由、Agent 和模型提供方验证断连清理。"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock
from urllib.parse import urlencode

import anyio
import anyio.lowlevel
import httpx
import pytest
from fastapi import FastAPI

from xpeech.agent import execute
from xpeech.agent import loop as loop_module
from xpeech.agent.history import YamlHistoryRepository
from xpeech.agent.loop import AgentLoop
from xpeech.agent.runner import AgentRunner
from xpeech.agent.server.auth import require_user_jwt
from xpeech.agent.server.dependencies import CHAT_GUARD
from xpeech.agent.server.routes.chat import chat, router, stop_chat
from xpeech.agent.server.schema import InboundMessage, InputText
from xpeech.provider import helper
from xpeech.provider.litellm_provider import LiteLLMProvider
from xpeech.provider.schema import LLMParameters


@pytest.fixture
def streaming_chat(monkeypatch, tmp_path):
    state = SimpleNamespace(
        semaphore=asyncio.Semaphore(1),
        reading=asyncio.Event(),
        provider_called=asyncio.Event(),
        closed=asyncio.Event(),
        closing=asyncio.Event(),
        close_gate=None,
        block=False,
        emitted=0,
        responses=[],  # 保留响应引用，避免垃圾回收掩盖未关闭流的问题。
    )

    class SDKStream:
        def __aiter__(self):
            return self

        async def __anext__(self):
            state.reading.set()
            if state.block:
                await asyncio.Event().wait()
            state.emitted += 1
            delta = SimpleNamespace(content="hello", reasoning_content=None, tool_calls=None)
            return SimpleNamespace(choices=[SimpleNamespace(delta=delta, finish_reason=None)])

        async def aclose(self):
            # 模拟关闭 HTTP 连接时的调度点，验证清理不会被 AnyIO 取消打断。
            await anyio.lowlevel.checkpoint()
            state.closing.set()
            if state.close_gate is not None:
                await state.close_gate.wait()
            state.closed.set()

    async def fake_acompletion(**kwargs):
        return SDKStream()

    monkeypatch.setattr(helper, "acompletion", fake_acompletion)
    monkeypatch.setattr(helper, "LLM_PARALLEL_SEMAPHORE", state.semaphore)
    monkeypatch.setattr(
        loop_module,
        "build_system_prompt",
        AsyncMock(return_value={"role": "system", "content": "system"}),
    )
    monkeypatch.setattr(loop_module, "build_user_prompt", AsyncMock(return_value={"role": "user", "content": "hi"}))
    provider = LiteLLMProvider(
        api_key="test",
        api_base="https://example.test",
        default_model="test",
        parameters=LLMParameters(),
    )

    async def call_provider(**kwargs):
        response = await provider.chat(**kwargs)
        state.responses.append(response)
        state.provider_called.set()
        return response

    agent = AgentLoop.__new__(AgentLoop)
    agent.workspace = tmp_path
    agent.channel = None
    agent.user_organization = None
    agent.history = YamlHistoryRepository(tmp_path / "history")
    state.history = agent.history
    agent.compressor = SimpleNamespace(should_compress=AsyncMock(return_value=False))
    agent.tools = []
    agent.max_iterations = 2
    agent.chat = call_provider
    runner = AgentRunner("cleanup-user", config=SimpleNamespace(path=SimpleNamespace(workspace_base_path=tmp_path)))
    runner._agent_loop = agent
    message = InboundMessage(
        user_id="cleanup-user",
        session_id="00000000-0000-7000-8000-000000000000",
        sender_name="test",
        channel="test",
        session_metadata={},
        content=[InputText(text="hi")],
        files=[],
    )
    monkeypatch.setattr(execute, "_prepare_chat", AsyncMock(return_value=(runner, message)))
    # 测试直接调用路由函数/自建 ASGI 应用，不经过 ContextASGIMiddleware，这里补上请求上下文。
    monkeypatch.setattr(loop_module, "get_request_id", lambda: "test-request-id")
    state.auth = SimpleNamespace(user_id=message.user_id, channel="test")
    state.stream = chat(
        auth=state.auth,
        user_id=message.user_id,
        session_id=message.session_id,
        sender_name=message.sender_name,
        session_metadata={},
        content=message.content,
        files=[],
        json_output=False,
    )
    return state


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_consumer_exit_while_route_paused_at_yield_returns_permit(streaming_chat, cancel):
    state = streaming_chat
    received = asyncio.Event()

    async def consume():
        async for event in state.stream:
            assert event == {"event": "assistant", "context": "hello"}
            received.set()
            if cancel:
                await asyncio.Event().wait()
            break

    task = asyncio.create_task(consume())
    if cancel:
        await asyncio.wait_for(received.wait(), timeout=1)
        assert state.semaphore.locked()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        await asyncio.wait_for(task, timeout=1)

    await asyncio.wait_for(state.closed.wait(), timeout=1)
    await asyncio.wait_for(state.semaphore.acquire(), timeout=1)
    assert state.semaphore.locked()
    state.semaphore.release()
    # 中断响应后的清理不能将响应标记为正常完成。
    with pytest.raises(ValueError, match="Content is not available"):
        _ = state.responses[0].content


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["model", "next_chunk", "permit", "backpressure"])
async def test_stop_api_cancels_only_matching_stream_and_returns_permit(streaming_chat, phase):
    state = streaming_chat
    state.block = phase == "model"
    if phase == "permit":
        await state.semaphore.acquire()
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[require_user_jwt] = lambda: state.auth
    body = urlencode({"content": '[{"text":"hi"}]'}).encode()
    body_delivered = False
    blocked_send = asyncio.Event()
    drain = asyncio.Event()
    ended = asyncio.Event()

    async def receive():
        nonlocal body_delivered
        if not body_delivered:
            body_delivered = True
            return {"type": "http.request", "body": body, "more_body": False}
        await asyncio.Event().wait()

    async def send(message):
        if message["type"] == "http.response.start":
            assert message["status"] == 200
        elif message.get("body", b"").startswith(b"data:"):
            if phase == "backpressure":
                while state.emitted < 3:
                    await anyio.lowlevel.checkpoint()
                blocked_send.set()
                await drain.wait()
            elif phase == "next_chunk":
                state.reading.clear()
                state.block = True
        elif message.get("more_body") is False:
            ended.set()

    session_id = "00000000-0000-7000-8000-000000000000"
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/chat",
        "raw_path": b"/chat",
        "query_string": b"",
        "root_path": "",
        "headers": [
            (b"content-type", b"application/x-www-form-urlencoded"),
            (b"x-session-id", session_id.encode()),
            (b"x-sender-name", b"test"),
        ],
        "client": ("127.0.0.1", 12345),
        "server": ("test", 80),
    }
    task = asyncio.create_task(app(scope, receive, send))
    try:
        ready = (
            blocked_send if phase == "backpressure"
            else state.provider_called if phase == "permit"
            else state.reading
        )
        await asyncio.wait_for(ready.wait(), timeout=1)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            other_session = await client.post(
                "/chat/stop", headers={"x-session-id": "018f0000-0000-7000-8000-000000000001"},
            )
            assert other_session.json() == {"stopped": False}
            owner = state.auth
            state.auth = SimpleNamespace(user_id="other-user", channel="test")
            other_user = await client.post("/chat/stop", headers={"x-session-id": session_id})
            assert other_user.json() == {"stopped": False}
            state.auth = owner
            assert state.semaphore.locked()

            stopped = await asyncio.wait_for(
                client.post("/chat/stop", headers={"x-session-id": session_id}), timeout=2,
            )
            assert stopped.status_code == 200
            assert stopped.json() == {"stopped": True}
            saved = await state.history.load(owner.user_id, session_id, channel="test")
            assert saved[0] == {"role": "user", "content": "hi"}
            if phase == "permit":
                # 被取消的排队请求从未获得名额，不能误归还其他任务的名额。
                assert state.semaphore.locked()
                state.semaphore.release()
            else:
                await asyncio.wait_for(state.closed.wait(), timeout=1)
            await asyncio.wait_for(state.semaphore.acquire(), timeout=1)
            assert state.semaphore.locked()  # 恰好归还一次。
            state.semaphore.release()
            assert state.responses
            with pytest.raises(ValueError, match="Content is not available"):
                _ = state.responses[0].content
            again = await client.post("/chat/stop", headers={"x-session-id": session_id})
            assert again.json() == {"stopped": False}

        drain.set()
        await asyncio.wait_for(task, timeout=2)
        assert ended.is_set()
        # SSE 退出后会话锁也应释放，可以继续发送消息。
        assert not await CHAT_GUARD.is_active(owner.user_id, session_id)
    finally:
        drain.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_repeated_stop_and_cancelled_stop_request_do_not_interrupt_cleanup(streaming_chat):
    state = streaming_chat
    state.block = True
    state.close_gate = asyncio.Event()

    async def consume():
        return [event async for event in state.stream]

    consumer = asyncio.create_task(consume())
    first = None
    second = None
    try:
        await asyncio.wait_for(state.reading.wait(), timeout=1)
        first = asyncio.create_task(stop_chat(state.auth.user_id, "00000000-0000-7000-8000-000000000000"))
        await asyncio.wait_for(state.closing.wait(), timeout=1)
        # 关闭上游 HTTP 连接仍在等待时，并发名额已经可以复用。
        await asyncio.wait_for(state.semaphore.acquire(), timeout=1)
        assert state.semaphore.locked()
        state.semaphore.release()
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        second = asyncio.create_task(stop_chat(state.auth.user_id, "00000000-0000-7000-8000-000000000000"))
        await asyncio.sleep(0)
        assert not second.done()
        state.close_gate.set()
        assert await asyncio.wait_for(second, timeout=1) == {"stopped": True}
        await asyncio.wait_for(consumer, timeout=1)
        assert state.closed.is_set()
        await asyncio.wait_for(state.semaphore.acquire(), timeout=1)
        assert state.semaphore.locked()
        state.semaphore.release()
    finally:
        state.close_gate.set()
        tasks = [task for task in (first, second, consumer) if task is not None]
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("after_first_chunk", [False, True])
async def test_cancel_while_waiting_for_model_returns_permit(streaming_chat, after_first_chunk):
    state = streaming_chat
    if after_first_chunk:
        await anext(state.stream)
        state.reading.clear()
    state.block = True
    task = asyncio.create_task(anext(state.stream))
    await asyncio.wait_for(state.reading.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    await asyncio.wait_for(state.closed.wait(), timeout=1)
    await asyncio.wait_for(state.semaphore.acquire(), timeout=1)
    state.semaphore.release()


@pytest.mark.asyncio
async def test_sse_disconnect_under_backpressure_returns_permit(streaming_chat):
    state = streaming_chat
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[require_user_jwt] = lambda: state.auth
    disconnect = asyncio.Event()
    body = urlencode({"content": '[{"text":"hi"}]'}).encode()
    body_delivered = False

    async def receive():
        nonlocal body_delivered
        if not body_delivered:
            body_delivered = True
            return {"type": "http.request", "body": body, "more_body": False}
        await disconnect.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        if message["type"] == "http.response.start":
            assert message["status"] == 200
        elif message.get("body", b"").startswith(b"data:"):
            # 暂停读取，让 FastAPI 的生产任务填满有限缓冲区。
            while state.emitted < 3:
                await anyio.lowlevel.checkpoint()
            disconnect.set()
            await anyio.sleep_forever()

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/chat",
        "raw_path": b"/chat",
        "query_string": b"",
        "root_path": "",
        "headers": [
            (b"content-type", b"application/x-www-form-urlencoded"),
            (b"x-session-id", b"00000000-0000-7000-8000-000000000000"),
            (b"x-sender-name", b"test"),
        ],
        "client": ("127.0.0.1", 12345),
        "server": ("test", 80),
    }
    await asyncio.wait_for(app(scope, receive, send), timeout=3)

    assert disconnect.is_set()
    await asyncio.wait_for(state.semaphore.acquire(), timeout=1)
    assert state.semaphore.locked()
    state.semaphore.release()
    await asyncio.wait_for(state.closed.wait(), timeout=1)
