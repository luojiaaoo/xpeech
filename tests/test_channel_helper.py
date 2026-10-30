import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from xpeech.subapp.channel import helper
from xpeech.subapp.channel.schema import ChatEventType, Message, TextData
from xpeech.utils.channel_jwt import ChannelJWTAuth

TEST_AUTH = ChannelJWTAuth(
    "test",
    "test-channel-key-that-is-at-least-32-bytes",
)


@pytest.mark.asyncio
@pytest.mark.parametrize("session_id", [None, "018f0000-0000-7000-8000-000000000001"])
async def test_chat_and_question_forward_session_header(monkeypatch, session_id):
    requests = []
    real_client = httpx.AsyncClient

    def respond(request):
        requests.append(request)
        return httpx.Response(200, content=b"ok")

    monkeypatch.setattr(helper.httpx, "AsyncClient", lambda **kwargs: real_client(
        transport=httpx.MockTransport(respond), **kwargs,
    ))
    stream = await helper.open_chat_stream(
        "alice", "Alice", "[]", "{}", "http://backend.test/chat", TEST_AUTH, session_id=session_id,
    )
    await stream.aclose()
    await helper.submit_question("alice", "yes", "http://backend.test", TEST_AUTH, session_id=session_id)
    assert [request.url.path for request in requests] == ["/chat", "/answer_question"]
    for request in requests:
        assert request.headers.get("x-session-id") == session_id
        assert request.headers["x-user-id"] == "alice"


class _JsonErrorStream(httpx.AsyncByteStream):
    async def __aiter__(self):
        yield b'{"detail":"Session already has an active chat request"}'


@pytest.mark.asyncio
async def test_iter_chat_events_reads_streaming_error_response(monkeypatch: pytest.MonkeyPatch):
    request = httpx.Request("POST", "http://backend.test/chat")
    response = httpx.Response(409, request=request, stream=_JsonErrorStream())
    client = SimpleNamespace(aclose=AsyncMock())
    open_chat_stream = AsyncMock(return_value=helper.ChatStream(response=response, _client=client))
    monkeypatch.setattr(helper, "open_chat_stream", open_chat_stream)
    message = Message(
        message_id="om_message",
        chat_id="oc_chat",
        user_id="E1001",
        sender_name="Alice",
        content=[TextData(text="hello")],
        timestamp=0,
        session_metadata={},
    )

    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        _ = [
            event
            async for event in helper.iter_chat_events(
                [message],
                "http://backend.test/chat",
                TEST_AUTH,
            )
        ]

    assert exc_info.value.response.json() == {"detail": "Session already has an active chat request"}
    assert response.is_closed
    client.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_iter_chat_events_groups_text_chunks_until_end_event(monkeypatch: pytest.MonkeyPatch):
    class FakeEventSource:
        def __init__(self, _response):
            pass

        async def aiter_sse(self):
            for payload in [
                {"event": "thinking", "context": "first "},
                {"event": "thinking", "context": "second"},
                {"event": "thinking_end", "context": ""},
                {"event": "assistant", "context": "answer"},
                {"event": "assistant_end", "context": ""},
                {"event": "command", "context": "/new"},
            ]:
                yield SimpleNamespace(data=json.dumps(payload))

    request = httpx.Request("POST", "http://backend.test/chat")
    response = httpx.Response(200, request=request)
    client = SimpleNamespace(aclose=AsyncMock())
    monkeypatch.setattr(helper, "open_chat_stream", AsyncMock(return_value=helper.ChatStream(response, client)))
    monkeypatch.setattr(helper, "EventSource", FakeEventSource)
    message = Message(
        message_id="om_message",
        chat_id="oc_chat",
        user_id="E1001",
        sender_name="Alice",
        content=[TextData(text="hello")],
        timestamp=0,
        session_metadata={},
    )

    events = helper.iter_chat_events(
        [message],
        "http://backend.test/chat",
        TEST_AUTH,
    )
    thinking = await anext(events)
    assert thinking.event == ChatEventType.THINKING
    assert not isinstance(thinking.context, str)
    assert [chunk async for chunk in thinking.context] == ["first ", "second"]

    assistant = await anext(events)
    assert assistant.event == ChatEventType.ASSISTANT
    assert not isinstance(assistant.context, str)
    assert [chunk async for chunk in assistant.context] == ["answer"]

    command = await anext(events)
    assert command.event == ChatEventType.COMMAND
    assert command.context == "/new"
    await events.aclose()
