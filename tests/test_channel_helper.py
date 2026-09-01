import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from xpeech.channel import helper
from xpeech.channel.schema import ChatEventType, Message, TextData


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
        session_id="E1001",
        sender_name="Alice",
        content=[TextData(text="hello")],
        timestamp=0,
        session_metadata={},
    )

    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        _ = [event async for event in helper.iter_chat_events([message], "http://backend.test/chat")]

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
        session_id="E1001",
        sender_name="Alice",
        content=[TextData(text="hello")],
        timestamp=0,
        session_metadata={},
    )

    events = helper.iter_chat_events([message], "http://backend.test/chat")
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
