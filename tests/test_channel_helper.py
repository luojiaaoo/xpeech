import json
from io import BytesIO
from threading import get_ident
from unittest.mock import AsyncMock
from urllib.parse import parse_qs

import httpx
import pytest

from xpeech.config.settings import (
    ClaimExtractionError,
    ExtraUserInfoConfig,
    JWTChannelConfig,
    extract_string_claim,
    settings,
)
from xpeech.subapp.channel import helper
from xpeech.subapp.channel.feishu.backend import iter_chat_events
from xpeech.subapp.channel.schema import ChatEventType, Message, TextData
from xpeech.utils.channel_jwt import UserJWTAuth

TEST_CHANNEL = "test"
TEST_SECRET_KEY = "test-channel-key-that-is-at-least-32-bytes"
SESSION_ID = "018f0000-0000-7000-8000-000000000001"


@pytest.mark.asyncio
async def test_multipart_upload_reads_files_off_the_event_loop(monkeypatch):
    monkeypatch.setitem(settings.jwt.channels, TEST_CHANNEL, JWTChannelConfig(secret_key=TEST_SECRET_KEY))
    event_loop_thread = get_ident()
    reads = []

    class File(BytesIO):
        def read(self, *args):
            reads.append(get_ident())
            assert reads[-1] != event_loop_thread
            return super().read(*args)

    async def respond(request):
        body = await request.aread()
        assert b'filename="report.txt"' in body
        assert b"file content" in body
        return httpx.Response(200, content=b"ok")

    client_class = httpx.AsyncClient
    monkeypatch.setattr(helper.httpx, "AsyncClient", lambda **kwargs: client_class(
        transport=httpx.MockTransport(respond), **kwargs,
    ))
    async with helper.open_chat_stream(
        helper.BackendUserParams("alice", TEST_CHANNEL), "Alice", "[]", "{}", "http://backend.test/chat",
        session_id=SESSION_ID, files=[("files", ("report.txt", File(b"file content")))],
    ) as (response, _events):
        assert response.status_code == 200
    assert reads


@pytest.mark.asyncio
async def test_backend_requests_bind_user_in_jwt_subject(monkeypatch):
    monkeypatch.setitem(settings.jwt.channels, TEST_CHANNEL, JWTChannelConfig(secret_key=TEST_SECRET_KEY))
    requests = []
    client_class = httpx.AsyncClient

    def respond(request):
        requests.append(request)
        return httpx.Response(200, content=b"ok")

    monkeypatch.setattr(helper.httpx, "AsyncClient", lambda **kwargs: client_class(
        transport=httpx.MockTransport(respond), **kwargs,
    ))
    async with helper.open_chat_stream(
        helper.BackendUserParams("alice", TEST_CHANNEL), "Alice", "[]", "{}", "http://backend.test/chat",
        session_id=SESSION_ID,
    ) as (response, _events):
        assert response.status_code == 200
    await helper.submit_question("alice", "yes", "http://backend.test", TEST_CHANNEL, session_id=SESSION_ID)
    await helper.stop_chat("alice", "http://backend.test", TEST_CHANNEL, session_id=SESSION_ID)
    await helper.fetch_file(helper.BackendUserParams("alice", TEST_CHANNEL), "report.txt", "http://backend.test")
    assert [request.url.path for request in requests] == ["/chat", "/answer_question", "/chat/stop", "/files"]
    assert requests[2].method == "POST"
    assert requests[2].content == b""
    for request in requests:
        assert "x-user-id" not in request.headers
        if request.url.path == "/files":
            assert "x-session-id" not in request.headers
        else:
            assert request.headers["x-session-id"] == SESSION_ID
        token = request.headers["authorization"].removeprefix("Bearer ")
        auth = UserJWTAuth(token)
        assert auth.channel == TEST_CHANNEL
        assert auth.user_id == "alice"
        assert set(auth.claims) == {"exp", "sub"}


class _JsonErrorStream(httpx.AsyncByteStream):
    async def __aiter__(self):
        yield b'{"detail":"Session already has an active chat request"}'


@pytest.mark.asyncio
async def test_iter_chat_events_reads_streaming_error_response(monkeypatch: pytest.MonkeyPatch):
    response = httpx.Response(409, stream=_JsonErrorStream())
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda _request: response))
    monkeypatch.setattr(helper.httpx, "AsyncClient", lambda **kwargs: client)
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
            async for event in iter_chat_events(
                [message],
                "http://backend.test/chat",
                session_id=SESSION_ID,
            )
        ]

    assert exc_info.value.response.json() == {"detail": "Session already has an active chat request"}
    assert response.is_closed
    assert client.is_closed


@pytest.mark.asyncio
async def test_sse_request_handles_multiline_events_and_split_utf8(monkeypatch):
    payload = (
        ': ping\r\n\r\n'
        'data: {"event":"user",\r\n'
        f'data: "context":"你好","turn_id":"{SESSION_ID}"}}\r\n\r\n'
        'data: invalid json\r\n\r\n'
        f'data: {{"event":"assistant_end","context":"","turn_id":"{SESSION_ID}"}}\r\n\r\n'
    ).encode()

    class ByteStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            for byte in payload:
                yield bytes([byte])

    response = httpx.Response(200, headers={"Content-Type": "text/event-stream"}, stream=ByteStream())

    def respond(request):
        assert request.headers["accept"] == "text/event-stream"
        return response

    client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    monkeypatch.setattr(helper.httpx, "AsyncClient", lambda **kwargs: client)
    async with helper.sse_request("GET", "http://backend.test/events", headers={}) as (upstream, events):
        assert upstream is response
        assert [event async for event in events] == [
            {"event": "user", "context": "你好", "turn_id": SESSION_ID},
            {"event": "assistant_end", "context": "", "turn_id": SESSION_ID},
        ]
    assert response.is_closed
    assert client.is_closed


@pytest.mark.asyncio
async def test_iter_chat_events_groups_text_chunks_until_end_event(monkeypatch: pytest.MonkeyPatch):
    lookup = AsyncMock(return_value={"department": "server-team"})
    monkeypatch.setattr(ExtraUserInfoConfig, "resolve_session_metadata", lookup)
    payloads = [
        {"event": "user", "context": "hello"},
        {"event": "thinking", "context": "first "},
        {"event": "thinking", "context": "second"},
        {"event": "thinking_end", "context": ""},
        {"event": "assistant", "context": "answer"},
        {"event": "assistant_end", "context": ""},
        {"event": "command", "context": "/new"},
        {"event": "command", "context": "这条消息将用于引导后续回复"},
    ]
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(
            200, headers={"Content-Type": "text/event-stream"},
            content="".join(f"data: {json.dumps({**payload, 'turn_id': SESSION_ID})}\n\n" for payload in payloads),
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    monkeypatch.setattr(helper.httpx, "AsyncClient", lambda **kwargs: client)
    message = Message(
        message_id="om_message",
        chat_id="oc_chat",
        user_id="E1001",
        sender_name="Alice",
        content=[TextData(text="hello")],
        timestamp=0,
        session_metadata={"department": "client-team", "trace": "original"},
    )

    events = iter_chat_events(
        [message],
        "http://backend.test/chat",
        session_id=SESSION_ID,
    )
    thinking = await anext(events)
    lookup.assert_awaited_once_with("E1001")
    [request] = requests
    metadata = parse_qs(request.content.decode())["session_metadata"][0]
    assert json.loads(metadata) == {"department": "server-team", "trace": "original"}
    assert message.session_metadata == {"department": "client-team", "trace": "original"}
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
    queued = await anext(events)
    assert queued.event == ChatEventType.COMMAND
    assert queued.context == "这条消息将用于引导后续回复"
    await events.aclose()
    assert client.is_closed


def test_extract_string_claim_resolves_nested_feishu_userinfo():
    userinfo = {
        "code": 0,
        "data": {
            "employee_no": "oauth-user-42",
            "name": "OAuth User",
        },
    }

    assert extract_string_claim(userinfo, ".data.employee_no") == "oauth-user-42"
    assert extract_string_claim(userinfo, ".data.name") == "OAuth User"
    assert extract_string_claim(
        {"data": {"items": [{"id": "oauth-user-42"}]}}, ".data.items[0].id"
    ) == "oauth-user-42"
    with pytest.raises(ClaimExtractionError):
        extract_string_claim(userinfo, "data.employee_no")  # bare dotted path is not valid jq
    with pytest.raises(ClaimExtractionError):
        extract_string_claim(userinfo, ".data.employee_no.value")  # cannot index a string
    with pytest.raises(ClaimExtractionError):
        extract_string_claim(userinfo, ".data.missing")
    with pytest.raises(ClaimExtractionError):
        extract_string_claim({"sub": 42}, ".sub")  # non-string claim value
    with pytest.raises(ClaimExtractionError):
        extract_string_claim({"tags": ["a", "b"]}, ".tags[]")  # must yield exactly one value
