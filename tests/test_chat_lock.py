import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from xpeech.agent.loop import AgentLoop, QuestionEvent
from xpeech.agent.server.api import app
from xpeech.agent.server.auth import AuthenticatedChannel
from xpeech.agent.server.dependencies import (
    CHAT_GUARD,
    acquire_chat_session,
    content,
    sender_name_header,
    session_id_header,
    settings,
)
from xpeech.agent.server.routes.chat import answer_question
from xpeech.utils.channel_jwt import ChannelJWTAuth, create_channel_access_token

UUID7_SESSION_ID = "018f0000-0000-7000-8000-000000000001"


def test_content_dependency_returns_parsed_list():
    parsed = content('[{"text": "hello"}]')

    assert parsed == [{"text": "hello"}]


def test_sender_name_header_is_required():
    parameters = app.openapi()["paths"]["/chat"]["post"]["parameters"]
    sender_name = next(parameter for parameter in parameters if parameter["name"] == "x-sender-name")

    assert sender_name["in"] == "header"
    assert sender_name["required"] is True


def test_session_id_header_is_optional_without_exposing_default_value():
    for path in ("/chat", "/chat_sync", "/answer_question"):
        parameters = app.openapi()["paths"][path]["post"]["parameters"]
        session_id = next(parameter for parameter in parameters if parameter["name"] == "x-session-id")
        assert session_id["in"] == "header"
        assert session_id["required"] is False
        assert "default" not in session_id["schema"]


def test_only_builtin_main_session_channels_may_omit_session_id():
    assert session_id_header(AuthenticatedChannel("web_client", []), None) == "default"
    assert session_id_header(AuthenticatedChannel("feishu", []), None) == "default"

    with pytest.raises(HTTPException) as error:
        session_id_header(AuthenticatedChannel("crm", []), None)
    assert error.value.status_code == 422
    assert error.value.detail == "x-session-id is required for this channel"

    assert session_id_header(AuthenticatedChannel("crm", []), UUID7_SESSION_ID) == UUID7_SESSION_ID


@pytest.mark.parametrize(
    "session_id",
    [None, UUID7_SESSION_ID],
)
@pytest.mark.parametrize("json_output", [None, False, True])
def test_chat_initializes_all_tools_for_every_session(
    tmp_path,
    monkeypatch,
    session_id,
    json_output,
):
    from xpeech.agent import execute as execute_module

    messages = []

    async def run(*, message):
        messages.append(message)
        yield {"event": "assistant", "context": "ok"}

    create = AsyncMock(return_value=SimpleNamespace(workspace=tmp_path, run=run))
    monkeypatch.setattr(execute_module.AgentRunner, "create", create)
    monkeypatch.setattr(settings.llm, "support_json_output", True)
    token = create_channel_access_token(ChannelJWTAuth("web_client", settings.jwt.web_client_secret_key))
    headers = {
        "Authorization": f"Bearer {token}",
        "x-user-id": "user",
        "x-sender-name": "Alice",
    }
    if session_id is not None:
        headers["x-session-id"] = session_id

    data = {"content": '[{"text":"hello"}]'}
    if json_output is not None:
        data["json_output"] = str(json_output).lower()
    response = TestClient(app).post(
        "/chat",
        headers=headers,
        data=data,
    )

    assert response.status_code == 200
    create.assert_awaited_once_with("user", not_interactive=False, channel_tags=[])
    assert messages[0].session_id == (session_id or "default")
    assert messages[0].json_output is (json_output is True)


@pytest.mark.parametrize(
    "session_id",
    ["default", "chat-1", "01234567-89ab-4def-8123-456789abcdef"],
)
def test_chat_rejects_non_uuidv7_session_ids(session_id):
    token = create_channel_access_token(ChannelJWTAuth("web_client", settings.jwt.web_client_secret_key))
    response = TestClient(app).post(
        "/chat",
        headers={
            "Authorization": f"Bearer {token}",
            "x-user-id": "user",
            "x-session-id": session_id,
            "x-sender-name": "Alice",
        },
        data={"content": '[{"text":"hello"}]'},
    )

    assert response.status_code == 422
    assert response.json() == {"detail": "Invalid session_id"}


def test_unsupported_json_output_returns_http_error_before_starting_stream(monkeypatch):
    from xpeech.agent import execute as execute_module

    monkeypatch.setattr(settings.llm, "support_json_output", False)
    create = AsyncMock()
    monkeypatch.setattr(execute_module.AgentRunner, "create", create)
    token = create_channel_access_token(ChannelJWTAuth("web_client", settings.jwt.web_client_secret_key))
    response = TestClient(app).post(
        "/chat",
        headers={
            "Authorization": f"Bearer {token}",
            "x-user-id": "user",
            "x-sender-name": "Alice",
        },
        data={"content": '[{"text":"hello"}]', "json_output": "true"},
    )

    assert response.status_code == 400
    assert response.headers["content-type"].startswith("application/json")
    assert response.json() == {"detail": "The configured model does not support JSON output"}
    create.assert_not_awaited()


def test_sender_name_header_decodes_and_rejects_blank_values():
    assert sender_name_header("demo-user") == "demo-user"
    assert sender_name_header("%E5%BC%A0%E4%B8%89") == "张三"

    with pytest.raises(HTTPException) as exc_info:
        sender_name_header("%20")

    assert exc_info.value.status_code == 422


@pytest.mark.asyncio
async def test_chat_returns_conflict_for_active_session():
    assert await CHAT_GUARD.try_acquire("user", "busy-session")
    try:
        lease = acquire_chat_session("user", "busy-session")
        with pytest.raises(HTTPException) as exc_info:
            await anext(lease)
    finally:
        await CHAT_GUARD.release("user", "busy-session")

    assert exc_info.value.status_code == 409


@pytest.mark.asyncio
async def test_dependency_release_clears_active_session():
    lease = acquire_chat_session("user", "released-session")
    assert await anext(lease) is None
    assert await CHAT_GUARD.is_active("user", "released-session")

    with pytest.raises(StopAsyncIteration):
        await anext(lease)

    assert not await CHAT_GUARD.is_active("user", "released-session")


@pytest.mark.asyncio
async def test_answer_wakes_only_the_matching_user_session():
    first = QuestionEvent(event=asyncio.Event())
    second = QuestionEvent(event=asyncio.Event())
    AgentLoop.SESSION_QUESTION_EVENT[("user", "chat-1")] = first
    AgentLoop.SESSION_QUESTION_EVENT[("user", "chat-2")] = second
    try:
        response = await answer_question("user", "chat-1", "yes")
    finally:
        AgentLoop.SESSION_QUESTION_EVENT.clear()

    assert response == {"message": "Answer received"}
    assert first.answer == "yes"
    assert first.event.is_set()
    assert not second.event.is_set()
