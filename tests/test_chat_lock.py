import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from xpeech.agent.loop import AgentLoop, QuestionEvent
from xpeech.agent.server.api import app
from xpeech.agent.server.dependencies import (
    CHAT_GUARD,
    acquire_chat_session,
    content,
    sender_name_header,
    settings,
)
from xpeech.agent.server.routes.chat import answer_question
from xpeech.utils.jwt_auth import create_access_token


def test_content_dependency_returns_parsed_list():
    parsed = content('[{"text": "hello"}]')

    assert parsed == [{"text": "hello"}]


def test_sender_name_header_is_required():
    parameters = app.openapi()["paths"]["/chat"]["post"]["parameters"]
    sender_name = next(parameter for parameter in parameters if parameter["name"] == "x-sender-name")

    assert sender_name["in"] == "header"
    assert sender_name["required"] is True


def test_session_id_header_defaults_for_chat_and_answer():
    for path in ("/chat", "/answer_question"):
        parameters = app.openapi()["paths"][path]["post"]["parameters"]
        session_id = next(
            parameter for parameter in parameters if parameter["name"] == "x-session-id"
        )
        assert session_id["in"] == "header"
        assert session_id["required"] is False
        assert session_id["schema"]["default"] == "default"


@pytest.mark.parametrize(
    ("session_id", "include_background_tools"),
    [(None, True), ("default", True), ("chat-1", False)],
)
@pytest.mark.parametrize("json_output", [None, False, True])
def test_chat_background_tools_are_available_only_for_default_session(
    tmp_path,
    monkeypatch,
    session_id,
    include_background_tools,
    json_output,
):
    from xpeech.agent.server.routes import chat as chat_routes

    messages = []

    async def run(*, message):
        messages.append(message)
        yield {"event": "assistant", "context": "ok"}

    create = AsyncMock(return_value=SimpleNamespace(workspace=tmp_path, run=run))
    monkeypatch.setattr(chat_routes.AgentRunner, "create", create)
    monkeypatch.setattr(settings.llm, "support_json_output", True)
    headers = {
        "Authorization": f"Bearer {create_access_token()}",
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
    create.assert_awaited_once_with("user", include_background_tools=include_background_tools)
    assert messages[0].session_id == (session_id or "default")
    assert messages[0].json_output is (json_output is True)


def test_unsupported_json_output_returns_http_error_before_starting_stream(monkeypatch):
    from xpeech.agent.server.routes import chat as chat_routes

    monkeypatch.setattr(settings.llm, "support_json_output", False)
    create = AsyncMock()
    monkeypatch.setattr(chat_routes.AgentRunner, "create", create)
    response = TestClient(app).post(
        "/chat",
        headers={
            "Authorization": f"Bearer {create_access_token()}",
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
