import pytest
from fastapi import HTTPException

from xpeech.agent.server.api import app
from xpeech.agent.server.dependencies import CHAT_GUARD, acquire_chat_session, content, sender_name_header


def test_content_dependency_returns_parsed_list():
    parsed = content('[{"text": "hello"}]')

    assert parsed == [{"text": "hello"}]


def test_sender_name_header_is_required():
    parameters = app.openapi()["paths"]["/chat"]["post"]["parameters"]
    sender_name = next(parameter for parameter in parameters if parameter["name"] == "sender-name")

    assert sender_name["in"] == "header"
    assert sender_name["required"] is True


def test_sender_name_header_decodes_and_rejects_blank_values():
    assert sender_name_header("demo-user") == "demo-user"
    assert sender_name_header("%E5%BC%A0%E4%B8%89") == "张三"

    with pytest.raises(HTTPException) as exc_info:
        sender_name_header("%20")

    assert exc_info.value.status_code == 422


@pytest.mark.asyncio
async def test_chat_returns_conflict_for_active_session():
    assert await CHAT_GUARD.try_acquire("busy-session")
    try:
        lease = acquire_chat_session("busy-session")
        with pytest.raises(HTTPException) as exc_info:
            await anext(lease)
    finally:
        await CHAT_GUARD.release("busy-session")

    assert exc_info.value.status_code == 409


@pytest.mark.asyncio
async def test_dependency_release_clears_active_session():
    lease = acquire_chat_session("released-session")
    assert await anext(lease) is None
    assert await CHAT_GUARD.is_active("released-session")

    with pytest.raises(StopAsyncIteration):
        await anext(lease)

    assert not await CHAT_GUARD.is_active("released-session")
