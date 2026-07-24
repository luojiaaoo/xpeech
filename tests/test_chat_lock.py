import pytest
from fastapi import HTTPException

from xpeech.agent.server.api import CHAT_GUARD, acquire_chat_session, content


def test_content_dependency_returns_parsed_list():
    parsed = content('[{"text": "hello"}]')

    assert parsed == [{"text": "hello"}]


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
