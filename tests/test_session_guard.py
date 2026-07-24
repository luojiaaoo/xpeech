import pytest

from xpeech.agent.server.session_guard import SessionChatGuard


class TestSessionChatGuard:
    @pytest.mark.asyncio
    async def test_rejects_second_chat_for_same_session(self):
        guard = SessionChatGuard()

        assert await guard.try_acquire("session")
        assert not await guard.try_acquire("session")

        await guard.release("session")
        assert await guard.try_acquire("session")

    @pytest.mark.asyncio
    async def test_allows_different_sessions(self):
        guard = SessionChatGuard()

        assert await guard.try_acquire("session-a")
        assert await guard.try_acquire("session-b")
