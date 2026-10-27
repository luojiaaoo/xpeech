import pytest

from xpeech.agent.server.session_guard import SessionChatGuard


class TestSessionChatGuard:
    @pytest.mark.asyncio
    async def test_rejects_second_chat_for_same_session(self):
        guard = SessionChatGuard()

        assert await guard.try_acquire("user", "session")
        assert not await guard.try_acquire("user", "session")

        await guard.release("user", "session")
        assert await guard.try_acquire("user", "session")

    @pytest.mark.asyncio
    async def test_allows_different_sessions(self):
        guard = SessionChatGuard()

        assert await guard.try_acquire("user", "session-a")
        assert await guard.try_acquire("user", "session-b")

    @pytest.mark.asyncio
    async def test_allows_same_session_id_for_different_users(self):
        guard = SessionChatGuard()

        assert await guard.try_acquire("user-a", "session")
        assert await guard.try_acquire("user-b", "session")
