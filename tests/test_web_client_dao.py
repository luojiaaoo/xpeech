from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlmodel import SQLModel

from xpeech.channel.web_client.dao import (
    AuthenticationSession,
    DuplicateUsernameError,
    User,
    WebClientDAO,
)


@pytest.mark.asyncio
async def test_web_client_dao_manages_users_and_sessions(tmp_path: Path):
    database_path = tmp_path / "web-client.db"
    dao = WebClientDAO(database_path)
    generated_hashes: list[str] = []

    def default_admin_hash() -> str:
        generated_hashes.append("admin-hash")
        return "admin-hash"

    await dao.initialize(default_admin_hash)
    try:
        admin = await dao.get_user_by_username("ADMIN")
        assert admin is not None
        assert admin.password_hash == "admin-hash"
        assert admin.is_admin is True
        assert generated_hashes == ["admin-hash"]

        user = await dao.create_user(
            username="Alice",
            password_hash="alice-hash",
            is_admin=False,
        )
        assert user.id is not None
        assert (await dao.get_user_by_username("alice")) == user

        await dao.create_session(
            token_hash="valid-token",
            user_id=user.id,
            expires_at=datetime.now(UTC) + timedelta(hours=1),
        )
        session_user = await dao.get_user_for_session("valid-token")
        assert session_user == user

        updated = await dao.update_user(
            user.id,
            password_hash="new-hash",
            is_admin=True,
            is_active=False,
        )
        assert updated is not None
        assert updated.password_hash == "new-hash"
        assert updated.is_admin is True
        assert updated.is_active is False
        assert await dao.get_user_for_session("valid-token") is None

        with pytest.raises(DuplicateUsernameError):
            await dao.create_user(
                username="ALICE",
                password_hash="duplicate-hash",
                is_admin=False,
            )

        await dao.initialize(default_admin_hash)
        assert generated_hashes == ["admin-hash"]
        assert [current.username for current in await dao.list_users()] == [
            "admin",
            "Alice",
        ]
    finally:
        await dao.close()

    assert database_path.is_file()
    assert User.metadata is not SQLModel.metadata
    assert AuthenticationSession.metadata is User.metadata
