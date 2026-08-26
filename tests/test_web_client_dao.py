from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlmodel import SQLModel

from xpeech.channel.web_client.dao import (
    AuthenticationSession,
    DuplicateSessionIdError,
    ProtectedAdminDeletionError,
    ProtectedAdminIdentityError,
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
        admin = await dao.get_user_by_session_id("admin")
        assert admin is not None
        assert admin.session_id == "admin"
        assert admin.password_hash == "admin-hash"
        assert admin.is_admin is True
        assert generated_hashes == ["admin-hash"]

        with pytest.raises(ProtectedAdminIdentityError):
            await dao.update_user(
                admin.id,
                username="renamed-admin",
                session_id="renamed-admin-session",
            )

        with pytest.raises(ProtectedAdminDeletionError):
            await dao.delete_user(admin.id)

        user = await dao.create_user(
            username="Alice",
            session_id="alice-session",
            password_hash="alice-hash",
            is_admin=False,
        )
        assert user.id is not None
        assert (await dao.get_user_by_session_id("alice-session")) == user

        await dao.create_session(
            token_hash="valid-token",
            user_id=user.id,
            expires_at=datetime.now(UTC) + timedelta(hours=1),
        )
        await dao.create_session(
            token_hash="other-token",
            user_id=user.id,
            expires_at=datetime.now(UTC) + timedelta(hours=1),
        )
        session_user = await dao.get_user_for_session("valid-token")
        assert session_user == user

        assert await dao.change_password(
            user.id,
            password_hash="changed-hash",
            keep_token_hash="valid-token",
        ) is True
        assert (await dao.get_user_for_session("valid-token")).password_hash == "changed-hash"
        assert await dao.get_user_for_session("other-token") is None

        updated = await dao.update_user(
            user.id,
            username="AliceUpdated",
            session_id="alice-updated-session",
            password_hash="new-hash",
            is_admin=True,
            is_active=False,
        )
        assert updated is not None
        assert updated.username == "AliceUpdated"
        assert updated.session_id == "alice-updated-session"
        assert updated.password_hash == "new-hash"
        assert updated.is_admin is True
        assert updated.is_active is False
        assert await dao.get_user_for_session("valid-token") is None

        duplicate_name_user = await dao.create_user(
            username="AliceUpdated",
            session_id="another-session",
            password_hash="duplicate-hash",
            is_admin=False,
        )
        assert duplicate_name_user.username == updated.username

        await dao.create_session(
            token_hash="deleted-user-token",
            user_id=duplicate_name_user.id,
            expires_at=datetime.now(UTC) + timedelta(hours=1),
        )
        assert await dao.delete_user(duplicate_name_user.id) is True
        assert await dao.get_user_by_session_id("another-session") is None
        assert await dao.get_user_for_session("deleted-user-token") is None
        assert await dao.delete_user(duplicate_name_user.id) is False

        with pytest.raises(DuplicateSessionIdError):
            await dao.create_user(
                username="Bob",
                session_id="alice-updated-session",
                password_hash="duplicate-hash",
                is_admin=False,
            )

        await dao.initialize(default_admin_hash)
        assert generated_hashes == ["admin-hash"]
        assert [current.username for current in await dao.list_users()] == [
            "admin",
            "AliceUpdated",
        ]
    finally:
        await dao.close()

    assert database_path.is_file()
    assert User.metadata is not SQLModel.metadata
    assert AuthenticationSession.metadata is User.metadata
