import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import inspect
from sqlmodel import SQLModel, select
from sqlmodel.ext.asyncio.session import AsyncSession

from xpeech.subapp.channel.web_client.dao import (
    AuthenticationSession,
    AuthenticationSessionChannel,
    DuplicateUserIdError,
    ProtectedAdminDeletionError,
    ProtectedAdminIdentityError,
    User,
    WebClientDAO,
    create_db_and_tables,
)
from xpeech.utils.database import get_async_database_engine

pytestmark = pytest.mark.usefixtures("postgres_database")


@pytest.mark.asyncio
async def test_web_client_migrates_legacy_auth_tables():
    engine = get_async_database_engine()
    dao = WebClientDAO(engine)
    await dao.initialize(lambda: "admin-hash")
    user = await dao.create_user(
        username="Alice", user_id="alice", password_hash="alice-hash", is_admin=False,
    )
    await dao.create_session(
        token_hash="legacy-token",
        user_id=user.id,
        expires_at=datetime.now(UTC) + timedelta(hours=1),
        channel="crm",
    )
    # Recreate the legacy table layout with existing rows and foreign keys.
    async with engine.begin() as connection:
        await connection.exec_driver_sql("ALTER TABLE web_client_users RENAME TO users")
        await connection.exec_driver_sql("ALTER TABLE web_client_sessions RENAME TO sessions")

    await asyncio.gather(create_db_and_tables(engine), create_db_and_tables(engine))

    async with engine.connect() as connection:
        tables = await connection.run_sync(lambda conn: set(inspect(conn).get_table_names()))
        assert tables == {"web_client_users", "web_client_sessions", "web_session_channels"}
    session_user = await dao.get_user_for_session("legacy-token")
    assert session_user is not None
    assert session_user.id == user.id
    assert session_user.user_id == "alice"
    assert session_user.channel == "crm"
    assert await dao.delete_user(user.id)
    async with AsyncSession(engine) as session:
        assert (await session.exec(select(AuthenticationSession))).first() is None
        assert (await session.exec(select(AuthenticationSessionChannel))).first() is None


@pytest.mark.asyncio
async def test_web_client_dao_manages_users_and_sessions(tmp_path: Path):
    dao = WebClientDAO(get_async_database_engine())
    generated_hashes: list[str] = []

    def default_admin_hash() -> str:
        generated_hashes.append("admin-hash")
        return "admin-hash"

    await dao.initialize(default_admin_hash)
    try:
        admin = await dao.get_user_by_user_id("admin")
        assert admin is not None
        assert admin.user_id == "admin"
        assert admin.password_hash == "admin-hash"
        assert admin.is_admin is True
        assert generated_hashes == ["admin-hash"]

        with pytest.raises(ProtectedAdminIdentityError):
            await dao.update_user(
                admin.id,
                username="renamed-admin",
                user_id="renamed-admin-session",
            )

        with pytest.raises(ProtectedAdminDeletionError):
            await dao.delete_user(admin.id)

        user = await dao.create_user(
            username="Alice",
            user_id="alice-session",
            password_hash="alice-hash",
            is_admin=False,
        )
        assert user.id is not None
        assert (await dao.get_user_by_user_id("alice-session")) == user

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
            user_id="alice-updated-session",
            password_hash="new-hash",
            is_admin=True,
            is_active=False,
        )
        assert updated is not None
        assert updated.username == "AliceUpdated"
        assert updated.user_id == "alice-updated-session"
        assert updated.password_hash == "new-hash"
        assert updated.is_admin is True
        assert updated.is_active is False
        assert await dao.get_user_for_session("valid-token") is None

        duplicate_name_user = await dao.create_user(
            username="AliceUpdated",
            user_id="another-session",
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
        assert await dao.get_user_by_user_id("another-session") is None
        assert await dao.get_user_for_session("deleted-user-token") is None
        assert await dao.delete_user(duplicate_name_user.id) is False

        with pytest.raises(DuplicateUserIdError):
            await dao.create_user(
                username="Bob",
                user_id="alice-updated-session",
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

    assert User.metadata is not SQLModel.metadata
    assert AuthenticationSession.metadata is User.metadata
