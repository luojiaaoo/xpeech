import asyncio
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from sqlalchemy import inspect
from sqlmodel import Session, select

from xpeech.config.settings import DatabaseConfig
from xpeech.utils.database import database_url, get_async_database_engine, get_database_engine


def test_engine_getters_reuse_the_same_instances():
    assert get_database_engine() is get_database_engine()
    assert get_async_database_engine() is get_async_database_engine()


def test_database_url_preserves_special_characters_and_hides_password():
    config = DatabaseConfig(username="user@name", password="p@ss:/?#word")
    url = database_url(config)
    assert url.drivername == "postgresql+psycopg"
    assert url.username == "user@name"
    assert url.password == "p@ss:/?#word"
    assert "p@ss" not in str(url)
    assert "p@ss" not in repr(config)


@pytest.mark.asyncio
async def test_shared_engines_use_test_schema(postgres_database):
    from sqlalchemy import text

    from xpeech.agent.record import ConversationRecord, ConversationRecordRepository, create_db_and_tables
    from xpeech.agent.server.routes.statistics import (
        ConversationStatisticsRepository,
        get_statistics_repository,
    )

    await create_db_and_tables(get_async_database_engine())
    await ConversationRecordRepository(get_async_database_engine()).append(ConversationRecord(
        user_id="isolated-user", session_id="session", request_id=str(uuid4()),
        sender_name="Alice", channel="test", user_question="hello", model_response="hi",
        input_tokens=1, output_tokens=1, model_call_count=1, duration_s=0.1,
    ))
    for repository in (ConversationStatisticsRepository(get_async_database_engine()), get_statistics_repository()):
        assert (await repository.overview())["question_count"] == 1
        assert (await repository.latest_records())[0].user_id == "isolated-user"

    engine = get_database_engine()
    try:
        with Session(engine) as session:
            assert session.execute(text("SELECT current_schema()")).scalar_one().startswith("test_")
            assert session.exec(select(ConversationRecord.user_id)).all() == ["isolated-user"]
    finally:
        engine.dispose()


@pytest.mark.asyncio
async def test_all_sqlmodel_tables_share_one_database(postgres_database):
    from xpeech.agent.record import ConversationRecord, ConversationRecordRepository, create_db_and_tables
    from xpeech.subapp.channel.web_client.dao import AuthenticationSession, User, WebClientDAO
    from xpeech.subapp.lark_token_manager.dao import DeviceAuthorization, TokenSession, TokenStore

    engine = get_async_database_engine()
    dao = WebClientDAO(engine=engine)
    tokens = TokenStore(get_database_engine())
    try:
        # Exercise concurrent schema initialization, as with backend replicas.
        await asyncio.gather(create_db_and_tables(engine), create_db_and_tables(engine))
        await dao.initialize(lambda: "admin-hash")
        tokens.initialize()
        tokens.ensure("employee")
        tokens.update_session("employee", {"access_expires_at": 9999999999})
        await ConversationRecordRepository(engine).append(ConversationRecord(
            user_id="employee", session_id="session", request_id=str(uuid4()),
            sender_name="Alice", channel="test", user_question="hello", model_response="hi",
            input_tokens=1, output_tokens=1, model_call_count=1, duration_s=0.1,
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
        ))
        with tokens.engine.connect() as connection:
            assert {
                "conversation_records", "users", "sessions", "lark_token_sessions",
                "lark_device_authorizations",
            } <= set(inspect(connection).get_table_names())
        with Session(tokens.engine) as session:
            record = session.exec(select(ConversationRecord)).one()
            assert record.created_at == datetime(2026, 1, 1, tzinfo=UTC)
            assert session.exec(select(User.user_id)).all() == ["admin"]
            assert session.exec(select(TokenSession.access_expires_at)).one() == 9999999999
        assert AuthenticationSession.metadata is User.metadata
        assert DeviceAuthorization.metadata is TokenSession.metadata
    finally:
        tokens.close()
        await dao.close()


def test_read_committed_readers_see_new_commits_without_blocking_writers(postgres_database):
    from xpeech.subapp.lark_token_manager.dao import TokenSession, TokenStore

    engine = get_database_engine()
    store = TokenStore(engine)
    store.initialize()
    store.ensure("employee")
    try:
        with (
            engine.connect() as reader,
            Session(reader) as session,
        ):
            assert reader.get_isolation_level() == "READ COMMITTED"
            assert session.exec(select(TokenSession.access_token)).one() is None
            store.update_session("employee", {"access_token": "updated"})
            assert session.exec(select(TokenSession.access_token)).one() == "updated"
        assert store.row("employee").access_token == "updated"
    finally:
        store.close()


@pytest.mark.asyncio
async def test_async_engines_use_read_committed(postgres_database):
    engine = get_async_database_engine()
    try:
        async with engine.connect() as connection:
            assert await connection.get_isolation_level() == "READ COMMITTED"
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_statistics_buckets_preserve_timezone_and_monday_week_numbers(postgres_database):
    from xpeech.agent.record import ConversationRecord, ConversationRecordRepository, create_db_and_tables
    from xpeech.agent.server.routes.statistics import ConversationStatisticsRepository

    engine = get_async_database_engine()
    try:
        await create_db_and_tables(engine)
        for timestamp in [datetime(2025, 12, 31, 17, tzinfo=UTC), datetime(2026, 1, 4, 17, tzinfo=UTC)]:
            await ConversationRecordRepository(engine).append(ConversationRecord(
                user_id="employee", session_id="session", request_id=str(uuid4()),
                sender_name="Alice", channel="test", user_question="hello", model_response="hi",
                input_tokens=1, output_tokens=1, model_call_count=1, duration_s=0.1, created_at=timestamp,
            ))
        repository = ConversationStatisticsRepository(engine)
        expected = {
            "hour": ["2026-01-01T01:00:00", "2026-01-05T01:00:00"],
            "day": ["2026-01-01", "2026-01-05"],
            "week": ["2026-W00", "2026-W01"],
            "month": ["2026-01"],
        }
        for granularity, buckets in expected.items():
            rows = await repository.timeseries(granularity=granularity, timezone_offset_minutes=480)
            assert [row["bucket"] for row in rows] == buckets
        rows, total = await repository.users(timezone_offset_minutes=480)
        assert total == 1
        assert rows[0]["active_day_count"] == 2
    finally:
        await engine.dispose()
