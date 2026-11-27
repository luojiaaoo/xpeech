"""PostgreSQL integration tests use a disposable schema, never application tables."""

import asyncio
import json
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from sqlalchemy.engine import make_url
from sqlmodel import create_engine


@pytest.fixture
def mock_builtin_skill_dirs(monkeypatch):
    """Stub built-in discovery while preserving real workspace scans."""
    from xpeech.agent.skills import skill

    iter_skill_files = skill._iter_skill_files

    def configure(directories):
        builtin_scan = AsyncMock(return_value=[directory / "SKILL.md" for directory in directories])

        async def scan(directory, excluded_names=()):
            if directory == skill.BUILTIN_SKILLS_DIR:
                return await builtin_scan()
            return await iter_skill_files(directory, excluded_names)

        monkeypatch.setattr(skill, "_iter_skill_files", scan)
        return builtin_scan

    return configure


@pytest.fixture
def postgres_database(monkeypatch, tmp_path):
    url = os.environ.get("XPEECH_TEST_DATABASE_URL")
    if not url:
        pytest.skip("Set XPEECH_TEST_DATABASE_URL to run PostgreSQL integration tests")

    from xpeech.agent.server.routes import statistics
    from xpeech.config.settings import settings
    from xpeech.utils import database

    schema = f"test_{uuid4().hex}"
    base_url = make_url(url).set(drivername="postgresql+psycopg")
    admin_engine = create_engine(base_url, isolation_level="AUTOCOMMIT")
    with admin_engine.connect() as connection:
        connection.exec_driver_sql(f'CREATE SCHEMA "{schema}"')
    test_url = base_url.update_query_dict({"options": f"-c timezone=UTC -c search_path={schema}"})
    monkeypatch.setattr(database, "database_url", lambda config=None: test_url)
    monkeypatch.setattr(settings.path, "session_record_yaml_path", tmp_path / "record")
    sync_engine = database.create_database_engine()
    engine = database.create_async_database_engine()
    monkeypatch.setattr(database, "_engine", sync_engine)
    monkeypatch.setattr(database, "_async_engine", engine)
    monkeypatch.setattr(statistics, "_statistics_repository", statistics.ConversationStatisticsRepository(engine))
    try:
        yield test_url
    finally:
        asyncio.run(engine.dispose())
        sync_engine.dispose()
        with admin_engine.connect() as connection:
            connection.exec_driver_sql(f'DROP SCHEMA "{schema}" CASCADE')
        admin_engine.dispose()


@pytest.fixture(autouse=True)
def fake_vllm_tokenizer(monkeypatch):
    """Keep unit tests offline while preserving payload-sensitive token counts."""
    from xpeech.utils import helper

    async def tokenize(payload):
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return {"count": max(1, len(serialized) // 8)}

    monkeypatch.setattr(helper, "_vllm_tokenize", tokenize)


class _ModelPermits:
    """Inspect the model lease set in the isolated fake Redis."""

    key = "xpeech:llm:permits:v1"

    async def count(self) -> int:
        from xpeech.utils.redis_state import client

        async with client.redis_client() as redis:
            return await redis.zcard(self.key)

    async def wait_for(self, expected: int) -> None:
        async def wait():
            while await self.count() != expected:
                await asyncio.sleep(0.01)

        await asyncio.wait_for(wait(), timeout=1)


@pytest.fixture
def model_permits():
    return _ModelPermits()


@pytest.fixture
def make_openai_sdk_client():
    """Build an SDK-shaped client with a test completion method."""
    def make_client(create=None):
        if create is None:
            create = AsyncMock(side_effect=AssertionError("Unexpected model request"))
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    return make_client


@pytest.fixture
def backend_openai_client(monkeypatch, make_openai_sdk_client):
    """Supply the SDK client for route tests that bypass the backend lifespan."""
    from xpeech.agent.server.api import app

    client = make_openai_sdk_client()
    monkeypatch.setattr(app.state, "openai_client", client, raising=False)
    return client


@pytest.fixture(autouse=True)
def isolated_redis(monkeypatch):
    """Exercise Redis commands/Lua without accessing application Redis data."""
    import fakeredis
    from fakeredis.aioredis import FakeRedis

    from xpeech.utils.redis_state import client as redis_connection

    server = fakeredis.FakeServer()

    def client(*args, **kwargs):
        return FakeRedis(server=server, decode_responses=True)

    monkeypatch.setattr(redis_connection, "redis_client", client)
    return server
