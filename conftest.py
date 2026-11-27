"""PostgreSQL integration tests use a disposable schema, never application tables."""

import asyncio
import os
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
