"""PostgreSQL integration tests use a disposable schema, never application tables."""

import asyncio
import os
from uuid import uuid4

import pytest
from sqlalchemy.engine import make_url
from sqlmodel import create_engine


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
