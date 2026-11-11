"""Shared PostgreSQL engine configuration for synchronous and async services."""

from sqlalchemy import Engine, MetaData, text
from sqlalchemy.engine import URL
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlmodel import create_engine

from ..config.settings import DatabaseConfig, settings

# Serialize schema creation across backend replicas and the other services.
_SCHEMA_LOCK = 708248659012


def database_url(config: DatabaseConfig | None = None) -> URL:
    config = config or settings.database
    return URL.create(
        "postgresql+psycopg",
        username=config.username,
        password=config.password.get_secret_value(),
        host=config.host,
        port=config.port,
        database=config.name,
        query={
            "sslmode": config.sslmode,
            "connect_timeout": str(config.connect_timeout),
            "options": "-c timezone=UTC",
        },
    )


def _engine_options(config: DatabaseConfig) -> dict:
    return {
        "isolation_level": "READ COMMITTED",
        "pool_pre_ping": True,
        "pool_size": config.pool_size,
        "max_overflow": config.max_overflow,
    }


def create_database_engine(config: DatabaseConfig | None = None) -> Engine:
    config = config or settings.database
    return create_engine(database_url(config), **_engine_options(config))


def create_async_database_engine(config: DatabaseConfig | None = None) -> AsyncEngine:
    config = config or settings.database
    return create_async_engine(database_url(config), **_engine_options(config))


_engine = create_database_engine()
_async_engine = create_async_database_engine()


def get_database_engine() -> Engine:
    """Return this process's shared synchronous engine."""
    return _engine


def get_async_database_engine() -> AsyncEngine:
    """Return this process's shared asynchronous engine."""
    return _async_engine


def create_tables(engine: Engine, metadata: MetaData) -> None:
    with engine.begin() as connection:
        connection.execute(text("SELECT pg_advisory_xact_lock(:key)"), {"key": _SCHEMA_LOCK})
        metadata.create_all(connection)


async def create_tables_async(engine: AsyncEngine, metadata: MetaData) -> None:
    async with engine.begin() as connection:
        await connection.execute(text("SELECT pg_advisory_xact_lock(:key)"), {"key": _SCHEMA_LOCK})
        await connection.run_sync(metadata.create_all)
