import pytest
from sqlalchemy import event
from sqlalchemy.orm import registry
from sqlmodel import Field, Session, SQLModel, create_engine, select
from sqlmodel.ext.asyncio.session import AsyncSession

from xpeech.subapp.channel.web_client.dao import create_web_client_engine
from xpeech.utils.sqlite import configure_sqlite_connection


class _EntryModel(SQLModel, registry=registry()):
    pass


class Entry(_EntryModel, table=True):
    value: int = Field(primary_key=True)


def test_sqlite_writer_can_commit_while_reader_keeps_its_snapshot(tmp_path):
    engine = create_engine(f"sqlite:///{(tmp_path / 'concurrent.db').as_posix()}")
    try:
        # Include an existing database created before WAL was enabled.
        Entry.metadata.create_all(engine)
        with Session(engine) as session:
            session.add(Entry(value=1))
            session.commit()
        engine.dispose()
        event.listen(engine, "connect", configure_sqlite_connection)
        with Session(engine) as reader, Session(engine) as writer:
            for session in (reader, writer):
                connection = session.connection()
                assert connection.exec_driver_sql("PRAGMA journal_mode").scalar_one() == "wal"
                assert connection.exec_driver_sql("PRAGMA busy_timeout").scalar_one() == 30000
                session.rollback()

            # sqlite3's legacy transaction control needs an explicit read BEGIN.
            reader.connection().exec_driver_sql("BEGIN")
            assert reader.exec(select(Entry.value)).all() == [1]
            writer.add(Entry(value=2))
            writer.commit()
            assert reader.exec(select(Entry.value)).all() == [1]
            reader.rollback()
            assert reader.exec(select(Entry.value).order_by(Entry.value)).all() == [1, 2]
    finally:
        engine.dispose()


@pytest.mark.asyncio
async def test_async_sqlite_connections_allow_concurrent_read_and_write(tmp_path):
    engine = create_web_client_engine(tmp_path / "web-client.db")
    try:
        async with engine.begin() as connection:
            await connection.run_sync(Entry.metadata.create_all)
        async with AsyncSession(engine) as session:
            session.add(Entry(value=1))
            await session.commit()

        async with AsyncSession(engine) as reader, AsyncSession(engine) as writer:
            for session in (reader, writer):
                connection = await session.connection()
                assert (await connection.exec_driver_sql("PRAGMA journal_mode")).scalar_one() == "wal"
                assert (await connection.exec_driver_sql("PRAGMA busy_timeout")).scalar_one() == 30000
                assert (await connection.exec_driver_sql("PRAGMA foreign_keys")).scalar_one() == 1
                await session.rollback()

            connection = await reader.connection()
            await connection.exec_driver_sql("BEGIN")
            assert (await reader.exec(select(Entry.value))).all() == [1]
            writer.add(Entry(value=2))
            await writer.commit()
            assert (await reader.exec(select(Entry.value))).all() == [1]
            await reader.rollback()
            assert (await reader.exec(select(Entry.value).order_by(Entry.value))).all() == [1, 2]
    finally:
        await engine.dispose()
