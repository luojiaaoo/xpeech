from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import aiofiles
import yaml
from sqlalchemy import Index, event
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.orm import registry
from sqlmodel import Field, SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from ..config.settings import settings
from ..utils.file_io import ensure_path, ensure_path_async
from ..utils.helper import LiteralDumper
from ..utils.sqlite import configure_sqlite_connection

TABLE_NAME = "conversation_records"
_record_registry = registry()
_record_database_path = settings.path.session_record_path.resolve()
ensure_path(_record_database_path.parent)
record_engine = create_async_engine(
    f"sqlite+aiosqlite:///{_record_database_path.as_posix()}",
    echo=False,
)
event.listen(record_engine.sync_engine, "connect", configure_sqlite_connection)


class _RecordSQLModel(SQLModel, registry=_record_registry):
    """为对话记录提供独立于其他 engine 的 metadata。"""


class ConversationRecord(_RecordSQLModel, table=True):
    """一次用户对话及其模型使用量，对应 conversation_records 表。"""

    __tablename__ = TABLE_NAME
    __table_args__ = (
        Index(
            "ix_conversation_records_sender_session_record",
            "sender_name",
            "user_id",
            "id",
        ),
        Index(
            "ix_conversation_records_created_sender_session",
            "created_at",
            "sender_name",
            "user_id",
        ),
    )

    id: int | None = Field(default=None, primary_key=True)
    user_id: str = Field(index=True)
    session_id: str = Field(index=True)
    request_id: str = Field(index=True)
    sender_name: str = Field(index=True)
    user_question: str
    model_response: str
    input_tokens: int
    output_tokens: int
    model_call_count: int
    duration_s: float = Field(ge=0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC), index=True)
    channel: str = Field(index=True)


async def create_db_and_tables(engine: AsyncEngine = record_engine) -> None:
    """创建对话记录表。"""
    async with engine.begin() as connection:
        await connection.run_sync(_RecordSQLModel.metadata.create_all)


def record_yaml_path(request_id: str, engine: AsyncEngine = record_engine) -> Path:
    """将服务端生成的请求 UUID 映射到记录数据库旁的独立 YAML 目录。"""
    if str(UUID(request_id)) != request_id:
        raise ValueError("Invalid request ID")
    return Path(engine.url.database).resolve().with_suffix("") / f"{request_id}.yaml"


class SqliteConversationRecordRepository:
    """向统一的 SQLite 文件追加各会话的对话记录。"""

    def __init__(self, engine: AsyncEngine = record_engine) -> None:
        self._engine = engine

    async def append(
        self, record: ConversationRecord, *, messages: list[dict[str, Any]] | None = None,
    ) -> None:
        """保存本轮新增消息的 YAML，并向统一的 SQLite 文件追加记录。"""
        if messages is not None:
            path = record_yaml_path(record.request_id, self._engine)
            await ensure_path_async(path.parent)
            serialized = yaml.dump(
                messages, Dumper=LiteralDumper, default_flow_style=False,
                allow_unicode=True, indent=4, sort_keys=False, width=1000,
            )
            async with aiofiles.open(path, "w", encoding="utf-8") as file:
                await file.write(serialized)
        async with AsyncSession(self._engine) as session:
            session.add(record)
            await session.commit()
