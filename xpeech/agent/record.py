from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.orm import registry
from sqlmodel import Field, SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from ..config.settings import settings
from ..utils.helper import ensure_path

TABLE_NAME = "conversation_records"
_record_registry = registry()
_record_database_path = settings.path.session_record_path.expanduser().resolve()
ensure_path(_record_database_path.parent)
record_engine = create_async_engine(
    f"sqlite+aiosqlite:///{_record_database_path.as_posix()}",
    echo=False,
    connect_args={"timeout": 30},
)


class _RecordSQLModel(SQLModel, registry=_record_registry):
    """为对话记录提供独立于其他 engine 的 metadata。"""


class ConversationRecord(_RecordSQLModel, table=True):
    """一次用户对话及其模型使用量，对应 conversation_records 表。"""

    __tablename__ = TABLE_NAME

    id: int | None = Field(default=None, primary_key=True)
    session_id: str
    sender_name: str
    user_question: str
    model_response: str
    input_tokens: int
    output_tokens: int
    model_call_count: int


async def create_db_and_tables(engine: AsyncEngine = record_engine) -> None:
    async with engine.begin() as connection:
        await connection.run_sync(_RecordSQLModel.metadata.create_all)


class SqliteConversationRecordRepository:
    """向统一的 SQLite 文件追加各会话的对话记录。"""

    def __init__(self, engine: AsyncEngine = record_engine) -> None:
        self._engine = engine

    async def append(self, record: ConversationRecord) -> None:
        """异步向统一的 SQLite 文件追加一条会话记录。"""
        async with AsyncSession(self._engine) as session:
            session.add(record)
            await session.commit()
