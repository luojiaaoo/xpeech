from datetime import UTC, datetime
from typing import Any, Literal

from async_lru import alru_cache
from sqlalchemy import asc, desc, distinct, func
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.orm import registry
from sqlmodel import Field, SQLModel, select
from sqlmodel.ext.asyncio.session import AsyncSession

from ..config.settings import settings
from ..utils.helper import ensure_path

TABLE_NAME = "conversation_records"
STATISTICS_CACHE_TTL_S = 5
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
    session_id: str = Field(index=True)
    sender_name: str = Field(index=True)
    user_question: str
    model_response: str
    input_tokens: int
    output_tokens: int
    model_call_count: int
    duration_s: float = Field(ge=0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC), index=True)


async def create_db_and_tables(engine: AsyncEngine = record_engine) -> None:
    """创建对话记录表。"""
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


Granularity = Literal["hour", "day", "week", "month"]


class SqliteConversationStatisticsRepository:
    """直接在对话记录数据库中完成看板所需的只读聚合查询。"""

    def __init__(self, engine: AsyncEngine = record_engine) -> None:
        self._engine = engine

    @staticmethod
    def _with_filters(
        statement: Any,
        *,
        start_at: datetime | None = None,
        end_at: datetime | None = None,
        sender_name: str | None = None,
        session_id: str | None = None,
        after_id: int | None = None,
    ) -> Any:
        if start_at is not None:
            statement = statement.where(ConversationRecord.created_at >= start_at)
        if end_at is not None:
            statement = statement.where(ConversationRecord.created_at < end_at)
        if sender_name is not None:
            statement = statement.where(ConversationRecord.sender_name == sender_name)
        if session_id is not None:
            statement = statement.where(ConversationRecord.session_id == session_id)
        if after_id is not None:
            statement = statement.where(ConversationRecord.id > after_id)
        return statement

    @alru_cache(maxsize=256, ttl=STATISTICS_CACHE_TTL_S)
    async def overview(
        self,
        *,
        start_at: datetime | None = None,
        end_at: datetime | None = None,
        sender_name: str | None = None,
    ) -> dict[str, Any]:
        statement = select(
            func.count(ConversationRecord.id).label("question_count"),
            func.count(distinct(ConversationRecord.sender_name)).label("active_user_count"),
            func.count(distinct(ConversationRecord.session_id)).label("session_count"),
            func.coalesce(func.sum(ConversationRecord.model_call_count), 0).label("model_call_count"),
            func.coalesce(func.sum(ConversationRecord.input_tokens), 0).label("input_tokens"),
            func.coalesce(func.sum(ConversationRecord.output_tokens), 0).label("output_tokens"),
            func.avg(ConversationRecord.duration_s).label("average_duration_s"),
            func.max(ConversationRecord.created_at).label("data_as_of"),
        )
        statement = self._with_filters(
            statement,
            start_at=start_at,
            end_at=end_at,
            sender_name=sender_name,
        )
        async with AsyncSession(self._engine) as session:
            row = (await session.exec(statement)).one()
        return dict(row._mapping)

    @alru_cache(maxsize=256, ttl=STATISTICS_CACHE_TTL_S)
    async def timeseries(
        self,
        *,
        granularity: Granularity,
        timezone_offset_minutes: int,
        start_at: datetime | None = None,
        end_at: datetime | None = None,
        sender_name: str | None = None,
    ) -> list[dict[str, Any]]:
        formats: dict[Granularity, str] = {
            "hour": "%Y-%m-%dT%H:00:00",
            "day": "%Y-%m-%d",
            "week": "%Y-W%W",
            "month": "%Y-%m",
        }
        modifier = f"{timezone_offset_minutes:+d} minutes"
        bucket = func.strftime(
            formats[granularity],
            ConversationRecord.created_at,
            modifier,
        ).label("bucket")
        statement = (
            select(
                bucket,
                func.count(ConversationRecord.id).label("question_count"),
                func.count(distinct(ConversationRecord.sender_name)).label("active_user_count"),
                func.count(distinct(ConversationRecord.session_id)).label("session_count"),
                func.coalesce(func.sum(ConversationRecord.model_call_count), 0).label("model_call_count"),
                func.coalesce(func.sum(ConversationRecord.input_tokens), 0).label("input_tokens"),
                func.coalesce(func.sum(ConversationRecord.output_tokens), 0).label("output_tokens"),
                func.avg(ConversationRecord.duration_s).label("average_duration_s"),
            )
            .where(ConversationRecord.created_at.is_not(None))
            .group_by(bucket)
            .order_by(asc(bucket))
        )
        statement = self._with_filters(
            statement,
            start_at=start_at,
            end_at=end_at,
            sender_name=sender_name,
        )
        async with AsyncSession(self._engine) as session:
            rows = (await session.exec(statement)).all()
        return [dict(row._mapping) for row in rows]

    @alru_cache(maxsize=256, ttl=STATISTICS_CACHE_TTL_S)
    async def users(
        self,
        *,
        timezone_offset_minutes: int,
        start_at: datetime | None = None,
        end_at: datetime | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        question_count = func.count(ConversationRecord.id).label("question_count")
        local_day = func.date(
            ConversationRecord.created_at,
            f"{timezone_offset_minutes:+d} minutes",
        )
        statement = (
            select(
                ConversationRecord.sender_name,
                question_count,
                func.count(distinct(local_day)).label("active_day_count"),
                func.count(distinct(ConversationRecord.session_id)).label("session_count"),
                func.coalesce(func.sum(ConversationRecord.model_call_count), 0).label("model_call_count"),
                func.coalesce(func.sum(ConversationRecord.input_tokens), 0).label("input_tokens"),
                func.coalesce(func.sum(ConversationRecord.output_tokens), 0).label("output_tokens"),
                func.avg(ConversationRecord.duration_s).label("average_duration_s"),
                func.max(ConversationRecord.created_at).label("last_active_at"),
            )
            .group_by(ConversationRecord.sender_name)
            .order_by(desc(question_count), asc(ConversationRecord.sender_name))
            .offset(offset)
            .limit(limit)
        )
        statement = self._with_filters(statement, start_at=start_at, end_at=end_at)
        total_statement = self._with_filters(
            select(func.count(distinct(ConversationRecord.sender_name))),
            start_at=start_at,
            end_at=end_at,
        )
        async with AsyncSession(self._engine) as session:
            rows = (await session.exec(statement)).all()
            total = (await session.exec(total_statement)).one()
        return [dict(row._mapping) for row in rows], int(total)

    @alru_cache(maxsize=256, ttl=STATISTICS_CACHE_TTL_S)
    async def sessions(
        self,
        *,
        start_at: datetime | None = None,
        end_at: datetime | None = None,
        sender_name: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        question_count = func.count(ConversationRecord.id).label("question_count")
        statement = (
            select(
                ConversationRecord.session_id,
                ConversationRecord.sender_name,
                question_count,
                func.coalesce(func.sum(ConversationRecord.model_call_count), 0).label("model_call_count"),
                func.coalesce(func.sum(ConversationRecord.input_tokens), 0).label("input_tokens"),
                func.coalesce(func.sum(ConversationRecord.output_tokens), 0).label("output_tokens"),
                func.avg(ConversationRecord.duration_s).label("average_duration_s"),
                func.min(ConversationRecord.created_at).label("first_active_at"),
                func.max(ConversationRecord.created_at).label("last_active_at"),
            )
            .group_by(ConversationRecord.session_id, ConversationRecord.sender_name)
            .order_by(desc(func.max(ConversationRecord.id)))
            .offset(offset)
            .limit(limit)
        )
        statement = self._with_filters(
            statement,
            start_at=start_at,
            end_at=end_at,
            sender_name=sender_name,
        )
        filtered_groups = self._with_filters(
            select(ConversationRecord.session_id, ConversationRecord.sender_name),
            start_at=start_at,
            end_at=end_at,
            sender_name=sender_name,
        ).group_by(ConversationRecord.session_id, ConversationRecord.sender_name)
        total_statement = select(func.count()).select_from(filtered_groups.subquery())
        async with AsyncSession(self._engine) as session:
            rows = (await session.exec(statement)).all()
            total = (await session.exec(total_statement)).one()
        return [dict(row._mapping) for row in rows], int(total)

    @alru_cache(maxsize=256, ttl=STATISTICS_CACHE_TTL_S)
    async def latest_records(self, *, limit: int = 20) -> list[ConversationRecord]:
        statement = select(ConversationRecord).order_by(desc(ConversationRecord.id)).limit(limit)
        async with AsyncSession(self._engine) as session:
            return list((await session.exec(statement)).all())

    @alru_cache(maxsize=256, ttl=STATISTICS_CACHE_TTL_S)
    async def records(
        self,
        *,
        start_at: datetime | None = None,
        end_at: datetime | None = None,
        sender_name: str | None = None,
        session_id: str | None = None,
        after_id: int | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[list[ConversationRecord], int]:
        statement = select(ConversationRecord)
        statement = self._with_filters(
            statement,
            start_at=start_at,
            end_at=end_at,
            sender_name=sender_name,
            session_id=session_id,
            after_id=after_id,
        )
        order = asc(ConversationRecord.id) if after_id is not None else desc(ConversationRecord.id)
        statement = statement.order_by(order).offset(offset).limit(limit)

        total_statement = select(func.count(ConversationRecord.id))
        total_statement = self._with_filters(
            total_statement,
            start_at=start_at,
            end_at=end_at,
            sender_name=sender_name,
            session_id=session_id,
            after_id=after_id,
        )
        async with AsyncSession(self._engine) as session:
            records = list((await session.exec(statement)).all())
            total = (await session.exec(total_statement)).one()
        return records, int(total)
