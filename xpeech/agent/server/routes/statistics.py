from datetime import UTC, datetime
from typing import Annotated, Any, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from async_lru import alru_cache
from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from pydantic import BaseModel
from sqlalchemy import asc, desc, distinct, func
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from ...record import ConversationRecord, record_engine

STATISTICS_CACHE_TTL_S = 5

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
        session_ids: tuple[str, ...] = (),
        question_keyword: str | None = None,
        after_id: int | None = None,
    ) -> Any:
        if start_at is not None:
            statement = statement.where(ConversationRecord.created_at >= start_at)
        if end_at is not None:
            statement = statement.where(ConversationRecord.created_at < end_at)
        if sender_name is not None:
            statement = statement.where(ConversationRecord.sender_name == sender_name)
        if session_ids:
            statement = statement.where(ConversationRecord.session_id.in_(session_ids))
        if question_keyword:
            statement = statement.where(
                ConversationRecord.user_question.contains(question_keyword, autoescape=True)
            )
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
        users = self._with_filters(
            select(ConversationRecord.sender_name, ConversationRecord.session_id),
            start_at=start_at,
            end_at=end_at,
            sender_name=sender_name,
        ).group_by(ConversationRecord.sender_name, ConversationRecord.session_id)
        active_user_count = select(func.count()).select_from(users.subquery()).scalar_subquery()
        statement = select(
            func.count(ConversationRecord.id).label("question_count"),
            active_user_count.label("active_user_count"),
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
        per_user_statement = (
            select(
                bucket,
                ConversationRecord.sender_name,
                ConversationRecord.session_id,
                func.count(ConversationRecord.id).label("question_count"),
                func.coalesce(func.sum(ConversationRecord.model_call_count), 0).label("model_call_count"),
                func.coalesce(func.sum(ConversationRecord.input_tokens), 0).label("input_tokens"),
                func.coalesce(func.sum(ConversationRecord.output_tokens), 0).label("output_tokens"),
                func.sum(ConversationRecord.duration_s).label("duration_sum"),
                func.count(ConversationRecord.duration_s).label("duration_count"),
            )
            .where(ConversationRecord.created_at.is_not(None))
            .group_by(bucket, ConversationRecord.sender_name, ConversationRecord.session_id)
        )
        per_user_statement = self._with_filters(
            per_user_statement,
            start_at=start_at,
            end_at=end_at,
            sender_name=sender_name,
        )
        per_user = per_user_statement.subquery()
        statement = (
            select(
                per_user.c.bucket,
                func.sum(per_user.c.question_count).label("question_count"),
                func.count().label("active_user_count"),
                func.count(distinct(per_user.c.session_id)).label("session_count"),
                func.sum(per_user.c.model_call_count).label("model_call_count"),
                func.sum(per_user.c.input_tokens).label("input_tokens"),
                func.sum(per_user.c.output_tokens).label("output_tokens"),
                (
                    func.sum(per_user.c.duration_sum)
                    / func.nullif(func.sum(per_user.c.duration_count), 0)
                ).label("average_duration_s"),
            )
            .group_by(per_user.c.bucket)
            .order_by(asc(per_user.c.bucket))
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
                ConversationRecord.session_id,
                question_count,
                func.count(distinct(local_day)).label("active_day_count"),
                func.count(distinct(ConversationRecord.session_id)).label("session_count"),
                func.coalesce(func.sum(ConversationRecord.model_call_count), 0).label("model_call_count"),
                func.coalesce(func.sum(ConversationRecord.input_tokens), 0).label("input_tokens"),
                func.coalesce(func.sum(ConversationRecord.output_tokens), 0).label("output_tokens"),
                func.avg(ConversationRecord.duration_s).label("average_duration_s"),
                func.max(ConversationRecord.created_at).label("last_active_at"),
            )
            .group_by(ConversationRecord.sender_name, ConversationRecord.session_id)
            .order_by(
                desc(question_count),
                asc(ConversationRecord.sender_name),
                asc(ConversationRecord.session_id),
            )
            .offset(offset)
            .limit(limit)
        )
        statement = self._with_filters(statement, start_at=start_at, end_at=end_at)
        filtered_users = self._with_filters(
            select(ConversationRecord.sender_name, ConversationRecord.session_id),
            start_at=start_at,
            end_at=end_at,
        ).group_by(ConversationRecord.sender_name, ConversationRecord.session_id)
        total_statement = select(func.count()).select_from(filtered_users.subquery())
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
    async def latest_data_at(
        self,
        *,
        start_at: datetime | None = None,
        end_at: datetime | None = None,
    ) -> datetime | None:
        statement = select(ConversationRecord.created_at).where(ConversationRecord.created_at.is_not(None))
        statement = self._with_filters(statement, start_at=start_at, end_at=end_at)
        statement = statement.order_by(desc(ConversationRecord.created_at)).limit(1)
        async with AsyncSession(self._engine) as session:
            return (await session.exec(statement)).first()

    @alru_cache(maxsize=256, ttl=STATISTICS_CACHE_TTL_S)
    async def records(
        self,
        *,
        start_at: datetime | None = None,
        end_at: datetime | None = None,
        sender_name: str | None = None,
        session_ids: tuple[str, ...] = (),
        question_keyword: str | None = None,
        after_id: int | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[list[ConversationRecord], dict[str, Any]]:
        statement = select(ConversationRecord)
        statement = self._with_filters(
            statement,
            start_at=start_at,
            end_at=end_at,
            sender_name=sender_name,
            session_ids=session_ids,
            question_keyword=question_keyword,
            after_id=after_id,
        )
        order = asc(ConversationRecord.id) if after_id is not None else desc(ConversationRecord.id)
        statement = statement.order_by(order).offset(offset).limit(limit)

        summary_statement = select(
            func.count(ConversationRecord.id).label("total"),
            func.coalesce(func.sum(ConversationRecord.input_tokens), 0).label("input_tokens"),
            func.coalesce(func.sum(ConversationRecord.output_tokens), 0).label("output_tokens"),
        )
        summary_statement = self._with_filters(
            summary_statement,
            start_at=start_at,
            end_at=end_at,
            sender_name=sender_name,
            session_ids=session_ids,
            question_keyword=question_keyword,
            after_id=after_id,
        )
        async with AsyncSession(self._engine) as session:
            records = list((await session.exec(statement)).all())
            summary = (await session.exec(summary_statement)).one()
        return records, dict(summary._mapping)


router = APIRouter(prefix="/statistics", tags=["statistics"])
_statistics_repository = SqliteConversationStatisticsRepository()


class OverviewResponse(BaseModel):
    question_count: int
    active_user_count: int
    session_count: int
    model_call_count: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    average_tokens_per_question: float
    average_duration_s: float | None
    data_as_of: datetime | None


class TimeseriesPoint(BaseModel):
    bucket: str
    question_count: int
    active_user_count: int
    session_count: int
    model_call_count: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    average_duration_s: float | None


class TimeseriesResponse(BaseModel):
    granularity: Granularity
    timezone: str
    data: list[TimeseriesPoint]


class UserStatistics(BaseModel):
    sender_name: str
    session_id: str
    question_count: int
    active_day_count: int
    session_count: int
    model_call_count: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    average_duration_s: float | None
    last_active_at: datetime | None


class UserStatisticsResponse(BaseModel):
    data: list[UserStatistics]
    total: int
    limit: int
    offset: int


class SessionStatistics(BaseModel):
    session_id: str
    sender_name: str
    question_count: int
    model_call_count: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    average_duration_s: float | None
    first_active_at: datetime | None
    last_active_at: datetime | None


class SessionStatisticsResponse(BaseModel):
    data: list[SessionStatistics]
    total: int
    limit: int
    offset: int


class ConversationRecordResponse(BaseModel):
    id: int
    created_at: datetime
    duration_s: float
    session_id: str
    sender_name: str
    user_question: str
    model_response: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    model_call_count: int


class LatestRecordsResponse(BaseModel):
    data: list[ConversationRecordResponse]
    latest_id: int | None


class UpdatesResponse(BaseModel):
    has_updates: bool
    data_as_of: datetime | None


class RecordsResponse(LatestRecordsResponse):
    total: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    limit: int
    offset: int


def get_statistics_repository() -> SqliteConversationStatisticsRepository:
    return _statistics_repository


StatisticsRepository = Annotated[
    SqliteConversationStatisticsRepository,
    Depends(get_statistics_repository),
]


def _as_database_datetime(value: datetime | None) -> datetime | None:
    """SQLite stores these UTC values without preserving timezone metadata."""
    if value is None:
        return None
    if value.tzinfo is None:
        return value
    return value.astimezone(UTC).replace(tzinfo=None)


def _as_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _period(start_at: datetime | None, end_at: datetime | None) -> tuple[datetime | None, datetime | None]:
    normalized_start = _as_database_datetime(start_at)
    normalized_end = _as_database_datetime(end_at)
    if normalized_start is not None and normalized_end is not None and normalized_start >= normalized_end:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="start_at must be earlier than end_at",
        )
    return normalized_start, normalized_end


def _timezone_offset_minutes(timezone_name: str, reference: datetime | None) -> int:
    try:
        timezone = ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Unknown timezone: {timezone_name}",
        )
    reference_utc = reference or datetime.now(UTC)
    if reference_utc.tzinfo is None:
        reference_utc = reference_utc.replace(tzinfo=UTC)
    offset = reference_utc.astimezone(timezone).utcoffset()
    return 0 if offset is None else int(offset.total_seconds() // 60)


def _rounded_duration(value: float | None) -> float | None:
    return None if value is None else round(float(value), 3)


def _record_response(record: ConversationRecord) -> ConversationRecordResponse:
    return ConversationRecordResponse(
        id=record.id,
        created_at=_as_utc(record.created_at),
        duration_s=_rounded_duration(record.duration_s),
        session_id=record.session_id,
        sender_name=record.sender_name,
        user_question=record.user_question,
        model_response=record.model_response,
        input_tokens=record.input_tokens,
        output_tokens=record.output_tokens,
        total_tokens=record.input_tokens + record.output_tokens,
        model_call_count=record.model_call_count,
    )


@router.get("", response_model=OverviewResponse)
async def overview(
    repository: StatisticsRepository,
    start_at: Annotated[datetime | None, Query(description="统计开始时间，包含该时间")] = None,
    end_at: Annotated[datetime | None, Query(description="统计结束时间，不包含该时间")] = None,
    sender_name: Annotated[str | None, Query(description="按发送者用户名筛选")] = None,
) -> OverviewResponse:
    """返回领导看板顶部使用规模与资源消耗指标。"""
    normalized_start, normalized_end = _period(start_at, end_at)
    values = await repository.overview(
        start_at=normalized_start,
        end_at=normalized_end,
        sender_name=sender_name,
    )
    question_count = int(values["question_count"])
    input_tokens = int(values["input_tokens"])
    output_tokens = int(values["output_tokens"])
    total_tokens = input_tokens + output_tokens
    return OverviewResponse(
        question_count=question_count,
        active_user_count=int(values["active_user_count"]),
        session_count=int(values["session_count"]),
        model_call_count=int(values["model_call_count"]),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        average_tokens_per_question=round(total_tokens / question_count, 2) if question_count else 0,
        average_duration_s=_rounded_duration(values["average_duration_s"]),
        data_as_of=_as_utc(values["data_as_of"]),
    )


@router.get("/timeseries", response_model=TimeseriesResponse)
async def timeseries(
    repository: StatisticsRepository,
    granularity: Annotated[Granularity, Query(description="聚合粒度")] = "day",
    timezone: Annotated[str, Query(description="分桶使用的 IANA 时区")] = "Asia/Shanghai",
    start_at: Annotated[datetime | None, Query(description="统计开始时间，包含该时间")] = None,
    end_at: Annotated[datetime | None, Query(description="统计结束时间，不包含该时间")] = None,
    sender_name: Annotated[str | None, Query(description="按发送者用户名筛选")] = None,
) -> TimeseriesResponse:
    """按小时、日、周或月返回问答量、活跃用户和 Token 趋势。"""
    normalized_start, normalized_end = _period(start_at, end_at)
    offset_minutes = _timezone_offset_minutes(timezone, start_at)
    rows = await repository.timeseries(
        granularity=granularity,
        timezone_offset_minutes=offset_minutes,
        start_at=normalized_start,
        end_at=normalized_end,
        sender_name=sender_name,
    )
    data = []
    for row in rows:
        input_tokens = int(row["input_tokens"])
        output_tokens = int(row["output_tokens"])
        data.append(
            TimeseriesPoint(
                bucket=row["bucket"],
                question_count=int(row["question_count"]),
                active_user_count=int(row["active_user_count"]),
                session_count=int(row["session_count"]),
                model_call_count=int(row["model_call_count"]),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                average_duration_s=_rounded_duration(row["average_duration_s"]),
            )
        )
    return TimeseriesResponse(granularity=granularity, timezone=timezone, data=data)


@router.get("/users", response_model=UserStatisticsResponse)
async def users(
    repository: StatisticsRepository,
    start_at: Annotated[datetime | None, Query(description="统计开始时间，包含该时间")] = None,
    end_at: Annotated[datetime | None, Query(description="统计结束时间，不包含该时间")] = None,
    timezone: Annotated[str, Query(description="活跃天数使用的 IANA 时区")] = "Asia/Shanghai",
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> UserStatisticsResponse:
    """返回按问答次数倒序排列的用户活跃度排行。"""
    normalized_start, normalized_end = _period(start_at, end_at)
    offset_minutes = _timezone_offset_minutes(timezone, start_at)
    rows, total = await repository.users(
        timezone_offset_minutes=offset_minutes,
        start_at=normalized_start,
        end_at=normalized_end,
        limit=limit,
        offset=offset,
    )
    data = []
    for row in rows:
        input_tokens = int(row["input_tokens"])
        output_tokens = int(row["output_tokens"])
        data.append(
            UserStatistics(
                sender_name=row["sender_name"],
                session_id=row["session_id"],
                question_count=int(row["question_count"]),
                active_day_count=int(row["active_day_count"]),
                session_count=int(row["session_count"]),
                model_call_count=int(row["model_call_count"]),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                average_duration_s=_rounded_duration(row["average_duration_s"]),
                last_active_at=_as_utc(row["last_active_at"]),
            )
        )
    return UserStatisticsResponse(data=data, total=total, limit=limit, offset=offset)


@router.get("/sessions", response_model=SessionStatisticsResponse)
async def sessions(
    repository: StatisticsRepository,
    start_at: Annotated[datetime | None, Query(description="统计开始时间，包含该时间")] = None,
    end_at: Annotated[datetime | None, Query(description="统计结束时间，不包含该时间")] = None,
    sender_name: Annotated[str | None, Query(description="按发送者用户名筛选")] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> SessionStatisticsResponse:
    """返回最近活跃的会话及其累计用量。"""
    normalized_start, normalized_end = _period(start_at, end_at)
    rows, total = await repository.sessions(
        start_at=normalized_start,
        end_at=normalized_end,
        sender_name=sender_name,
        limit=limit,
        offset=offset,
    )
    data = []
    for row in rows:
        input_tokens = int(row["input_tokens"])
        output_tokens = int(row["output_tokens"])
        data.append(
            SessionStatistics(
                session_id=row["session_id"],
                sender_name=row["sender_name"],
                question_count=int(row["question_count"]),
                model_call_count=int(row["model_call_count"]),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                average_duration_s=_rounded_duration(row["average_duration_s"]),
                first_active_at=_as_utc(row["first_active_at"]),
                last_active_at=_as_utc(row["last_active_at"]),
            )
        )
    return SessionStatisticsResponse(data=data, total=total, limit=limit, offset=offset)


@router.get("/records/latest", response_model=LatestRecordsResponse)
async def latest_records(
    response: Response,
    repository: StatisticsRepository,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> LatestRecordsResponse:
    """返回 ID 最大的完整问答记录，默认用于大屏滚动展示。"""
    response.headers["Cache-Control"] = "no-store"
    records = await repository.latest_records(limit=limit)
    data = [_record_response(record) for record in records]
    return LatestRecordsResponse(data=data, latest_id=data[0].id if data else None)


@router.get("/updates", response_model=UpdatesResponse)
async def updates(
    response: Response,
    repository: StatisticsRepository,
    data_as_of: Annotated[datetime | None, Query(description="面板当前数据截止时间")] = None,
    start_at: Annotated[datetime | None, Query(description="统计开始时间，包含该时间")] = None,
    end_at: Annotated[datetime | None, Query(description="统计结束时间，不包含该时间")] = None,
) -> UpdatesResponse:
    """轻量检查统计范围内是否存在比面板更新的数据。"""
    response.headers["Cache-Control"] = "no-store"
    normalized_start, normalized_end = _period(start_at, end_at)
    normalized_data_as_of = _as_database_datetime(data_as_of)
    latest = await repository.latest_data_at(start_at=normalized_start, end_at=normalized_end)
    has_updates = latest is not None and (
        normalized_data_as_of is None or latest > normalized_data_as_of
    )
    return UpdatesResponse(has_updates=has_updates, data_as_of=_as_utc(latest))


@router.get("/records", response_model=RecordsResponse)
async def records(
    response: Response,
    repository: StatisticsRepository,
    start_at: Annotated[datetime | None, Query(description="统计开始时间，包含该时间")] = None,
    end_at: Annotated[datetime | None, Query(description="统计结束时间，不包含该时间")] = None,
    sender_name: Annotated[str | None, Query(description="按发送者用户名筛选")] = None,
    session_id: Annotated[list[str] | None, Query(description="按一个或多个会话 ID 筛选")] = None,
    keyword: Annotated[str | None, Query(max_length=200, description="按问题内容关键词筛选")] = None,
    after_id: Annotated[int | None, Query(ge=0, description="只返回 ID 大于该值的新记录")] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> RecordsResponse:
    """分页查询完整问答；传 after_id 时按 ID 正序返回增量记录。"""
    response.headers["Cache-Control"] = "no-store"
    normalized_start, normalized_end = _period(start_at, end_at)
    session_ids = tuple(dict.fromkeys(value.strip() for value in (session_id or []) if value.strip()))
    normalized_keyword = keyword.strip() if keyword and keyword.strip() else None
    result, summary = await repository.records(
        start_at=normalized_start,
        end_at=normalized_end,
        sender_name=sender_name,
        session_ids=session_ids,
        question_keyword=normalized_keyword,
        after_id=after_id,
        limit=limit,
        offset=offset,
    )
    data = [_record_response(record) for record in result]
    latest_id = max((record.id for record in data), default=None)
    input_tokens = int(summary["input_tokens"])
    output_tokens = int(summary["output_tokens"])
    return RecordsResponse(
        data=data,
        latest_id=latest_id,
        total=int(summary["total"]),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        limit=limit,
        offset=offset,
    )
