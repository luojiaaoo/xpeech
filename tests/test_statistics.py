import asyncio
from datetime import UTC, datetime
from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from xpeech.agent.record import (
    ConversationRecord,
    SqliteConversationRecordRepository,
    create_db_and_tables,
)
from xpeech.agent.server.api import app
from xpeech.agent.server.routes.statistics import (
    STATISTICS_CACHE_TTL_S,
    SqliteConversationStatisticsRepository,
    get_statistics_repository,
)
from xpeech.utils.jwt_auth import create_access_token


async def _seed_statistics_database(database_path: Path) -> AsyncEngine:
    engine = create_async_engine(f"sqlite+aiosqlite:///{database_path.as_posix()}")
    await create_db_and_tables(engine)
    repository = SqliteConversationRecordRepository(engine)
    records = [
        ConversationRecord(
            session_id="session-alice",
            sender_name="alice",
            user_question="first question",
            model_response="first answer",
            input_tokens=10,
            output_tokens=2,
            model_call_count=1,
            created_at=datetime(2026, 8, 18, 15, 59, tzinfo=UTC),
            duration_s=1.25,
        ),
        ConversationRecord(
            session_id="session-bob",
            sender_name="alice",
            user_question="second question",
            model_response="second answer",
            input_tokens=20,
            output_tokens=4,
            model_call_count=2,
            created_at=datetime(2026, 8, 18, 16, 1, tzinfo=UTC),
            duration_s=2.5,
        ),
        ConversationRecord(
            session_id="session-alice",
            sender_name="alice",
            user_question="latest question",
            model_response="latest answer",
            input_tokens=30,
            output_tokens=6,
            model_call_count=3,
            created_at=datetime(2026, 8, 19, 1, 2, tzinfo=UTC),
            duration_s=3.75,
        ),
    ]
    for record in records:
        await repository.append(record)
    await engine.dispose()
    return engine


def _clear_statistics_caches() -> None:
    for method_name in (
        "overview",
        "timeseries",
        "users",
        "sessions",
        "latest_records",
        "latest_data_at",
        "records",
    ):
        getattr(SqliteConversationStatisticsRepository, method_name).cache_clear()


def test_statistics_api_exposes_dashboard_aggregates_and_complete_latest_records(tmp_path: Path):
    engine = asyncio.run(_seed_statistics_database(tmp_path / "statistics.db"))
    repository = SqliteConversationStatisticsRepository(engine)
    _clear_statistics_caches()
    app.dependency_overrides[get_statistics_repository] = lambda: repository
    token = create_access_token()
    headers = {"Authorization": f"Bearer {token}"}

    try:
        with TestClient(app) as client:
            unauthorized = client.get("/statistics/records/latest")
            overview = client.get("/statistics", headers=headers)
            cached_overview = client.get("/statistics", headers=headers)
            timeseries = client.get(
                "/statistics/timeseries",
                params={"granularity": "day", "timezone": "Asia/Shanghai"},
                headers=headers,
            )
            users = client.get("/statistics/users", headers=headers)
            sessions = client.get("/statistics/sessions", headers=headers)
            latest = client.get(
                "/statistics/records/latest",
                params={"limit": 2},
                headers=headers,
            )
            no_updates = client.get(
                "/statistics/updates",
                params={"data_as_of": "2026-08-19T01:02:00Z"},
                headers=headers,
            )
            has_updates = client.get(
                "/statistics/updates",
                params={"data_as_of": "2026-08-18T16:01:00Z"},
                headers=headers,
            )
            incremental = client.get(
                "/statistics/records",
                params={"after_id": 1},
                headers=headers,
            )
            searched = client.get(
                "/statistics/records",
                params=[
                    ("session_id", "session-alice"),
                    ("session_id", "session-bob"),
                    ("keyword", "second"),
                    ("limit", "10"),
                ],
                headers=headers,
            )
        cache_info = SqliteConversationStatisticsRepository.overview.cache_info()
    finally:
        app.dependency_overrides.pop(get_statistics_repository, None)
        _clear_statistics_caches()
        asyncio.run(engine.dispose())

    assert unauthorized.status_code == 401
    assert overview.status_code == 200
    assert cached_overview.json() == overview.json()
    assert STATISTICS_CACHE_TTL_S == 5
    assert cache_info.hits == 1
    assert cache_info.misses == 1
    assert overview.json() == {
        "question_count": 3,
        "active_user_count": 2,
        "session_count": 2,
        "model_call_count": 6,
        "input_tokens": 60,
        "output_tokens": 12,
        "total_tokens": 72,
        "average_tokens_per_question": 24.0,
        "average_duration_s": 2.5,
        "data_as_of": "2026-08-19T01:02:00Z",
    }
    assert [point["bucket"] for point in timeseries.json()["data"]] == ["2026-08-18", "2026-08-19"]
    assert [user["sender_name"] for user in users.json()["data"]] == ["alice", "alice"]
    assert [user["session_id"] for user in users.json()["data"]] == [
        "session-alice",
        "session-bob",
    ]
    assert users.json()["total"] == 2
    assert users.json()["data"][0]["active_day_count"] == 2
    assert [session["session_id"] for session in sessions.json()["data"]] == [
        "session-alice",
        "session-bob",
    ]

    latest_body = latest.json()
    assert latest.status_code == 200
    assert latest.headers["cache-control"] == "no-store"
    assert latest_body["latest_id"] == 3
    assert [record["id"] for record in latest_body["data"]] == [3, 2]
    assert latest_body["data"][0]["user_question"] == "latest question"
    assert latest_body["data"][0]["model_response"] == "latest answer"
    assert latest_body["data"][0]["duration_s"] == 3.75
    assert latest_body["data"][0]["total_tokens"] == 36

    assert no_updates.json() == {
        "has_updates": False,
        "data_as_of": "2026-08-19T01:02:00Z",
    }
    assert no_updates.headers["cache-control"] == "no-store"
    assert has_updates.json() == {
        "has_updates": True,
        "data_as_of": "2026-08-19T01:02:00Z",
    }

    incremental_body = incremental.json()
    assert incremental.status_code == 200
    assert incremental_body["total"] == 2
    assert incremental_body["total_tokens"] == 60
    assert incremental_body["latest_id"] == 3
    assert [record["id"] for record in incremental_body["data"]] == [2, 3]

    searched_body = searched.json()
    assert searched.status_code == 200
    assert searched_body["total"] == 1
    assert searched_body["input_tokens"] == 20
    assert searched_body["output_tokens"] == 4
    assert searched_body["total_tokens"] == 24
    assert [record["id"] for record in searched_body["data"]] == [2]


def test_statistics_api_rejects_invalid_filters(tmp_path: Path):
    engine = asyncio.run(_seed_statistics_database(tmp_path / "invalid-filters.db"))
    repository = SqliteConversationStatisticsRepository(engine)
    app.dependency_overrides[get_statistics_repository] = lambda: repository
    headers = {"Authorization": f"Bearer {create_access_token()}"}

    try:
        with TestClient(app) as client:
            invalid_period = client.get(
                "/statistics",
                params={"start_at": "2026-08-20T00:00:00Z", "end_at": "2026-08-19T00:00:00Z"},
                headers=headers,
            )
            invalid_timezone = client.get(
                "/statistics/timeseries",
                params={"timezone": "Mars/Olympus"},
                headers=headers,
            )
    finally:
        app.dependency_overrides.pop(get_statistics_repository, None)
        _clear_statistics_caches()
        asyncio.run(engine.dispose())

    assert invalid_period.status_code == 422
    assert invalid_period.json() == {"detail": "start_at must be earlier than end_at"}
    assert invalid_timezone.status_code == 422
    assert invalid_timezone.json() == {"detail": "Unknown timezone: Mars/Olympus"}
