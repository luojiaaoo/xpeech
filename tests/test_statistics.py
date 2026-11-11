import asyncio
import csv
import io
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest
import yaml
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncEngine

from xpeech.agent.record import (
    ConversationRecord,
    ConversationRecordRepository,
    create_db_and_tables,
)
from xpeech.agent.server.api import app
from xpeech.agent.server.routes.statistics import (
    STATISTICS_CACHE_TTL_S,
    ConversationStatisticsRepository,
    get_statistics_repository,
)
from xpeech.utils.database import get_async_database_engine
from xpeech.utils.internal_jwt import create_internal_access_token

pytestmark = pytest.mark.usefixtures("postgres_database")


def test_statistics_timezone_does_not_depend_on_system_tzdata():
    environment = {**os.environ, "PYTHONTZPATH": ""}
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from datetime import datetime; "
                "from zoneinfo import ZoneInfo; "
                "print(datetime(2026, 8, 19, tzinfo=ZoneInfo('Asia/Shanghai')).utcoffset())"
            ),
        ],
        check=True,
        capture_output=True,
        env=environment,
        text=True,
    )

    assert result.stdout.strip() == "8:00:00"


async def _seed_statistics_database() -> AsyncEngine:
    engine = get_async_database_engine()
    await create_db_and_tables(engine)
    repository = ConversationRecordRepository(engine)
    records = [
        ConversationRecord(
            request_id=str(uuid4()),
            user_id="session-alice",
            session_id="018f0000-0000-7000-8000-000000000001",
            channel="web_client",
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
            request_id=str(uuid4()),
            user_id="session-bob",
            session_id="00000000-0000-7000-8000-000000000000",
            channel="feishu",
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
            request_id=str(uuid4()),
            user_id="session-alice",
            session_id="018f0000-0000-7000-8000-000000000002",
            channel="feishu",
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
        await repository.append(record, messages=[
            {"role": "user", "content": record.user_question},
            {"role": "assistant", "content": record.model_response},
        ])
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
        "channels",
    ):
        getattr(ConversationStatisticsRepository, method_name).cache_clear()


def test_statistics_api_exposes_dashboard_aggregates_and_complete_latest_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    engine = asyncio.run(_seed_statistics_database())
    repository = ConversationStatisticsRepository(engine)
    _clear_statistics_caches()
    app.dependency_overrides[get_statistics_repository] = lambda: repository
    token = create_internal_access_token()
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
                    ("user_id", "session-alice"),
                    ("user_id", "session-bob"),
                    ("keyword", "second"),
                    ("limit", "10"),
                ],
                headers=headers,
            )
        cache_info = ConversationStatisticsRepository.overview.cache_info()
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
    assert [user["user_id"] for user in users.json()["data"]] == [
        "session-alice",
        "session-bob",
    ]
    assert users.json()["total"] == 2
    assert users.json()["data"][0]["active_day_count"] == 2
    assert [session["user_id"] for session in sessions.json()["data"]] == [
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
    assert latest_body["data"][0]["channel"] == "feishu"
    assert [record["session_id"] for record in latest_body["data"]] == [
        "018f0000-0000-7000-8000-000000000002",
        "00000000-0000-7000-8000-000000000000",
    ]

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
    engine = asyncio.run(_seed_statistics_database())
    repository = ConversationStatisticsRepository(engine)
    app.dependency_overrides[get_statistics_repository] = lambda: repository
    token = create_internal_access_token()
    headers = {"Authorization": f"Bearer {token}"}

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


def test_statistics_deduplicates_users_across_names_and_channels(tmp_path: Path):
    engine = asyncio.run(_seed_statistics_database())

    async def append_renamed_user():
        await ConversationRecordRepository(engine).append(ConversationRecord(
            request_id=str(uuid4()),
            user_id="session-alice", sender_name="renamed-alice", channel="web_client",
            session_id="018f0000-0000-7000-8000-000000000003",
            user_question="renamed user question", model_response="answer",
            input_tokens=1, output_tokens=1, model_call_count=1, duration_s=1,
            created_at=datetime(2026, 8, 19, 2, tzinfo=UTC),
        ))

    asyncio.run(append_renamed_user())
    repository = ConversationStatisticsRepository(engine)
    app.dependency_overrides[get_statistics_repository] = lambda: repository
    headers = {"Authorization": f"Bearer {create_internal_access_token()}"}
    try:
        with TestClient(app) as client:
            overview = client.get("/statistics", headers=headers).json()
            users = client.get("/statistics/users", headers=headers).json()
            recent = client.get("/statistics/sessions", headers=headers).json()
            trend = client.get("/statistics/timeseries", headers=headers).json()
            filtered = client.get("/statistics/users", params={"channel": "web_client"}, headers=headers).json()
            records = client.get("/statistics/records", params={"user_id": "session-alice"}, headers=headers).json()
    finally:
        app.dependency_overrides.pop(get_statistics_repository, None)
        _clear_statistics_caches()
        asyncio.run(engine.dispose())

    assert overview["active_user_count"] == 2
    assert "session_count" not in overview
    assert users["total"] == recent["total"] == 2
    for result in (users, recent):
        assert result["data"][0]["user_id"] == "session-alice"
        assert result["data"][0]["sender_name"] == "renamed-alice"
        assert result["data"][0]["question_count"] == 3
        assert "session_count" not in result["data"][0]
    assert [point["active_user_count"] for point in trend["data"]] == [1, 2]
    assert all("session_count" not in point for point in trend["data"])
    assert filtered["total"] == 1
    assert filtered["data"][0]["question_count"] == 2
    assert records["total"] == 3


def test_channel_queries_and_export_include_all_matching_records(tmp_path: Path):
    engine = asyncio.run(_seed_statistics_database())

    async def append_export_records():
        repository = ConversationRecordRepository(engine)
        for index in range(105):
            await repository.append(ConversationRecord(
                request_id=str(uuid4()),
                user_id="session-alice", sender_name="alice", channel="feishu",
                session_id="00000000-0000-7000-8000-000000000000",
                user_question=f'export question {index}, "quoted"\n第二行',
                model_response="=1+1", input_tokens=2, output_tokens=1,
                model_call_count=1, duration_s=1,
            ))
        await repository.append(ConversationRecord(
            request_id=str(uuid4()),
            user_id="session-alice", sender_name="alice", channel="web_client",
            session_id="00000000-0000-7000-8000-000000000000",
            user_question="export question from another channel", model_response="excluded",
            input_tokens=100, output_tokens=100, model_call_count=1, duration_s=1,
        ))

    asyncio.run(append_export_records())
    repository = ConversationStatisticsRepository(engine)
    app.dependency_overrides[get_statistics_repository] = lambda: repository
    headers = {"Authorization": f"Bearer {create_internal_access_token()}"}
    params = [("channel", "feishu"), ("user_id", "session-alice"), ("user_id", "session-bob"), ("keyword", " export question ")]
    try:
        with TestClient(app) as client:
            channels = client.get("/statistics/channels", headers=headers)
            searched = client.get("/statistics/records", params=[*params, ("limit", "10")], headers=headers)
            exported = client.get("/statistics/records/export", params=params, headers=headers)
            missing_channel = client.get("/statistics/records/export", headers=headers)
            empty_channel = client.get("/statistics/records/export", params={"channel": ""}, headers=headers)
            unauthorized = client.get("/statistics/records/export", params=params)
            empty_export = client.get("/statistics/records/export", params={"channel": "nonexistent"}, headers=headers)
    finally:
        app.dependency_overrides.pop(get_statistics_repository, None)
        _clear_statistics_caches()
        asyncio.run(engine.dispose())

    assert channels.status_code == 200
    assert {"web_client", "feishu"} <= set(channels.json())
    assert searched.status_code == 200
    assert searched.json()["total"] == 105
    assert len(searched.json()["data"]) == 10
    assert searched.json()["total_tokens"] == 315
    assert all(record["channel"] == "feishu" for record in searched.json()["data"])
    assert exported.status_code == 200
    assert exported.content.startswith(b"\xef\xbb\xbf")
    assert exported.headers["content-type"] == "text/csv; charset=utf-8"
    assert exported.headers["content-disposition"] == 'attachment; filename="conversation-records.csv"'
    assert exported.headers["cache-control"] == "no-store"
    rows = list(csv.DictReader(io.StringIO(exported.content.decode("utf-8-sig"))))
    assert len(rows) == 105
    assert {row["渠道"] for row in rows} == {"feishu"}
    assert {row["用户 ID"] for row in rows} == {"session-alice"}
    assert rows[0]["用户问题"] == 'export question 104, "quoted"\n第二行'
    assert rows[0]["模型回答"] == "'=1+1"
    assert "session_id" not in rows[0]
    assert missing_channel.status_code == empty_channel.status_code == 422
    assert unauthorized.status_code == 401
    assert len(list(csv.reader(io.StringIO(empty_export.content.decode("utf-8-sig"))))) == 1


def test_statistics_downloads_the_yaml_for_one_record(tmp_path: Path):
    engine = asyncio.run(_seed_statistics_database())
    repository = ConversationStatisticsRepository(engine)
    app.dependency_overrides[get_statistics_repository] = lambda: repository
    headers = {"Authorization": f"Bearer {create_internal_access_token()}"}
    try:
        with TestClient(app) as client:
            record = client.get("/statistics/records/latest", headers=headers).json()["data"][0]
            path = f"/statistics/records/{record['id']}/yaml"
            downloaded = client.get(path, headers=headers)
            unauthorized = client.get(path)
            missing_record = client.get("/statistics/records/999/yaml", headers=headers)
            invalid_id = client.get("/statistics/records/0/yaml", headers=headers)
            archive = tmp_path / "record" / f"{record['request_id']}.yaml"
            archive.unlink()
            missing_file = client.get(path, headers=headers)
            outside = tmp_path / "outside.yaml"
            outside.write_text("private contents")
            archive.symlink_to(outside)
            linked_file = client.get(path, headers=headers)
    finally:
        app.dependency_overrides.pop(get_statistics_repository, None)
        _clear_statistics_caches()
        asyncio.run(engine.dispose())

    assert downloaded.status_code == 200
    assert downloaded.headers["content-type"] == "application/yaml"
    assert downloaded.headers["content-disposition"] == f'attachment; filename="{record["request_id"]}.yaml"'
    assert downloaded.headers["cache-control"] == "no-store"
    assert yaml.safe_load(downloaded.content) == [
        {"role": "user", "content": "latest question"},
        {"role": "assistant", "content": "latest answer"},
    ]
    assert unauthorized.status_code == 401
    assert invalid_id.status_code == 422
    assert missing_record.status_code == missing_file.status_code == linked_file.status_code == 404
