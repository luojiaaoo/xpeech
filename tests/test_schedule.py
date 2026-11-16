"""Channel-owned scheduling: real persistence, management HTTP, and sync execution."""

import asyncio
import json
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock
from urllib.parse import parse_qs

import httpx
import pytest
import pytest_asyncio
from pydantic import ValidationError
from sqlalchemy import create_engine, inspect

from xpeech.agent.server.api import app as backend_app
from xpeech.agent.tools.schedule import client as tool_client
from xpeech.agent.tools.schedule.schema import FeishuScheduleCreateRequest
from xpeech.agent.tools.schedule.tool import (
    feishu_schedule_cancel,
    feishu_schedule_cron,
    feishu_schedule_list,
    feishu_schedule_once,
)
from xpeech.agent.tools.schedule.tool_args import (
    FeishuScheduleCancelArgs,
    FeishuScheduleArgs,
    FeishuScheduleCronArgs,
    FeishuScheduleOnceArgs,
)
from xpeech.subapp.channel.feishu import app as service
from xpeech.subapp.channel.feishu import schedule
from xpeech.utils.channel_jwt import UserJWTAuth, UserTokenParams, create_token
from xpeech.utils.internal_jwt import create_internal_access_token


def headers(user="alice"):
    return {"Authorization": f"Bearer {create_internal_access_token()}", "x-user-id": user}


def payload(**updates):
    return {
        "prompt": "task",
        "run_at": None,
        "cron": "0 9 * * *",
        "sender_name": "Alice",
        "session_metadata": {"open_id": "ou_alice"},
        **updates,
    }


@pytest_asyncio.fixture
async def running(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'jobs.db'}", connect_args={"check_same_thread": False})
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json=[{"event": "assistant", "context": "result"}])
        )
    )
    channel = SimpleNamespace(send=AsyncMock(return_value=SimpleNamespace(success=True)))
    runtime = schedule.FeishuScheduleRuntime("http://backend.test", client, channel)
    schedule.start_background_scheduler(engine, runtime=runtime)
    monkeypatch.setattr(schedule.random, "uniform", lambda *_: 0)
    try:
        yield SimpleNamespace(engine=engine, client=client, channel=channel, runtime=runtime)
    finally:
        await schedule.stop_background_scheduler()
        await client.aclose()
        engine.dispose()


@pytest_asyncio.fixture
async def api(running):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=service.app), base_url="http://feishu.test"
    ) as client:
        yield client


@pytest.mark.parametrize(
    "values",
    [{}, {"run_at": "2099-01-01", "cron": "0 9 * * *"}, {"run_at": "invalid"}, {"prompt": " ", "cron": "0 9 * * *"}],
)
def test_trigger_validation(values):
    with pytest.raises(ValidationError):
        FeishuScheduleCreateRequest(**{
            "prompt": "task", "run_at": None, "cron": None,
            "sender_name": "Alice", "session_metadata": {"open_id": "ou_alice"}, **values,
        })


@pytest.mark.asyncio
async def test_management_json_and_ownership(api):
    created = await api.post("/schedules", headers=headers(), json=payload())
    assert created.status_code == 200
    task = created.json()["task"]
    assert task["prompt"] == "task"
    assert datetime.fromisoformat(task["next_run_time"]).tzinfo is not None
    assert (await api.get("/schedules", headers=headers())).json() == {"tasks": [task]}
    assert (await api.get("/schedules", headers=headers("bob"))).json() == {"tasks": []}
    url = f"/schedules/{task['job_id']}"
    assert (await api.delete(url, headers=headers("bob"))).status_code == 404
    cancelled = await api.delete(url, headers=headers())
    assert cancelled.json() == {"job_id": task["job_id"], "cancelled": True}
    assert (await api.delete(url, headers=headers())).status_code == 404
    assert (await api.get("/schedules", headers=headers())).json() == {"tasks": []}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "auth",
    [
        {},
        {"Authorization": "Bearer invalid"},
        {"Authorization": "Bearer " + create_token(UserTokenParams("feishu", user_id="alice"))},
    ],
)
async def test_management_requires_internal_jwt(api, auth):
    for method, path in [("GET", "/schedules"), ("POST", "/schedules"), ("DELETE", "/schedules/missing")]:
        assert (await api.request(method, path, headers=auth, json=payload())).status_code == 401


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "updates,status",
    [
        ({"session_metadata": {}}, 422),
        ({"sender_name": " "}, 422),
        ({"run_at": "2000-01-01T00:00:00Z", "cron": None}, 400),
        ({"cron": "* * * * *"}, 422),
        ({"cron": "15 9 * * *"}, 422),
        ({"cron": "*/30 * * * *"}, 422),
        ({"cron": "invalid"}, 422),
        ({"run_at": "2099-01-01"}, 422),
        ({"channel": "web_client"}, 422),
    ],
)
async def test_management_validation(api, updates, status):
    assert (await api.post("/schedules", headers=headers(), json=payload(**updates))).status_code == status
    assert (await api.get("/schedules", headers=headers())).json() == {"tasks": []}


@pytest.mark.asyncio
async def test_concurrent_creation_enforces_limit(api):
    results = await asyncio.gather(*(api.post("/schedules", headers=headers(), json=payload()) for _ in range(8)))
    assert sorted(r.status_code for r in results) == [200] * 4 + [400] * 4
    assert (await api.post("/schedules", headers=headers("bob"), json=payload())).status_code == 200


@pytest.mark.asyncio
async def test_persistence_skips_downtime_and_ignores_old_table(running):
    scheduler = schedule.get_background_scheduler()
    now = datetime.now(scheduler.timezone)
    common = {"prompt": "task", "user_id": "alice", "sender_name": "Alice", "session_metadata": {"open_id": "ou_alice"}}
    once = schedule.schedule_feishu_task(run_at=now + timedelta(hours=1), cron=None, **common)
    recurring = schedule.schedule_feishu_task(run_at=None, cron="0 9 * * *", **common)
    assert recurring.max_instances == 1
    await schedule.stop_background_scheduler()
    future = now + timedelta(days=2)
    restored = schedule.start_background_scheduler(running.engine, runtime=running.runtime, now=future)
    assert restored.get_job(once.id) is None
    assert restored.get_job(recurring.id).next_run_time > future
    assert inspect(running.engine).get_table_names() == ["feishu_scheduled_jobs"]


@pytest.mark.asyncio
async def test_tools_query_then_cancel(api, monkeypatch):
    monkeypatch.setattr(backend_app.state, "feishu_url", "http://feishu.test", raising=False)
    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        tool_client, "AsyncClient", lambda **kw: real_client(transport=httpx.ASGITransport(app=service.app), **kw)
    )
    created = json.loads(
        await feishu_schedule_cron(
            FeishuScheduleCronArgs(prompt="task", cron="0 9 * * *"), {"open_id": "ou_alice"}, "Alice", "alice", "feishu"
        )
    )
    tasks = json.loads(await feishu_schedule_list("alice"))["tasks"]
    assert tasks == [created["task"]]
    result = json.loads(await feishu_schedule_cancel(FeishuScheduleCancelArgs(job_id=tasks[0]["job_id"]), "alice"))
    assert result["cancelled"] is True
    with pytest.raises(ValueError):
        await feishu_schedule_cancel(FeishuScheduleCancelArgs(job_id=tasks[0]["job_id"]), "alice")


async def execute():
    await schedule.run_feishu_background_task(
        job_id="job-1",
        prompt="original task",
        user_id="alice",
        sender_name="张三",
        session_metadata={"open_id": "ou_alice"},
        schedule_type="cron",
        schedule_value="0 9 * * *",
    )


@pytest.mark.asyncio
async def test_sync_execution_identity_and_result(running):
    requests = []

    async def respond(request):
        requests.append(request)
        auth = UserJWTAuth(request.headers["authorization"].removeprefix("Bearer "))
        assert auth.user_id == "alice" and auth.channel == "feishu"
        assert str(request.url) == "http://backend.test/chat_sync"
        data = parse_qs(request.content.decode())
        prompt = json.loads(data["content"][0])[0]["text"]
        assert "memory/HISTORY.md" in prompt and prompt.endswith("original task")
        assert json.loads(data["session_metadata"][0]) == {"open_id": "ou_alice", "source": "scheduled_task"}
        return httpx.Response(
            200, json=[{"event": "assistant", "context": "first"}, {"event": "assistant", "context": "final result"}]
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        schedule._runtime = schedule.FeishuScheduleRuntime("http://backend.test", client, running.channel)
        await execute()
    assert len(requests) == 1
    args = running.channel.send.await_args.kwargs
    assert args["to"] == "ou_alice"
    assert "final result" in str(args["message"])


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["http", "timeout", "error", "empty", "invalid"])
async def test_execution_failure_delivers_notice_without_retry(running, failure):
    requests = []

    async def respond(request):
        requests.append(request)
        if failure == "timeout":
            raise httpx.ReadTimeout("timeout", request=request)
        if failure == "http":
            return httpx.Response(503)
        events = {
            "error": [{"event": "error", "context": "failed"}],
            "empty": [{"event": "assistant", "context": " "}],
            "invalid": {"unexpected": True},
        }
        return httpx.Response(200, json=events[failure])

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        schedule._runtime = schedule.FeishuScheduleRuntime("http://backend.test", client, running.channel)
        await execute()
    assert len(requests) == 1
    assert schedule.BACKGROUND_TASK_FAILURE_MESSAGE in str(running.channel.send.await_args)


@pytest.mark.asyncio
async def test_delivery_failure_does_not_escape(running):
    running.channel.send.side_effect = RuntimeError("send failed")
    await execute()
    assert not schedule._running_tasks


@pytest.mark.asyncio
async def test_shutdown_cancels_inflight_execution(running):
    started = asyncio.Event()

    async def respond(request):
        started.set()
        await asyncio.Event().wait()

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        schedule._runtime = schedule.FeishuScheduleRuntime("http://backend.test", client, running.channel)
        task = asyncio.create_task(execute())
        await started.wait()
        await schedule.stop_background_scheduler()
        assert task.cancelled()
        assert not schedule._running_tasks
        running.channel.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_channel_lifespan_owns_connection_and_scheduler(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'lifecycle.db'}", connect_args={"check_same_thread": False})
    started, stopped = asyncio.Event(), asyncio.Event()

    async def start():
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    monkeypatch.setattr(service, "FeishuBridge", lambda *a: SimpleNamespace(start=start, channel=SimpleNamespace()))
    monkeypatch.setattr(service, "get_database_engine", lambda: engine)
    monkeypatch.setattr(service.app.state, "chat_url", "http://backend.test", raising=False)
    try:
        async with service.lifespan(service.app):
            await started.wait()
            assert schedule.get_background_scheduler().running
            assert (await service.health())["status"] == "ok"
        assert stopped.is_set() and schedule._scheduler is None
        assert (await service.health()).status_code == 503
    finally:
        engine.dispose()


@pytest.mark.parametrize("path,method", [("/background_message", "GET"), ("/internal/scheduled-run", "POST")])
def test_old_backend_routes_removed(path, method):
    from fastapi.testclient import TestClient

    assert TestClient(backend_app).request(method, path).status_code == 404


@pytest.mark.asyncio
async def test_one_time_job_actually_runs_and_is_removed(running):
    sent = asyncio.Event()

    async def send(**kwargs):
        sent.set()
        return SimpleNamespace(success=True)

    running.channel.send.side_effect = send
    scheduler = schedule.get_background_scheduler()
    job = schedule.schedule_feishu_task(
        prompt="run now",
        run_at=datetime.now(scheduler.timezone) + timedelta(milliseconds=150),
        cron=None,
        user_id="alice",
        sender_name="Alice",
        session_metadata={"open_id": "ou_alice"},
    )
    await asyncio.wait_for(sent.wait(), timeout=5)
    assert scheduler.get_job(job.id) is None
    assert "result" in str(running.channel.send.await_args)


@pytest.mark.asyncio
async def test_postgres_persistence(postgres_database):
    from xpeech.utils.database import get_database_engine

    engine = get_database_engine()
    async with httpx.AsyncClient() as client:
        runtime = schedule.FeishuScheduleRuntime("http://backend.test", client, SimpleNamespace())
        try:
            schedule.start_background_scheduler(engine, runtime=runtime)
            job = schedule.schedule_feishu_task(
                prompt="postgres task",
                run_at=None,
                cron="0 9 * * *",
                user_id="alice",
                sender_name="Alice",
                session_metadata={"open_id": "ou_alice"},
            )
            await schedule.stop_background_scheduler()
            restored = schedule.start_background_scheduler(engine, runtime=runtime)
            assert restored.get_job(job.id).kwargs["prompt"] == "postgres task"
            assert "feishu_scheduled_jobs" in inspect(engine).get_table_names()
        finally:
            await schedule.stop_background_scheduler()


@pytest.mark.parametrize("value", ["2099-01-01", 4070908800, "4070908800", "明天9点", None, ""])
def test_once_rejects_ambiguous_time(value):
    with pytest.raises(ValidationError):
        FeishuScheduleOnceArgs(prompt="task", run_at=value)


@pytest.mark.parametrize("value", ["0 0 9 * * *", "*/15 * * * *", "0 25 * * *", "0 9 * * 7", ""])
def test_cron_rejects_invalid_expression(value):
    with pytest.raises(ValidationError):
        FeishuScheduleCronArgs(prompt="task", cron=value)


@pytest.mark.parametrize("tool,trigger", [
    (feishu_schedule_cron, "cron"),
    (feishu_schedule_once, "run_at"),
])
def test_creation_tool_schema(tool, trigger):
    from xpeech.agent.tools.helper import as_tool

    parameters = as_tool(tool)["function"]["parameters"]
    assert set(parameters["properties"]) == {"prompt", trigger}
    assert set(parameters["required"]) == {"prompt", trigger}
    assert parameters["properties"][trigger]["type"] == "string"
    assert parameters["additionalProperties"] is False


@pytest.mark.parametrize("values", [
    {"run_at": None, "cron": None},
    {"run_at": "2099-01-01T09:00:00+08:00", "cron": "0 9 * * *"},
])
def test_creation_requires_exactly_one_trigger(values):
    with pytest.raises(ValidationError, match="exactly one"):
        FeishuScheduleArgs(prompt="task", **values)


def test_trigger_whitespace_and_weekday_semantics():
    from zoneinfo import ZoneInfo
    from xpeech.agent.tools.schedule.tool_args import parse_schedule_cron

    assert FeishuScheduleCreateRequest(**payload(run_at=" ", cron=" 0  9 * * mon-fri ")).cron == "0 9 * * mon-fri"
    assert FeishuScheduleCreateRequest(**payload(run_at="2099-01-01T09:00:00", cron=" ")).cron is None
    timezone = ZoneInfo("Asia/Shanghai")
    now = datetime(2099, 1, 1, tzinfo=timezone)
    for weekday in ("mon", "0"):
        next_run = parse_schedule_cron(f"0 9 * * {weekday}", timezone=timezone).get_next_fire_time(None, now)
        assert next_run.weekday() == 0 and next_run.hour == 9


@pytest.mark.asyncio
@pytest.mark.parametrize("tool,args,expected_type,expected_value", [
    (
        feishu_schedule_once,
        FeishuScheduleOnceArgs(prompt="task", run_at="2099-01-01T09:00:00+08:00"),
        "run_at", "2099-01-01T09:00:00+08:00",
    ),
    (
        feishu_schedule_cron,
        FeishuScheduleCronArgs(prompt="task", cron=" 0  9 * * mon-fri "),
        "cron", "0 9 * * mon-fri",
    ),
])
async def test_creation_tools(api, monkeypatch, tool, args, expected_type, expected_value):
    monkeypatch.setattr(backend_app.state, "feishu_url", "http://feishu.test", raising=False)
    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        tool_client, "AsyncClient", lambda **kw: real_client(transport=httpx.ASGITransport(app=service.app), **kw)
    )
    created = json.loads(await tool(args, {"open_id": "ou_alice"}, "Alice", "alice", "feishu"))
    assert created["task"]["schedule_type"] == expected_type
    assert created["task"]["schedule_value"] == expected_value
    assert created["task"]["prompt"] == "task"
    assert len(json.loads(await feishu_schedule_list("alice"))["tasks"]) == 1


@pytest.mark.parametrize("minute", ["0", "30", "0,30"])
def test_cron_allows_only_hour_and_half_hour(minute):
    from xpeech.agent.tools.schedule.tool_args import parse_schedule_cron

    expression = f"{minute} * * * *"
    assert FeishuScheduleCronArgs(prompt="task", cron=expression).cron == expression
    trigger = parse_schedule_cron(expression)
    assert str(next(field for field in trigger.fields if field.name == "minute")) == minute


@pytest.mark.parametrize("minute", ["*", "*/30", "15", "0,15,30,45", "0-30", "0/30", "30,0", "00"])
def test_cron_rejects_other_minute_syntax(minute):
    from xpeech.agent.tools.schedule.tool_args import parse_schedule_cron

    expression = f"{minute} * * * *"
    with pytest.raises(ValidationError, match="minute field must be exactly"):
        FeishuScheduleCronArgs(prompt="task", cron=expression)
    with pytest.raises(ValueError, match="minute field must be exactly"):
        parse_schedule_cron(expression)


@pytest.mark.parametrize("values,missing", [
    ({}, {"run_at", "cron"}),
    ({"run_at": "2099-01-01T09:00:00+08:00"}, {"cron"}),
    ({"cron": "0 9 * * *"}, {"run_at"}),
])
def test_creation_rejects_omitted_trigger_fields(values, missing):
    with pytest.raises(ValidationError) as exc:
        FeishuScheduleArgs(prompt="task", **values)
    assert {error["loc"][0] for error in exc.value.errors() if error["type"] == "missing"} == missing


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["run_at", "cron"])
async def test_management_requires_explicit_trigger_fields(api, field):
    data = payload()
    del data[field]
    response = await api.post("/schedules", headers=headers(), json=data)
    assert response.status_code == 422
    assert any(error["loc"] == ["body", field] and error["type"] == "missing" for error in response.json()["detail"])
    assert (await api.get("/schedules", headers=headers())).json() == {"tasks": []}


@pytest.mark.parametrize("model,trigger,other_trigger,value", [
    (FeishuScheduleCronArgs, "cron", "run_at", "0 9 * * *"),
    (FeishuScheduleOnceArgs, "run_at", "cron", "2099-01-01T09:00:00+08:00"),
])
def test_creation_tool_args_require_only_their_trigger(model, trigger, other_trigger, value):
    for invalid in ({}, {trigger: None}, {trigger: value, other_trigger: None}):
        with pytest.raises(ValidationError):
            model(prompt="task", **invalid)
    with pytest.raises(ValidationError, match="prompt must be a non-empty string"):
        model(prompt=" ", **{trigger: value})


@pytest.mark.asyncio
@pytest.mark.parametrize("tool,args", [
    (feishu_schedule_cron, FeishuScheduleCronArgs(prompt="task", cron="0 9 * * *")),
    (feishu_schedule_once, FeishuScheduleOnceArgs(prompt="task", run_at="2099-01-01T09:00:00+08:00")),
])
@pytest.mark.parametrize("updates,error", [
    ({"channel": "web_client"}, "only available"),
    ({"session_metadata": {}}, "session_metadata.open_id"),
    ({"sender_name": " "}, "sender_name"),
    ({"user_id": None}, "user_id"),
])
async def test_creation_tools_require_feishu_context(monkeypatch, tool, args, updates, error):
    request = AsyncMock()
    monkeypatch.setattr("xpeech.agent.tools.schedule.tool.feishu_request", request)
    context = {
        "session_metadata": {"open_id": "ou_alice"},
        "sender_name": "Alice", "user_id": "alice", "channel": "feishu", **updates,
    }
    with pytest.raises(ValueError, match=error):
        await tool(args, **context)
    request.assert_not_awaited()
