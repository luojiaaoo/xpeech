import asyncio
import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from threading import get_ident
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
import pytest_asyncio
from pydantic import ValidationError

import xpeech.subapp.scheduler.app as scheduler_service
from xpeech.agent.runner import AgentRunner
from xpeech.agent.server.api import app as backend_app
from xpeech.agent.server.schema import BackgroundMessageChannel, FeishuBackgroundMessage
from xpeech.agent.tools.helper import as_tool
from xpeech.agent.tools.schedule import (
    FeishuScheduleArgs,
    FeishuScheduleCancelArgs,
    feishu_schedule,
    feishu_schedule_cancel,
    feishu_schedule_list,
)
from xpeech.agent.tools.schedule import client as scheduler_client
from xpeech.config.settings import settings
from xpeech.subapp.scheduler import background
from xpeech.subapp.scheduler.background import (
    BACKGROUND_TASK_FAILURE_MESSAGE,
    FeishuBackgroundMessageQueue,
    cancel_scheduled_task,
    list_scheduled_tasks,
    run_feishu_background_task,
    schedule_feishu_task,
    start_background_scheduler,
    stop_background_scheduler,
)
from xpeech.utils.channel_jwt import ChannelJWTAuth, create_channel_access_token
from xpeech.utils.internal_jwt import (
    create_internal_access_token,
    decode_internal_access_token,
)

TEST_SCHEDULER_URL = "http://127.0.0.1:8090"
TEST_BACKEND_URL = "http://127.0.0.1:7878"


def drain_queue(queue: FeishuBackgroundMessageQueue) -> None:
    while not queue.empty():
        queue.get_nowait()


@pytest.fixture
def scheduler_http(monkeypatch):
    monkeypatch.setattr(
        backend_app.state,
        "scheduler_url",
        TEST_SCHEDULER_URL,
        raising=False,
    )
    client_class = httpx.AsyncClient
    transport = httpx.ASGITransport(app=scheduler_service.app)
    monkeypatch.setattr(
        scheduler_client,
        "AsyncClient",
        lambda **kwargs: client_class(transport=transport, **kwargs),
    )


@pytest.fixture
def backend_http(monkeypatch):
    client_class = httpx.AsyncClient
    transport = httpx.ASGITransport(app=backend_app, raise_app_exceptions=False)
    requests = []

    async def record_request(request):
        requests.append(request)
        token = request.headers["authorization"].removeprefix("Bearer ")
        decode_internal_access_token(token)
        assert str(request.url) == f"{TEST_BACKEND_URL}/internal/scheduled-run"
        assert request.headers["x-user-id"] == "session-1"

    monkeypatch.setattr(
        background.httpx,
        "AsyncClient",
        lambda **kwargs: client_class(
            transport=transport,
            event_hooks={"request": [record_request]},
            **kwargs,
        ),
    )
    return requests


@pytest_asyncio.fixture(autouse=True)
async def clean_background_state(monkeypatch):
    monkeypatch.setattr(
        scheduler_service.app.state,
        "backend_url",
        TEST_BACKEND_URL,
        raising=False,
    )
    stop_background_scheduler()
    drain_queue(background.BACKGROUND_MESSAGE_QUEUES[BackgroundMessageChannel.FEISHU])
    yield
    stop_background_scheduler()
    await asyncio.sleep(0)
    drain_queue(background.BACKGROUND_MESSAGE_QUEUES[BackgroundMessageChannel.FEISHU])


@pytest.mark.parametrize(
    "values",
    [
        {},
        {"run_at": "2099-01-01T00:00:00", "cron": "0 9 * * *"},
    ],
)
def test_schedule_requires_exactly_one_time_field(values):
    with pytest.raises(ValidationError, match="exactly one"):
        FeishuScheduleArgs(prompt="task", **values)


def test_schedule_rejects_invalid_iso_time():
    with pytest.raises(ValidationError):
        FeishuScheduleArgs(prompt="task", run_at="tomorrow morning")


def test_schedule_treats_blank_time_fields_as_missing():
    cron_args = FeishuScheduleArgs(prompt="task", run_at="", cron="0 9 * * *")
    assert cron_args.run_at is None
    assert cron_args.cron == "0 9 * * *"

    run_at_args = FeishuScheduleArgs(
        prompt="task",
        run_at="2099-01-01T00:00:00",
        cron=" \t",
    )
    assert run_at_args.run_at == datetime(2099, 1, 1)
    assert run_at_args.cron is None


def test_schedule_tool_explains_mutually_exclusive_time_fields():
    function_schema = as_tool(feishu_schedule)["function"]
    description = function_schema["description"]
    run_at_description = function_schema["parameters"]["properties"]["run_at"]["description"]

    assert "run_at 和 cron 只能有一个有效值" in description
    assert "可以不带时区" in description
    assert "按服务端系统本地时区解释" in run_at_description
    assert "另一个省略不传即可" not in description
    assert '不要传字符串 "None"' not in description


@pytest.mark.asyncio
async def test_schedule_api_offloads_database_calls(monkeypatch, scheduler_http):
    event_loop_thread = get_ident()
    worker_threads = []

    def create_task(**kwargs):
        worker_threads.append(get_ident())
        return SimpleNamespace(id="job-1", next_run_time=None)

    def list_tasks(user_id):
        worker_threads.append(get_ident())
        return []

    def cancel_task(**kwargs):
        worker_threads.append(get_ident())

    monkeypatch.setattr(scheduler_service, "schedule_feishu_task", create_task)
    monkeypatch.setattr(scheduler_service, "list_scheduled_tasks", list_tasks)
    monkeypatch.setattr(scheduler_service, "cancel_scheduled_task", cancel_task)

    await feishu_schedule(
        FeishuScheduleArgs(prompt="task", run_at="2099-01-01T00:00:00+08:00"),
        channel="feishu",
        session_metadata={"open_id": "ou_1"},
        sender_name="Alice",
        user_id="session-1",
    )
    await feishu_schedule_list(user_id="session-1")
    await feishu_schedule_cancel(
        FeishuScheduleCancelArgs(job_id="job-1"),
        user_id="session-1",
    )

    assert len(worker_threads) == 3
    assert all(thread_id != event_loop_thread for thread_id in worker_threads)


@pytest.mark.asyncio
async def test_schedule_uses_system_timezone_and_rejects_past_and_invalid_cron(tmp_path: Path):
    scheduler = start_background_scheduler(tmp_path / "schedule.db")
    now = datetime.now(scheduler.timezone)

    job = schedule_feishu_task(
        prompt="task",
        run_at=(now + timedelta(hours=1)).replace(tzinfo=None),
        cron=None,
        user_id="session-1",
        sender_name="Alice",
        channel="feishu",
        session_metadata={"open_id": "ou_1"},
    )
    assert job.next_run_time.tzinfo == scheduler.timezone

    with pytest.raises(ValueError, match="future"):
        schedule_feishu_task(
            prompt="past",
            run_at=now - timedelta(seconds=1),
            cron=None,
            user_id="session-1",
            sender_name="Alice",
            channel="feishu",
            session_metadata={"open_id": "ou_1"},
        )
    with pytest.raises(ValueError):
        schedule_feishu_task(
            prompt="bad cron",
            run_at=None,
            cron="not a cron",
            user_id="session-1",
            sender_name="Alice",
            channel="feishu",
            session_metadata={"open_id": "ou_1"},
        )
    for cron in ("* * * * *", "*/5 * * * *", "1,* * * * *"):
        with pytest.raises(ValueError, match="minute field must not contain"):
            schedule_feishu_task(
                prompt="too frequent cron",
                run_at=None,
                cron=cron,
                user_id="session-1",
                sender_name="Alice",
                channel="feishu",
                session_metadata={"open_id": "ou_1"},
            )


@pytest.mark.asyncio
async def test_restart_removes_missed_date_and_advances_cron(tmp_path: Path):
    database = tmp_path / "schedule.db"
    scheduler = start_background_scheduler(database)
    base = datetime.now(scheduler.timezone)
    date_job = schedule_feishu_task(
        prompt="once",
        run_at=base + timedelta(hours=1),
        cron=None,
        user_id="session-1",
        sender_name="Alice",
        channel="feishu",
        session_metadata={"open_id": "ou_1"},
    )
    cron_job = schedule_feishu_task(
        prompt="repeat",
        run_at=None,
        cron="0 * * * *",
        user_id="session-1",
        sender_name="Alice",
        channel="feishu",
        session_metadata={"open_id": "ou_1"},
    )
    stop_background_scheduler()
    await asyncio.sleep(0)

    future_now = base + timedelta(hours=2)
    restored = start_background_scheduler(database, now=future_now)
    assert restored.get_job(date_job.id) is None
    restored_cron = restored.get_job(cron_job.id)
    assert restored_cron is not None
    assert restored_cron.func_ref == "xpeech.subapp.scheduler.background:run_feishu_background_task"
    assert restored_cron.next_run_time > future_now


@pytest.mark.asyncio
async def test_restart_migrates_legacy_channel_from_session_metadata(tmp_path: Path, monkeypatch):
    database = tmp_path / "schedule.db"
    scheduler = start_background_scheduler(database)
    run_at = datetime.now(scheduler.timezone) + timedelta(hours=1)
    # channel 独立成字段之前写入的任务 kwargs 里没有 channel，当时的函数签名允许；
    # 绕过 add_job 对今日签名的校验来构造这样的旧行。
    monkeypatch.setattr("apscheduler.job.check_callable_args", lambda *args, **kwargs: None)
    scheduler.add_job(
        run_feishu_background_task,
        trigger="date",
        run_date=run_at,
        id="legacy-job",
        kwargs={
            "job_id": "legacy-job",
            "prompt": "task",
            "user_id": "session-1",
            "sender_name": "Alice",
            "session_metadata": {"channel": "feishu", "open_id": "ou_1", "custom": "kept"},
            "schedule_type": "run_at",
            "schedule_value": run_at.isoformat(),
        },
    )
    stop_background_scheduler()
    await asyncio.sleep(0)

    restored = start_background_scheduler(database)
    job = restored.get_job("legacy-job")
    assert job.kwargs["channel"] == "feishu"
    assert job.kwargs["session_metadata"] == {"open_id": "ou_1", "custom": "kept"}
    assert list_scheduled_tasks("session-1")[0].channel == "feishu"

    stop_background_scheduler()
    await asyncio.sleep(0)
    stable = start_background_scheduler(database)
    assert stable.get_job("legacy-job").kwargs == job.kwargs


@pytest.mark.asyncio
async def test_tools_capture_context_list_across_channels_and_enforce_cancel_owner(tmp_path: Path, scheduler_http):
    scheduler = start_background_scheduler(tmp_path / "schedule.db")
    args = FeishuScheduleArgs(
        prompt="daily report",
        run_at=datetime.now(scheduler.timezone) + timedelta(hours=1),
    )
    metadata = {"open_id": "ou_1", "email": "a@example.test"}
    result = json.loads(
        await feishu_schedule(
            args,
            session_metadata=metadata,
            sender_name="Alice",
            user_id="session-1",
            channel="feishu",
        )
    )
    job = scheduler.get_job(result["job_id"])
    assert job is not None
    assert job.kwargs["channel"] == "feishu"
    assert job.kwargs["session_metadata"] == metadata
    assert job.kwargs["sender_name"] == "Alice"
    assert job.kwargs["user_id"] == "session-1"
    assert "session_id" not in job.kwargs

    tasks = json.loads(await feishu_schedule_list(user_id="session-1"))
    assert tasks == [
        {
            "job_id": result["job_id"],
            "channel": "feishu",
            "prompt": "daily report",
            "schedule_type": "run_at",
            "schedule_value": result["next_run_time"],
            "next_run_time": result["next_run_time"],
        }
    ]
    assert await feishu_schedule_list(user_id="another-user") == "[]"

    with pytest.raises(ValueError, match="not available"):
        await feishu_schedule_cancel(
            FeishuScheduleCancelArgs(job_id=result["job_id"]),
            user_id="another-user",
        )
    with pytest.raises(ValueError, match="not available"):
        cancel_scheduled_task(user_id="session-1", job_id="missing")
    assert await feishu_schedule_cancel(
        FeishuScheduleCancelArgs(job_id=result["job_id"]),
        user_id="session-1",
    )
    assert list_scheduled_tasks("session-1") == []


@pytest.mark.asyncio
async def test_user_can_have_at_most_four_scheduled_tasks(tmp_path: Path):
    scheduler = start_background_scheduler(tmp_path / "schedule.db")
    run_at = datetime.now(scheduler.timezone) + timedelta(hours=1)
    common = {
        "run_at": run_at,
        "cron": None,
        "user_id": "session-1",
        "sender_name": "Alice",
        "channel": "feishu",
        "session_metadata": {"open_id": "ou_1"},
    }

    jobs = [schedule_feishu_task(prompt=f"task-{index}", **common) for index in range(4)]
    assert len(list_scheduled_tasks("session-1")) == 4

    with pytest.raises(ValueError, match="maximum 4"):
        schedule_feishu_task(prompt="task-5", **common)

    schedule_feishu_task(prompt="other-user-task", **{**common, "user_id": "user-2"})
    assert len(list_scheduled_tasks("user-2")) == 1

    cancel_scheduled_task(user_id="session-1", job_id=jobs[0].id)
    replacement = schedule_feishu_task(prompt="replacement", **common)
    assert replacement.id
    assert len(list_scheduled_tasks("session-1")) == 4


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "channel, metadata, error",
    [
        ("web", {"open_id": "ou_1"}, "only available for Feishu"),
        ("feishu", {"open_id": "  "}, "non-empty string"),
        ("feishu", {}, "non-empty string"),
    ],
)
async def test_feishu_schedule_validates_delivery_context(tmp_path: Path, channel, metadata, error):
    scheduler = start_background_scheduler(tmp_path / "schedule.db")
    args = FeishuScheduleArgs(
        prompt="task",
        run_at=datetime.now(scheduler.timezone) + timedelta(hours=1),
    )
    with pytest.raises(ValueError, match=error):
        await feishu_schedule(
            args,
            session_metadata=metadata,
            sender_name="Alice",
            user_id="session-1",
            channel=channel,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "result, expected",
    [
        ("completed", "completed"),
        (RuntimeError("model failed"), BACKGROUND_TASK_FAILURE_MESSAGE),
        ("", BACKGROUND_TASK_FAILURE_MESSAGE),
    ],
)
async def test_background_agent_queues_success_or_fixed_failure(
    monkeypatch,
    backend_http,
    result,
    expected,
):
    captured = {}
    monkeypatch.setattr(background.random, "uniform", lambda start, end: 0)

    class FakeRunner:
        async def run_not_interactive(self, message):
            captured["message"] = message
            if isinstance(result, Exception):
                raise result
            return [
                {"event": "assistant", "context": "draft"},
                {"event": "tool_call", "context": "tool"},
                {"event": "assistant", "context": result},
                {"event": "token_usage", "context": "usage"},
            ]

    async def create(cls, user_id, **kwargs):
        captured["user_id"] = user_id
        return FakeRunner()

    monkeypatch.setattr(AgentRunner, "create", classmethod(create))
    metadata = {"open_id": "ou_1", "custom": "kept"}
    await run_feishu_background_task(
        job_id="job-1",
        prompt="scheduled prompt",
        user_id="session-1",
        sender_name="Alice",
        channel="feishu",
        session_metadata=metadata,
        schedule_type="cron",
        schedule_value="0 9 * * *",
    )

    queued = background.BACKGROUND_MESSAGE_QUEUES[BackgroundMessageChannel.FEISHU].get_nowait()
    assert queued.open_id == "ou_1"
    assert queued.content == expected
    assert queued.schedule_type == "cron"
    assert queued.schedule_value == "0 9 * * *"
    assert captured["user_id"] == "session-1"
    assert captured["message"].sender_name == "Alice"
    assert re.fullmatch(
        r"[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}",
        captured["message"].session_id,
    )
    scheduled_prompt = captured["message"].content[0].text
    assert scheduled_prompt.startswith("## 定时任务运行说明")
    assert "本次运行由后台定时任务触发" in scheduled_prompt
    assert "追加到 `memory/HISTORY.md`" in scheduled_prompt
    assert "不要覆盖已有内容" in scheduled_prompt
    assert "最终回复会直接发送给用户" in scheduled_prompt
    assert "必须呈现原始任务要求的实际结果" in scheduled_prompt
    assert "不能仅回复“执行记录已追加”" in scheduled_prompt
    assert "调度类型：cron" in scheduled_prompt
    assert "调度表达式：0 9 * * *" in scheduled_prompt
    assert scheduled_prompt.endswith("scheduled prompt")
    assert captured["message"].channel == "feishu"
    assert captured["message"].session_metadata == {**metadata, "source": "scheduled_task"}
    assert metadata == {"open_id": "ou_1", "custom": "kept"}
    assert len(backend_http) == 1  # Failures must not replay the Agent execution.


@pytest.mark.asyncio
async def test_background_agent_waits_for_random_delay(monkeypatch, backend_http):
    sleep = AsyncMock()
    monkeypatch.setattr(background.asyncio, "sleep", sleep)
    uniform = MagicMock(return_value=73.5)
    monkeypatch.setattr(background.random, "uniform", uniform)

    class FakeRunner:
        async def run_not_interactive(self, message):
            return [{"event": "assistant", "context": "completed"}]

    async def create(cls, user_id, **kwargs):
        return FakeRunner()

    monkeypatch.setattr(AgentRunner, "create", classmethod(create))
    await run_feishu_background_task(
        job_id="job-delayed",
        prompt="scheduled prompt",
        user_id="session-1",
        sender_name="Alice",
        channel="feishu",
        session_metadata={"open_id": "ou_1"},
        schedule_type="run_at",
        schedule_value="2099-01-01T00:00:00+08:00",
    )

    uniform.assert_called_once_with(0, background.MAX_BACKGROUND_TASK_DELAY_SECONDS)
    sleep.assert_awaited_once_with(73.5)


def test_background_message_and_queue_are_strictly_typed():
    queue = FeishuBackgroundMessageQueue()
    schedule = {"schedule_type": "cron", "schedule_value": "0 9 * * *"}
    with pytest.raises(ValidationError):
        FeishuBackgroundMessage(open_id=" ", content="result", **schedule)
    with pytest.raises(ValidationError):
        FeishuBackgroundMessage(open_id="ou_1", content=" ", **schedule)
    with pytest.raises(ValidationError):
        FeishuBackgroundMessage(open_id="ou_1", content="result")
    with pytest.raises(TypeError):
        queue.put_nowait({"open_id": "ou_1", "content": "result"})


@pytest.mark.asyncio
async def test_background_message_long_poll_returns_immediately(monkeypatch):
    queue = FeishuBackgroundMessageQueue()
    monkeypatch.setattr(
        scheduler_service,
        "BACKGROUND_MESSAGE_QUEUES",
        {BackgroundMessageChannel.FEISHU: queue},
    )
    message = FeishuBackgroundMessage(
        open_id="ou_1",
        content="result",
        schedule_type="cron",
        schedule_value="0 9 * * *",
    )
    queue.put_nowait(message)
    assert await scheduler_service.poll_background_message(BackgroundMessageChannel.FEISHU) == message


@pytest.mark.asyncio
async def test_background_message_long_poll_times_out_with_204(monkeypatch):
    monkeypatch.setattr(
        scheduler_service,
        "BACKGROUND_MESSAGE_QUEUES",
        {BackgroundMessageChannel.FEISHU: FeishuBackgroundMessageQueue()},
    )
    monkeypatch.setattr(scheduler_service, "BACKGROUND_MESSAGE_LONG_POLL_SECONDS", 0.001)
    response = await scheduler_service.poll_background_message(BackgroundMessageChannel.FEISHU)
    assert response.status_code == 204


@pytest.mark.asyncio
async def test_direct_backend_proxies_background_poll_to_scheduler(scheduler_http, monkeypatch):
    token = create_channel_access_token(ChannelJWTAuth("feishu", settings.jwt.feishu_secret_key))
    queue = background.BACKGROUND_MESSAGE_QUEUES[BackgroundMessageChannel.FEISHU]
    queue.put_nowait(
        FeishuBackgroundMessage(
            open_id="ou_1",
            content="completed",
            schedule_type="cron",
            schedule_value="0 9 * * *",
        )
    )
    monkeypatch.setattr(scheduler_service, "BACKGROUND_MESSAGE_LONG_POLL_SECONDS", 0.001)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=backend_app),
        base_url="http://test",
        headers={"Authorization": f"Bearer {token}"},
    ) as client:
        response = await client.get("/background_message?channel=feishu")
        assert response.status_code == 200
        assert response.json() == {
            "channel": "feishu",
            "open_id": "ou_1",
            "content": "completed",
            "schedule_type": "cron",
            "schedule_value": "0 9 * * *",
        }
        response = await client.get("/background_message?channel=feishu")
        assert response.status_code == 204


@pytest.mark.asyncio
async def test_only_scheduler_lifespan_starts_scheduling(tmp_path, monkeypatch):
    from xpeech.agent.server import server

    monkeypatch.setattr(settings.path, "session_schedule_path", tmp_path / "schedule.db")
    monkeypatch.setattr(server, "create_db_and_tables", AsyncMock())
    monkeypatch.setattr(server, "close_all_mcp_connections", AsyncMock())
    monkeypatch.setattr(server, "record_engine", SimpleNamespace(dispose=AsyncMock()))

    async with server.lifespan(backend_app):
        assert background._scheduler is None

    async with scheduler_service.lifespan(scheduler_service.app):
        assert background.get_background_scheduler().running
    assert background._scheduler is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "app, method, path",
    [
        (scheduler_service.app, "POST", "/schedules"),
        (scheduler_service.app, "GET", "/schedules"),
        (scheduler_service.app, "DELETE", "/schedules/job-1"),
        (scheduler_service.app, "GET", "/background_message?channel=feishu"),
        (backend_app, "POST", "/internal/scheduled-run"),
    ],
)
async def test_scheduler_and_execution_endpoints_require_jwt(app, method, path):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.request(method, path, headers={"x-user-id": "alice"})
    assert response.status_code == 401


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "app, method, path",
    [
        (scheduler_service.app, "GET", "/schedules"),
        (backend_app, "POST", "/internal/scheduled-run"),
    ],
)
@pytest.mark.parametrize("channel", ["feishu", "web_client"])
async def test_channel_jwt_cannot_access_internal_endpoints(app, method, path, channel):
    keys = {
        "feishu": settings.jwt.feishu_secret_key,
        "web_client": settings.jwt.web_client_secret_key,
    }
    token = create_channel_access_token(ChannelJWTAuth(channel, keys[channel]))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.request(
            method,
            path,
            headers={
                "Authorization": f"Bearer {token}",
                "x-user-id": "alice",
            },
        )

    assert response.status_code == 401


@pytest.mark.asyncio
async def test_scheduler_api_validates_context_and_time(tmp_path, scheduler_http):
    start_background_scheduler(tmp_path / "schedule.db")
    payload = {
        "prompt": "task",
        "run_at": "2000-01-01T00:00:00+08:00",
        "sender_name": "Alice",
        "channel": "feishu",
        "session_metadata": {"open_id": "ou_1"},
    }
    with pytest.raises(ValueError, match="future"):
        await scheduler_client.scheduler_request("POST", "/schedules", "alice", payload=payload)

    payload["run_at"] = "2099-01-01T00:00:00+08:00"
    payload["channel"] = "web"
    with pytest.raises(httpx.HTTPStatusError) as error:
        await scheduler_client.scheduler_request("POST", "/schedules", "alice", payload=payload)
    assert error.value.response.status_code == 422
    assert list_scheduled_tasks("alice") == []


@pytest.mark.asyncio
async def test_scheduler_api_limit_is_shared_across_http_clients(tmp_path, scheduler_http):
    start_background_scheduler(tmp_path / "schedule.db")
    args = FeishuScheduleArgs(prompt="task", cron="0 9 * * *")
    for _ in range(4):
        await feishu_schedule(
            args,
            channel="feishu",
            session_metadata={"open_id": "ou_1"},
            sender_name="Alice",
            user_id="alice",
        )
    with pytest.raises(ValueError, match="maximum 4"):
        await feishu_schedule(
            args,
            channel="feishu",
            session_metadata={"open_id": "ou_1"},
            sender_name="Alice",
            user_id="alice",
        )


@pytest.mark.asyncio
async def test_execution_rejects_invalid_user_id_before_running(monkeypatch):
    create = AsyncMock()
    monkeypatch.setattr(AgentRunner, "create", create)
    transport = httpx.ASGITransport(app=backend_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/internal/scheduled-run",
            headers={"Authorization": f"Bearer {create_internal_access_token()}", "x-user-id": "../escape"},
            json={
                "job_id": "job-1",
                "prompt": "task",
                "sender_name": "Alice",
                "channel": "feishu",
                "session_metadata": {},
                "schedule_type": "cron",
                "schedule_value": "0 9 * * *",
            },
        )
    assert response.status_code == 422
    create.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["unavailable", "timeout", "invalid-result"])
async def test_dispatch_failures_queue_one_failure_message_without_retry(monkeypatch, failure):
    calls = []

    async def handler(request):
        calls.append(request)
        if failure == "timeout":
            raise httpx.ReadTimeout("upstream timed out", request=request)
        if failure == "unavailable":
            return httpx.Response(503)
        return httpx.Response(200, json={"content": "   "})

    client_class = httpx.AsyncClient
    monkeypatch.setattr(
        background.httpx,
        "AsyncClient",
        lambda **kwargs: client_class(transport=httpx.MockTransport(handler), **kwargs),
    )
    monkeypatch.setattr(background.random, "uniform", lambda *_: 0)
    await run_feishu_background_task(
        job_id="job-1",
        prompt="task",
        user_id="alice",
        sender_name="Alice",
        channel="feishu",
        session_metadata={"open_id": "ou_1"},
        schedule_type="cron",
        schedule_value="0 9 * * *",
    )
    queue = background.BACKGROUND_MESSAGE_QUEUES[BackgroundMessageChannel.FEISHU]
    assert queue.get_nowait().content == BACKGROUND_TASK_FAILURE_MESSAGE
    assert queue.empty()
    assert len(calls) == 1
