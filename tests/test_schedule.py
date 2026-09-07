import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path
from threading import get_ident
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from pydantic import ValidationError

import xpeech.agent.background as background
import xpeech.agent.tools.schedule as schedule_tools
from xpeech.agent.background import (
    BACKGROUND_TASK_FAILURE_MESSAGE,
    BackgroundMessageChannel,
    FeishuBackgroundMessage,
    FeishuBackgroundMessageQueue,
    cancel_scheduled_task,
    list_scheduled_tasks,
    run_feishu_background_task,
    schedule_feishu_task,
    start_background_scheduler,
    stop_background_scheduler,
)
from xpeech.agent.runner import AgentRunner
from xpeech.agent.server.routes import chat
from xpeech.agent.tools.helper import as_tool
from xpeech.agent.tools.schedule import (
    FeishuScheduleArgs,
    FeishuScheduleCancelArgs,
    feishu_schedule,
    feishu_schedule_cancel,
    feishu_schedule_list,
)


def drain_queue(queue: FeishuBackgroundMessageQueue) -> None:
    while not queue.empty():
        queue.get_nowait()


@pytest_asyncio.fixture(autouse=True)
async def clean_background_state():
    stop_background_scheduler()
    drain_queue(
        background.BACKGROUND_MESSAGE_QUEUES[BackgroundMessageChannel.FEISHU]
    )
    yield
    stop_background_scheduler()
    await asyncio.sleep(0)
    drain_queue(
        background.BACKGROUND_MESSAGE_QUEUES[BackgroundMessageChannel.FEISHU]
    )


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
    run_at_description = function_schema["parameters"]["properties"]["run_at"][
        "description"
    ]

    assert "run_at 和 cron 只能有一个有效值" in description
    assert "可以不带时区" in description
    assert "按服务端系统本地时区解释" in run_at_description
    assert "另一个省略不传即可" not in description
    assert '不要传字符串 "None"' not in description


@pytest.mark.asyncio
async def test_schedule_tools_offload_scheduler_database_calls(monkeypatch):
    event_loop_thread = get_ident()
    worker_threads = []

    def create_task(**kwargs):
        worker_threads.append(get_ident())
        return SimpleNamespace(id="job-1", next_run_time=None)

    def list_tasks(session_id):
        worker_threads.append(get_ident())
        return []

    def cancel_task(**kwargs):
        worker_threads.append(get_ident())

    monkeypatch.setattr(schedule_tools, "schedule_feishu_task", create_task)
    monkeypatch.setattr(schedule_tools, "list_scheduled_tasks", list_tasks)
    monkeypatch.setattr(schedule_tools, "cancel_scheduled_task", cancel_task)

    await schedule_tools.feishu_schedule(
        FeishuScheduleArgs(prompt="task", run_at="2099-01-01T00:00:00+08:00"),
        session_metadata={"channel": "feishu", "open_id": "ou_1"},
        sender_name="Alice",
        session_id="session-1",
    )
    await schedule_tools.feishu_schedule_list(session_id="session-1")
    await schedule_tools.feishu_schedule_cancel(
        FeishuScheduleCancelArgs(job_id="job-1"),
        session_id="session-1",
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
        session_id="session-1",
        sender_name="Alice",
        session_metadata={"channel": "feishu", "open_id": "ou_1"},
    )
    assert job.next_run_time.tzinfo == scheduler.timezone

    with pytest.raises(ValueError, match="future"):
        schedule_feishu_task(
            prompt="past",
            run_at=now - timedelta(seconds=1),
            cron=None,
            session_id="session-1",
            sender_name="Alice",
            session_metadata={"channel": "feishu", "open_id": "ou_1"},
        )
    with pytest.raises(ValueError):
        schedule_feishu_task(
            prompt="bad cron",
            run_at=None,
            cron="not a cron",
            session_id="session-1",
            sender_name="Alice",
            session_metadata={"channel": "feishu", "open_id": "ou_1"},
        )
    for cron in ("* * * * *", "*/5 * * * *", "1,* * * * *"):
        with pytest.raises(ValueError, match="minute field must not contain"):
            schedule_feishu_task(
                prompt="too frequent cron",
                run_at=None,
                cron=cron,
                session_id="session-1",
                sender_name="Alice",
                session_metadata={"channel": "feishu", "open_id": "ou_1"},
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
        session_id="session-1",
        sender_name="Alice",
        session_metadata={"channel": "feishu", "open_id": "ou_1"},
    )
    cron_job = schedule_feishu_task(
        prompt="repeat",
        run_at=None,
        cron="0 * * * *",
        session_id="session-1",
        sender_name="Alice",
        session_metadata={"channel": "feishu", "open_id": "ou_1"},
    )
    stop_background_scheduler()
    await asyncio.sleep(0)

    future_now = base + timedelta(hours=2)
    restored = start_background_scheduler(database, now=future_now)
    assert restored.get_job(date_job.id) is None
    restored_cron = restored.get_job(cron_job.id)
    assert restored_cron is not None
    assert restored_cron.next_run_time > future_now


@pytest.mark.asyncio
async def test_tools_capture_context_list_across_channels_and_enforce_cancel_owner(tmp_path: Path):
    scheduler = start_background_scheduler(tmp_path / "schedule.db")
    args = FeishuScheduleArgs(
        prompt="daily report",
        run_at=datetime.now(scheduler.timezone) + timedelta(hours=1),
    )
    metadata = {"channel": "feishu", "open_id": "ou_1", "email": "a@example.test"}
    result = json.loads(
        await feishu_schedule(
            args,
            session_metadata=metadata,
            sender_name="Alice",
            session_id="session-1",
        )
    )
    job = scheduler.get_job(result["job_id"])
    assert job is not None
    assert job.kwargs["session_metadata"] == metadata
    assert job.kwargs["sender_name"] == "Alice"

    tasks = json.loads(await feishu_schedule_list(session_id="session-1"))
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
    assert await feishu_schedule_list(session_id="another-session") == "[]"

    with pytest.raises(ValueError, match="not available"):
        cancel_scheduled_task(session_id="another-session", job_id=result["job_id"])
    with pytest.raises(ValueError, match="not available"):
        cancel_scheduled_task(session_id="session-1", job_id="missing")
    assert await feishu_schedule_cancel(
        FeishuScheduleCancelArgs(job_id=result["job_id"]),
        session_id="session-1",
    )
    assert list_scheduled_tasks("session-1") == []


@pytest.mark.asyncio
async def test_session_can_have_at_most_eight_scheduled_tasks(tmp_path: Path):
    scheduler = start_background_scheduler(tmp_path / "schedule.db")
    run_at = datetime.now(scheduler.timezone) + timedelta(hours=1)
    common = {
        "run_at": run_at,
        "cron": None,
        "session_id": "session-1",
        "sender_name": "Alice",
        "session_metadata": {"channel": "feishu", "open_id": "ou_1"},
    }

    jobs = [
        schedule_feishu_task(prompt=f"task-{index}", **common)
        for index in range(8)
    ]
    assert len(list_scheduled_tasks("session-1")) == 8

    with pytest.raises(ValueError, match="maximum 8"):
        schedule_feishu_task(prompt="task-9", **common)

    cancel_scheduled_task(session_id="session-1", job_id=jobs[0].id)
    replacement = schedule_feishu_task(prompt="replacement", **common)
    assert replacement.id
    assert len(list_scheduled_tasks("session-1")) == 8


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "metadata, error",
    [
        ({"channel": "web", "open_id": "ou_1"}, "only available for Feishu"),
        ({"channel": "feishu", "open_id": "  "}, "non-empty string"),
        ({"channel": "feishu"}, "non-empty string"),
    ],
)
async def test_feishu_schedule_validates_delivery_context(tmp_path: Path, metadata, error):
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
            session_id="session-1",
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
    result,
    expected,
):
    captured = {}
    monkeypatch.setattr(background.random, "uniform", lambda start, end: 0)

    class FakeRunner:
        async def run_once(self, message, *, background=False):
            captured["message"] = message
            captured["background"] = background
            if isinstance(result, Exception):
                raise result
            return result

    async def create(cls, session_id, **kwargs):
        captured["session_id"] = session_id
        return FakeRunner()

    monkeypatch.setattr(AgentRunner, "create", classmethod(create))
    metadata = {"channel": "feishu", "open_id": "ou_1", "custom": "kept"}
    await run_feishu_background_task(
        job_id="job-1",
        prompt="scheduled prompt",
        session_id="session-1",
        sender_name="Alice",
        session_metadata=metadata,
        schedule_type="cron",
        schedule_value="0 9 * * *",
    )

    queued = background.BACKGROUND_MESSAGE_QUEUES[
        BackgroundMessageChannel.FEISHU
    ].get_nowait()
    assert queued.open_id == "ou_1"
    assert queued.content == expected
    assert captured["session_id"] == "session-1"
    assert captured["background"] is True
    assert captured["message"].sender_name == "Alice"
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
    assert captured["message"].session_metadata == {**metadata, "source": "scheduled_task"}
    assert metadata == {"channel": "feishu", "open_id": "ou_1", "custom": "kept"}


@pytest.mark.asyncio
async def test_background_agent_waits_for_random_delay(monkeypatch):
    sleep = AsyncMock()
    monkeypatch.setattr(background.asyncio, "sleep", sleep)
    uniform = MagicMock(return_value=73.5)
    monkeypatch.setattr(background.random, "uniform", uniform)

    class FakeRunner:
        async def run_once(self, message, *, background=False):
            return "completed"

    async def create(cls, session_id, **kwargs):
        return FakeRunner()

    monkeypatch.setattr(AgentRunner, "create", classmethod(create))
    await run_feishu_background_task(
        job_id="job-delayed",
        prompt="scheduled prompt",
        session_id="session-1",
        sender_name="Alice",
        session_metadata={"channel": "feishu", "open_id": "ou_1"},
        schedule_type="run_at",
        schedule_value="2099-01-01T00:00:00+08:00",
    )

    uniform.assert_called_once_with(0, background.MAX_BACKGROUND_TASK_DELAY_SECONDS)
    sleep.assert_awaited_once_with(73.5)


def test_background_message_and_queue_are_strictly_typed():
    queue = FeishuBackgroundMessageQueue()
    with pytest.raises(ValidationError):
        FeishuBackgroundMessage(open_id=" ", content="result")
    with pytest.raises(ValidationError):
        FeishuBackgroundMessage(open_id="ou_1", content=" ")
    with pytest.raises(TypeError):
        queue.put_nowait({"open_id": "ou_1", "content": "result"})


@pytest.mark.asyncio
async def test_background_message_long_poll_returns_immediately(monkeypatch):
    queue = FeishuBackgroundMessageQueue()
    monkeypatch.setattr(
        chat,
        "BACKGROUND_MESSAGE_QUEUES",
        {BackgroundMessageChannel.FEISHU: queue},
    )
    message = FeishuBackgroundMessage(open_id="ou_1", content="result")
    queue.put_nowait(message)
    assert await chat.poll_background_message(BackgroundMessageChannel.FEISHU) == message


@pytest.mark.asyncio
async def test_background_message_long_poll_times_out_with_204(monkeypatch):
    monkeypatch.setattr(
        chat,
        "BACKGROUND_MESSAGE_QUEUES",
        {BackgroundMessageChannel.FEISHU: FeishuBackgroundMessageQueue()},
    )
    monkeypatch.setattr(chat, "BACKGROUND_MESSAGE_LONG_POLL_SECONDS", 0.001)
    response = await chat.poll_background_message(BackgroundMessageChannel.FEISHU)
    assert response.status_code == 204
