from __future__ import annotations

import asyncio
import random
from threading import Lock
from datetime import datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Literal
from uuid import uuid4

from apscheduler.job import Job
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..config.settings import settings
from .server.schema import InboundMessage, InputText

BACKGROUND_TASK_FAILURE_MESSAGE = "定时任务执行失败，请稍后重试。"
MAX_BACKGROUND_TASK_DELAY_SECONDS = 3 * 60
MAX_SCHEDULED_TASKS_PER_SESSION = 8
SCHEDULED_TASK_PROMPT_PREFIX = (
    "## 定时任务运行说明\n\n"
    "本次运行由后台定时任务触发，不是用户实时发起的新对话。\n"
    "完成任务后，请将本次定时任务的执行记录追加到 `memory/HISTORY.md`，"
    "不要覆盖已有内容。记录必须以 `[YYYY-MM-DD HH:MM]` 开头，并包含任务内容和执行结果摘要。\n\n"
    "## 原始定时任务提示词\n"
)


class BackgroundMessageChannel(StrEnum):
    """Supported delivery channels for background messages."""

    FEISHU = "feishu"


class FeishuBackgroundMessage(BaseModel):
    """An Agent result waiting to be delivered by the Feishu worker."""

    model_config = ConfigDict(frozen=True)

    channel: Literal[BackgroundMessageChannel.FEISHU] = BackgroundMessageChannel.FEISHU
    open_id: str = Field(min_length=1)
    content: str = Field(min_length=1)

    @field_validator("open_id", "content")
    @classmethod
    def validate_non_blank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must not be blank")
        return value


class FeishuBackgroundMessageQueue(asyncio.Queue[FeishuBackgroundMessage]):
    """Queue that rejects values other than ``FeishuBackgroundMessage``."""

    @staticmethod
    def _validate(item: object) -> FeishuBackgroundMessage:
        if not isinstance(item, FeishuBackgroundMessage):
            raise TypeError("background message queue only accepts FeishuBackgroundMessage")
        return item

    async def put(self, item: FeishuBackgroundMessage) -> None:
        await super().put(self._validate(item))

    def put_nowait(self, item: FeishuBackgroundMessage) -> None:
        super().put_nowait(self._validate(item))


class ScheduledTask(BaseModel):
    """Serializable description of an active background task."""

    job_id: str
    channel: str
    prompt: str
    schedule_type: Literal["run_at", "cron"]
    schedule_value: str
    next_run_time: str | None


BACKGROUND_MESSAGE_QUEUES: dict[
    BackgroundMessageChannel,
    FeishuBackgroundMessageQueue,
] = {
    BackgroundMessageChannel.FEISHU: FeishuBackgroundMessageQueue(),
}
_scheduler: AsyncIOScheduler | None = None
_schedule_lock = Lock()


def _scheduler_database_path(database_path: str | Path | None = None) -> Path:
    path = (
        Path(database_path)
        if database_path is not None
        else settings.path.schedule_path
    )
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _sqlite_url(path: Path) -> str:
    return f"sqlite:///{path.as_posix()}"


def get_background_scheduler() -> AsyncIOScheduler:
    """Return the running scheduler or fail when the application has not started it."""
    if _scheduler is None or not _scheduler.running:
        raise RuntimeError("Background scheduler is not running")
    return _scheduler


def _skip_jobs_missed_while_stopped(scheduler: AsyncIOScheduler, now: datetime) -> None:
    for job in scheduler.get_jobs():
        next_run_time = job.next_run_time
        if next_run_time is None or next_run_time > now:
            continue
        if isinstance(job.trigger, DateTrigger):
            scheduler.remove_job(job.id)
            logger.info("Removed expired one-time background job job_id={}", job.id)
            continue

        next_future_run = job.trigger.get_next_fire_time(None, now + timedelta(microseconds=1))
        if next_future_run is None:
            scheduler.remove_job(job.id)
            logger.info("Removed exhausted background job job_id={}", job.id)
            continue

        scheduler.modify_job(job.id, next_run_time=next_future_run)
        logger.info(
            "Advanced missed background job job_id={} next_run_time={}",
            job.id,
            next_future_run.isoformat(),
        )


def start_background_scheduler(
    database_path: str | Path | None = None,
    *,
    now: datetime | None = None,
) -> AsyncIOScheduler:
    """Start the persistent scheduler without replaying runs missed during downtime."""
    global _scheduler

    if _scheduler is not None and _scheduler.running:
        return _scheduler

    path = _scheduler_database_path(database_path)
    scheduler = AsyncIOScheduler(
        jobstores={"default": SQLAlchemyJobStore(url=_sqlite_url(path))},
    )
    timezone = scheduler.timezone
    scheduler.start(paused=True)
    try:
        current_time = now or datetime.now(timezone)
        if current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=timezone)
        else:
            current_time = current_time.astimezone(timezone)
        _skip_jobs_missed_while_stopped(scheduler, current_time)
        scheduler.resume()
    except Exception:
        scheduler.shutdown(wait=False)
        raise

    _scheduler = scheduler
    logger.info("Background scheduler started database={}", path)
    return scheduler


def stop_background_scheduler() -> None:
    """Stop the scheduler if it is running."""
    global _scheduler

    scheduler = _scheduler
    _scheduler = None
    if scheduler is not None and scheduler.running:
        scheduler.shutdown(wait=False)
        logger.info("Background scheduler stopped")


def _normalize_run_at(run_at: datetime, timezone) -> datetime:
    normalized = (
        run_at.replace(tzinfo=timezone)
        if run_at.tzinfo is None
        else run_at.astimezone(timezone)
    )
    if normalized <= datetime.now(timezone):
        raise ValueError("run_at must be in the future")
    return normalized


def schedule_feishu_task(
    *,
    prompt: str,
    run_at: datetime | None,
    cron: str | None,
    session_id: str,
    sender_name: str,
    session_metadata: dict[str, str],
) -> Job:
    """Register a persistent Feishu Agent task and return its APScheduler job."""
    scheduler = get_background_scheduler()
    timezone = scheduler.timezone

    if (run_at is None) == (cron is None):
        raise ValueError("exactly one of run_at and cron must be provided")

    if run_at is not None:
        normalized_run_at = _normalize_run_at(run_at, timezone)
        trigger = DateTrigger(run_date=normalized_run_at, timezone=timezone)
        schedule_type = "run_at"
        schedule_value = normalized_run_at.isoformat()
    else:
        assert cron is not None
        cron_fields = cron.split()
        if cron_fields and "*" in cron_fields[0]:
            raise ValueError("cron minute field must not contain '*'")
        trigger = CronTrigger.from_crontab(cron, timezone=timezone)
        if trigger.get_next_fire_time(None, datetime.now(timezone)) is None:
            raise ValueError("cron does not produce a future run time")
        schedule_type = "cron"
        schedule_value = cron

    # Keep the count-and-add operation atomic inside the supported single process.
    with _schedule_lock:
        session_job_count = sum(
            job.kwargs.get("session_id") == session_id
            for job in scheduler.get_jobs()
        )
        if session_job_count >= MAX_SCHEDULED_TASKS_PER_SESSION:
            raise ValueError(
                "Scheduled task limit reached for this session "
                f"(maximum {MAX_SCHEDULED_TASKS_PER_SESSION})"
            )

        job_id = uuid4().hex
        return scheduler.add_job(
            run_feishu_background_task,
            trigger=trigger,
            id=job_id,
            name="Feishu scheduled Agent task",
            kwargs={
                "job_id": job_id,
                "prompt": prompt,
                "session_id": session_id,
                "sender_name": sender_name,
                "session_metadata": dict(session_metadata),
                "schedule_type": schedule_type,
                "schedule_value": schedule_value,
            },
            coalesce=False,
            max_instances=1,
            misfire_grace_time=1,
            replace_existing=False,
        )


def list_scheduled_tasks(session_id: str) -> list[ScheduledTask]:
    """List all active scheduled tasks owned by a session, regardless of channel."""
    tasks = []
    for job in get_background_scheduler().get_jobs():
        if job.kwargs.get("session_id") != session_id:
            continue
        metadata = job.kwargs.get("session_metadata") or {}
        tasks.append(
            ScheduledTask(
                job_id=job.id,
                channel=str(metadata.get("channel", "unknown")),
                prompt=str(job.kwargs.get("prompt", "")),
                schedule_type=job.kwargs["schedule_type"],
                schedule_value=job.kwargs["schedule_value"],
                next_run_time=job.next_run_time.isoformat() if job.next_run_time else None,
            )
        )
    return sorted(
        tasks,
        key=lambda task: (
            task.next_run_time is None,
            task.next_run_time or "",
            task.job_id,
        ),
    )


def cancel_scheduled_task(*, session_id: str, job_id: str) -> None:
    """Remove a job only when it belongs to the requesting session."""
    scheduler = get_background_scheduler()
    job = scheduler.get_job(job_id)
    if job is None or job.kwargs.get("session_id") != session_id:
        raise ValueError("Scheduled task is not available for this session")
    scheduler.remove_job(job_id)


async def run_feishu_background_task(
    *,
    job_id: str,
    prompt: str,
    session_id: str,
    sender_name: str,
    session_metadata: dict[str, str],
    schedule_type: str,
    schedule_value: str,
) -> None:
    """Run a scheduled Agent task and enqueue its user-facing Feishu result."""
    
    delay_seconds = random.uniform(0, MAX_BACKGROUND_TASK_DELAY_SECONDS)
    logger.info(
        "Feishu scheduled Agent task waiting job_id={} delay_seconds={:.2f}",
        job_id,
        delay_seconds,
    )
    await asyncio.sleep(delay_seconds)

    open_id = session_metadata["open_id"]
    try:
        # Keep this import lazy so persisted jobs can import this module without a
        # runner/background/registry import cycle.
        from .runner import AgentRunner

        metadata = dict(session_metadata)
        metadata["source"] = "scheduled_task"
        message = InboundMessage(
            session_id=session_id,
            sender_name=sender_name,
            session_metadata=metadata,
            content=[
                InputText(
                    text=(
                        f"{SCHEDULED_TASK_PROMPT_PREFIX}"
                        f"调度类型：{schedule_type}\n"
                        f"调度表达式：{schedule_value}\n\n"
                        f"{prompt}"
                    )
                )
            ],
            timestamp=datetime.now().astimezone(),
            files=[],
        )
        runner = await AgentRunner.create(session_id)
        content = await runner.run_once(message, background=True)
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("Scheduled Agent task completed without model output")
    except Exception:
        logger.exception(
            "Scheduled Agent task failed job_id={} session_id={}",
            job_id,
            session_id,
        )
        content = BACKGROUND_TASK_FAILURE_MESSAGE

    await BACKGROUND_MESSAGE_QUEUES[BackgroundMessageChannel.FEISHU].put(
        FeishuBackgroundMessage(open_id=open_id, content=content),
    )
