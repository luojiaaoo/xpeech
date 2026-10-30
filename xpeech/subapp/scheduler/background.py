from __future__ import annotations

import asyncio
import random
from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock
from uuid import uuid4

import httpx
from apscheduler.job import Job
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger
from loguru import logger

from ...agent.server.schema import BackgroundMessageChannel, FeishuBackgroundMessage
from ...agent.tools.schedule.schema import (
    ScheduledExecution,
    ScheduledExecutionResult,
    ScheduledTask,
)
from ...config.settings import settings
from ...utils.internal_jwt import create_internal_access_token

BACKGROUND_TASK_FAILURE_MESSAGE = "定时任务执行失败，请稍后重试。"
EXECUTION_TIMEOUT_SECONDS = 3600.0
MAX_BACKGROUND_TASK_DELAY_SECONDS = 3 * 60
MAX_SCHEDULED_TASKS_PER_USER = 4


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
        else settings.path.session_schedule_path
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


def _migrate_legacy_channel_kwargs(scheduler: AsyncIOScheduler) -> None:
    """把旧任务 session_metadata 里的渠道名迁移到 job kwargs 的 channel 字段。

    channel 成为独立字段之前创建的任务把渠道名放在 session_metadata["channel"]，
    kwargs 里没有 channel，迁移后才能列出渠道并在触发时传给执行函数。
    """
    for job in scheduler.get_jobs():
        kwargs = dict(job.kwargs)
        metadata = dict(kwargs.get("session_metadata") or {})
        legacy_channel = metadata.pop("channel", None)
        if not legacy_channel:
            continue
        kwargs["channel"] = kwargs.get("channel") or legacy_channel
        kwargs["session_metadata"] = metadata
        scheduler.modify_job(job.id, kwargs=kwargs)
        logger.info(
            "Migrated legacy scheduled job job_id={} channel={}",
            job.id,
            kwargs["channel"],
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
        # TODO: 所有环境的 schedule.db 都用新代码启动过一次（启动日志不再出现
        #  "Migrated legacy scheduled job"）后，删除 _migrate_legacy_channel_kwargs 及此调用。
        _migrate_legacy_channel_kwargs(scheduler)
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
    user_id: str,
    sender_name: str,
    channel: str,
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
        user_job_count = sum(
            job.kwargs.get("user_id") == user_id
            for job in scheduler.get_jobs()
        )
        if user_job_count >= MAX_SCHEDULED_TASKS_PER_USER:
            raise ValueError(
                "Scheduled task limit reached for this user "
                f"(maximum {MAX_SCHEDULED_TASKS_PER_USER})"
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
                "user_id": user_id,
                "sender_name": sender_name,
                "channel": channel,
                "session_metadata": dict(session_metadata),
                "schedule_type": schedule_type,
                "schedule_value": schedule_value,
            },
            coalesce=True,
            max_instances=1,
            misfire_grace_time=10,
            replace_existing=False,
        )


def list_scheduled_tasks(user_id: str) -> list[ScheduledTask]:
    """List all active scheduled tasks owned by a user, regardless of channel."""
    tasks = []
    for job in get_background_scheduler().get_jobs():
        if job.kwargs.get("user_id") != user_id:
            continue
        tasks.append(
            ScheduledTask(
                job_id=job.id,
                channel=str(job.kwargs.get("channel", "unknown")),
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


def cancel_scheduled_task(*, user_id: str, job_id: str) -> None:
    """Remove a job only when it belongs to the requesting user."""
    scheduler = get_background_scheduler()
    job = scheduler.get_job(job_id)
    if job is None or job.kwargs.get("user_id") != user_id:
        raise ValueError("Scheduled task is not available for this user")
    scheduler.remove_job(job_id)


async def run_feishu_background_task(
    *,
    job_id: str,
    prompt: str,
    user_id: str,
    sender_name: str,
    channel: str,
    session_metadata: dict[str, str],
    schedule_type: str,
    schedule_value: str,
) -> None:
    """Trigger the user-affine backend and enqueue its user-facing result.

    HTTP calls are not retried: a lost response may follow a successful run.
    """

    delay_seconds = random.uniform(0, MAX_BACKGROUND_TASK_DELAY_SECONDS)
    logger.info(
        "Feishu scheduled Agent task waiting job_id={} delay_seconds={:.2f}",
        job_id,
        delay_seconds,
    )
    await asyncio.sleep(delay_seconds)

    open_id = session_metadata["open_id"]
    from .app import app

    try:
        request = ScheduledExecution(
            job_id=job_id,
            prompt=prompt,
            sender_name=sender_name,
            channel=channel,
            session_metadata=session_metadata,
            schedule_type=schedule_type,
            schedule_value=schedule_value,
        )
        async with httpx.AsyncClient(timeout=EXECUTION_TIMEOUT_SECONDS) as client:
            response = await client.post(
                f"{app.state.backend_url}/internal/scheduled-run",
                headers={
                    "Authorization": f"Bearer {create_internal_access_token()}",
                    "x-user-id": user_id,
                },
                json=request.model_dump(mode="json"),
            )
            response.raise_for_status()
            result = ScheduledExecutionResult.model_validate(response.json())
    except (httpx.HTTPError, ValueError):
        logger.exception(
            "Scheduled Agent task failed job_id={} user_id={}",
            job_id,
            user_id,
        )
        result = ScheduledExecutionResult(
            content=BACKGROUND_TASK_FAILURE_MESSAGE,
            schedule_type=schedule_type,
            schedule_value=schedule_value,
        )

    await BACKGROUND_MESSAGE_QUEUES[BackgroundMessageChannel.FEISHU].put(
        FeishuBackgroundMessage(
            open_id=open_id,
            content=result.content,
            schedule_type=result.schedule_type,
            schedule_value=result.schedule_value,
        ),
    )
