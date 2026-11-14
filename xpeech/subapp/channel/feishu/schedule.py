from __future__ import annotations

import asyncio
import json
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from threading import Lock
from typing import Any
from urllib.parse import quote
from uuid import uuid4

import httpx
from apscheduler.job import Job
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger
from loguru import logger
from sqlalchemy.engine import Engine

from ....agent.tools.schedule.schema import ScheduledTask
from ....utils.channel_jwt import UserTokenParams, create_token
from .cards import build_feishu_background_task_card
from .channel import XpeechFeishuChannel
from .delivery import ensure_send_success

SCHEDULED_TASK_PROMPT_PREFIX = (
    "## 定时任务运行说明\n\n"
    "本次运行由后台定时任务触发，不是用户实时发起的新对话。\n"
    "请完整执行下方的原始定时任务。完成任务后、给出最终回复前，"
    "请将本次定时任务的执行记录追加到 `memory/HISTORY.md`，"
    "不要覆盖已有内容。记录必须以 `[YYYY-MM-DD HH:MM]` 开头，并包含任务内容和执行结果摘要。\n\n"
    "最终回复会直接发送给用户，必须呈现原始任务要求的实际结果。"
    "不要提及后台调度、`memory/HISTORY.md`、"
    "执行记录或追加动作，更不能仅回复“执行记录已追加”等操作确认。\n\n"
    "## 原始定时任务提示词\n"
)


BACKGROUND_TASK_FAILURE_MESSAGE = "定时任务执行失败，请稍后重试。"
EXECUTION_TIMEOUT_SECONDS = 3600.0
MAX_BACKGROUND_TASK_DELAY_SECONDS = 3 * 60
MAX_SCHEDULED_TASKS_PER_USER = 4


@dataclass(frozen=True)
class FeishuScheduleRuntime:
    backend_url: str
    client: httpx.AsyncClient
    channel: XpeechFeishuChannel


_scheduler: AsyncIOScheduler | None = None
_runtime: FeishuScheduleRuntime | None = None
_schedule_lock = Lock()
_running_tasks: set[asyncio.Task[Any]] = set()


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
    engine: Engine,
    *,
    runtime: FeishuScheduleRuntime,
    now: datetime | None = None,
    event_loop: asyncio.AbstractEventLoop | None = None,
) -> AsyncIOScheduler:
    """Start the persistent scheduler without replaying runs missed during downtime."""
    global _scheduler, _runtime

    if _scheduler is not None and _scheduler.running:
        return _scheduler

    jobstore = SQLAlchemyJobStore(engine=engine, tablename="feishu_scheduled_jobs")
    scheduler = AsyncIOScheduler(
        jobstores={"default": jobstore},
        event_loop=event_loop,
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
        _runtime = runtime
        scheduler.resume()
    except Exception:
        scheduler.shutdown(wait=False)
        raise

    _scheduler = scheduler
    logger.info("Background scheduler started")
    return scheduler


async def stop_background_scheduler() -> None:
    """Stop scheduling and cancel running calls before closing their dependencies."""
    global _scheduler, _runtime
    scheduler, _scheduler = _scheduler, None
    if scheduler is not None and scheduler.running:
        scheduler.shutdown(wait=False)
        # APScheduler schedules shutdown on the event loop.
        await asyncio.sleep(0)
    tasks = list(_running_tasks)
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    _runtime = None


def _normalize_run_at(run_at: datetime, timezone) -> datetime:
    normalized = run_at.replace(tzinfo=timezone) if run_at.tzinfo is None else run_at.astimezone(timezone)
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
        user_job_count = sum(job.kwargs.get("user_id") == user_id for job in scheduler.get_jobs())
        if user_job_count >= MAX_SCHEDULED_TASKS_PER_USER:
            raise ValueError(f"Scheduled task limit reached for this user (maximum {MAX_SCHEDULED_TASKS_PER_USER})")

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
                "session_metadata": dict(session_metadata),
                "schedule_type": schedule_type,
                "schedule_value": schedule_value,
            },
            coalesce=True,
            max_instances=1,
            misfire_grace_time=10,
            replace_existing=False,
        )


def describe_scheduled_task(job: Job) -> ScheduledTask:
    """Serialize the job itself, including a one-time job already dispatched."""
    return ScheduledTask(
        job_id=job.id,
        prompt=job.kwargs["prompt"],
        schedule_type=job.kwargs["schedule_type"],
        schedule_value=job.kwargs["schedule_value"],
        next_run_time=job.next_run_time.isoformat() if job.next_run_time else None,
    )


def list_scheduled_tasks(user_id: str) -> list[ScheduledTask]:
    """List active Feishu tasks owned by the requesting user."""
    tasks = [
        describe_scheduled_task(job)
        for job in get_background_scheduler().get_jobs()
        if job.kwargs.get("user_id") == user_id
    ]
    return sorted(tasks, key=lambda task: (task.next_run_time is None, task.next_run_time or "", task.job_id))


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
    session_metadata: dict[str, str],
    schedule_type: str,
    schedule_value: str,
) -> None:
    """Run through the standard synchronous chat endpoint, then deliver locally."""
    runtime = _runtime
    if runtime is None:
        raise RuntimeError("Feishu scheduler runtime is not available")
    task = asyncio.current_task()
    _running_tasks.add(task)
    try:
        await asyncio.sleep(random.uniform(0, MAX_BACKGROUND_TASK_DELAY_SECONDS))
        try:
            prompt_text = (
                f"{SCHEDULED_TASK_PROMPT_PREFIX}调度类型：{schedule_type}\n调度表达式：{schedule_value}\n\n{prompt}"
            )
            response = await runtime.client.post(
                f"{runtime.backend_url}/chat_sync",
                headers={
                    "Authorization": f"Bearer {create_token(UserTokenParams('feishu', user_id=user_id))}",
                    "x-sender-name": quote(sender_name, safe=""),
                },
                data={
                    "content": json.dumps([{"text": prompt_text}], ensure_ascii=False),
                    "session_metadata": json.dumps(
                        {**session_metadata, "source": "scheduled_task"},
                        ensure_ascii=False,
                    ),
                },
                timeout=EXECUTION_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            events = response.json()
            if not isinstance(events, list) or any(not isinstance(event, dict) for event in events):
                raise ValueError("Invalid synchronous chat response")
            if any(event.get("event") == "error" for event in events):
                raise ValueError("Scheduled chat returned an error event")
            content = next(
                (
                    event["context"]
                    for event in reversed(events)
                    if event.get("event") == "assistant"
                    and isinstance(event.get("context"), str)
                    and event["context"].strip()
                ),
                None,
            )
            if content is None:
                raise ValueError("Scheduled chat completed without model output")
        except (httpx.HTTPError, ValueError):
            logger.exception("Scheduled execution failed job_id={} user_id={}", job_id, user_id)
            content = BACKGROUND_TASK_FAILURE_MESSAGE
        try:
            result = await runtime.channel.send(
                to=session_metadata["open_id"],
                message={
                    "card": build_feishu_background_task_card(
                        content,
                        schedule_type=schedule_type,
                        schedule_value=schedule_value,
                    )
                },
            )
            ensure_send_success(result, operation="send scheduled result")
        except Exception:  # noqa: BLE001 - delivery failure must not stop future jobs.
            logger.exception("Scheduled delivery failed job_id={} user_id={}", job_id, user_id)
    finally:
        _running_tasks.discard(task)
