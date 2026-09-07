import json
from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, Field, field_validator, model_validator

from ...utils.helper import ensure_async
from ..background import (
    cancel_scheduled_task,
    list_scheduled_tasks,
    schedule_feishu_task,
)


def _require_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


class FeishuScheduleArgs(BaseModel):
    """飞书后台定时 Agent 任务参数。

    run_at 和 cron 只能有一个有效值。
    """

    prompt: Annotated[
        str,
        Field(description="定时执行时交给 Agent 的完整任务提示词，不能为空"),
    ]
    run_at: Annotated[
        datetime | None,
        Field(
            description=(
                "单次任务的 ISO 8601 执行时间；可以不带时区，"
                "不带时区时按服务端系统本地时区解释"
            )
        ),
    ] = None
    cron: Annotated[
        str | None,
        Field(description="周期任务的标准 5 段 cron 表达式"),
    ] = None

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, value: str) -> str:
        return _require_text(value, "prompt")

    @field_validator("run_at", "cron", mode="before")
    @classmethod
    def normalize_blank_trigger(cls, value: object) -> object | None:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @model_validator(mode="after")
    def validate_trigger_choice(self):
        if (self.run_at is None) == (self.cron is None):
            raise ValueError("exactly one of run_at and cron must have a value")
        return self


class FeishuScheduleCancelArgs(BaseModel):
    job_id: Annotated[str, Field(description="要取消的定时任务 ID")]

    @field_validator("job_id")
    @classmethod
    def validate_job_id(cls, value: str) -> str:
        return _require_text(value, "job_id")


async def feishu_schedule(
    args: FeishuScheduleArgs,
    session_metadata: dict[str, str] | None,
    sender_name: str | None,
    session_id: str | None,
) -> str:
    """创建飞书后台定时 Agent 任务。

    Args:
        prompt: 定时执行时交给 Agent 的完整任务提示词，不能为空。
        run_at: 单次任务的 ISO 8601 执行时间，与 cron 二选一。
            可以不带时区，不带时区时按服务端系统本地时区解释。
            示例："2026-09-07T09:00:00+08:00" 或 "2026-09-07T09:00:00"
        cron: 周期任务的标准 5 段 cron 表达式，与 run_at 二选一。
            示例："0 9 * * *"

    Returns:
        包含 job_id 和 next_run_time 的字典。

    Examples:
        单次任务，明天早上9点提醒开会::

            >>> feishu_schedule(
            ...     prompt="提醒用户开会",
            ...     run_at="2026-09-08T09:00:00+08:00"
            ... )
            {"job_id": "abc123", "next_run_time": "2026-09-08T09:00:00+08:00"}

        周期任务，每天早上9点发笑话::

            >>> feishu_schedule(
            ...     prompt="给用户发一个笑话",
            ...     cron="0 9 * * *"
            ... )
            {"job_id": "def456", "next_run_time": "2026-09-08T09:00:00+08:00"}

    Note:
        run_at 和 cron 只能有一个有效值。
    """
    if not session_metadata or session_metadata.get("channel") != "feishu":
        raise ValueError("feishu_schedule is only available for Feishu sessions")

    open_id = session_metadata.get("open_id")
    _require_text(open_id, "session_metadata.open_id")
    validated_session_id = _require_text(session_id, "session_id")
    validated_sender_name = _require_text(sender_name, "sender_name")

    job = await ensure_async(schedule_feishu_task)(
        prompt=args.prompt,
        run_at=args.run_at,
        cron=args.cron,
        session_id=validated_session_id,
        sender_name=validated_sender_name,
        session_metadata=session_metadata,
    )
    return json.dumps(
        {
            "job_id": job.id,
            "next_run_time": job.next_run_time.isoformat() if job.next_run_time else None,
        },
        ensure_ascii=False,
    )


async def feishu_schedule_list(session_id: str | None) -> str:
    """列出当前会话所有渠道仍有效的后台定时任务。"""
    validated_session_id = _require_text(session_id, "session_id")
    tasks = await ensure_async(list_scheduled_tasks)(validated_session_id)
    return json.dumps(
        [task.model_dump() for task in tasks],
        ensure_ascii=False,
    )


async def feishu_schedule_cancel(
    args: FeishuScheduleCancelArgs,
    session_id: str | None,
) -> str:
    """取消当前会话拥有的后台定时任务。"""
    validated_session_id = _require_text(session_id, "session_id")
    await ensure_async(cancel_scheduled_task)(
        session_id=validated_session_id,
        job_id=args.job_id,
    )
    return f"Scheduled task cancelled successfully: {args.job_id}"
