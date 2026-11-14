from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, Field, field_validator, model_validator


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
        Field(description=("单次任务的 ISO 8601 执行时间；可以不带时区，不带时区时按服务端系统本地时区解释")),
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
