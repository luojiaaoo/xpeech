from datetime import datetime
from typing import Annotated

from apscheduler.triggers.cron import CronTrigger
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def _require_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def parse_schedule_cron(value: str, *, timezone=None) -> CronTrigger:
    fields = value.split()
    if len(fields) != 5:
        raise ValueError(
            "cron must have exactly 5 fields: minute hour day month day_of_week; "
            "e.g. '0 9 * * *' (no seconds)"
        )
    if fields[0] not in {"0", "30", "0,30"}:
        raise ValueError(
            "cron minute field must be exactly '0', '30', or '0,30'; "
            "wildcards, steps, ranges, and other minutes are not allowed"
        )
    try:
        return CronTrigger.from_crontab(" ".join(fields), timezone=timezone)
    except ValueError as exc:
        raise ValueError(
            f"Invalid cron: {exc}. Use 5 fields; weekdays are mon-sun (0=Monday, 6=Sunday)"
        ) from exc


CRON_DESCRIPTION = (
    "周期任务：5 段 cron，依次为 分 时 日 月 星期，不含秒，按服务端系统本地时区执行。"
    "分钟字段仅允许 0、30 或 0,30，禁止 *、步长、范围及其他分钟值；每半小时写 0,30 * * * *。"
    "每天9点：0 9 * * *；工作日9点：0 9 * * mon-fri。"
    "星期推荐使用 mon-sun，数字 0=周一、6=周日，不支持7。"
)

RUN_AT_DESCRIPTION = (
    "单次任务：未来的 ISO 8601 日期和时间，例如 2099-01-01T09:00:00+08:00。"
    "请按当前日期计算，不能直接传‘明天9点’、纯日期或 Unix 时间戳。"
    "建议带时区偏移；不带时区按服务端系统本地时区解释。"
)


def validate_schedule_cron(value: str) -> str:
    parse_schedule_cron(value)
    return " ".join(value.split())


def validate_run_at_format(value: object) -> object:
    if isinstance(value, str):
        value = value.strip() or None
    if value is not None and not isinstance(value, datetime):
        if not isinstance(value, str) or not any(separator in value for separator in ("T", "t", " ")):
            raise ValueError(
                "run_at must be an ISO 8601 date and time, e.g. '2099-01-01T09:00:00+08:00'; "
                "not a date or Unix timestamp"
            )
    return value


class FeishuScheduleCancelArgs(BaseModel):
    job_id: Annotated[str, Field(description="先调用 feishu_schedule_list，使用返回的 job_id 原值；不要使用任务名称或自行编造 ID")]

    @field_validator("job_id")
    @classmethod
    def validate_job_id(cls, value: str) -> str:
        return _require_text(value, "job_id")


class _FeishuScheduleBaseArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: Annotated[
        str,
        Field(description="定时执行时交给 Agent 的完整任务提示词，不能为空"),
    ]

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, value: str) -> str:
        return _require_text(value, "prompt")


class FeishuScheduleOnceArgs(_FeishuScheduleBaseArgs):
    """创建单次任务。"""

    run_at: Annotated[datetime, Field(description=RUN_AT_DESCRIPTION)]

    @field_validator("run_at", mode="before")
    @classmethod
    def validate_run_at(cls, value: object) -> object:
        return validate_run_at_format(value)


class FeishuScheduleCronArgs(_FeishuScheduleBaseArgs):
    """创建周期任务。"""

    cron: Annotated[str, Field(description=CRON_DESCRIPTION)]

    @field_validator("cron")
    @classmethod
    def validate_cron(cls, value: str) -> str:
        return validate_schedule_cron(value)


class FeishuScheduleArgs(_FeishuScheduleBaseArgs):
    """调度接口的创建参数；两个时间字段都必填，且只能有一个非空值。"""

    run_at: Annotated[
        datetime | None,
        Field(description=(
            RUN_AT_DESCRIPTION + "此字段必填；周期任务传 JSON null，与 cron 必须且只能有一个非空值"
        )),
    ]
    cron: Annotated[
        str | None,
        Field(description=(
            CRON_DESCRIPTION + "此字段必填；单次任务传 JSON null，与 run_at 必须且只能有一个非空值"
        )),
    ]

    @field_validator("run_at", "cron", mode="before")
    @classmethod
    def normalize_blank_trigger(cls, value: object) -> object | None:
        if isinstance(value, str) and not value.strip():
            return None
        return value.strip() if isinstance(value, str) else value

    @field_validator("run_at", mode="before")
    @classmethod
    def validate_run_at(cls, value: object) -> object:
        return validate_run_at_format(value)

    @field_validator("cron")
    @classmethod
    def validate_cron(cls, value: str | None) -> str | None:
        if value is not None:
            return validate_schedule_cron(value)
        return value

    @model_validator(mode="after")
    def validate_trigger_choice(self):
        if (self.run_at is None) == (self.cron is None):
            raise ValueError(
                "Provide exactly one of run_at and cron: one-time task uses run_at with cron=null; "
                "recurring task uses cron with run_at=null"
            )
        return self
