"""Structured contracts for the Feishu channel task management API."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .tool_args import FeishuScheduleArgs


class ScheduledTask(BaseModel):
    job_id: str
    prompt: str
    schedule_type: Literal["run_at", "cron"]
    schedule_value: str
    next_run_time: str | None


class FeishuScheduleCreateRequest(FeishuScheduleArgs):
    model_config = ConfigDict(extra="forbid")
    sender_name: str = Field(min_length=1)
    session_metadata: dict[str, str]

    @model_validator(mode="after")
    def validate_delivery_context(self):
        if not self.sender_name.strip():
            raise ValueError("sender_name must be a non-empty string")
        if not self.session_metadata.get("open_id", "").strip():
            raise ValueError("session_metadata.open_id must be a non-empty string")
        return self


class ScheduleResponse(BaseModel):
    task: ScheduledTask


class ScheduleListResponse(BaseModel):
    tasks: list[ScheduledTask]


class ScheduleCancelResponse(BaseModel):
    job_id: str
    cancelled: Literal[True] = True
