"""Shared HTTP contracts for scheduled Agent execution and task management."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ScheduledTask(BaseModel):
    """Serializable description of an active background task."""

    job_id: str
    channel: str
    prompt: str
    schedule_type: Literal["run_at", "cron"]
    schedule_value: str
    next_run_time: str | None


class ScheduledExecution(BaseModel):
    """A scheduled run sent to the backend through the user-affinity gateway."""

    job_id: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    sender_name: str = Field(min_length=1)
    session_metadata: dict[str, str]
    schedule_type: Literal["run_at", "cron"]
    schedule_value: str


class ScheduledExecutionResult(BaseModel):
    content: str = Field(min_length=1)
    schedule_type: Literal["run_at", "cron"]
    schedule_value: str

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Scheduled Agent task completed without model output")
        return value
