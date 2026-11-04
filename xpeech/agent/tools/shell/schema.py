"""Shell tool arguments and execution results."""

from dataclasses import dataclass

from pydantic import BaseModel, Field

DEFAULT_EXEC_TIMEOUT_MS = 120_000
MAX_EXEC_TIMEOUT_MS = 240_000


class ShellArgs(BaseModel):
    command: str = Field(description="Bash command to execute")
    timeout_ms: int = Field(
        default=DEFAULT_EXEC_TIMEOUT_MS,
        gt=0,
        le=MAX_EXEC_TIMEOUT_MS,
        description="Command execution timeout in milliseconds (maximum 240000).",
    )


@dataclass(frozen=True)
class ShellCommandResult:
    stdout: bytes
    stderr: bytes
    returncode: int | None
    timed_out: bool = False
