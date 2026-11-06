"""Results and default timeout for sandbox command execution."""

from dataclasses import dataclass

DEFAULT_EXEC_TIMEOUT_MS = 120_000


@dataclass(frozen=True)
class CommandResult:
    stdout: bytes
    stderr: bytes
    returncode: int | None
    timed_out: bool = False
