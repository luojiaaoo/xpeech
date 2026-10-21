"""Safely execute configured command templates without an implicit shell."""

from __future__ import annotations

import asyncio
import re
import shlex
from collections.abc import Collection, Mapping
from contextlib import suppress
from typing import Literal

CommandTemplateErrorReason = Literal[
    "invalid",
    "empty",
    "missing_variable",
    "start",
    "timeout",
    "failed",
    "non_utf8",
]


class CommandTemplateError(RuntimeError):
    """A configuration or execution failure without command output or secrets."""

    def __init__(
        self,
        reason: CommandTemplateErrorReason,
        *,
        operation: str,
        variable: str | None = None,
        returncode: int | None = None,
    ) -> None:
        if reason == "invalid":
            message = f"{operation} is invalid"
        elif reason == "empty":
            message = f"{operation} cannot be empty"
        elif reason == "missing_variable":
            message = f"{operation} requires ${variable} or ${{{variable}}}"
        elif reason == "start":
            message = f"{operation} could not start"
        elif reason == "timeout":
            message = f"{operation} timed out"
        elif reason == "failed":
            message = f"{operation} failed with exit code {returncode}"
        else:
            message = f"{operation} must output UTF-8 text"
        super().__init__(message)
        self.reason = reason
        self.operation = operation
        self.variable = variable
        self.returncode = returncode


def _variable_pattern(name: str) -> re.Pattern[str]:
    escaped_name = re.escape(name)
    return re.compile(rf"\$(?:\{{{escaped_name}\}}|{escaped_name}(?![a-zA-Z0-9_]))")


async def _terminate_process(process: asyncio.subprocess.Process) -> None:
    if process.returncode is None:
        with suppress(ProcessLookupError):
            process.kill()
    with suppress(ProcessLookupError):
        await process.wait()


async def execute_command_template(
    command_template: str,
    variables: Mapping[str, str],
    *,
    operation: str,
    timeout: float,
    required_variables: Collection[str] = (),
) -> str:
    """Execute a command template and return stripped UTF-8 stdout.

    The template is split before variables are substituted, so each replacement
    remains within its configured argument. No shell is implicitly started.
    """
    try:
        command = shlex.split(command_template)
    except ValueError as exc:
        raise CommandTemplateError("invalid", operation=operation) from exc
    if not command:
        raise CommandTemplateError("empty", operation=operation)

    patterns = {name: _variable_pattern(name) for name in variables}
    for name in required_variables:
        pattern = patterns.get(name) or _variable_pattern(name)
        if not any(pattern.search(argument) for argument in command):
            raise CommandTemplateError(
                "missing_variable",
                operation=operation,
                variable=name,
            )

    for name, replacement in variables.items():
        pattern = patterns[name]
        command = [
            pattern.sub(lambda _match, value=replacement: value, argument)
            for argument in command
        ]

    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except OSError as exc:
        raise CommandTemplateError("start", operation=operation) from exc

    try:
        stdout, _stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
    except TimeoutError as exc:
        await _terminate_process(process)
        raise CommandTemplateError("timeout", operation=operation) from exc
    except asyncio.CancelledError:
        await _terminate_process(process)
        raise

    if process.returncode != 0:
        raise CommandTemplateError(
            "failed",
            operation=operation,
            returncode=process.returncode,
        )
    try:
        return stdout.decode("utf-8").strip()
    except UnicodeDecodeError as exc:
        raise CommandTemplateError("non_utf8", operation=operation) from exc
