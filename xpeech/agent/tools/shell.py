import asyncio
import json
import os
import platform
import shutil
import signal
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent

from loguru import logger
from pydantic import BaseModel, Field

from . import sandbox
from .helper import is_direct_python_pip_exec, is_power_command

DEFAULT_EXEC_TIMEOUT_MS = 120_000
MAX_EXEC_TIMEOUT_MS = 240_000
SHELL_CHECK_TIMEOUT = 5.0
SHELL_BLOCK_RULES: dict[Callable[[str], bool], str] = {
    is_power_command: "shutdown/reboot/poweroff are prohibited.",
    is_direct_python_pip_exec: (
        "run Python through 'uv run python/pip' instead of calling 'python/pip' directly."
    ),
}


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


async def _shell_requires_confirmation(payload: str) -> bool:
    """用 dcg 检查 Bash；仅明确允许的命令免确认，检查失败时退回用户确认。"""
    command = json.loads(payload).get("command")
    if not isinstance(command, str) or not command.strip():
        return True

    process = None
    try:
        async with asyncio.timeout(SHELL_CHECK_TIMEOUT):
            process = await asyncio.create_subprocess_exec(
                "dcg",
                "test",
                "--stdin",
                "--dialect",
                "posix",
                "--with-packs",
                "core.filesystem,core.git,system",
                "--format",
                "json",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            stdout, _ = await process.communicate(command.encode("utf-8"))
        if process.returncode != 0:
            return True
        result = json.loads(stdout)
        return not (isinstance(result, dict) and result.get("decision") == "allow")
    except (OSError, TimeoutError, ValueError) as exc:
        logger.warning("Shell dcg check unavailable; requiring user confirmation: {}", exc)
        return True
    finally:
        if process is not None and process.returncode is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
            await process.wait()


async def _stop_process(process: asyncio.subprocess.Process) -> None:
    """Stop the isolated shell process group and reap the shell process."""
    if process.returncode is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        try:
            process.kill()
        except ProcessLookupError:
            pass
    try:
        await asyncio.wait_for(process.wait(), timeout=3.0)
    except TimeoutError:
        try:
            process.kill()
        except ProcessLookupError:
            pass
        await process.wait()


def _guard_command(command: str) -> str:
    """Apply explicit block rules; filesystem access is enforced by bubblewrap."""
    cmd = command.strip()

    for check, message in SHELL_BLOCK_RULES.items():
        if check(cmd):
            raise RuntimeError(f"Command blocked by safety guard: {message}")

    return cmd


async def _run_wrapped_command(
    command: str,
    workspace: Path,
    timeout_ms: int = DEFAULT_EXEC_TIMEOUT_MS,
) -> ShellCommandResult:
    wrapped_command = sandbox.wrap_command(command, workspace)
    logger.info(f"Running wrapped command: {wrapped_command}")
    with tempfile.TemporaryFile(mode="w+b") as stdout_f, tempfile.TemporaryFile(mode="w+b") as stderr_f:
        process = await asyncio.create_subprocess_exec(
            "bash",
            "-l",
            "-c",
            wrapped_command,
            stdin=subprocess.DEVNULL,
            stdout=stdout_f,
            stderr=stderr_f,
            cwd=workspace,
            start_new_session=True,
        )
        timed_out = False
        try:
            await asyncio.wait_for(process.wait(), timeout=timeout_ms / 1000)
        except TimeoutError:
            timed_out = True
            logger.error(
                "Shell command timed out after {}ms; terminating process group pid={}",
                timeout_ms,
                process.pid,
            )
            await _stop_process(process)
        except asyncio.CancelledError:
            await _stop_process(process)
            raise

        stdout_f.seek(0)
        stderr_f.seek(0)
        return ShellCommandResult(stdout_f.read(), stderr_f.read(), process.returncode, timed_out)


def _format_command_output(
    stdout: bytes,
    stderr: bytes,
    returncode: int | None,
    *,
    timed_out: bool = False,
    timeout_ms: int = DEFAULT_EXEC_TIMEOUT_MS,
) -> str:
    output_parts = []

    if timed_out:
        output_parts.append(f"ERROR: Command timed out after {timeout_ms / 1000:g}s (process terminated)")

    if stdout:
        output_parts.append(stdout.decode("utf-8", errors="replace"))

    if stderr:
        stderr_text = stderr.decode("utf-8", errors="replace")
        if stderr_text.strip():
            output_parts.append(f"STDERR:\n{stderr_text}")

    output_parts.append(f"Exit code: {returncode}")
    return "\n".join(output_parts) if output_parts else "(no output)"


async def _ensure_workspace_uv_venv(workspace: Path) -> None:
    if (workspace / ".venv").exists():
        return
    result = await _run_wrapped_command("uv venv .venv", workspace)
    if result.returncode != 0:
        output = _format_command_output(
            result.stdout, result.stderr, result.returncode, timed_out=result.timed_out
        )
        raise RuntimeError(f"Failed to initialize workspace Python environment with `uv venv .venv`.\n{output}")


def build_shell_tools(workspace: str | Path):
    if platform.system() != "Linux":
        raise RuntimeError("Shell tool requires Linux with bubblewrap; Windows and other platforms are not supported.")
    missing = [name for name in ("bwrap", "bash", "uv") if shutil.which(name) is None]
    if missing:
        raise RuntimeError(f"Shell tool requires executable(s) on PATH: {', '.join(missing)}")
    workspace = Path(workspace).expanduser().resolve()
    if not workspace.exists() or not workspace.is_dir():
        raise ValueError(f"Invalid workspace: {workspace}")

    async def shell(args: ShellArgs) -> str:
        command = _guard_command(args.command)
        await _ensure_workspace_uv_venv(workspace)
        result = await _run_wrapped_command(command, workspace, timeout_ms=args.timeout_ms)
        return _format_command_output(
            result.stdout,
            result.stderr,
            result.returncode,
            timed_out=result.timed_out,
            timeout_ms=args.timeout_ms,
        )

    shell.__doc__ = dedent(
        """
            Execute a bash command in the current workspace and return the output.

            Commands run on Linux inside a bubblewrap sandbox. The current
            workspace is isolated from other workspaces.

            The workspace is writable; system paths and built-in skills are
            read-only. Workspace files persist across calls; background
            processes do not.

            Commands time out after 120 seconds by default; timeout_ms can
            request up to 240 seconds. On timeout, partial output is returned.

            Run Python and pip through uv:
            uv run python script.py
            uv pip install package
        """
    ).lstrip()
    return shell
