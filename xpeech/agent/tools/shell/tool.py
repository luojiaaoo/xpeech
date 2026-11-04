"""Build and execute the sandboxed shell tool."""

import asyncio
import os
import platform
import shlex
import shutil
import signal
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path
from textwrap import dedent

from loguru import logger

from .. import sandbox
from ..helper import is_direct_python_pip_exec
from .schema import DEFAULT_EXEC_TIMEOUT_MS, ShellArgs, ShellCommandResult

SHELL_BLOCK_RULES: dict[Callable[[str], bool], str] = {
    is_direct_python_pip_exec: (
        "run Python through 'uv run python/pip' instead of calling 'python/pip' directly."
    ),
}


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
    wrapped_command = shlex.join(sandbox.wrap_sandbox_command(command, workspace))
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


def build_shell_tools(workspace: Path):
    if platform.system() != "Linux":
        raise RuntimeError("Shell tool requires Linux with bubblewrap; Windows and other platforms are not supported.")
    missing = [name for name in (*sandbox.SANDBOX_EXECUTABLES, "uv") if shutil.which(name) is None]
    if missing:
        raise RuntimeError(f"Shell tool requires executable(s) on PATH: {', '.join(missing)}")
    workspace = workspace.resolve()
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
