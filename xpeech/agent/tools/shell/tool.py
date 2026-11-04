"""Build and execute the sandboxed shell tool."""

import platform
import shutil
from collections.abc import Callable
from pathlib import Path
from textwrap import dedent

from loguru import logger

from .. import sandbox
from ..helper import is_direct_python_pip_exec
from .schema import DEFAULT_EXEC_TIMEOUT_MS, ShellArgs, ShellCommandResult
from .session import sandbox_manager

SHELL_BLOCK_RULES: dict[Callable[[str], bool], str] = {
    is_direct_python_pip_exec: (
        "run Python through 'uv run python/pip' instead of calling 'python/pip' directly."
    ),
}


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
    result = await sandbox_manager.run(command, workspace, timeout_ms)
    if result.timed_out:
        logger.error("Shell command timed out after {}ms; terminating command process group", timeout_ms)
    return result


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
    # All conversations share this lock. Check inside it so another call
    # cannot use a .venv directory while uv is still initializing it.
    async with sandbox_manager.use(workspace) as session, session.initialization_lock:
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
    missing = [name for name in (*sandbox.NETWORK_SANDBOX_EXECUTABLES, "uv") if shutil.which(name) is None]
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
        f"""
            Execute a bash command in the current workspace and return the output.

            Commands run on Linux inside a bubblewrap sandbox. The current
            workspace is isolated from other workspaces.

            The workspace is writable; system paths and built-in skills are
            read-only. Calls in this user workspace reuse the same sandbox,
            including /tmp, network and background processes. Multiple commands
            can run concurrently, with separate output, timeouts and cancellation.
            After {sandbox_manager.idle_timeout:g} seconds without active shell requests,
            the sandbox and all its processes are stopped. Workspace files remain on disk. Each call
            starts a new bash in the workspace; shell variables and cd do not
            carry over.

            Commands time out after 120 seconds by default; timeout_ms can
            request up to 240 seconds. On timeout, partial output is returned.

            Run Python and pip through uv:
            uv run python script.py
            uv pip install package
        """
    ).lstrip()
    return shell
