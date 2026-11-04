"""Build the OpenSandbox shell tool and format command results."""

import platform
from collections.abc import Callable
from pathlib import Path
from textwrap import dedent

from ....config.settings import OpenSandboxConfig, settings
from .opensandbox import OpenSandboxExecutor
from .schema import DEFAULT_EXEC_TIMEOUT_MS, ShellArgs

SHELL_BLOCK_RULES: dict[Callable[[str], bool], str] = {}


def _guard_command(command: str) -> str:
    cmd = command.strip()
    if not cmd:
        raise ValueError("Shell command must not be empty")
    for check, message in SHELL_BLOCK_RULES.items():
        if check(cmd):
            raise RuntimeError(f"Command blocked by safety guard: {message}")
    return cmd


def _format_command_output(
    stdout: bytes,
    stderr: bytes,
    returncode: int | None,
    *,
    timed_out: bool = False,
    timeout_ms: int = DEFAULT_EXEC_TIMEOUT_MS,
) -> str:
    parts = []
    if timed_out:
        parts.append(f"ERROR: Command timed out after {timeout_ms / 1000:g}s (process terminated)")
    if stdout:
        parts.append(stdout.decode("utf-8", errors="replace"))
    if stderr and stderr.strip():
        parts.append(f"STDERR:\n{stderr.decode('utf-8', errors='replace')}")
    parts.append(f"Exit code: {returncode}")
    return "\n".join(parts)


def build_shell_tools(workspace: Path, config: OpenSandboxConfig | None = None):
    if platform.system() != "Linux":
        raise RuntimeError("OpenSandbox shell requires a single Linux Docker host")
    workspace = workspace.resolve()
    if not workspace.is_dir():
        raise ValueError(f"Invalid workspace: {workspace}")
    executor = OpenSandboxExecutor(config or settings.tool.shell)

    async def shell(args: ShellArgs) -> str:
        command = _guard_command(args.command)
        result = await executor.run(command, workspace, timeout_ms=args.timeout_ms)
        return _format_command_output(
            result.stdout, result.stderr, result.returncode,
            timed_out=result.timed_out, timeout_ms=args.timeout_ms,
        )

    shell.__doc__ = dedent(
        """
        Execute a bash command in the current workspace inside OpenSandbox.

        The current user's workspace is mounted read-write and shared with
        file tools. Other users' workspaces are not mounted. Built-in skills
        and managed HOME configuration files are read-only. Python, uv, Node,
        npm and lark-cli are installed in the execution image.

        Workspace files persist across calls; background processes do not.
        Commands may queue when capacity is busy. They time out after 120
        seconds by default; timeout_ms can request up to 240 seconds.
        On timeout, partial output is returned.

        Miniforge provides Python, Node, uv, conda and mamba. Python/pip can
        be called directly; python and pip default to the current user's
        persistent Python environment. HOME is <workspace>/home.
        Examples: python script.py; pip install package; node script.js.
        """
    ).lstrip()
    return shell
