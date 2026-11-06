"""Shell tool definition and output presentation."""

from pathlib import Path
from textwrap import dedent

from pydantic import BaseModel, Field

from ...utils.security import sandbox
from ..skills.skill import SkillsLoader

DEFAULT_EXEC_TIMEOUT_MS = sandbox.DEFAULT_EXEC_TIMEOUT_MS
MAX_EXEC_TIMEOUT_MS = 240_000


class ShellArgs(BaseModel):
    command: str = Field(description="Bash command to execute")
    timeout_ms: int = Field(
        default=DEFAULT_EXEC_TIMEOUT_MS,
        gt=0,
        le=MAX_EXEC_TIMEOUT_MS,
        description="Command execution timeout in milliseconds (maximum 240000).",
    )


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


def build_shell_tools(workspace: Path, *, skills_loader: SkillsLoader):
    executor = sandbox.SandboxExecutor(workspace, skills_loader=skills_loader)

    async def shell(args: ShellArgs) -> str:
        result = await executor.run(args.command, timeout_ms=args.timeout_ms)
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
            After {executor.idle_timeout:g} seconds without active shell requests,
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
