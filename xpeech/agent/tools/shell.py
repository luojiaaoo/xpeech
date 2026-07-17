from pathlib import Path
from textwrap import dedent

import asyncio
import os
import platform
import re
import shlex
import shutil
import subprocess
import tempfile

from loguru import logger
from pydantic import BaseModel, Field

from ...utils.helper import ensure_path
from ..server.context import get_session_id
from . import sandbox
from ...config.settings import settings
from .helper import safe_resolve_workspace_path, is_direct_python_pip_exec

EXEC_TIMEOUT = 60 * 2
_MAX_OUTPUT = 10000
DENY_PATTERNS = [
    r"\brm\s+-[rf]{1,2}\b",  # rm -r, rm -rf, rm -fr
    r"\bdel\s+/[fq]\b",  # del /f, del /q
    r"\brmdir\s+/s\b",  # rmdir /s
    r"(?:^|[;&|]\s*)format\b",  # format (as standalone command only)
    r"\b(mkfs|diskpart)\b",  # disk operations
    r"\bdd\s+if=",  # dd
    r">\s*/dev/sd",  # write to disk
    r"\b(shutdown|reboot|poweroff)\b",  # system power
    r":\(\)\s*\{.*\};\s*:",  # fork bomb
]
_WORKSPACE_BOUNDARY_NOTE = (
    "\n\nNote: this is a hard policy boundary, not a transient failure. "
    "Do NOT retry with shell tricks (symlinks, base64 piping, alternative "
    "tools). Do NOT use absolute paths (e.g., "
    "/usr/bin, /bin, /usr/local/bin) to bypass restrictions — these "
    "will also be intercepted. Always invoke commands directly by name "
    "(e.g., 'curl' instead of '/usr/bin/curl'). "
    "If the user genuinely needs this resource, tell them you cannot "
    "reach it under the current restrict policy"
)

_BENIGN_DEVICE_PATHS: frozenset[str] = frozenset(
    {
        "/dev/null",
        "/dev/zero",
        "/dev/full",
        "/dev/random",
        "/dev/urandom",
        "/dev/stdin",
        "/dev/stdout",
        "/dev/stderr",
        "/dev/tty",
    }
)


class ShellArgs(BaseModel):
    command: str = Field(description="Bash command to execute")


def _is_benign_device_path(path: str) -> bool:
    """Return True for kernel device files that should never be workspace-blocked."""
    if path in _BENIGN_DEVICE_PATHS:
        return True
    return path.startswith("/dev/fd/")


async def _stop_process(process: asyncio.subprocess.Process) -> None:
    """Stop and reap the shell process without managing its background children."""
    if process.returncode is not None:
        return
    try:
        process.terminate()
    except ProcessLookupError:
        return
    try:
        await asyncio.wait_for(process.wait(), timeout=3.0)
    except asyncio.TimeoutError:
        try:
            process.kill()
        except ProcessLookupError:
            return
        await process.wait()


def _extract_absolute_paths(command: str) -> list[str]:
    posix_paths = re.findall(r"(?:^|[\s|>'\"])(/[^\s\"'>;|<]+)", command)
    home_paths = re.findall(r"(?:^|[\s>'\"])(~[^\s\"'>;|<]*)", command)
    return posix_paths + home_paths


def _guard_command(command: str, workspace: str | Path) -> str:
    """Guard a command string from code injection attacks."""
    cmd = command.strip()
    lower = cmd.lower()

    # 正则判断
    for pattern in DENY_PATTERNS:
        if re.search(pattern, lower):
            raise RuntimeError(f"Command blocked by deny pattern filter: {cmd}")

    # 判断是否使用了 python 直接运行
    if is_direct_python_pip_exec(cmd):
        raise RuntimeError(
            "Command blocked by safety guard: run Python through 'uv run python/pip' instead of calling "
            "'python/pip' directly."
        )

    # 判断是否包含 ..\
    if "..\\" in cmd or "../" in cmd:
        raise RuntimeError(
            f"Command blocked by safety guard (path traversal ../ detected): {cmd}" + _WORKSPACE_BOUNDARY_NOTE
        )
    # 提取所有路径，判断是否在工作路径下
    for path in _extract_absolute_paths(cmd):
        try:
            expanded = os.path.expandvars(path.strip())
            if _is_benign_device_path(expanded):
                continue
            resolved = safe_resolve_workspace_path(
                expanded,
                workspace,
                include_builtin_skills_path=True,
            )
            if _is_benign_device_path(str(resolved)):
                continue
        except PermissionError:
            raise RuntimeError(
                f"Command blocked by safety guard (a path outside the workspace or built-in skills was detected.): {path}"
                + _WORKSPACE_BOUNDARY_NOTE
            ) from None
        except Exception:
            continue

    return cmd


_AGENT_BROWSER_PATTERN = re.compile(r"(?<![\w./-])agent-browser(?=\s|$)")
_NPM_INSTALL_PATTERN = re.compile(r"(?:^|&&|\|\||[;|\n])\s*npm\s+(?:install|i)\b[^;&|\n]*$")


def _inject_command_context(command: str, **context: object) -> tuple[str, dict[str, str]]:
    """Inject managed command arguments and sandbox environment variables."""
    env: dict[str, str] = {}

    if _AGENT_BROWSER_PATTERN.search(command):
        cdp_url = os.environ.get("CDP_URL", "").strip()
        if not cdp_url:
            raise RuntimeError("agent-browser requires the CDP_URL environment variable")
        session_id = context.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            raise RuntimeError("agent-browser requires a session ID in the current request context")
        quoted_cdp_url = shlex.quote(cdp_url)
        quoted_session_id = shlex.quote(session_id)
        args_suffix = f" --cdp {quoted_cdp_url} --session {quoted_session_id}"

        def inject(match: re.Match[str]) -> str:
            if _NPM_INSTALL_PATTERN.search(command[: match.start()]):
                return match.group(0)
            if re.match(r"\s+skills\b", command[match.end() :]):
                return match.group(0)
            return match.group(0) + args_suffix

        command = _AGENT_BROWSER_PATTERN.sub(inject, command)
        env["AGENT_BROWSER_SOCKET_DIR"] = str(settings.path.cache_path.resolve())
    return command, env


async def _run_wrapped_command(
    command: str,
    workspace: Path,
    env: dict[str, str] | None = None,
) -> tuple[bytes, bytes, int | None]:
    wrapped_command = sandbox.wrap_command(command, workspace, env=env)
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
        )
        try:
            await asyncio.wait_for(process.wait(), timeout=EXEC_TIMEOUT)
        except asyncio.TimeoutError:
            await _stop_process(process)
        except asyncio.CancelledError:
            await _stop_process(process)
            raise

        stdout_f.seek(0)
        stderr_f.seek(0)
        return stdout_f.read(), stderr_f.read(), process.returncode


def _format_command_output(stdout: bytes, stderr: bytes, returncode: int | None) -> str:
    output_parts = []

    if stdout:
        output_parts.append(stdout.decode("utf-8", errors="replace"))

    if stderr:
        stderr_text = stderr.decode("utf-8", errors="replace")
        if stderr_text.strip():
            output_parts.append(f"STDERR:\n{stderr_text}")

    output_parts.append(f"\nExit code: {returncode}")
    result = "\n".join(output_parts) if output_parts else "(no output)"

    max_len = _MAX_OUTPUT
    if len(result) > max_len:
        half = max_len // 2
        result = result[:half] + f"\n\n... ({len(result) - max_len:,} chars truncated) ...\n\n" + result[-half:]

    return result


async def _ensure_workspace_uv_venv(workspace: Path) -> None:
    if (workspace / ".venv").exists():
        return
    stdout, stderr, returncode = await _run_wrapped_command("uv venv .venv", workspace)
    if returncode != 0:
        result = _format_command_output(stdout, stderr, returncode)
        raise RuntimeError(f"Failed to initialize workspace Python environment with `uv venv .venv`.\n{result}")


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
        command = _guard_command(args.command, workspace)
        command, env = _inject_command_context(
            command,
            session_id=get_session_id(),
        )
        ensure_path(sandbox.get_sandbox_home())
        await _ensure_workspace_uv_venv(workspace)
        stdout, stderr, returncode = await _run_wrapped_command(command, workspace, env=env)
        return _format_command_output(stdout, stderr, returncode)

    shell.__doc__ = dedent(
        """
            Execute a bash command and return the output.

            Commands run on Linux inside a bubblewrap sandbox. The current
            workspace is isolated from other workspaces.
            Python must be run through uv, for example:
            uv run python script.py
        """
    ).lstrip()

    return shell
