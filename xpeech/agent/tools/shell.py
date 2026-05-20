from pydantic import BaseModel, Field
import shutil
from pathlib import Path
import platform
import asyncio
from loguru import logger
import os
from contextlib import suppress
import re
from ...utils.security.network import contains_internal_url
from ...utils.helper import is_relative_path, msys_to_win
from textwrap import dedent
from ..skills.skill import BUILTIN_SKILLS_DIR
import shlex
from ..server.context import get_session_id

EXEC_TIMEOUT = 60
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
    "tools, working_dir overrides). If the user genuinely needs this "
    "resource, tell them you cannot reach it under the current "
    "restrict_to_workspace policy and ask how to proceed."
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

# 如果是windows系统，检测git bash是否安装，如果未安装，抛出异常
if platform.system() == "Windows":
    _IS_WINDOWS = True
    git_path = shutil.which("git")
    if git_path is None:
        raise Exception("Git Bash is not installed. Please install Git Bash and try again.")
    bash_path = Path(git_path).parent.parent / "bin" / "bash"
    docstring_str = (
        "MSYS2 is a Bash emulator for Windows. It supports Windows-style paths. All absolute paths MUST follow the D:/dir1/dir2/file format (Drive letter + colon + forward slash). The /d/dir1/dir2/file format is STRICTLY FORBIDDEN.\n"
        "To avoid escaping issues, All paths MUST unconditionally use forward slashes (/). You MUST automatically convert any backslashes (\\) to forward slashes (/) without exception.\n"
        "example:\n"
        "ls -l D:/dir1/dir2/file\n"
        "ls -l dir3/file\n"
        "cat D:/dir1/dir2/file.txt"
    )
else:
    _IS_WINDOWS = False
    bash_path = Path(shutil.which("bash") or "/bin/bash")
    docstring_str = ""


class ShellArgs(BaseModel):
    command: str = Field(description="Bash command to execute")


def _is_benign_device_path(path: str) -> bool:
    """Return True for kernel device files that should never be workspace-blocked."""
    if path in _BENIGN_DEVICE_PATHS:
        return True
    return path.startswith("/dev/fd/")


async def _kill_process(process: asyncio.subprocess.Process) -> None:
    """Kill a subprocess and reap it to prevent zombies."""
    process.kill()
    try:
        with suppress(asyncio.TimeoutError):
            await asyncio.wait_for(process.wait(), timeout=5.0)
    finally:
        if not _IS_WINDOWS:
            try:
                os.waitpid(process.pid, os.WNOHANG)
            except (ProcessLookupError, ChildProcessError) as e:
                logger.debug("Process already reaped or not found: {}", e)


def _extract_absolute_paths(command: str) -> list[str]:
    win_paths = re.findall(r"\b[A-Za-z]:[\\/][^\s\"'|><;]*", command)
    posix_paths = re.findall(r"(?:^|[\s|>'\"])(/[^\s\"'>;|<]+)", command)
    home_paths = re.findall(r"(?:^|[\s>'\"])(~[^\s\"'>;|<]*)", command)
    return win_paths + posix_paths + home_paths


def rewrite_shell_command_for_skills(command: str, session_id: str | None) -> str:
    if not isinstance(session_id, str):
        raise ValueError("shell internal error: session_id must be a string")
    parts = shlex.split(command)
    # 如果包含 playwright-cli，则添加 -s={session_id}
    if "playwright-cli" in parts:
        parts.append(f"-s={session_id}")
    return shlex.join(parts)


def _guard_command(command: str, workspace: str, restrict_tools_to_workspace: bool) -> str | None:
    """Guard a command string from code injection attacks."""
    cmd = command.strip()
    lower = cmd.lower()

    # 正则判断
    for pattern in DENY_PATTERNS:
        if re.search(pattern, lower):
            raise RuntimeError(f"Command blocked by deny pattern filter: {cmd}")
    # 判断是否恶意访问内网接口
    if contains_internal_url(cmd):
        raise RuntimeError(f"Command blocked by safety guard (internal/private URL detected): {cmd}")

    # 为skill命令注入参数
    cmd = rewrite_shell_command_for_skills(cmd, get_session_id())

    # 是否开启工作路径限制
    if restrict_tools_to_workspace:
        # 拦截非工作路径下的文件操作
        ## 拦截 .. 符号
        if "..\\" in cmd or "../" in cmd:
            raise RuntimeError(
                f"Command blocked by safety guard (path traversal detected): {cmd}" + _WORKSPACE_BOUNDARY_NOTE
            )
        ## 提取所有路径，判断是否在工作路径下
        for i in _extract_absolute_paths(cmd):
            try:
                expanded = os.path.expandvars(i.strip())
                if _is_benign_device_path(expanded):
                    continue
                if not _IS_WINDOWS:
                    p = Path(expanded).expanduser().resolve()
                else:
                    if expanded.startswith("~"):
                        p = Path(expanded).expanduser().resolve()
                    else:
                        # windows系统下，需要将 MSYS2 路径转换为 Windows 路径
                        p = Path(msys_to_win(expanded)).resolve()
            except Exception:
                continue

            if _is_benign_device_path(str(p)):
                continue

            if not (
                is_relative_path(path_target=p, base=workspace)
                or is_relative_path(path_target=p, base=BUILTIN_SKILLS_DIR)
            ):
                raise RuntimeError(
                    f"Command blocked by safety guard (path outside working dir): {p}" + _WORKSPACE_BOUNDARY_NOTE
                )

    return None


def build_shell_tools(workspace: str, restrict_tools_to_workspace: bool):
    base = Path(workspace).expanduser().resolve()
    if not base.exists() or not base.is_dir():
        raise ValueError(f"Invalid workspace: {workspace}")

    async def shell(args: ShellArgs) -> str:
        command = args.command
        _guard_command(command, workspace, restrict_tools_to_workspace)
        process = await asyncio.create_subprocess_exec(
            bash_path,
            "-l",
            "-c",
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=base,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=EXEC_TIMEOUT,
            )
        except asyncio.TimeoutError:
            await _kill_process(process)
            raise TimeoutError(f"Command timed out after {EXEC_TIMEOUT} seconds")
        except asyncio.CancelledError:
            await _kill_process(process)
            raise

        output_parts = []

        if stdout:
            output_parts.append(stdout.decode("utf-8", errors="replace"))

        if stderr:
            stderr_text = stderr.decode("utf-8", errors="replace")
            if stderr_text.strip():
                output_parts.append(f"STDERR:\n{stderr_text}")

        output_parts.append(f"\nExit code: {process.returncode}")

        result = "\n".join(output_parts) if output_parts else "(no output)"

        max_len = _MAX_OUTPUT
        if len(result) > max_len:
            half = max_len // 2
            result = result[:half] + f"\n\n... ({len(result) - max_len:,} chars truncated) ...\n\n" + result[-half:]

        return result

    shell.__doc__ = dedent(
        f"""
            Execute a bash command and return the output.
            {docstring_str}
        """
    ).lstrip()

    return shell
