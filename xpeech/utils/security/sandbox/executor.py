"""Workspace command execution and Python environment initialization."""

import platform
import shutil
from pathlib import Path

from ....agent.skills.skill import SkillsLoader
from .isolation import NETWORK_SANDBOX_EXECUTABLES
from .policy import guard_command
from .schema import DEFAULT_EXEC_TIMEOUT_MS, CommandResult
from .session import sandbox_manager


class SandboxExecutor:
    """Run guarded commands with a shared sandbox and workspace Python environment."""

    def __init__(self, workspace: Path, *, skills_loader: SkillsLoader):
        if platform.system() != "Linux":
            raise RuntimeError("Shell tool requires Linux with bubblewrap; Windows and other platforms are not supported.")
        missing = [name for name in (*NETWORK_SANDBOX_EXECUTABLES, "uv") if shutil.which(name) is None]
        if missing:
            raise RuntimeError(f"Shell tool requires executable(s) on PATH: {', '.join(missing)}")
        self.workspace = workspace.resolve()
        if not self.workspace.is_dir():
            raise ValueError(f"Invalid workspace: {self.workspace}")
        self.skills_loader = skills_loader

    @property
    def idle_timeout(self) -> float:
        return sandbox_manager.idle_timeout

    async def ensure_python_environment(self) -> None:
        # Shared across tool instances and conversations. Check inside the lock
        # so commands cannot use a .venv that uv is still initializing.
        async with sandbox_manager.use(self.workspace) as session, session.initialization_lock:
            if (self.workspace / ".venv").exists():
                return
            result = await sandbox_manager.run(
                "uv venv .venv", self.workspace, skills_loader=self.skills_loader,
            )
            if result.returncode != 0:
                detail = (result.stdout + result.stderr).decode("utf-8", errors="replace")
                if result.timed_out:
                    detail = f"Command timed out after {DEFAULT_EXEC_TIMEOUT_MS / 1000:g}s.\n{detail}"
                raise RuntimeError(
                    f"Failed to initialize workspace Python environment with `uv venv .venv`.\n"
                    f"{detail}\nExit code: {result.returncode}"
                )

    async def run(self, command: str, timeout_ms: int = DEFAULT_EXEC_TIMEOUT_MS) -> CommandResult:
        command = guard_command(command)
        await self.ensure_python_environment()
        return await sandbox_manager.run(
            command, self.workspace, timeout_ms, skills_loader=self.skills_loader,
        )
