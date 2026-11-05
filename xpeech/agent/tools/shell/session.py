"""Reuse one isolated process tree per user workspace until it is idle."""

import asyncio
import base64
import json
import os
import signal
import sys
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from loguru import logger

from ....config.settings import settings
from ...server.context import get_lark_user_access_token, get_user_id
from ...skills.skill import SkillsLoader
from .. import sandbox
from .schema import ShellCommandResult

_WORKER_SOURCE = (Path(__file__).parent / "scripts" / "worker.py").read_text(encoding="utf-8")
SandboxScope = tuple[str | None, Path]


class SandboxConnectionError(RuntimeError):
    """The shared command transport ended before a response arrived."""


async def _stop_process(process: asyncio.subprocess.Process) -> None:
    if process.returncode is None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    await process.wait()


class SandboxSession:
    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.lock = asyncio.Lock()
        self.initialization_lock = asyncio.Lock()
        self.drained = asyncio.Condition(self.lock)
        self.users = 0
        self.process = None
        self.stderr = None
        self.idle_task = None
        self.wrapped_command: list[str] | None = None
        self.reader_task = None
        self.pending: dict[int, asyncio.Future[ShellCommandResult]] = {}
        self.next_request_id = 0
        self.retired = False

    async def start(self, skills_loader: SkillsLoader) -> None:
        """Start or reuse the worker while holding the lifecycle lock."""
        # The backend venv is deliberately not mounted. Use its base interpreter
        # and expose only that runtime when installed outside /usr or /opt.
        python = Path(sys._base_executable).resolve()
        command = await sandbox.wrap_sandbox_command(
            [str(python), "-I", "-u", "-c", _WORKER_SOURCE], self.workspace, skills_loader=skills_loader,
        )
        runtime = Path(sys.base_prefix).resolve()
        if not any(runtime.is_relative_to(root) for root in (Path("/usr"), Path("/opt"))):
            index = command.index("--chdir")
            command[index:index] = ["--ro-bind", str(runtime), str(runtime)]
        # Refresh mounts and environment on every call. Let active commands
        # finish before rebuilding the sandbox with a different configuration.
        while self.process is not None and self.wrapped_command != command and self.pending:
            await self.drained.wait()
        if self.retired:
            raise SandboxConnectionError("Shell sandbox has been closed")
        if self.process is not None:
            if (
                self.process.returncode is None
                and self.wrapped_command == command
                and self.reader_task is not None
                and not self.reader_task.done()
            ):
                return
            await self.stop()
        self.stderr = tempfile.TemporaryFile()  # noqa: SIM115 -- owned by the session; closed in stop().
        try:
            spawning = asyncio.create_task(
                asyncio.create_subprocess_exec(
                    *command,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=self.stderr,
                    cwd=self.workspace,
                    start_new_session=True,
                )
            )
            try:
                self.process = await asyncio.shield(spawning)
            except asyncio.CancelledError:
                self.process = await spawning
                raise
            async with asyncio.timeout(30):
                ready = await self.process.stdout.readline()
            if ready != b"ready\n":
                self.stderr.seek(0)
                detail = self.stderr.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"Sandbox failed to start: {detail}")
        except BaseException:
            await self.stop()
            raise
        self.wrapped_command = command
        self.reader_task = asyncio.create_task(self.receive(self.process))
        logger.info("Started persistent shell sandbox workspace={}", self.workspace)

    async def stop(self) -> None:
        """Close the transport and fail its pending requests under the lock."""
        reader = self.reader_task
        self.reader_task = None
        for response in self.pending.values():
            if not response.done():
                response.set_exception(SandboxConnectionError("Shell sandbox connection closed"))
        self.pending.clear()
        self.drained.notify_all()
        if reader is not None:
            reader.cancel()
            await asyncio.gather(reader, return_exceptions=True)
        if self.process is not None:
            await _stop_process(self.process)
            self.process = None
        if self.stderr is not None:
            self.stderr.close()
            self.stderr = None
        self.wrapped_command = None

    async def send(self, request: dict) -> None:
        self.process.stdin.write(json.dumps(request).encode("utf-8") + b"\n")
        await self.process.stdin.drain()

    async def receive(self, process: asyncio.subprocess.Process) -> None:
        """Use one reader to route out-of-order responses to their callers."""
        try:
            while True:
                size = int(await process.stdout.readline())
                message = json.loads(await process.stdout.readexactly(size))
                response = self.pending.get(message["id"])
                if response is None or response.done():
                    continue
                if "error" in message:
                    response.set_exception(RuntimeError(message["error"]))
                else:
                    response.set_result(
                        ShellCommandResult(
                            base64.b64decode(message["stdout"]),
                            base64.b64decode(message["stderr"]),
                            message["returncode"],
                            message["timed_out"],
                        )
                    )
        except (OSError, ValueError, KeyError, TypeError, asyncio.IncompleteReadError) as error:
            for response in self.pending.values():
                if not response.done():
                    response.set_exception(SandboxConnectionError(f"Shell sandbox transport failed: {error}"))

    async def _stop_if_current(self, process: asyncio.subprocess.Process) -> None:
        async with self.lock:
            if self.process is process:
                await self.stop()

    async def execute(
        self, command: str, timeout_ms: int, *, skills_loader: SkillsLoader,
    ) -> ShellCommandResult:
        # Only lifecycle changes and writes share the lock, never command waits.
        async with self.lock:
            await self.start(skills_loader)
            process = self.process
            request_id = self.next_request_id
            self.next_request_id += 1
            response = asyncio.get_running_loop().create_future()
            self.pending[request_id] = response
            try:
                # Tokens belong to the current request, never the cached session.
                await self.send(
                    {
                        "op": "run",
                        "id": request_id,
                        "command": command,
                        "timeout_ms": timeout_ms,
                        "env": {
                            "SHELL_USER_ID": str(get_user_id()),
                            "LARKSUITE_CLI_USER_ACCESS_TOKEN": get_lark_user_access_token(),
                        },
                    }
                )
            except BaseException:
                # A partial write leaves the shared stream unusable.
                await self.stop()
                await asyncio.gather(response, return_exceptions=True)
                raise
        try:
            async with asyncio.timeout(timeout_ms / 1000 + 5):
                return await asyncio.shield(response)
        except asyncio.CancelledError:
            try:
                async with self.lock:
                    if self.process is process and not response.done():
                        await self.send({"op": "cancel", "id": request_id})
                await asyncio.wait_for(asyncio.shield(response), 5)
            except (OSError, TimeoutError, SandboxConnectionError):
                await self._stop_if_current(process)
            except RuntimeError:
                # A per-command error still acknowledges cancellation.
                pass
            raise
        except (TimeoutError, SandboxConnectionError):
            # Never retry a command: it may already have produced side effects.
            await self._stop_if_current(process)
            raise
        finally:
            async with self.lock:
                self.pending.pop(request_id, None)
                self.drained.notify_all()
            if not response.done():
                response.cancel()
            await asyncio.gather(response, return_exceptions=True)


class SandboxManager:
    def __init__(self, idle_timeout: float | None = None):
        self.idle_timeout = settings.tool.shell.sandbox_idle_timeout_seconds if idle_timeout is None else idle_timeout
        self.sessions: dict[SandboxScope, SandboxSession] = {}
        self.closing = False

    @asynccontextmanager
    async def use(self, workspace: Path):
        """Keep a user/workspace session alive while a caller uses its resources."""
        workspace = workspace.resolve()
        scope = get_user_id(), workspace
        if self.closing:
            raise SandboxConnectionError("Shell sandboxes are shutting down")
        session = self.sessions.setdefault(scope, SandboxSession(workspace))
        session.users += 1  # Includes requests waiting for sandbox startup.
        if session.idle_task is not None:
            session.idle_task.cancel()
        try:
            yield session
        finally:
            session.users -= 1
            if not session.users and self.sessions.get(scope) is session:
                session.idle_task = asyncio.create_task(self._expire(scope, session))

    async def run(
        self, command: str, workspace: Path, timeout_ms: int, *, skills_loader: SkillsLoader,
    ) -> ShellCommandResult:
        async with self.use(workspace) as session:
            return await session.execute(command, timeout_ms, skills_loader=skills_loader)

    async def _expire(self, scope: SandboxScope, session: SandboxSession) -> None:
        await asyncio.sleep(self.idle_timeout)
        async with session.lock:
            if session.users or self.sessions.get(scope) is not session:
                return
            # Keep cleanup shielded from a request arriving during teardown.
            cleanup = asyncio.create_task(session.stop())
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                await cleanup
                raise
            if self.sessions.get(scope) is session and not session.users:
                del self.sessions[scope]
            logger.info("Reclaimed idle shell sandbox user_id={} workspace={}", *scope)

    async def close(self) -> None:
        self.closing = True
        sessions = list(self.sessions.values())
        self.sessions.clear()
        for session in sessions:
            session.retired = True
            if session.idle_task is not None:
                session.idle_task.cancel()

        async def stop(session):
            async with session.lock:
                await session.stop()

        try:
            await asyncio.gather(*(s.idle_task for s in sessions if s.idle_task is not None), return_exceptions=True)
            await asyncio.gather(*(stop(s) for s in sessions))
        finally:
            self.closing = False


sandbox_manager = SandboxManager()


async def close_all_shell_sandboxes() -> None:
    await sandbox_manager.close()
