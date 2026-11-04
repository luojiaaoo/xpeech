import asyncio
import platform
import shutil
from pathlib import Path

import pytest
import pytest_asyncio
from pydantic import ValidationError

from xpeech.agent.server.context import lark_user_access_token_var, session_id_var, user_id_var
from xpeech.agent.tools import sandbox
from xpeech.agent.tools.shell import tool as shell_module
from xpeech.agent.tools.shell.schema import ShellArgs, ShellCommandResult
from xpeech.agent.tools.shell.session import SandboxConnectionError, SandboxManager, SandboxSession
from xpeech.config.settings import Settings, ShellConfig, ToolConfig


def test_existing_tool_configs_keep_the_default_sandbox_lifetime():
    assert ToolConfig().shell.sandbox_idle_timeout_seconds == 600


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("-inf"), float("nan")])
def test_sandbox_idle_timeout_rejects_invalid_values(timeout):
    with pytest.raises(ValidationError, match="sandbox_idle_timeout_seconds"):
        ShellConfig(sandbox_idle_timeout_seconds=timeout)


def test_sandbox_idle_timeout_loads_from_toml(tmp_path, monkeypatch):
    template = (Path(__file__).resolve().parents[1] / "conf.toml.example").read_text(encoding="utf-8")
    (tmp_path / "conf.toml").write_text(
        template.replace("sandbox_idle_timeout_seconds = 600", "sandbox_idle_timeout_seconds = 0.15"),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    loaded = Settings()
    monkeypatch.setattr(sandbox.settings.tool, "shell", loaded.tool.shell)
    assert SandboxManager().idle_timeout == 0.15


@pytest.mark.asyncio
async def test_manager_reuses_resets_idle_deadline_and_recreates_after_expiry(tmp_path, monkeypatch):
    manager = SandboxManager(idle_timeout=0.1)
    stopped = asyncio.Event()

    async def execute(self, command, timeout_ms):
        return self

    async def stop(self):
        stopped.set()

    monkeypatch.setattr(SandboxSession, "execute", execute)
    monkeypatch.setattr(SandboxSession, "stop", stop)
    try:
        first = await manager.run("one", tmp_path, 100)
        await asyncio.sleep(0.06)
        assert await manager.run("two", tmp_path, 100) is first
        await asyncio.sleep(0.06)
        assert not stopped.is_set()
        await asyncio.wait_for(first.idle_task, 1)
        assert stopped.is_set()
        assert not manager.sessions
        assert await manager.run("three", tmp_path, 100) is not first
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_concurrent_requests_prevent_expiry_and_users_are_independent(tmp_path, monkeypatch):
    manager = SandboxManager(idle_timeout=0.02)
    started = asyncio.Queue()
    release = asyncio.Event()
    stopped = []

    async def execute(self, command, timeout_ms):
        if command.startswith("long"):
            started.put_nowait(command)
            await release.wait()
        return self

    async def stop(self):
        stopped.append(self)

    monkeypatch.setattr(SandboxSession, "execute", execute)
    monkeypatch.setattr(SandboxSession, "stop", stop)
    first = asyncio.create_task(manager.run("long-first", tmp_path / "a", 100))
    second = None
    try:
        assert await asyncio.wait_for(started.get(), 1) == "long-first"
        second = asyncio.create_task(manager.run("long-second", tmp_path / "a", 100))
        assert await asyncio.wait_for(started.get(), 1) == "long-second"
        other = await manager.run("other", tmp_path / "b", 100)
        await asyncio.sleep(0.06)
        assert stopped == [other]
        assert not first.done() and not second.done()
        release.set()
        assert await first is await second
    finally:
        release.set()
        await asyncio.gather(*(t for t in (first, second) if t is not None))
        await manager.close()


@pytest.mark.asyncio
async def test_request_during_expiry_waits_for_cleanup(tmp_path, monkeypatch):
    manager = SandboxManager(idle_timeout=0.01)
    stopping = asyncio.Event()
    release = asyncio.Event()
    cleaned = False

    async def execute(self, command, timeout_ms):
        async with self.lock:
            if command == "after":
                assert cleaned
            return self

    async def stop(self):
        nonlocal cleaned
        stopping.set()
        await release.wait()
        cleaned = True

    monkeypatch.setattr(SandboxSession, "execute", execute)
    monkeypatch.setattr(SandboxSession, "stop", stop)
    await manager.run("before", tmp_path, 100)
    await asyncio.wait_for(stopping.wait(), 1)
    after = asyncio.create_task(manager.run("after", tmp_path, 100))
    try:
        await asyncio.sleep(0)
        assert not after.done()
        release.set()
        await asyncio.wait_for(after, 1)
    finally:
        release.set()
        await manager.close()


@pytest.mark.asyncio
async def test_request_credentials_are_refreshed_without_replaying_commands(tmp_path, monkeypatch):
    session = SandboxSession(tmp_path)
    requests = []

    async def start():
        pass

    async def send(request):
        requests.append(request)
        session.pending[request["id"]].set_result(ShellCommandResult(b"", b"", 0))

    monkeypatch.setattr(session, "start", start)
    monkeypatch.setattr(session, "send", send)
    user_context = user_id_var.set("alice")
    try:
        for value in ("old-token", "new-token", None):
            context = lark_user_access_token_var.set(value)
            try:
                await session.execute("true", 100)
            finally:
                lark_user_access_token_var.reset(context)
        assert [r["env"]["LARKSUITE_CLI_USER_ACCESS_TOKEN"] for r in requests] == [
            "old-token",
            "new-token",
            None,
        ]
        assert all(r["env"]["SHELL_USER_ID"] == "alice" for r in requests)
    finally:
        user_id_var.reset(user_context)


@pytest.mark.asyncio
async def test_concurrent_first_shell_calls_share_initialization_then_execute_in_parallel(tmp_path, monkeypatch):
    manager = SandboxManager()
    initializing = asyncio.Event()
    initialized = asyncio.Event()
    executing = asyncio.Queue()
    finish = asyncio.Event()
    initializations = []

    async def run(command, workspace, timeout_ms=120_000):
        if command == "uv venv .venv":
            initializations.append(command)
            (workspace / ".venv").mkdir()
            initializing.set()
            await initialized.wait()
        else:
            executing.put_nowait(command)
            await finish.wait()
        return ShellCommandResult(command.encode(), b"", 0)

    monkeypatch.setattr(shell_module, "sandbox_manager", manager)
    monkeypatch.setattr(shell_module, "_run_wrapped_command", run)
    monkeypatch.setattr(shell_module.platform, "system", lambda: "Linux")
    monkeypatch.setattr(shell_module.shutil, "which", lambda name: f"/usr/bin/{name}")
    first_shell = shell_module.build_shell_tools(tmp_path)
    second_shell = shell_module.build_shell_tools(tmp_path)
    first = asyncio.create_task(first_shell(ShellArgs(command="first")))
    second = None
    try:
        await asyncio.wait_for(initializing.wait(), 1)
        second = asyncio.create_task(second_shell(ShellArgs(command="second")))
        await asyncio.sleep(0)
        assert executing.empty()
        initialized.set()
        commands = [await asyncio.wait_for(executing.get(), 1) for _ in range(2)]
        assert set(commands) == {"first", "second"}
        assert initializations == ["uv venv .venv"]
        finish.set()
        assert await first == "first\nExit code: 0"
        assert await second == "second\nExit code: 0"
    finally:
        initialized.set()
        finish.set()
        await asyncio.gather(*(t for t in (first, second) if t is not None), return_exceptions=True)
        await manager.close()


linux_sandbox = pytest.mark.skipif(
    platform.system() != "Linux" or any(shutil.which(name) is None for name in sandbox.SANDBOX_EXECUTABLES),
    reason="Requires Linux sandbox dependencies",
)


@pytest_asyncio.fixture
async def manager(tmp_path, monkeypatch, request):
    monkeypatch.setattr(sandbox.settings.path, "sandbox_home_path", tmp_path / "config-home")
    monkeypatch.setattr(sandbox, "iter_builtin_skill_dirs", list)
    if hasattr(request, "param"):
        monkeypatch.setattr(sandbox.settings.tool.shell, "sandbox_idle_timeout_seconds", request.param)
    manager = SandboxManager()
    try:
        yield manager
    finally:
        await manager.close()


async def wait_for_command_file(path, task):
    async with asyncio.timeout(5):
        while not path.exists():
            if task.done():
                pytest.fail(f"Command exited before signaling readiness: {task.result()}")
            await asyncio.sleep(0.01)


@linux_sandbox
@pytest.mark.asyncio
async def test_commands_overlap_in_one_sandbox_and_route_out_of_order_output(tmp_path, manager):
    workspace = tmp_path / "alice"
    workspace.mkdir()
    slow = asyncio.create_task(
        manager.run(
            "touch slow.started; while [ ! -f release ]; do sleep 0.01; done; "
            "printf slow; readlink /proc/self/ns/net; printf slow-error >&2",
            workspace,
            4_000,
        )
    )
    try:
        await wait_for_command_file(workspace / "slow.started", slow)
        fast = await manager.run(
            "printf fast; readlink /proc/self/ns/net; printf fast-error >&2",
            workspace,
            1_000,
        )
        assert not slow.done()
        assert fast.returncode == 0 and not fast.timed_out
        assert fast.stdout.startswith(b"fastnet:[") and fast.stderr == b"fast-error"
        (workspace / "release").touch()
        result = await slow
        assert result.returncode == 0 and not result.timed_out
        assert result.stdout == b"slow" + fast.stdout.removeprefix(b"fast")
        assert result.stderr == b"slow-error"
    finally:
        (workspace / "release").touch()
        await asyncio.gather(slow, return_exceptions=True)


@linux_sandbox
@pytest.mark.asyncio
async def test_concurrent_credentials_and_large_outputs_stay_with_their_request(tmp_path, manager):
    workspace = tmp_path / "alice"
    workspace.mkdir()

    async def command(index):
        context = lark_user_access_token_var.set(f"token-{index}")
        try:
            result = await manager.run(
                f"touch started-{index}; "
                "while [ $(find . -name 'started-*' | wc -l) -lt 4 ]; do sleep 0.01; done; "
                "printf '%s\\n' \"$LARKSUITE_CLI_USER_ACCESS_TOKEN\"; "
                f"head -c 100000 /dev/zero | tr '\\0' '{index}'; "
                f"printf error-{index} >&2",
                workspace,
                3_000,
            )
        finally:
            lark_user_access_token_var.reset(context)
        assert result.returncode == 0 and not result.timed_out
        assert result.stdout == f"token-{index}\n".encode() + str(index).encode() * 100000
        assert result.stderr == f"error-{index}".encode()

    await asyncio.gather(*(command(index) for index in range(4)))


@linux_sandbox
@pytest.mark.asyncio
async def test_real_sandbox_reuses_namespaces_tmp_and_background_processes_across_conversations(tmp_path, manager):
    workspace = tmp_path / "alice"
    workspace.mkdir()
    context = session_id_var.set("first-chat")
    try:
        result = await manager.run(
            "echo retained > /tmp/state; sleep 30 >/dev/null 2>&1 & echo $! > worker.pid; "
            "readlink /proc/self/ns/net; readlink /proc/self/ns/pid",
            workspace,
            2_000,
        )
    finally:
        session_id_var.reset(context)
    context = session_id_var.set("second-chat")
    try:
        again = await manager.run(
            "kill -0 $(cat worker.pid) && test $(cat /tmp/state) = retained && "
            "readlink /proc/self/ns/net && readlink /proc/self/ns/pid",
            workspace,
            2_000,
        )
    finally:
        session_id_var.reset(context)
    assert result.returncode == again.returncode == 0
    assert result.stdout == again.stdout
    other = tmp_path / "bob"
    other.mkdir()
    isolated = await manager.run("test ! -f /tmp/state; readlink /proc/self/ns/net", other, 2_000)
    assert isolated.returncode == 0
    assert isolated.stdout != result.stdout.splitlines(keepends=True)[0]


@linux_sandbox
@pytest.mark.asyncio
@pytest.mark.parametrize("termination", ["timeout", "cancel", "cancel_immediately"])
async def test_real_sandbox_survives_command_timeout_and_cancellation(tmp_path, manager, termination, monkeypatch):
    workspace = tmp_path / "alice"
    workspace.mkdir()
    initial = await manager.run("readlink /proc/self/ns/net", workspace, 2_000)
    manager.idle_timeout = 0.05
    sibling = asyncio.create_task(
        manager.run(
            "touch sibling.started; while [ ! -f release ]; do sleep 0.01; done; readlink /proc/self/ns/net",
            workspace,
            5_000,
        )
    )
    await wait_for_command_file(workspace / "sibling.started", sibling)
    sent = asyncio.Event()
    session = manager.sessions[workspace]
    send = session.send

    async def observe_send(request):
        await send(request)
        if request["op"] == "run":
            sent.set()

    monkeypatch.setattr(session, "send", observe_send)
    command = "echo partial; echo ready > ready; sleep 30 & wait"
    task = asyncio.create_task(manager.run(command, workspace, 100 if termination == "timeout" else 10_000))
    if termination.startswith("cancel"):
        if termination == "cancel_immediately":
            await asyncio.wait_for(sent.wait(), 5)
        else:
            async with asyncio.timeout(5):
                while not (workspace / "ready").exists():
                    if task.done():
                        pytest.fail(f"Command exited early: {task.result()}")
                    await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        result = await task
        assert result.timed_out and result.stdout == b"partial\n"
    await asyncio.sleep(0.1)
    assert not sibling.done()
    after = await manager.run("readlink /proc/self/ns/net", workspace, 2_000)
    assert after.returncode == 0 and after.stdout == initial.stdout
    (workspace / "release").touch()
    result = await sibling
    assert result.returncode == 0 and result.stdout == initial.stdout


@linux_sandbox
@pytest.mark.asyncio
@pytest.mark.parametrize("termination", ["crash", "shutdown"])
async def test_shared_transport_loss_finishes_all_requests_without_replaying_them(tmp_path, manager, termination):
    workspace = tmp_path / "alice"
    workspace.mkdir()
    tasks = [
        asyncio.create_task(
            manager.run(
                f"echo once >> attempts-{index}; touch started-{index}; sleep 30",
                workspace,
                40_000,
            )
        )
        for index in range(2)
    ]
    for index, task in enumerate(tasks):
        await wait_for_command_file(workspace / f"started-{index}", task)
    if termination == "shutdown":
        await asyncio.wait_for(manager.close(), 5)
    else:
        process = manager.sessions[workspace].process
        process.kill()
    results = await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 5)
    assert all(isinstance(result, SandboxConnectionError) for result in results)
    after = await manager.run("printf recovered", workspace, 1_000)
    assert after.returncode == 0 and after.stdout == b"recovered"
    assert all((workspace / f"attempts-{index}").read_text() == "once\n" for index in range(2))


@linux_sandbox
@pytest.mark.asyncio
@pytest.mark.parametrize("manager", [0.15], indirect=True)
async def test_real_idle_expiry_stops_detached_background_processes(tmp_path, manager):
    workspace = tmp_path / "alice"
    workspace.mkdir()
    result = await manager.run(
        "setsid bash -c 'while :; do echo x >> heartbeat; sleep 0.02; done' >/dev/null 2>&1 & "
        "while [ ! -s heartbeat ]; do sleep 0.01; done; echo volatile > /tmp/state",
        workspace,
        2_000,
    )
    assert result.returncode == 0
    async with asyncio.timeout(5):
        while manager.sessions:
            await asyncio.sleep(0.02)
    heartbeat = (workspace / "heartbeat").read_bytes()
    await asyncio.sleep(0.1)
    assert (workspace / "heartbeat").read_bytes() == heartbeat
    after = await manager.run("test ! -f /tmp/state && test -s heartbeat", workspace, 2_000)
    assert after.returncode == 0
