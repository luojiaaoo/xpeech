import asyncio
import shlex
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from opensandbox.exceptions import SandboxApiException
from opensandbox.models.execd import Execution, ExecutionComplete, ExecutionLogs, OutputMessage
from pydantic import ValidationError

from xpeech.agent.server.context import lark_user_access_token_var
from xpeech.agent.tools.shell import opensandbox as module
from xpeech.agent.tools.shell.shared_slots import FileLock, acquire_slot, read_record, write_record
from xpeech.config.settings import OpenSandboxConfig

SUPERVISOR = Path(__file__).resolve().parents[1] / "docker/sandbox/run_command.py"


def execution(stdout="", stderr="", code=0):
    def message(text):
        return OutputMessage(text=text, timestamp=0)
    return Execution(
        id="command-1", exit_code=code,
        complete=ExecutionComplete(timestamp=0, execution_time_in_millis=1),
        logs=ExecutionLogs(stdout=[message(stdout)], stderr=[message(stderr)]),
    )


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    base = tmp_path / "workspaces"
    workspace = base / "alice"
    workspace.mkdir(parents=True)
    home_configs = tmp_path / "configs"
    home_configs.mkdir()
    (home_configs / ".npmrc").write_text("registry=https://registry.npmmirror.com/")
    monkeypatch.setattr(module.settings.path, "workspace_base_path", base)
    monkeypatch.setattr(module.settings.path, "sandbox_home_path", home_configs)
    monkeypatch.setenv("OPENSANDBOX_API_KEY", "test-secret")
    monkeypatch.setenv("XPEECH_TOKEN_MANAGER_URL", "http://lark-token-manager:7883")
    config = OpenSandboxConfig(
        host_workspace_root="/host/workspaces", host_skills_root="/host/skills",
        host_home_config_root="/host/configs", host_cli_root="/host/lark-cli", state_path=tmp_path / "state",
        max_sandboxes=2, max_commands=2, max_creating=1, queue_timeout_seconds=0.15,
    )
    created = []

    async def create(*args, **kwargs):
        sandbox = SimpleNamespace(
            id=f"sandbox-{len(created)}", commands=SimpleNamespace(run=AsyncMock(return_value=execution()), interrupt=AsyncMock()),
            renew=AsyncMock(), kill=AsyncMock(), close=AsyncMock(),
            get_info=AsyncMock(side_effect=SandboxApiException(status_code=404)),
        )
        created.append(sandbox)
        return sandbox

    create_mock = AsyncMock(side_effect=create)
    connect_mock = AsyncMock(side_effect=lambda sandbox_id, **kwargs: next(s for s in created if s.id == sandbox_id))
    monkeypatch.setattr(module.Sandbox, "create", create_mock)
    monkeypatch.setattr(module.Sandbox, "connect", connect_mock)
    return SimpleNamespace(
        workspace=workspace, config=config, created=created, create=create_mock,
        connect=connect_mock, executor=module.OpenSandboxExecutor(config),
    )


@pytest.mark.asyncio
async def test_reuses_user_sandbox_across_executor_instances_and_mounts_only_user_files(runtime):
    r = runtime
    await r.executor.run("pwd", r.workspace, 1000)
    other = module.OpenSandboxExecutor(r.config)
    await other.run("node --version", r.workspace, 1000)
    r.create.assert_awaited_once()
    r.connect.assert_awaited_once()
    kwargs = r.create.await_args.kwargs
    volumes = kwargs["volumes"]
    assert volumes[0].host.path == "/host/workspaces/alice"
    assert volumes[0].mount_path == str(r.workspace)
    assert not volumes[0].read_only
    assert all(v.read_only for v in volumes[1:])
    assert not any(v.host.path == "/host/workspaces" for v in volumes)
    policy = kwargs["network_policy"]
    assert policy.egress[0].target == "172.17.0.1" and policy.egress[0].ports == [7883]
    assert any(rule.action == "deny" and rule.target == "172.16.0.0/12" for rule in policy.egress)
    assert kwargs["resource"] == {"cpu": "500m", "memory": "1Gi"}
    assert "env" not in kwargs  # no request credentials in persistent container env
    opts = r.created[0].commands.run.await_args.kwargs["opts"]
    assert opts.working_directory == str(r.workspace)
    assert opts.envs["UV_PYTHON"] == "/usr/local/bin/python-runtime"
    assert opts.envs["UV_PROJECT_ENVIRONMENT"].endswith("/.venv-opensandbox")
    assert opts.envs["HOME"] == str(r.workspace / "home")
    assert opts.envs["PATH"].startswith(str(r.workspace / ".venv-opensandbox/bin") + ":")
    assert "/opt/miniforge/envs/runtime/bin" in opts.envs["PATH"]
    assert opts.envs["XPEECH_TOKEN_MANAGER_URL"] == "http://172.17.0.1:7883"
    cli = next(v for v in volumes if v.name == "lark-cli")
    assert cli.host.path == "/host/lark-cli" and cli.mount_path == "/opt/xpeech/bin" and cli.read_only


@pytest.mark.asyncio
async def test_lru_eviction_kills_previous_user_before_reusing_capacity(runtime):
    r = runtime
    r.config.max_sandboxes = 1
    await r.executor.run("true", r.workspace, 1000)
    second = r.workspace.parent / "bob"
    second.mkdir()
    await r.executor.run("true", second, 1000)
    r.created[0].kill.assert_awaited_once()
    assert len(r.created) == 2
    record = read_record(r.config.state_path / "resident-0.json")
    assert record["sandbox_id"] == r.created[1].id
    assert record["busy"] is False


@pytest.mark.asyncio
async def test_interrupted_backend_record_is_destroyed_before_next_command(runtime):
    r = runtime
    await r.executor.run("true", r.workspace, 1000)
    path = r.config.state_path / "resident-0.json"
    record = read_record(path)
    record["busy"] = True
    write_record(path, record)
    await r.executor.run("true", r.workspace, 1000)
    r.created[0].kill.assert_awaited_once()
    assert len(r.created) == 2


@pytest.mark.asyncio
async def test_request_token_changes_remount_skills_and_do_not_persist_token(runtime):
    r = runtime
    token = lark_user_access_token_var.set("request-token")
    try:
        await r.executor.run("true", r.workspace, 1000)
    finally:
        lark_user_access_token_var.reset(token)
    options = r.created[0].commands.run.await_args.kwargs["opts"]
    assert options.envs["LARKSUITE_CLI_USER_ACCESS_TOKEN"] == "request-token"
    first_targets = {v.mount_path for v in r.create.await_args.kwargs["volumes"]}
    assert str(r.workspace / "skills/lark-skills-token") in first_targets
    await r.executor.run("true", r.workspace, 1000)
    r.created[0].kill.assert_awaited_once()
    options = r.created[1].commands.run.await_args.kwargs["opts"]
    assert options.envs["LARKSUITE_CLI_USER_ACCESS_TOKEN"] == ""
    assert "request-token" not in "".join(p.read_text() for p in r.config.state_path.glob("*.json"))


@pytest.mark.asyncio
async def test_cancellation_interrupts_and_destroys_remote_command_before_releasing_slot(runtime):
    r = runtime
    await r.executor.run("true", r.workspace, 1000)
    sandbox = r.created[0]
    started = asyncio.Event()

    async def run(_command, *, handlers, **kwargs):
        await handlers.on_init(SimpleNamespace(id="running-command"))
        started.set()
        await asyncio.Event().wait()

    sandbox.commands.run.side_effect = run
    task = asyncio.create_task(r.executor.run("sleep 100", r.workspace, 1000))
    await asyncio.wait_for(started.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    sandbox.commands.interrupt.assert_awaited_once_with("running-command")
    sandbox.kill.assert_awaited_once()
    assert read_record(r.config.state_path / "resident-0.json") == {}
    await r.executor.run("true", r.workspace, 1000)
    assert len(r.created) == 2


@pytest.mark.asyncio
async def test_failed_delete_retains_capacity_and_never_creates_a_replacement(runtime):
    r = runtime
    r.config.max_sandboxes = 1
    await r.executor.run("true", r.workspace, 1000)
    r.created[0].kill.side_effect = RuntimeError("Docker unavailable")
    second = r.workspace.parent / "bob"
    second.mkdir()
    with pytest.raises(RuntimeError, match="Docker unavailable"):
        await r.executor.run("true", second, 1000)
    assert len(r.created) == 1
    assert read_record(r.config.state_path / "resident-0.json")["sandbox_id"] == "sandbox-0"


@pytest.mark.asyncio
async def test_unknown_create_outcome_keeps_reserved_slot_until_remote_expiry(runtime):
    r = runtime
    r.config.max_sandboxes = 1
    r.create.side_effect = TimeoutError("Create outcome unknown")
    with pytest.raises(TimeoutError, match="unknown"):
        await r.executor.run("true", r.workspace, 1000)
    r.create.reset_mock()
    with pytest.raises(TimeoutError, match="slots are busy"):
        await r.executor.run("true", r.workspace, 1000)
    r.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_sdk_error_without_exit_status_is_not_reported_as_success(runtime):
    r = runtime
    await r.executor.run("true", r.workspace, 1000)
    r.created[0].commands.run.return_value = Execution()
    with pytest.raises(RuntimeError, match="without completion"):
        await r.executor.run("true", r.workspace, 1000)
    r.created[0].kill.assert_awaited_once()


@pytest.mark.asyncio
async def test_timeout_result_preserves_partial_output(runtime):
    r = runtime
    sandbox = SimpleNamespace(commands=SimpleNamespace(run=AsyncMock(return_value=execution(
        stdout="partial\n", stderr=module.TIMEOUT_MARKER.decode(), code=124,
    ))))
    result = await r.executor._run(sandbox, "sleep 10", r.workspace, 100)
    assert result.timed_out and result.returncode == 124
    assert result.stdout == b"partial\n" and result.stderr == b""


@pytest.mark.asyncio
async def test_global_slots_are_shared_and_cancelled_waiters_do_not_leak_locks(tmp_path):
    entered = asyncio.Event()

    async def waiter():
        async with acquire_slot(tmp_path, "command", 1, time.monotonic() + 10):
            entered.set()

    async with acquire_slot(tmp_path, "command", 1, time.monotonic() + 1):
        task = asyncio.create_task(waiter())
        await asyncio.sleep(0.05)
        assert not entered.is_set()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    await waiter()
    assert entered.is_set()


def test_file_lock_is_exclusive_across_backend_processes(tmp_path):
    path = tmp_path / "user.lock"
    lock = FileLock(path)
    assert lock.try_acquire()
    probe = "import fcntl,sys; f=open(sys.argv[1], 'a'); fcntl.flock(f, fcntl.LOCK_EX|fcntl.LOCK_NB)"
    try:
        result = subprocess.run([sys.executable, "-c", probe, str(path)], capture_output=True)
        assert result.returncode != 0 and b"BlockingIOError" in result.stderr
    finally:
        lock.release()
    assert subprocess.run([sys.executable, "-c", probe, str(path)], capture_output=True).returncode == 0


@pytest.mark.parametrize("field", ["host_workspace_root", "host_skills_root", "host_home_config_root", "host_cli_root"])
def test_host_mounts_require_absolute_paths(field):
    with pytest.raises(ValidationError, match="Docker host paths"):
        OpenSandboxConfig(**{field: "../private"})


@pytest.mark.asyncio
@pytest.mark.parametrize("termination", ["exit", "timeout", "cancel"])
async def test_image_supervisor_reaps_detached_jobs_and_keeps_partial_output(tmp_path, termination):
    heartbeat = tmp_path / "heartbeat"
    worker = "while true; do printf x >> heartbeat; sleep 0.05; done"
    command = f"setsid bash -c {shlex.quote(worker)} & while [ ! -s heartbeat ]; do sleep 0.01; done; echo partial; "
    command += "exit 7" if termination == "exit" else "sleep 30"
    process = await asyncio.create_subprocess_exec(
        sys.executable, str(SUPERVISOR), "0.3" if termination == "timeout" else "5", command,
        cwd=tmp_path, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    async with asyncio.timeout(5):
        while not heartbeat.exists():
            await asyncio.sleep(0.01)
        if termination == "cancel":
            process.terminate()
        stdout, stderr = await process.communicate()
    assert process.returncode == {"exit": 7, "timeout": 124, "cancel": 143}[termination]
    if termination != "cancel":
        assert b"partial" in stdout
    assert (module.TIMEOUT_MARKER in stderr) is (termination == "timeout")
    stopped = heartbeat.read_bytes()
    await asyncio.sleep(0.15)
    assert heartbeat.read_bytes() == stopped


def test_image_supervisor_caps_output_and_preserves_nonzero_exit(tmp_path):
    result = subprocess.run(
        [sys.executable, str(SUPERVISOR), "2", "head -c 300000 /dev/zero; echo failure >&2; exit 3"],
        cwd=tmp_path, capture_output=True, timeout=5,
    )
    assert result.returncode == 3
    assert len(result.stdout) < 132000 and b"truncated" in result.stdout
    assert b"failure" in result.stderr


@pytest.mark.asyncio
async def test_delete_acknowledgment_keeps_slot_until_sandbox_disappears(runtime):
    r = runtime
    await r.executor.run("true", r.workspace, 1000)
    sandbox = r.created[0]
    sandbox.get_info.side_effect = [SimpleNamespace(), SandboxApiException(status_code=404)]
    path = r.config.state_path / "resident-0.json"
    await r.executor._destroy(sandbox, path)
    assert sandbox.get_info.await_count == 2
    assert read_record(path) == {}
