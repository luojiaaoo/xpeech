import asyncio
import json
import os
import platform
import shlex
import shutil
import socket
import sys
from pathlib import Path

import pytest
import pytest_asyncio

from xpeech.agent.server.context import lark_user_access_token_var, user_id_var
from xpeech.agent.skills.skill import BUILTIN_SKILLS_DIR
from xpeech.agent.tools import sandbox
from xpeech.agent.tools.filesystem import ReadFileArgs, WriteFileArgs, build_file_tools
from xpeech.agent.tools.helper import safe_resolve_workspace_path
from xpeech.agent.tools.shell import tool as shell_module
from xpeech.agent.tools.shell.session import SandboxManager
from xpeech.agent.tools.shell.tool import _run_wrapped_command
from xpeech.exceptions import PathProtectionError


@pytest_asyncio.fixture(autouse=True)
async def shell_sessions(monkeypatch):
    manager = SandboxManager()
    monkeypatch.setattr(shell_module, "sandbox_manager", manager)
    try:
        yield manager
    finally:
        await manager.close()


def _option_pairs(args: list[str], option: str) -> list[tuple[str, str]]:
    return [(args[index + 1], args[index + 2]) for index, value in enumerate(args) if value == option]


def test_sandbox_homes_are_created_inside_each_workspace(tmp_path: Path) -> None:
    first_workspace = tmp_path / "first-workspace"
    second_workspace = tmp_path / "second-workspace"
    first_workspace.mkdir()
    second_workspace.mkdir()

    first_home = sandbox.get_sandbox_home(first_workspace)
    second_home = sandbox.get_sandbox_home(second_workspace)

    assert first_home == first_workspace / "home"
    assert second_home == second_workspace / "home"
    assert first_home.is_dir()
    assert second_home.is_dir()
    assert first_home != second_home


def test_sandbox_home_rejects_symlink(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "home").symlink_to(tmp_path)

    with pytest.raises(ValueError, match="must not be a symlink"):
        sandbox.get_sandbox_home(workspace)


def test_sandbox_uses_private_network_with_outbound_nat(tmp_path, monkeypatch):
    monkeypatch.setattr(sandbox.settings.path, "sandbox_home_path", tmp_path / "config-home")
    monkeypatch.setattr(sandbox, "iter_builtin_skill_dirs", lambda: iter(()))

    args = sandbox.wrap_sandbox_command("true", tmp_path)

    assert args[:4] == [
        sys.executable, "-I", str(Path(sandbox.__file__).parent / "scripts" / "sandbox_network.py"), "bwrap",
    ]
    assert "--unshare-ipc" in args
    assert "--unshare-net" in args
    assert "rootlesskit" not in args
    assert ("/etc/resolv.conf", "/etc/resolv.conf") not in _option_pairs(args, "--ro-bind-try")


def test_sandbox_can_share_host_network_without_rootlesskit(tmp_path, monkeypatch):
    monkeypatch.setattr(sandbox.settings.path, "sandbox_home_path", tmp_path / "config-home")
    monkeypatch.setattr(sandbox, "iter_builtin_skill_dirs", lambda: iter(()))

    args = sandbox.wrap_sandbox_command("true", tmp_path, network_isolation=False)

    assert args[0] == "bwrap"
    assert "rootlesskit" not in args
    assert not any(arg.startswith("--net=") for arg in args)
    assert "--unshare-net" not in args
    assert "--unshare-pid" in args
    assert "--unshare-ipc" in args
    assert ("/etc/resolv.conf", "/etc/resolv.conf") in _option_pairs(args, "--ro-bind-try")
    assert args[args.index("--cap-drop") + 1] == "ALL"
    assert _option_pairs(args, "--bind") == [(str(tmp_path), str(tmp_path))]
    assert dict(_option_pairs(args, "--setenv"))["HOME"] == str(tmp_path / "home")


def test_sandbox_workspace_does_not_expand_the_backend_home(tmp_path: Path, monkeypatch) -> None:
    backend_home = tmp_path / "backend-home"
    workspace = tmp_path / "~" / "workspace"
    workspace.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(backend_home))
    monkeypatch.setattr(sandbox.settings.path, "sandbox_home_path", tmp_path / "config-home")
    monkeypatch.setattr(sandbox, "iter_builtin_skill_dirs", lambda: iter(()))

    relative_workspace = Path("~/workspace")
    assert sandbox.get_sandbox_home(relative_workspace) == workspace / "home"
    args = sandbox.wrap_sandbox_command("pwd", relative_workspace)
    environment = dict(_option_pairs(args, "--setenv"))
    assert environment["HOME"] == str(workspace / "home")
    assert _option_pairs(args, "--bind") == [(str(workspace), str(workspace))]
    assert args[args.index("--chdir") + 1] == str(workspace)
    assert not backend_home.exists()


@pytest.mark.parametrize("workspace_kind", ["relative", "symlink"])
def test_resolver_normalizes_workspace(tmp_path, monkeypatch, workspace_kind):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(workspace, target_is_directory=True)
    monkeypatch.chdir(tmp_path)
    supplied_workspace = Path("workspace") if workspace_kind == "relative" else alias

    assert safe_resolve_workspace_path(Path("~/notes.txt"), supplied_workspace) == workspace / "home/notes.txt"
    assert safe_resolve_workspace_path(Path("notes.txt"), supplied_workspace) == workspace / "notes.txt"
    with pytest.raises(PathProtectionError, match="Path escapes workspace"):
        safe_resolve_workspace_path(Path("../outside.txt"), supplied_workspace)


@pytest.mark.parametrize(
    ("user_path", "relative_path"),
    [
        ("~", Path("home")),
        ("~/notes/todo.md", Path("home/notes/todo.md")),
        (r"~\notes\todo.md", Path(r"home/notes\todo.md")),
    ],
)
def test_tilde_paths_resolve_to_workspace_home(
    tmp_path: Path,
    user_path: str,
    relative_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    assert safe_resolve_workspace_path(Path(user_path), workspace) == workspace / relative_path


@pytest.mark.parametrize("user_path", ["~root", "~root/secret.txt"])
def test_named_user_homes_are_never_expanded_by_backend(tmp_path, user_path):
    with pytest.raises(PathProtectionError, match="Named-user home paths"):
        safe_resolve_workspace_path(Path(user_path), tmp_path)


@pytest.mark.parametrize(
    "user_path",
    [
        "~/.config/uv/uv.toml",
        "home/.config/uv/uv.toml",
    ],
)
def test_sandbox_home_config_paths_map_to_managed_files_for_reading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    user_path: str,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config_home = tmp_path / "sandbox-home"
    config_file = config_home / ".config/uv/uv.toml"
    config_file.parent.mkdir(parents=True)
    config_file.write_text("managed config", encoding="utf-8")
    monkeypatch.setattr(sandbox.settings.path, "sandbox_home_path", config_home)

    assert (
        safe_resolve_workspace_path(
            Path(user_path),
            workspace,
            protect_read_only_file=False,
        )
        == config_file
    )


def test_sandbox_home_config_paths_are_read_only_to_file_tools(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config_home = tmp_path / "sandbox-home"
    config_file = config_home / ".npmrc"
    config_home.mkdir()
    config_file.write_text("managed config", encoding="utf-8")
    monkeypatch.setattr(sandbox.settings.path, "sandbox_home_path", config_home)

    with pytest.raises(PathProtectionError, match="HOME config files are read-only"):
        safe_resolve_workspace_path(Path("~/.npmrc"), workspace)


def test_unmapped_sandbox_home_paths_stay_in_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config_home = tmp_path / "sandbox-home"
    config_home.mkdir()
    monkeypatch.setattr(sandbox.settings.path, "sandbox_home_path", config_home)

    assert safe_resolve_workspace_path(Path("~/notes/todo.md"), workspace) == workspace / "home/notes/todo.md"


@pytest.mark.asyncio
async def test_file_tools_can_read_but_not_write_mapped_home_configs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config_home = tmp_path / "sandbox-home"
    config_home.mkdir()
    config_file = config_home / ".npmrc"
    config_file.write_text("registry=https://example.invalid\n", encoding="utf-8")
    monkeypatch.setattr(sandbox.settings.path, "sandbox_home_path", config_home)
    _, _, read_file, write_file, _ = build_file_tools(workspace)

    result = await read_file(ReadFileArgs(path="~/.npmrc"))

    assert "registry=https://example.invalid" in result
    with pytest.raises(PathProtectionError, match="HOME config files are read-only"):
        await write_file(WriteFileArgs(path="~/.npmrc", content="replacement"))
    assert config_file.read_text(encoding="utf-8") == "registry=https://example.invalid\n"


def test_sandbox_command_uses_private_home_and_maps_mirror_configs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config_home = tmp_path / "sandbox-home"
    config_files = (
        Path(".config/uv/uv.toml"),
        Path(".pip/pip.conf"),
        Path(".npmrc"),
        Path("nested/custom/config.ini"),
    )
    for relative_path in config_files:
        config_file = config_home / relative_path
        config_file.parent.mkdir(parents=True, exist_ok=True)
        config_file.write_text("mirror config", encoding="utf-8")

    monkeypatch.setattr(sandbox.settings.path, "sandbox_home_path", config_home)
    monkeypatch.setattr(sandbox, "iter_builtin_skill_dirs", lambda: iter(()))
    user_id = "test-session"

    session_token = user_id_var.set(user_id)
    try:
        args = sandbox.wrap_sandbox_command("pwd", workspace)
    finally:
        user_id_var.reset(session_token)
    environment = dict(_option_pairs(args, "--setenv"))
    read_only_binds = set(_option_pairs(args, "--ro-bind"))
    writable_binds = set(_option_pairs(args, "--bind"))
    home = workspace / "home"

    assert environment["HOME"] == str(home)
    assert environment["UV_CACHE_DIR"] == str(home / ".cache/uv")
    assert environment["NPM_CONFIG_PREFIX"] == str(home / ".npm-global")
    assert environment["SHELL_USER_ID"] == user_id
    assert environment["PATH"].startswith(f"{home / '.local/bin'}:{home / '.npm-global/bin'}:")
    assert writable_binds == {(str(workspace), str(workspace))}
    assert {
        (str(config_home / relative_path), str(home / relative_path)) for relative_path in config_files
    } <= read_only_binds


def test_sandbox_command_selects_lark_skills_by_injected_token(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    def wrapped(token: str | None) -> tuple[dict[str, str], set[tuple[str, str]]]:
        context = lark_user_access_token_var.set(token)
        try:
            args = sandbox.wrap_sandbox_command("true", workspace)
        finally:
            lark_user_access_token_var.reset(context)
        return dict(_option_pairs(args, "--setenv")), set(_option_pairs(args, "--ro-bind"))

    plain_environment, plain_binds = wrapped(None)
    token_environment, token_binds = wrapped("u-request-token")
    skills_target = workspace / "skills"
    builtin_root = BUILTIN_SKILLS_DIR.resolve()

    assert "LARKSUITE_CLI_USER_ACCESS_TOKEN" not in plain_environment
    assert (str(builtin_root / "lark-skills"), str(skills_target / "lark-skills")) in plain_binds
    assert (str(builtin_root / "lark-cli-auth"), str(skills_target / "lark-cli-auth")) in plain_binds

    assert token_environment["LARKSUITE_CLI_USER_ACCESS_TOKEN"] == "u-request-token"
    assert (
        str(builtin_root / "lark-skills-token"),
        str(skills_target / "lark-skills-token"),
    ) in token_binds
    assert not any(target == str(skills_target / "lark-cli-auth") for _, target in token_binds)


@pytest.mark.skipif(
    platform.system() != "Linux" or any(
        shutil.which(name) is None for name in (*sandbox.NETWORK_SANDBOX_EXECUTABLES, "setsid")
    ),
    reason="Requires Linux with bubblewrap, slirp4netns, bash and setsid",
)
@pytest.mark.asyncio
@pytest.mark.parametrize("termination", ["exit", "timeout", "cancel"])
async def test_shell_sandbox_isolates_processes_and_cleans_up_background_jobs_on_close(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    termination: str,
    shell_sessions,
) -> None:
    workspace = tmp_path / "workspaces" / "current"
    workspace.mkdir(parents=True)
    config_home = tmp_path / "managed-home"
    config_home.mkdir()
    managed_config = config_home / ".audit-config"
    managed_config.write_text("read-only config", encoding="utf-8")
    monkeypatch.setattr(sandbox.settings.path, "sandbox_home_path", config_home)
    monkeypatch.setattr(sandbox, "iter_builtin_skill_dirs", list)
    heartbeat = workspace / "heartbeat"
    worker = "for ((i=0; i<40; i++)); do printf x >> heartbeat; sleep 0.05; done"
    command = (
        "readlink /proc/self/ns/pid > isolation.txt; "
        "readlink /proc/self/ns/net >> isolation.txt; "
        "readlink /proc/self/ns/ipc >> isolation.txt; "
        "awk '/^CapEff:/ {print $2}' /proc/self/status >> isolation.txt; "
        "if (printf modified > home/.audit-config) 2>/dev/null; then exit 1; fi; "
        f"setsid bash -c {shlex.quote(worker)} & "
        "while [ ! -s heartbeat ]; do sleep 0.01; done; "
        "echo partial-output; "
        + ("true" if termination == "exit" else "sleep 30")
    )
    task = asyncio.create_task(
        _run_wrapped_command(command, workspace, timeout_ms=1_000 if termination == "timeout" else 10_000)
    )
    if termination == "cancel":
        try:
            async with asyncio.timeout(5):
                while not heartbeat.exists() or heartbeat.stat().st_size == 0:
                    if task.done():
                        pytest.fail(f"Sandbox exited before starting background job: {task.result()}")
                    await asyncio.sleep(0.01)
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
    else:
        result = await task
        assert result.timed_out is (termination == "timeout")
        assert b"partial-output" in result.stdout
        if termination == "exit":
            assert result.returncode == 0

    await shell_sessions.close()

    pid_namespace, net_namespace, ipc_namespace, capabilities = (workspace / "isolation.txt").read_text().splitlines()
    assert pid_namespace != os.readlink("/proc/self/ns/pid")
    assert net_namespace != os.readlink("/proc/self/ns/net")
    assert ipc_namespace != os.readlink("/proc/self/ns/ipc")
    assert int(capabilities, 16) == 0
    assert managed_config.read_text(encoding="utf-8") == "read-only config"
    await asyncio.sleep(0.1)
    stopped_heartbeat = heartbeat.read_bytes()
    await asyncio.sleep(0.2)
    assert heartbeat.read_bytes() == stopped_heartbeat


@pytest.mark.skipif(
    platform.system() != "Linux" or any(
        shutil.which(name) is None for name in sandbox.NETWORK_SANDBOX_EXECUTABLES
    ) or not Path("/usr/bin/python3").exists(),
    reason="Requires Linux sandbox dependencies and system Python",
)
@pytest.mark.asyncio
async def test_concurrent_sandboxes_have_private_ports_and_ipc(tmp_path, monkeypatch):
    monkeypatch.setattr(sandbox.settings.path, "sandbox_home_path", tmp_path / "config-home")
    monkeypatch.setattr(sandbox, "iter_builtin_skill_dirs", lambda: iter(()))
    program = """
import json, os, socket, sys
tcp = socket.socket()
tcp.bind(('0.0.0.0', 43821))
tcp.listen()
abstract = socket.socket(socket.AF_UNIX)
abstract.bind(b'\\0xpeech-private-network')
probe = socket.socket()
probe.settimeout(0.5)
assert probe.connect_ex(('10.0.2.2', int(sys.argv[1]))) != 0
print(json.dumps({name: os.readlink('/proc/self/ns/' + name) for name in ('net', 'ipc')}), flush=True)
sys.stdin.readline()
"""
    processes = []
    namespaces = []
    with socket.socket() as host_listener:
        host_listener.bind(("127.0.0.1", 0))
        host_listener.listen()
        host_port = host_listener.getsockname()[1]
        try:
            for name in ("first", "second"):
                workspace = tmp_path / name
                workspace.mkdir()
                process = await asyncio.create_subprocess_exec(
                    *sandbox.wrap_sandbox_command(["/usr/bin/python3", "-c", program, str(host_port)], workspace),
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                processes.append(process)
                line = await asyncio.wait_for(process.stdout.readline(), timeout=20)
                if not line:
                    _, stderr = await process.communicate()
                    pytest.fail(f"Sandbox failed to start: {stderr.decode(errors='replace')}")
                namespaces.append(json.loads(line))
            for kind in ("net", "ipc"):
                assert namespaces[0][kind] != namespaces[1][kind]
                assert all(ns[kind] != os.readlink(f"/proc/self/ns/{kind}") for ns in namespaces)
            for process in processes:
                _, stderr = await asyncio.wait_for(process.communicate(b"\n"), timeout=10)
                assert process.returncode == 0, stderr.decode(errors="replace")
        finally:
            for process in processes:
                if process.returncode is None:
                    process.terminate()
                    await asyncio.wait_for(process.communicate(), timeout=10)
