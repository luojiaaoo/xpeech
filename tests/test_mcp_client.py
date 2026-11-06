import asyncio
import json
import os
import platform
import shlex
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from xpeech.agent.server.context import lark_user_access_token_var
from xpeech.agent.skills import skill as skill_module
from xpeech.agent.skills.skill import SkillsLoader
from xpeech.agent.tools import mcp_client
from xpeech.config.settings import MCPServerSettings, ToolConfig, settings


@pytest.fixture
def stdio_transport(tmp_path, monkeypatch):
    """Exercise connection ownership and sandbox wrapping without launching a server."""
    import mcp
    from mcp.client import stdio

    builtin_root = tmp_path / "builtin"
    builtin_root.mkdir()
    monkeypatch.setattr(skill_module, "BUILTIN_SKILLS_DIR", builtin_root)
    monkeypatch.setattr(settings.path, "sandbox_home_path", tmp_path / "managed-home")
    monkeypatch.setattr(mcp_client.platform, "system", lambda: "Linux")
    monkeypatch.setattr(mcp_client.shutil, "which", lambda name: f"/usr/bin/{name}")
    sessions = []
    calls = []
    started = asyncio.Event()
    release = asyncio.Event()

    @asynccontextmanager
    async def transport(params):
        yield params, None

    class Session:
        def __init__(self, params, _write, **_kwargs):
            self.params = params
            self.version = len(sessions) + 1
            self.closed = False
            sessions.append(self)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            self.closed = True

        async def initialize(self):
            pass

        async def list_tools(self):
            return SimpleNamespace(tools=[{"name": "echo", "description": f"session-{self.version}"}])

        async def call_tool(self, name, arguments):
            calls.append((self.version, name))
            if name == "wait":
                started.set()
                await release.wait()
            return {"content": [{"type": "text", "text": f"session-{self.version}"}]}

    monkeypatch.setattr(stdio, "stdio_client", transport)
    monkeypatch.setattr(mcp, "ClientSession", Session)
    return SimpleNamespace(
        sessions=sessions, calls=calls, started=started, release=release, builtin_root=builtin_root,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["add_skill", "remove_skill", "token", "home_config"])
async def test_stdio_rebuilds_when_wrapped_command_changes(tmp_path, stdio_transport, change):
    state = stdio_transport
    skill_dir = state.builtin_root / "new-skill"

    def add_skill():
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\nname: new-skill\n---\nInstructions")

    if change == "remove_skill":
        add_skill()
    registration = mcp_client.create_mcp_registration(
        server_name="demo", command="demo", workspace=tmp_path, skills_loader=SkillsLoader(tmp_path),
    )
    context = lark_user_access_token_var.set(None)
    try:
        await registration._get_tool_bindings()
        assert await registration.call_tool("echo", {}) == "session-1"
        assert len(state.sessions) == 1
        if change == "add_skill":
            add_skill()
        elif change == "remove_skill":
            (skill_dir / "SKILL.md").unlink()
        elif change == "home_config":
            config_home = settings.path.sandbox_home_path
            config_home.mkdir()
            (config_home / ".npmrc").write_text("managed config")

        tokens = ("old-token", "new-token", None) if change == "token" else (None,)
        for version, token in enumerate(tokens, start=2):
            lark_user_access_token_var.set(token)
            assert await registration.call_tool("echo", {}) == f"session-{version}"
            assert len(state.sessions) == version
            assert state.sessions[-2].closed
            params = state.sessions[-1].params
            assert registration._wrapped_command == [params.command, *params.args]
            if change == "token":
                if token is not None:
                    assert token in params.args
                else:
                    assert "--unsetenv" in params.args
            elif change in {"add_skill", "remove_skill"}:
                assert (str(skill_dir) in params.args) is (change == "add_skill")
        assert state.calls == [(1, "echo"), *[(version, "echo") for version in range(2, len(tokens) + 2)]]
    finally:
        lark_user_access_token_var.reset(context)
        await registration.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_stdio_rebuild_waits_for_active_calls_without_serializing_unchanged_calls(
    tmp_path, stdio_transport, cancel,
):
    state = stdio_transport
    registration = mcp_client.create_mcp_registration(
        server_name="demo", command="demo", workspace=tmp_path, skills_loader=SkillsLoader(tmp_path),
    )
    context = lark_user_access_token_var.set(None)
    pending = asyncio.create_task(registration.call_tool("wait", {}))
    rebuilding = None
    try:
        await asyncio.wait_for(state.started.wait(), 1)
        assert await asyncio.wait_for(registration.call_tool("fast", {}), 1) == "session-1"
        lark_user_access_token_var.set("new-token")
        rebuilding = asyncio.create_task(registration.call_tool("after", {}))
        await asyncio.sleep(0.02)
        assert not rebuilding.done()
        assert len(state.sessions) == 1
        assert not state.sessions[0].closed
        if cancel:
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
        else:
            state.release.set()
            assert await asyncio.wait_for(pending, 1) == "session-1"
        assert await asyncio.wait_for(rebuilding, 1) == "session-2"
        assert state.sessions[0].closed
        assert registration._active_calls == 0
        assert state.calls == [(1, "wait"), (1, "fast"), (2, "after")]
    finally:
        lark_user_access_token_var.reset(context)
        state.release.set()
        if rebuilding is not None:
            rebuilding.cancel()
        pending.cancel()
        await asyncio.gather(*(task for task in (pending, rebuilding) if task is not None), return_exceptions=True)
        await registration.aclose()


@pytest.mark.asyncio
async def test_stdio_discovery_refreshes_sandbox_and_tool_schemas(tmp_path, stdio_transport):
    registration = mcp_client.create_mcp_registration(
        server_name="demo", command="demo", workspace=tmp_path, skills_loader=SkillsLoader(tmp_path),
    )
    try:
        first = await registration.start_tool_discovery()
        assert first[0].schema["function"]["description"] == "session-1"
        assert await registration.start_tool_discovery() == first
        assert len(stdio_transport.sessions) == 1
        skill_dir = stdio_transport.builtin_root / "added"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\nname: added\n---\nInstructions")
        second = await registration.start_tool_discovery()
        assert second[0].schema["function"]["description"] == "session-2"
        assert len(stdio_transport.sessions) == 2
        assert stdio_transport.sessions[0].closed
    finally:
        await registration.aclose()


def test_registration_normalizes_workspace_without_expanding_home(tmp_path, monkeypatch):
    workspace = tmp_path / "~" / "workspace"
    workspace.mkdir(parents=True)
    alias = tmp_path / "alias"
    alias.symlink_to(workspace, target_is_directory=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "backend-home"))

    for supplied_workspace in (Path("~/workspace"), alias):
        registration = mcp_client.create_mcp_registration(
            server_name="demo", command="demo", workspace=supplied_workspace,
            skills_loader=SkillsLoader(supplied_workspace),
        )
        assert registration.config.workspace == workspace
        assert registration.config.workspace.as_uri() == workspace.as_uri()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("patterns", "expected"),
    [
        ([], set()),
        (["*"], {"read_file", "write_file", "delete_record", "tool.with.dots"}),
        (["read_file"], {"read_file"}),
        (["read_*", "write_*"], {"read_file", "write_file"}),
        (["write_?ile"], {"write_file"}),
        (["[rw]*"], {"read_file", "write_file"}),
        (["tool.*"], {"tool.with.dots"}),
        (["READ_*"], set()),
        (["mcp_demo_read_file"], set()),
        (["mcp_demo_*"], set()),
    ],
)
async def test_enabled_tools_match_original_names_with_globs(tmp_path, patterns, expected):
    names = {"read_file", "write_file", "delete_record", "tool.with.dots"}

    class Session:
        async def list_tools(self):
            return SimpleNamespace(tools=[
                {"name": name, "inputSchema": {"type": "object", "properties": {}}}
                for name in sorted(names)
            ])

    registration = mcp_client.create_mcp_registration(
        server_name="demo", command="demo", enabled_tools=patterns, workspace=tmp_path,
        skills_loader=SkillsLoader(tmp_path),
    )
    registration._session = Session()
    bindings = await registration._discover_tool_bindings()
    assert {binding.func.mcp_original_name for binding in bindings} == expected
    assert {binding.func.__name__ for binding in bindings} == {
        mcp_client._mcp_tool_name("demo", name) for name in expected
    }


def test_supports_the_three_mcp_transports(tmp_path):
    cases = [
        ({"command": "server"}, "stdio"),
        ({"url": "https://example.test/sse"}, "sse"),
        ({"url": "https://example.test/mcp"}, "streamable-http"),
        ({"url": "https://example.test/events", "transport": "sse"}, "sse"),
        ({"url": "https://example.test/mcp", "transport": "streamable-http"}, "streamable-http"),
    ]

    for kwargs, expected in cases:
        registration = mcp_client.create_mcp_registration(server_name="test", **kwargs, workspace=tmp_path, skills_loader=SkillsLoader(tmp_path))
        assert registration.config.transport == expected


@pytest.mark.parametrize("transport", ["streamable_http", "http-stream", "http"])
def test_rejects_nonstandard_transport_names(tmp_path, transport: str):
    with pytest.raises(ValueError, match="Unsupported MCP transport"):
        mcp_client.create_mcp_registration(
            server_name="test",
            url="https://example.test/mcp",
            transport=transport,
            workspace=tmp_path,
            skills_loader=SkillsLoader(tmp_path),
        )


def test_tls_verification_can_be_disabled_from_config(tmp_path):
    settings = ToolConfig.model_validate(
        {
            "mcpServers": {
                "test": {
                    "description": "查询客户资料和订单",
                    "url": "https://example.test/mcp",
                    "command_authorization_token": "token-helper --user $user_id",
                    "headers": {"Authorization": "Bearer ${authorization_token}"},
                    "verify_tls": False,
                    "connect_timeout": 45,
                }
            }
        }
    )
    registration = mcp_client.create_mcp_registration_from_config(
        "test",
        settings.mcp_servers["test"],
        workspace=tmp_path,
        skills_loader=SkillsLoader(tmp_path),
    )

    assert registration.config.verify_tls is False
    assert registration.config.description == "查询客户资料和订单"
    assert registration.config.connect_timeout == 45
    assert registration.config.command_authorization_token == "token-helper --user $user_id"
    assert registration.config.headers == {"Authorization": "Bearer ${authorization_token}"}
    assert registration.config.user_scoped is False


def test_user_scoped_is_loaded_from_config(tmp_path):
    config = MCPServerSettings(command="server", user_scoped=True)
    registration = mcp_client.create_mcp_registration_from_config("test", config, tmp_path, skills_loader=SkillsLoader(tmp_path))
    assert registration.config.user_scoped is True


def test_stdio_env_is_loaded_from_server_config(tmp_path):
    config = ToolConfig.model_validate({
        "mcpServers": {"local": {"command": "server", "env": {"API_KEY": "test-key"}}},
    })

    registration = mcp_client.create_mcp_registration_from_config(
        "local", config.mcp_servers["local"], workspace=tmp_path,
        skills_loader=SkillsLoader(tmp_path),
    )

    assert registration.config.env == {"API_KEY": "test-key"}


@pytest.mark.parametrize(("config", "message"), [
    ({"command": "server", "env": {"PATH": "/custom/bin"}}, "cannot configure PATH"),
    ({"command": "server", "env": {"HOME": "/custom/home"}}, "cannot configure HOME"),
    ({"url": "https://example.test/mcp", "env": {"API_KEY": "test-key"}}, "only supported for stdio"),
    ({"url": "https://example.test/sse", "env": {}}, "only supported for stdio"),
])
def test_env_restrictions_apply_to_config_and_direct_registration(tmp_path, config, message):
    with pytest.raises(ValueError, match=message):
        MCPServerSettings.model_validate(config)

    with pytest.raises(ValueError, match=message):
        mcp_client.create_mcp_registration(server_name="test", workspace=tmp_path, **config, skills_loader=SkillsLoader(tmp_path))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("user_id_placeholder", "token_placeholder"),
    [
        ("$user_id", "$authorization_token"),
        ("${user_id}", "${authorization_token}"),
    ],
)
async def test_resolves_per_user_authorization_headers_without_a_shell(
    monkeypatch: pytest.MonkeyPatch,
    user_id_placeholder: str,
    token_placeholder: str,
):
    process_args = None

    class FakeProcess:
        returncode = 0

        async def communicate(self):
            return b"secret-token\n", b""

    async def create_subprocess_exec(*args, **_kwargs):
        nonlocal process_args
        process_args = args
        return FakeProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess_exec)
    user_id = "user; echo not-a-command"

    headers = await mcp_client._resolve_http_headers(
        {
            "Authorization": f"Bearer {token_placeholder}",
            "X-User-ID": user_id_placeholder,
            "X-Literal": "$user_identifier",
        },
        f"token-helper --user {user_id_placeholder}",
        user_id,
    )

    assert process_args == ("token-helper", "--user", user_id)
    assert headers == {
        "Authorization": "Bearer secret-token",
        "X-User-ID": user_id,
        "X-Literal": "$user_identifier",
    }


@pytest.mark.asyncio
async def test_authorization_token_header_requires_token_command():
    with pytest.raises(ValueError, match="require command_authorization_token"):
        await mcp_client._resolve_http_headers(
            {"Authorization": "Bearer $authorization_token"},
            None,
            "user-1",
        )


@pytest.mark.asyncio
async def test_does_not_run_token_command_without_authorization_token_placeholder(
    monkeypatch: pytest.MonkeyPatch,
):
    async def unexpected_subprocess(*_args, **_kwargs):
        raise AssertionError("token command should not run")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", unexpected_subprocess)

    headers = await mcp_client._resolve_http_headers(
        {"X-User-ID": "$user_id"},
        "token-helper $user_id",
        "user-1",
    )

    assert headers == {"X-User-ID": "user-1"}


def test_authorization_token_command_requires_remote_mcp(tmp_path):
    with pytest.raises(ValueError, match="requires a remote URL"):
        mcp_client.create_mcp_registration(
            server_name="test",
            command="fake-server",
            command_authorization_token="token-helper $user_id",
            workspace=tmp_path,
            skills_loader=SkillsLoader(tmp_path),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("configured_env", [
    None,
    {},
    {"CUSTOM_SETTING": "value"},
])
async def test_stdio_uses_workspace_home_and_passes_arguments_through(tmp_path, monkeypatch, configured_env):
    import mcp
    from mcp.client import stdio

    monkeypatch.setenv("HOME", str(tmp_path / "backend-home"))
    monkeypatch.setattr(mcp_client.platform, "system", lambda: "Linux")
    # Stdio MCP must work with only the base sandbox dependencies installed.
    monkeypatch.setattr(
        mcp_client.shutil, "which",
        lambda name: f"/usr/bin/{name}" if name in ("bwrap", "bash") else None,
    )
    captured_params = []
    calls = []

    @asynccontextmanager
    async def fake_transport(params):
        captured_params.append(params)
        yield object(), object()

    class FakeSession:
        def __init__(self, *_args, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def initialize(self):
            return None

        async def call_tool(self, name, arguments):
            calls.append((name, arguments))
            return SimpleNamespace(content=[])

    monkeypatch.setattr(stdio, "stdio_client", fake_transport)
    monkeypatch.setattr(mcp, "ClientSession", FakeSession)

    for user in ("alice", "bob"):
        workspace = tmp_path / user
        workspace.mkdir()
        registration = mcp_client.create_mcp_registration(
            server_name="test", command="fake-server", args=["--config", "~/server-config", "literal ; $(touch injected)"],
            workspace=workspace, env=configured_env,
            skills_loader=SkillsLoader(workspace),
        )
        ready = asyncio.get_running_loop().create_future()
        close_event = asyncio.Event()
        owner = asyncio.create_task(registration._run_connection(ready, close_event))
        try:
            await ready
            params = captured_params[-1]
            assert Path(params.cwd) == workspace
            assert params.command == "bwrap"
            sandbox_command = [params.command, *params.args]
            environment = {
                sandbox_command[index + 1]: sandbox_command[index + 2]
                for index, argument in enumerate(sandbox_command) if argument == "--setenv"
            }
            assert environment["HOME"] == str(workspace / "home")
            home = workspace / "home"
            assert environment["PATH"].startswith(f"{home / '.local/bin'}:{home / '.npm-global/bin'}:")
            assert "--unshare-pid" in sandbox_command
            assert "--unshare-ipc" in sandbox_command
            assert "rootlesskit" not in sandbox_command
            assert "--net=slirp4netns" not in sandbox_command
            assert "--disable-host-loopback" not in sandbox_command
            assert "--port-driver=none" not in sandbox_command
            assert "--unshare-net" not in sandbox_command
            assert sandbox_command[sandbox_command.index("--cap-drop") + 1] == "ALL"
            assert sandbox_command[sandbox_command.index("--cap-add") + 1] == "CAP_DAC_OVERRIDE"
            assert sandbox_command[sandbox_command.index("--chdir") + 1] == str(workspace)
            assert params.env == configured_env
            assert (workspace / "home").is_dir()
            wrapped = sandbox_command[sandbox_command.index("--") + 1:]
            assert wrapped[:2] == ["bash", "-c"]
            assert shlex.split(wrapped[2]) == [
                "fake-server", "--config", "~/server-config", "literal ; $(touch injected)",
            ]
            assert registration.config.env == configured_env
            arguments = {"path": "~/notes.txt", "remote_path": "/remote/data", "text": "~/literal"}
            await registration.call_tool("example", arguments)
            assert calls[-1] == ("example", arguments)
        finally:
            close_event.set()
            await owner

    assert captured_params[0].args != captured_params[1].args
    assert not (tmp_path / "backend-home").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(("system", "bwrap"), [("Windows", "/usr/bin/bwrap"), ("Linux", None)])
async def test_stdio_refuses_to_launch_without_sandbox(tmp_path, monkeypatch, system, bwrap):
    from mcp.client import stdio

    def unexpected_transport(_params):
        raise AssertionError("Must not launch an unsandboxed process")

    monkeypatch.setattr(mcp_client.platform, "system", lambda: system)
    monkeypatch.setattr(mcp_client.shutil, "which", lambda _name: bwrap)
    monkeypatch.setattr(stdio, "stdio_client", unexpected_transport)
    registration = mcp_client.create_mcp_registration(
        server_name="test", command="fake-server", workspace=tmp_path,
        skills_loader=SkillsLoader(tmp_path),
    )
    ready = asyncio.get_running_loop().create_future()

    await registration._run_connection(ready, asyncio.Event())

    with pytest.raises(RuntimeError, match="requires Linux with bubblewrap"):
        await ready
    assert registration._session is None


@pytest.mark.asyncio
@pytest.mark.parametrize("missing", ["bash"])
async def test_stdio_refuses_to_launch_without_required_executable(tmp_path, monkeypatch, missing):
    from mcp.client import stdio

    def unexpected_transport(_params):
        raise AssertionError("Must not launch a process without the required executable")

    monkeypatch.setattr(mcp_client.platform, "system", lambda: "Linux")
    monkeypatch.setattr(mcp_client.shutil, "which", lambda name: None if name == missing else f"/usr/bin/{name}")
    monkeypatch.setattr(stdio, "stdio_client", unexpected_transport)
    registration = mcp_client.create_mcp_registration(
        server_name="test", command="fake-server", workspace=tmp_path,
        skills_loader=SkillsLoader(tmp_path),
    )
    ready = asyncio.get_running_loop().create_future()

    await registration._run_connection(ready, asyncio.Event())

    with pytest.raises(RuntimeError, match=missing):
        await ready
    assert registration._session is None


@pytest.mark.skipif(
    platform.system() != "Linux" or any(shutil.which(name) is None for name in mcp_client.SANDBOX_EXECUTABLES)
    or shutil.which("python3", path="/usr/bin:/bin") is None,
    reason="Requires Linux with bubblewrap and system Python",
)
@pytest.mark.asyncio
async def test_stdio_sandbox_isolates_files_and_cleans_up_children(tmp_path, monkeypatch):
    workspace = tmp_path / "workspaces" / "alice"
    workspace.mkdir(parents=True)
    other_workspace = tmp_path / "workspaces" / "bob"
    other_workspace.mkdir()
    other_file = other_workspace / "secret.txt"
    other_file.write_text("other user data")
    backend_home = tmp_path / "backend-home"
    backend_home.mkdir()
    backend_file = backend_home / "secret.txt"
    backend_file.write_text("backend data")
    config_home = tmp_path / "managed-home"
    config_home.mkdir()
    (config_home / ".mcp-config").write_text("managed config")
    monkeypatch.setenv("HOME", str(backend_home))
    monkeypatch.setattr(settings.path, "sandbox_home_path", config_home)
    monkeypatch.setattr(skill_module, "iter_builtin_skill_dirs", lambda: iter(()))
    (workspace / "worker.py").write_text(
        "import time\nfrom pathlib import Path\nwhile True:\n"
        "    with Path('heartbeat').open('ab') as file:\n        file.write(b'x')\n"
        "    time.sleep(0.02)\n"
    )
    server = workspace / "server.py"
    server.write_text('''
import json
import os
import subprocess
import sys
from pathlib import Path

def can_read(path):
    try:
        path.read_bytes()
        return True
    except OSError:
        return False

for line in sys.stdin:
    request = json.loads(line)
    if "id" not in request:
        continue
    method = request["method"]
    if method == "initialize":
        result = {
            "protocolVersion": request["params"]["protocolVersion"],
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "sandbox-test", "version": "1"},
        }
    elif method == "tools/list":
        result = {"tools": [{
            "name": "inspect", "description": "Inspect sandbox isolation",
            "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        }]}
    elif method == "tools/call":
        home = Path.home()
        config = home / ".mcp-config"
        data = {
            "cwd": os.getcwd(), "home": str(home),
            "backend_readable": can_read(Path(request["params"]["arguments"]["path"])),
            "other_readable": can_read(Path(sys.argv[1])),
            "config": config.read_text(),
            "literal": sys.argv[2], "custom": os.environ["CUSTOM_SETTING"],
            "pid_namespace": os.readlink("/proc/self/ns/pid"),
            "net_namespace": os.readlink("/proc/self/ns/net"),
            "ipc_namespace": os.readlink("/proc/self/ns/ipc"),
            "capabilities": next(line for line in Path("/proc/self/status").read_text().splitlines() if line.startswith("CapEff:")),
        }
        try:
            config.write_text("modified")
            data["config_writable"] = True
        except OSError:
            data["config_writable"] = False
        (home / "result.txt").write_text("workspace data")
        subprocess.Popen(
            [sys.executable, "-u", "worker.py"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
        )
        result = {"content": [{"type": "text", "text": json.dumps(data)}], "isError": False}
    else:
        result = {}
    print(json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": result}), flush=True)
''')
    literal = "spaces ; $(touch injected) 'quotes'"
    registration = mcp_client.create_mcp_registration(
        server_name="sandbox-test", command=shutil.which("python3", path="/usr/bin:/bin"),
        args=["-u", str(server), str(other_file), literal], workspace=workspace,
        env={"CUSTOM_SETTING": "configured"},
        skills_loader=SkillsLoader(workspace),
    )
    try:
        bindings = await registration._get_tool_bindings()
        assert len(bindings) == 1
        model = bindings[0].func.__annotations__["args"]
        result = json.loads(await bindings[0].func(model(path=str(backend_file))))
        assert result["cwd"] == str(workspace)
        assert result["home"] == str(workspace / "home")
        assert not result["backend_readable"]
        assert not result["other_readable"]
        assert result["config"] == "managed config"
        assert not result["config_writable"]
        assert result["literal"] == literal
        assert result["custom"] == "configured"
        assert result["pid_namespace"] != os.readlink("/proc/self/ns/pid")
        assert result["net_namespace"] == os.readlink("/proc/self/ns/net")
        assert result["ipc_namespace"] != os.readlink("/proc/self/ns/ipc")
        assert int(result["capabilities"].split()[1], 16) == 1 << 1  # CAP_DAC_OVERRIDE
        assert (workspace / "home/result.txt").read_text() == "workspace data"
        assert not (workspace / "injected").exists()
        async with asyncio.timeout(5):
            while not (workspace / "heartbeat").exists():
                await asyncio.sleep(0.01)
    finally:
        await registration.aclose()

    stopped_heartbeat = (workspace / "heartbeat").read_bytes()
    await asyncio.sleep(0.15)
    assert (workspace / "heartbeat").read_bytes() == stopped_heartbeat
    assert backend_file.read_text() == "backend data"
    assert (config_home / ".mcp-config").read_text() == "managed config"


@pytest.mark.asyncio
async def test_streamable_http_passes_tls_setting_to_httpx(tmp_path, monkeypatch: pytest.MonkeyPatch):
    import mcp
    from mcp.client import streamable_http

    client_kwargs = {}

    class FakeHttpClient:
        def __init__(self, **kwargs):
            client_kwargs.update(kwargs)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

    @asynccontextmanager
    async def fake_transport(_url, *, http_client):
        assert isinstance(http_client, FakeHttpClient)
        yield object(), object(), None

    class FakeSession:
        def __init__(self, *_args, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def initialize(self):
            return None

    monkeypatch.setattr(mcp_client.httpx, "AsyncClient", FakeHttpClient)
    monkeypatch.setattr(streamable_http, "streamable_http_client", fake_transport)
    monkeypatch.setattr(mcp, "ClientSession", FakeSession)

    def unexpected_sandbox(*_args):
        raise AssertionError("Remote MCP must not launch a local sandbox")

    monkeypatch.setattr(mcp_client, "wrap_sandbox_command", unexpected_sandbox)

    registration = mcp_client.create_mcp_registration(
        server_name="test",
        url="https://example.test/mcp",
        headers={"X-User-ID": "$user_id"},
        verify_tls=False,
        connect_timeout=45,
        workspace=tmp_path,
        skills_loader=SkillsLoader(tmp_path),
    )
    registration.user_id = "user-42"
    ready = asyncio.get_running_loop().create_future()
    close_event = asyncio.Event()
    owner = asyncio.create_task(registration._run_connection(ready, close_event))

    await ready
    close_event.set()
    await owner

    assert client_kwargs["verify"] is False
    assert client_kwargs["timeout"].connect == 45
    assert client_kwargs["headers"] == {"X-User-ID": "user-42"}
    assert not (tmp_path / "home").exists()


@pytest.mark.asyncio
async def test_connection_error_is_passed_through_exit_stack(tmp_path, monkeypatch: pytest.MonkeyPatch):
    import mcp
    from mcp.client import streamable_http

    exit_errors = []

    class FakeHttpClient:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

    @asynccontextmanager
    async def fake_transport(_url, *, http_client):
        assert isinstance(http_client, FakeHttpClient)
        try:
            yield object(), object(), None
        except BaseException as exc:
            exit_errors.append(exc)
            raise

    class FailingSession:
        def __init__(self, *_args, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def initialize(self):
            raise ConnectionError("initialize failed")

    monkeypatch.setattr(mcp_client.httpx, "AsyncClient", FakeHttpClient)
    monkeypatch.setattr(streamable_http, "streamable_http_client", fake_transport)
    monkeypatch.setattr(mcp, "ClientSession", FailingSession)

    registration = mcp_client.create_mcp_registration(
        server_name="test",
        url="https://example.test/mcp",
        verify_tls=False,
        workspace=tmp_path,
        skills_loader=SkillsLoader(tmp_path),
    )
    ready = asyncio.get_running_loop().create_future()
    owner = asyncio.create_task(registration._run_connection(ready, asyncio.Event()))

    await owner
    with pytest.raises(ConnectionError, match="initialize failed"):
        await ready

    assert len(exit_errors) == 1
    assert isinstance(exit_errors[0], ConnectionError)


@pytest.mark.asyncio
async def test_collect_mcp_tool_does_not_swallow_cancellation_groups(tmp_path, monkeypatch: pytest.MonkeyPatch):
    registration = mcp_client.create_mcp_registration(
        server_name="test",
        command="fake-server", workspace=tmp_path,
        skills_loader=SkillsLoader(tmp_path),
    )

    async def cancelled():
        raise BaseExceptionGroup("cancelled", [asyncio.CancelledError()])

    monkeypatch.setattr(registration, "_get_tool_bindings", cancelled)

    with pytest.raises(BaseExceptionGroup):
        [item async for item in mcp_client.collect_mcp_tool(registration)]


@pytest.mark.asyncio
async def test_collect_mcp_tool_skips_unavailable_server(tmp_path, monkeypatch: pytest.MonkeyPatch):
    registration = mcp_client.create_mcp_registration(
        server_name="test",
        command="fake-server", workspace=tmp_path,
        skills_loader=SkillsLoader(tmp_path),
    )

    async def unavailable():
        raise ConnectionError("server unavailable")

    monkeypatch.setattr(registration, "_get_tool_bindings", unavailable)

    assert [item async for item in mcp_client.collect_mcp_tool(registration)] == []


@pytest.mark.asyncio
@pytest.mark.parametrize("user_scoped", [False, True])
@pytest.mark.parametrize("transport", ["stdio", "sse", "streamable-http"])
async def test_connections_are_reused_and_expire(tmp_path, monkeypatch: pytest.MonkeyPatch, user_scoped, transport):
    await mcp_client.close_all_mcp_connections()
    monkeypatch.setattr(mcp_client, "MCP_CONNECTION_TTL_SECONDS", 0.06)
    monkeypatch.setattr(skill_module, "iter_builtin_skill_dirs", list)

    class FakeSession:
        async def list_tools(self):
            return SimpleNamespace(
                tools=[
                    SimpleNamespace(
                        name="echo",
                        description="Echo",
                        inputSchema={
                            "type": "object",
                            "properties": {"text": {"type": "string"}},
                            "required": ["text"],
                        },
                    )
                ]
            )

        async def call_tool(self, _name, arguments):
            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text=arguments["text"])]
            )

    async def run_fake_connection(registration, ready, close_event):
        registration._session = FakeSession()
        ready.set_result(None)
        await close_event.wait()
        registration._session = None

    monkeypatch.setattr(
        mcp_client.MCPServerRegistration,
        "_run_connection",
        run_fake_connection,
    )
    server_options = {"command": "fake-server"} if transport == "stdio" else {"url": "https://example.test/mcp"}
    config = MCPServerSettings(**server_options, transport=transport, user_scoped=user_scoped)
    isolated_by_user = transport == "stdio" or user_scoped

    first = await mcp_client.get_session_mcp_registration_from_config("session-1", "demo", config, tmp_path, skills_loader=SkillsLoader(tmp_path))
    reused = await mcp_client.get_session_mcp_registration_from_config("session-1", "demo", config, tmp_path, skills_loader=SkillsLoader(tmp_path))
    isolated = await mcp_client.get_session_mcp_registration_from_config("session-2", "demo", config, tmp_path, skills_loader=SkillsLoader(tmp_path))

    assert first is reused
    assert (first is not isolated) is isolated_by_user
    assert set(mcp_client.MCP_CONNECTIONS) == (
        {("session-1", tmp_path), ("session-2", tmp_path)} if isolated_by_user else {first.connection_scope}
    )
    if transport == "stdio":
        assert first.config.workspace == tmp_path
        assert isolated.config.workspace == tmp_path

    binding = (await first._get_tool_bindings())[0]
    await isolated._get_tool_bindings()
    model = binding.func.__annotations__["args"]

    await asyncio.sleep(0.04)
    assert await binding.func(model(text="ok")) == "ok"
    await asyncio.sleep(0.04)
    assert set(mcp_client.MCP_CONNECTIONS) == {first.connection_scope}
    await asyncio.sleep(0.03)
    assert mcp_client.MCP_CONNECTIONS == {}
    await mcp_client.close_all_mcp_connections()


@pytest.mark.asyncio
@pytest.mark.parametrize("options, user_scoped", [
    ({}, False),
    ({"user_scoped": False}, False),
    ({"user_scoped": True}, True),
    ({"command_authorization_token": "token-helper $user_id"}, True),
    ({"command_authorization_token": "token-helper $user_id", "user_scoped": False}, True),
    ({"command_authorization_token": "token-helper $user_id", "user_scoped": True}, True),
    ({"command": "demo"}, True),
    ({"command": "demo", "user_scoped": False}, True),
    ({"command": "demo", "user_scoped": True}, True),
    ({"command": "demo", "transport": "stdio", "user_scoped": False}, True),
])
async def test_connection_scope_and_cleanup(tmp_path, options, user_scoped):
    await mcp_client.close_all_mcp_connections()
    config = MCPServerSettings(**(options if "command" in options else {"url": "https://example.test/mcp", **options}))
    try:
        first, reused, other = await asyncio.gather(*(
            mcp_client.get_session_mcp_registration_from_config(user_id, "demo", config, tmp_path, skills_loader=SkillsLoader(tmp_path))
            for user_id in ("user-1", "user-1", "user-2")
        ))
        assert first is reused
        assert (first is not other) is user_scoped
        assert first.user_id == ("user-1" if user_scoped else None)
        assert other.user_id == ("user-2" if user_scoped else None)

        await mcp_client.close_session_mcp_connections("user-1")
        assert first._retired is user_scoped
        assert not other._retired
        assert await mcp_client.get_session_mcp_registration_from_config(
            "user-2", "demo", config, tmp_path,
            skills_loader=SkillsLoader(tmp_path),
        ) is other
    finally:
        await mcp_client.close_all_mcp_connections()
    assert first._retired
    assert other._retired
    assert not mcp_client._SESSION_EXPIRY_HANDLES


@pytest.mark.asyncio
@pytest.mark.parametrize("stdio", [True, False])
@pytest.mark.parametrize("access", [
    {"channel_tag": "crm"},
    {"user_organization": "company/engineering"},
    {"not_interactive": True},
])
async def test_stdio_cache_separates_skill_visibility(tmp_path, stdio, access, monkeypatch):
    await mcp_client.close_all_mcp_connections()
    monkeypatch.setattr(mcp_client.settings.path, "cache_path", tmp_path / "cache")
    config = MCPServerSettings(**({"command": "demo"} if stdio else {"url": "https://example.test/mcp"}))
    try:
        first = await mcp_client.get_session_mcp_registration_from_config(
            "alice", "demo", config, tmp_path, skills_loader=SkillsLoader(tmp_path),
        )
        changed = await mcp_client.get_session_mcp_registration_from_config(
            "alice", "demo", config, tmp_path, skills_loader=SkillsLoader(tmp_path, **access),
        )
        again = await mcp_client.get_session_mcp_registration_from_config(
            "alice", "demo", config, tmp_path, skills_loader=SkillsLoader(tmp_path, **access),
        )
        assert (first is not changed) is stdio
        assert again is changed
    finally:
        await mcp_client.close_all_mcp_connections()


@pytest.mark.asyncio
async def test_shared_connections_ignore_user_workspaces_but_separate_configs(tmp_path, monkeypatch):
    await mcp_client.close_all_mcp_connections()
    monkeypatch.setattr(mcp_client.settings.path, "cache_path", tmp_path / "cache")
    config = MCPServerSettings(url="https://example.test/mcp")
    try:
        first = await mcp_client.get_session_mcp_registration_from_config("user-1", "demo", config, tmp_path, skills_loader=SkillsLoader(tmp_path))
        changed_config = await mcp_client.get_session_mcp_registration_from_config(
            "user-2", "demo", config.model_copy(update={"headers": {"X-Test": "other"}}), tmp_path,
            skills_loader=SkillsLoader(tmp_path),
        )
        changed_workspace = await mcp_client.get_session_mcp_registration_from_config(
            "user-2", "demo", config, tmp_path / "other",
            skills_loader=SkillsLoader(tmp_path / "other"),
        )
        scoped = await mcp_client.get_session_mcp_registration_from_config(
            "user-1", "demo", config.model_copy(update={"user_scoped": True}), tmp_path,
            skills_loader=SkillsLoader(tmp_path),
        )
        assert changed_workspace is first
        assert first.config.workspace == tmp_path / "cache" / "mcp" / "demo"
        assert first.config.workspace.is_dir()
        assert len({id(first), id(changed_config), id(scoped)}) == 3
        assert scoped.config.workspace == tmp_path
    finally:
        await mcp_client.close_all_mcp_connections()


@pytest.mark.asyncio
@pytest.mark.parametrize("options", [
    {"user_scoped": True},
    {"command_authorization_token": "token-helper $user_id"},
    {"command": "demo", "user_scoped": False},
    {"command": "demo", "transport": "stdio"},
])
async def test_user_workspaces_have_independent_cache_expiry_and_cleanup(tmp_path, options):
    await mcp_client.close_all_mcp_connections()
    config = MCPServerSettings(**(options if "command" in options else {"url": "https://example.test/mcp", **options}))
    workspace = tmp_path / "first"
    other_workspace = tmp_path / "second"
    workspace.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(workspace, target_is_directory=True)
    try:
        first = await mcp_client.get_session_mcp_registration_from_config("user-1", "demo", config, workspace, skills_loader=SkillsLoader(workspace))
        reused = await mcp_client.get_session_mcp_registration_from_config("user-1", "demo", config, alias, skills_loader=SkillsLoader(alias))
        other = await mcp_client.get_session_mcp_registration_from_config("user-1", "demo", config, other_workspace, skills_loader=SkillsLoader(other_workspace))
        assert first is reused
        assert first is not other
        assert set(mcp_client.MCP_CONNECTIONS) == {("user-1", workspace), ("user-1", other_workspace)}

        first_handle = mcp_client._SESSION_EXPIRY_HANDLES[first.connection_scope]
        other_handle = mcp_client._SESSION_EXPIRY_HANDLES[other.connection_scope]
        await mcp_client._refresh_session_expiry(first)
        assert first_handle.cancelled()
        assert mcp_client._SESSION_EXPIRY_HANDLES[other.connection_scope] is other_handle
        await mcp_client._expire_session(first.connection_scope, first_handle)
        assert not first._retired

        other_handle.cancel()
        await mcp_client._expire_session(other.connection_scope, other_handle)
        assert other._retired
        assert not first._retired
        replacement = await mcp_client.get_session_mcp_registration_from_config(
            "user-1", "demo", config, other_workspace,
            skills_loader=SkillsLoader(other_workspace),
        )
        assert replacement is not other

        await mcp_client.close_session_mcp_connections("user-1", alias)
        assert first._retired
        assert not replacement._retired
        assert set(mcp_client.MCP_CONNECTIONS) == {("user-1", other_workspace)}
        await mcp_client.close_session_mcp_connections("user-1")
        assert replacement._retired
        assert not mcp_client.MCP_CONNECTIONS
        assert not mcp_client._SESSION_EXPIRY_HANDLES
    finally:
        await mcp_client.close_all_mcp_connections()


@pytest.mark.asyncio
async def test_failed_connection_is_retried_immediately_by_next_request(tmp_path, monkeypatch: pytest.MonkeyPatch):
    await mcp_client.close_all_mcp_connections()
    monkeypatch.setattr(mcp_client, "MCP_CONNECTION_TTL_SECONDS", 0.2)
    attempts = 0

    async def run_slow_connection(_registration, _ready, _close_event):
        nonlocal attempts
        attempts += 1
        await asyncio.Event().wait()

    monkeypatch.setattr(
        mcp_client.MCPServerRegistration,
        "_run_connection",
        run_slow_connection,
    )
    config = MCPServerSettings(url="https://example.test/mcp", connect_timeout=0.02)
    registration = await mcp_client.get_session_mcp_registration_from_config(
        "session-1",
        "unreachable",
        config,
        workspace=tmp_path,
        skills_loader=SkillsLoader(tmp_path),
    )

    with pytest.raises(asyncio.TimeoutError):
        await registration._get_tool_bindings()
    assert attempts == 1

    cached = await mcp_client.get_session_mcp_registration_from_config(
        "session-1",
        "unreachable",
        config,
        workspace=tmp_path,
        skills_loader=SkillsLoader(tmp_path),
    )
    assert cached is registration
    with pytest.raises(asyncio.TimeoutError):
        await cached._get_tool_bindings()
    assert attempts == 2
    await mcp_client.close_all_mcp_connections()


@pytest.mark.asyncio
async def test_terminated_connection_reconnects_on_next_call_without_replaying_failed_tool(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    failed_calls = []

    class TerminatedSession:
        async def call_tool(self, _name, arguments):
            failed_calls.append(arguments)
            raise ConnectionError("connection closed")

    class HealthySession:
        async def call_tool(self, _name, arguments):
            return {"content": [{"type": "text", "text": "reconnected"}]}

    registration = mcp_client.create_mcp_registration(
        server_name="test",
        command="fake-server", workspace=tmp_path,
        skills_loader=SkillsLoader(tmp_path),
    )
    registration._session = TerminatedSession()
    reconnect_attempts = 0

    async def fake_connect():
        nonlocal reconnect_attempts
        reconnect_attempts += 1
        registration._session = HealthySession()

    monkeypatch.setattr(registration, "_connect", fake_connect)

    assert await registration.call_tool("echo", {}) == "(MCP server is temporarily unavailable)"
    assert reconnect_attempts == 0
    assert failed_calls == [{}]
    assert await registration.call_tool("echo", {}) == "reconnected"
    assert reconnect_attempts == 1
    assert failed_calls == [{}]
    await registration.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["connecting", "discovering"])
async def test_background_discovery_reports_failure_and_retries_immediately(tmp_path, monkeypatch, phase):
    registration = mcp_client.create_mcp_registration(server_name="demo", command="demo", workspace=tmp_path, skills_loader=SkillsLoader(tmp_path))
    attempts = 0

    class Session:
        async def list_tools(self):
            if attempts == 1 and phase == "discovering":
                raise ConnectionError("secret-authorization-token")
            return SimpleNamespace(tools=[])

    async def connect():
        nonlocal attempts
        attempts += 1
        if attempts == 1 and phase == "connecting":
            raise ConnectionError("secret-authorization-token")
        registration._session = Session()

    monkeypatch.setattr(registration, "_connect", connect)
    assert registration.get_status()["status"] == "disconnected"
    task = registration.start_tool_discovery()
    assert registration.start_tool_discovery() is task
    assert registration.get_status()["status"] == "connecting"
    assert await task == []
    status = registration.get_status()
    assert status["status"] == "unavailable"
    assert status["error"] == "ConnectionError"
    assert "retry_after_seconds" not in status
    assert "secret-authorization-token" not in str(status)
    await asyncio.sleep(0)
    assert attempts == 1

    retry = registration.start_tool_discovery()
    assert retry is not task
    assert registration.start_tool_discovery() is retry
    await retry
    assert attempts == 2
    assert registration.get_status()["status"] == "ready"
    assert registration.get_status()["error"] is None
    assert registration.get_status()["tool_count"] == 0
    await registration.aclose()
    await registration.start_tool_discovery()
    assert attempts == 3
    assert registration.get_status()["status"] == "ready"
    await registration.aclose()


@pytest.mark.asyncio
async def test_new_discovery_waits_for_failed_connection_cleanup_without_being_cancelled(tmp_path, monkeypatch):
    cleanup_started = asyncio.Event()
    release_cleanup = asyncio.Event()
    old_close = asyncio.Event()

    class OldSession:
        async def call_tool(self, _name, arguments):
            raise ConnectionError("connection closed")

    class NewSession:
        async def list_tools(self):
            return SimpleNamespace(tools=[{"name": "echo"}])

    async def old_owner():
        await old_close.wait()
        cleanup_started.set()
        await release_cleanup.wait()

    async def run_connection(ready, close_event):
        session = NewSession()
        registration._session = session
        ready.set_result(None)
        try:
            await close_event.wait()
        finally:
            if registration._session is session:
                registration._session = None

    registration = mcp_client.create_mcp_registration(server_name="demo", command="demo", workspace=tmp_path, skills_loader=SkillsLoader(tmp_path))
    registration._session = OldSession()
    registration._owner_task = asyncio.create_task(old_owner())
    registration._close_event = old_close
    monkeypatch.setattr(registration, "_run_connection", run_connection)
    call = asyncio.create_task(registration.call_tool("echo", {}))
    discovery = None
    try:
        await asyncio.wait_for(cleanup_started.wait(), 1)
        discovery = registration.start_tool_discovery()
        await asyncio.sleep(0)
        assert not discovery.done()
        release_cleanup.set()
        assert await asyncio.wait_for(call, 1) == "(MCP server is temporarily unavailable)"
        bindings = await asyncio.wait_for(discovery, 1)
        assert [binding.func.__name__ for binding in bindings] == ["mcp_demo_echo"]
        assert registration.get_status()["status"] == "ready"
    finally:
        release_cleanup.set()
        await registration.aclose()
        await asyncio.gather(*(task for task in (call, discovery) if task is not None), return_exceptions=True)


@pytest.mark.skipif(
    platform.system() != "Linux" or any(shutil.which(name) is None for name in mcp_client.SANDBOX_EXECUTABLES)
    or shutil.which("python3", path="/usr/bin:/bin") is None,
    reason="Requires Linux with sandbox dependencies and system Python",
)
@pytest.mark.asyncio
async def test_stdio_server_reconnects_immediately_after_failed_initialization(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(settings.path, "sandbox_home_path", tmp_path / "managed-home")
    monkeypatch.setattr(skill_module, "iter_builtin_skill_dirs", list)
    server = workspace / "server.py"
    server.write_text('''
import json
import sys
from pathlib import Path

counter = Path("attempts")
attempt = int(counter.read_text()) + 1 if counter.exists() else 1
counter.write_text(str(attempt))
for line in sys.stdin:
    request = json.loads(line)
    if "id" not in request:
        continue
    method = request["method"]
    if method == "initialize":
        if attempt == 1:
            sys.exit(1)
        result = {
            "protocolVersion": request["params"]["protocolVersion"],
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "reconnect-test", "version": "1"},
        }
    elif method == "tools/list":
        result = {"tools": [{"name": "echo", "inputSchema": {"type": "object", "properties": {}}}]}
    elif method == "tools/call":
        result = {"content": [{"type": "text", "text": "connected"}], "isError": False}
    else:
        result = {}
    print(json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": result}), flush=True)
''', encoding="utf-8")
    registration = mcp_client.create_mcp_registration(
        server_name="reconnect", command=shutil.which("python3", path="/usr/bin:/bin"),
        args=["-u", str(server)], workspace=workspace,
        skills_loader=SkillsLoader(workspace),
    )
    try:
        assert await asyncio.wait_for(registration.start_tool_discovery(), 10) == []
        assert registration.get_status()["status"] == "unavailable"
        bindings = await asyncio.wait_for(registration.start_tool_discovery(), 10)
        assert [binding.func.__name__ for binding in bindings] == ["mcp_reconnect_echo"]
        assert await registration.call_tool("echo", {}) == "connected"
        assert (workspace / "attempts").read_text() == "2"
    finally:
        await registration.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["connecting", "discovering"])
@pytest.mark.parametrize("close_all", [False, True])
async def test_closing_session_cancels_background_discovery(tmp_path, monkeypatch, phase, close_all):
    await mcp_client.close_all_mcp_connections()
    started = asyncio.Event()
    stopped = asyncio.Event()

    class Session:
        async def list_tools(self):
            started.set()
            await asyncio.Event().wait()

    async def run_connection(registration, ready, close_event):
        try:
            if phase == "connecting":
                started.set()
                await asyncio.Event().wait()
            registration._session = Session()
            ready.set_result(None)
            await close_event.wait()
        finally:
            stopped.set()
            registration._session = None

    monkeypatch.setattr(mcp_client.MCPServerRegistration, "_run_connection", run_connection)
    registration = await mcp_client.get_session_mcp_registration_from_config(
        "closing-test", "demo", MCPServerSettings(command="demo", user_scoped=True), workspace=tmp_path,
        skills_loader=SkillsLoader(tmp_path),
    )
    task = registration.start_tool_discovery()
    await asyncio.wait_for(started.wait(), timeout=1)
    close = mcp_client.close_all_mcp_connections() if close_all else mcp_client.close_session_mcp_connections("closing-test")
    await asyncio.wait_for(close, timeout=1)
    assert task.cancelled()
    assert stopped.is_set()
    assert registration._owner_task is None
    assert registration._session is None
    assert registration.get_status()["status"] == "disconnected"
    assert registration.connection_scope not in mcp_client.MCP_CONNECTIONS
    assert await asyncio.wait_for(registration.start_tool_discovery(), timeout=0.2) == []
    assert await registration.call_tool("echo", {}) == "(MCP server is temporarily unavailable)"
    assert registration._owner_task is None


@pytest.mark.asyncio
async def test_late_failure_from_old_session_does_not_close_new_session(tmp_path, monkeypatch):
    started = asyncio.Event()
    fail = asyncio.Event()

    class OldSession:
        async def call_tool(self, _name, arguments):
            started.set()
            await fail.wait()
            raise ConnectionError("old connection closed")

    registration = mcp_client.create_mcp_registration(server_name="demo", command="demo", workspace=tmp_path, skills_loader=SkillsLoader(tmp_path))
    registration._session = OldSession()
    call = asyncio.create_task(registration.call_tool("echo", {}))
    await asyncio.wait_for(started.wait(), timeout=1)
    new_session = object()
    registration._session = new_session
    fail.set()
    assert await call == "(MCP server is temporarily unavailable)"
    assert registration._session is new_session
    assert registration._last_error is None
    await registration.aclose()


@pytest.mark.asyncio
async def test_discovery_after_tool_reconnect_uses_new_session_schemas(tmp_path, monkeypatch):
    attempts = 0

    class Session:
        def __init__(self, version):
            self.version = version

        async def list_tools(self):
            return SimpleNamespace(tools=[{"name": f"version_{self.version}"}])

    async def run_connection(registration, ready, close_event):
        nonlocal attempts
        attempts += 1
        session = Session(attempts)
        registration._session = session
        ready.set_result(None)
        try:
            await close_event.wait()
        finally:
            if registration._session is session:
                registration._session = None

    monkeypatch.setattr(mcp_client.MCPServerRegistration, "_run_connection", run_connection)
    registration = mcp_client.create_mcp_registration(server_name="demo", command="demo", workspace=tmp_path, skills_loader=SkillsLoader(tmp_path))
    try:
        first = await registration.start_tool_discovery()
        assert [binding.func.__name__ for binding in first] == ["mcp_demo_version_1"]
        registration._mark_unavailable(ConnectionError("connection closed"))
        await registration.aclose()
        # A previously registered tool can reconnect before the next runner
        # asks for discovery. It must not cause the old schemas to be reused.
        await registration._ensure_session()
        second = await registration.start_tool_discovery()
        assert [binding.func.__name__ for binding in second] == ["mcp_demo_version_2"]
        assert attempts == 2
    finally:
        await registration.aclose()
