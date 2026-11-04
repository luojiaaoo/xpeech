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

from xpeech.agent.tools import mcp_client
from xpeech.agent.tools import sandbox
from xpeech.config.settings import MCPServerSettings, ToolConfig


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
        registration = mcp_client.create_mcp_registration(server_name="test", **kwargs, workspace=tmp_path)
        assert registration.config.transport == expected


@pytest.mark.parametrize("transport", ["streamable_http", "http-stream", "http"])
def test_rejects_nonstandard_transport_names(tmp_path, transport: str):
    with pytest.raises(ValueError, match="Unsupported MCP transport"):
        mcp_client.create_mcp_registration(
            server_name="test",
            url="https://example.test/mcp",
            transport=transport,
            workspace=tmp_path,
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
    )

    assert registration.config.verify_tls is False
    assert registration.config.description == "查询客户资料和订单"
    assert registration.config.connect_timeout == 45
    assert registration.config.command_authorization_token == "token-helper --user $user_id"
    assert registration.config.headers == {"Authorization": "Bearer ${authorization_token}"}


def test_stdio_env_is_loaded_from_server_config(tmp_path):
    config = ToolConfig.model_validate({
        "mcpServers": {"local": {"command": "server", "env": {"API_KEY": "test-key"}}},
    })

    registration = mcp_client.create_mcp_registration_from_config(
        "local", config.mcp_servers["local"], workspace=tmp_path,
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
        mcp_client.create_mcp_registration(server_name="test", workspace=tmp_path, **config)


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
    monkeypatch.setattr(mcp_client.shutil, "which", lambda name: f"/usr/bin/{name}")
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
            assert environment["PATH"].startswith(f"{workspace}/home/.local/bin:{workspace}/home/.npm-global/bin:")
            assert "--unshare-pid" in sandbox_command
            assert sandbox_command[sandbox_command.index("--cap-drop") + 1] == "ALL"
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
    )
    ready = asyncio.get_running_loop().create_future()

    await registration._run_connection(ready, asyncio.Event())

    with pytest.raises(RuntimeError, match="requires Linux with bubblewrap"):
        await ready
    assert registration._session is None


@pytest.mark.skipif(
    platform.system() != "Linux" or shutil.which("bwrap") is None
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
    monkeypatch.setattr(sandbox.settings.path, "sandbox_home_path", config_home)
    monkeypatch.setattr(sandbox, "iter_builtin_skill_dirs", lambda: iter(()))
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
        assert int(result["capabilities"].split()[1], 16) == 0
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
    )

    async def unavailable():
        raise ConnectionError("server unavailable")

    monkeypatch.setattr(registration, "_get_tool_bindings", unavailable)

    assert [item async for item in mcp_client.collect_mcp_tool(registration)] == []


@pytest.mark.asyncio
async def test_connections_are_reused_per_session_and_expire(tmp_path, monkeypatch: pytest.MonkeyPatch):
    await mcp_client.close_all_mcp_connections()
    monkeypatch.setattr(mcp_client, "MCP_CONNECTION_TTL_SECONDS", 0.06)

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
    config = MCPServerSettings(command="fake-server")

    first = await mcp_client.get_session_mcp_registration_from_config("session-1", "demo", config, tmp_path)
    reused = await mcp_client.get_session_mcp_registration_from_config("session-1", "demo", config, tmp_path)
    isolated = await mcp_client.get_session_mcp_registration_from_config("session-2", "demo", config, tmp_path)

    assert first is reused
    assert first is not isolated
    assert set(mcp_client.MCP_CONNECTIONS) == {"session-1", "session-2"}

    binding = (await first._get_tool_bindings())[0]
    await isolated._get_tool_bindings()
    model = binding.func.__annotations__["args"]

    await asyncio.sleep(0.04)
    assert await binding.func(model(text="ok")) == "ok"
    await asyncio.sleep(0.04)
    assert set(mcp_client.MCP_CONNECTIONS) == {"session-1"}
    await asyncio.sleep(0.03)
    assert mcp_client.MCP_CONNECTIONS == {}
    await mcp_client.close_all_mcp_connections()


@pytest.mark.asyncio
async def test_failed_connection_is_skipped_during_cooldown_then_retried(tmp_path, monkeypatch: pytest.MonkeyPatch):
    await mcp_client.close_all_mcp_connections()
    monkeypatch.setattr(mcp_client, "MCP_CONNECTION_TTL_SECONDS", 0.2)
    monkeypatch.setattr(mcp_client, "MCP_RECONNECT_DELAY_SECONDS", 0.06)
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
    )

    with pytest.raises(asyncio.TimeoutError):
        await registration._get_tool_bindings()
    assert attempts == 1

    cached = await mcp_client.get_session_mcp_registration_from_config(
        "session-1",
        "unreachable",
        config,
        workspace=tmp_path,
    )
    assert cached is registration
    assert await cached._get_tool_bindings() == []
    assert attempts == 1

    await asyncio.sleep(0.07)
    retried = await mcp_client.get_session_mcp_registration_from_config(
        "session-1",
        "unreachable",
        config,
        workspace=tmp_path,
    )
    assert retried is registration
    with pytest.raises(asyncio.TimeoutError):
        await retried._get_tool_bindings()
    assert attempts == 2
    await mcp_client.close_all_mcp_connections()


@pytest.mark.asyncio
async def test_terminated_connection_does_not_reconnect_during_cooldown(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    class TerminatedSession:
        async def call_tool(self, _name, arguments):
            raise ConnectionError("connection closed")

    registration = mcp_client.create_mcp_registration(
        server_name="test",
        command="fake-server", workspace=tmp_path,
    )
    registration._session = TerminatedSession()
    reconnect_attempts = 0

    async def fake_close():
        registration._session = None

    async def fake_connect():
        nonlocal reconnect_attempts
        reconnect_attempts += 1

    monkeypatch.setattr(registration, "aclose", fake_close)
    monkeypatch.setattr(registration, "_connect", fake_connect)

    assert await registration.call_tool("echo", {}) == "(MCP server is temporarily unavailable)"
    assert await registration.call_tool("echo", {}) == "(MCP server is temporarily unavailable)"
    assert reconnect_attempts == 0


@pytest.mark.asyncio
async def test_background_discovery_reports_failure_and_retries_after_cooldown(tmp_path, monkeypatch):
    monkeypatch.setattr(mcp_client, "MCP_RECONNECT_DELAY_SECONDS", 60)
    registration = mcp_client.create_mcp_registration(server_name="demo", command="demo", workspace=tmp_path)
    attempts = 0

    class Session:
        async def list_tools(self):
            return SimpleNamespace(tools=[])

    async def connect():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
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
    assert 0 < status["retry_after_seconds"] <= 60
    assert "secret-authorization-token" not in str(status)
    assert registration.start_tool_discovery() is task
    assert attempts == 1

    registration._retry_after = asyncio.get_running_loop().time() - 1
    await registration.start_tool_discovery()
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
        "closing-test", "demo", MCPServerSettings(command="demo"), workspace=tmp_path,
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
    assert "closing-test" not in mcp_client.MCP_CONNECTIONS
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

    registration = mcp_client.create_mcp_registration(server_name="demo", command="demo", workspace=tmp_path)
    registration._session = OldSession()
    call = asyncio.create_task(registration.call_tool("echo", {}))
    await asyncio.wait_for(started.wait(), timeout=1)
    new_session = object()
    registration._session = new_session
    fail.set()
    assert await call == "(MCP server is temporarily unavailable)"
    assert registration._session is new_session
    assert registration._retry_after is None
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
    registration = mcp_client.create_mcp_registration(server_name="demo", command="demo", workspace=tmp_path)
    try:
        first = await registration.start_tool_discovery()
        assert [binding.func.__name__ for binding in first] == ["mcp_demo_version_1"]
        registration._start_retry_cooldown(ConnectionError("connection closed"))
        await registration.aclose()
        registration._retry_after = asyncio.get_running_loop().time() - 1
        # A previously registered tool can reconnect before the next runner
        # asks for discovery. It must not cause the old schemas to be reused.
        await registration._ensure_session()
        second = await registration.start_tool_discovery()
        assert [binding.func.__name__ for binding in second] == ["mcp_demo_version_2"]
        assert attempts == 2
    finally:
        await registration.aclose()
