import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from xpeech.agent.tools import mcp_client
from xpeech.config.settings import MCPServerSettings, ToolConfig


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
async def test_enabled_tools_match_original_names_with_globs(patterns, expected):
    names = {"read_file", "write_file", "delete_record", "tool.with.dots"}

    class Session:
        async def list_tools(self):
            return SimpleNamespace(tools=[
                {"name": name, "inputSchema": {"type": "object", "properties": {}}}
                for name in sorted(names)
            ])

    registration = mcp_client.create_mcp_registration(
        server_name="demo", command="demo", enabled_tools=patterns,
    )
    registration._session = Session()
    bindings = await registration._discover_tool_bindings()
    assert {binding.func.mcp_original_name for binding in bindings} == expected
    assert {binding.func.__name__ for binding in bindings} == {
        mcp_client._mcp_tool_name("demo", name) for name in expected
    }


def test_supports_the_three_mcp_transports():
    cases = [
        ({"command": "server"}, "stdio"),
        ({"url": "https://example.test/sse"}, "sse"),
        ({"url": "https://example.test/mcp"}, "streamable-http"),
        ({"url": "https://example.test/events", "transport": "sse"}, "sse"),
        ({"url": "https://example.test/mcp", "transport": "streamable-http"}, "streamable-http"),
    ]

    for kwargs, expected in cases:
        registration = mcp_client.create_mcp_registration(server_name="test", **kwargs)
        assert registration.config.transport == expected


@pytest.mark.parametrize("transport", ["streamable_http", "http-stream", "http"])
def test_rejects_nonstandard_transport_names(transport: str):
    with pytest.raises(ValueError, match="Unsupported MCP transport"):
        mcp_client.create_mcp_registration(
            server_name="test",
            url="https://example.test/mcp",
            transport=transport,
        )


def test_tls_verification_can_be_disabled_from_config():
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
    )

    assert registration.config.verify_tls is False
    assert registration.config.description == "查询客户资料和订单"
    assert registration.config.connect_timeout == 45
    assert registration.config.command_authorization_token == "token-helper --user $user_id"
    assert registration.config.headers == {"Authorization": "Bearer ${authorization_token}"}


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


def test_authorization_token_command_requires_remote_mcp():
    with pytest.raises(ValueError, match="requires a remote URL"):
        mcp_client.create_mcp_registration(
            server_name="test",
            command="fake-server",
            command_authorization_token="token-helper $user_id",
        )


@pytest.mark.asyncio
async def test_streamable_http_passes_tls_setting_to_httpx(monkeypatch: pytest.MonkeyPatch):
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

    registration = mcp_client.create_mcp_registration(
        server_name="test",
        url="https://example.test/mcp",
        headers={"X-User-ID": "$user_id"},
        verify_tls=False,
        connect_timeout=45,
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


@pytest.mark.asyncio
async def test_connection_error_is_passed_through_exit_stack(monkeypatch: pytest.MonkeyPatch):
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
    )
    ready = asyncio.get_running_loop().create_future()
    owner = asyncio.create_task(registration._run_connection(ready, asyncio.Event()))

    await owner
    with pytest.raises(ConnectionError, match="initialize failed"):
        await ready

    assert len(exit_errors) == 1
    assert isinstance(exit_errors[0], ConnectionError)


@pytest.mark.asyncio
async def test_collect_mcp_tool_does_not_swallow_cancellation_groups(monkeypatch: pytest.MonkeyPatch):
    registration = mcp_client.create_mcp_registration(
        server_name="test",
        command="fake-server",
    )

    async def cancelled():
        raise BaseExceptionGroup("cancelled", [asyncio.CancelledError()])

    monkeypatch.setattr(registration, "_get_tool_bindings", cancelled)

    with pytest.raises(BaseExceptionGroup):
        [item async for item in mcp_client.collect_mcp_tool(registration)]


@pytest.mark.asyncio
async def test_collect_mcp_tool_skips_unavailable_server(monkeypatch: pytest.MonkeyPatch):
    registration = mcp_client.create_mcp_registration(
        server_name="test",
        command="fake-server",
    )

    async def unavailable():
        raise ConnectionError("server unavailable")

    monkeypatch.setattr(registration, "_get_tool_bindings", unavailable)

    assert [item async for item in mcp_client.collect_mcp_tool(registration)] == []


@pytest.mark.asyncio
async def test_connections_are_reused_per_session_and_expire(monkeypatch: pytest.MonkeyPatch):
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

    first = await mcp_client.get_session_mcp_registration_from_config("session-1", "demo", config)
    reused = await mcp_client.get_session_mcp_registration_from_config("session-1", "demo", config)
    isolated = await mcp_client.get_session_mcp_registration_from_config("session-2", "demo", config)

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
async def test_failed_connection_is_skipped_during_cooldown_then_retried(monkeypatch: pytest.MonkeyPatch):
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
    )

    with pytest.raises(asyncio.TimeoutError):
        await registration._get_tool_bindings()
    assert attempts == 1

    cached = await mcp_client.get_session_mcp_registration_from_config(
        "session-1",
        "unreachable",
        config,
    )
    assert cached is registration
    assert await cached._get_tool_bindings() == []
    assert attempts == 1

    await asyncio.sleep(0.07)
    retried = await mcp_client.get_session_mcp_registration_from_config(
        "session-1",
        "unreachable",
        config,
    )
    assert retried is registration
    with pytest.raises(asyncio.TimeoutError):
        await retried._get_tool_bindings()
    assert attempts == 2
    await mcp_client.close_all_mcp_connections()


@pytest.mark.asyncio
async def test_terminated_connection_does_not_reconnect_during_cooldown(
    monkeypatch: pytest.MonkeyPatch,
):
    class TerminatedSession:
        async def call_tool(self, _name, arguments):
            raise ConnectionError("connection closed")

    registration = mcp_client.create_mcp_registration(
        server_name="test",
        command="fake-server",
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
async def test_background_discovery_reports_failure_and_retries_after_cooldown(monkeypatch):
    monkeypatch.setattr(mcp_client, "MCP_RECONNECT_DELAY_SECONDS", 60)
    registration = mcp_client.create_mcp_registration(server_name="demo", command="demo")
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
async def test_closing_session_cancels_background_discovery(monkeypatch, phase, close_all):
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
        "closing-test", "demo", MCPServerSettings(command="demo"),
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
async def test_late_failure_from_old_session_does_not_close_new_session(monkeypatch):
    started = asyncio.Event()
    fail = asyncio.Event()

    class OldSession:
        async def call_tool(self, _name, arguments):
            started.set()
            await fail.wait()
            raise ConnectionError("old connection closed")

    registration = mcp_client.create_mcp_registration(server_name="demo", command="demo")
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
async def test_discovery_after_tool_reconnect_uses_new_session_schemas(monkeypatch):
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
    registration = mcp_client.create_mcp_registration(server_name="demo", command="demo")
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
