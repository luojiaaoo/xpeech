import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from xpeech.agent.tools import mcp_client
from xpeech.config.settings import ToolConfig


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
                "url": "https://example.test/mcp",
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
    assert registration.config.connect_timeout == 45


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
        verify_tls=False,
        connect_timeout=45,
    )
    ready = asyncio.get_running_loop().create_future()
    close_event = asyncio.Event()
    owner = asyncio.create_task(registration._run_connection(ready, close_event))

    await ready
    close_event.set()
    await owner

    assert client_kwargs["verify"] is False
    assert client_kwargs["timeout"].connect == 45


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
    config = {"command": "fake-server"}

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
    monkeypatch.setattr(mcp_client, "MCP_CONNECT_TIMEOUT_SECONDS", 0.02)
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
    config = {"url": "https://example.test/mcp"}
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
