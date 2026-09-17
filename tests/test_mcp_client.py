import asyncio
from types import SimpleNamespace

import pytest

from xpeech.agent.tools import mcp_client


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
    assert mcp_client.MCP_CONNECTIONS == {}
    await mcp_client.close_all_mcp_connections()


@pytest.mark.asyncio
async def test_failed_connection_is_cached_until_session_expires(monkeypatch: pytest.MonkeyPatch):
    await mcp_client.close_all_mcp_connections()
    monkeypatch.setattr(mcp_client, "MCP_CONNECT_TIMEOUT_SECONDS", 0.02)
    monkeypatch.setattr(mcp_client, "MCP_CONNECTION_TTL_SECONDS", 0.06)
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
    assert retried is not registration
    with pytest.raises(asyncio.TimeoutError):
        await retried._get_tool_bindings()
    assert attempts == 2
    await mcp_client.close_all_mcp_connections()
