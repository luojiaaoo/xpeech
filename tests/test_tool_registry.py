from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from xpeech.agent.tool_executor import ToolExecutor
from xpeech.agent.tools import mcp_client, registry
from xpeech.config.settings import ToolConfig
from xpeech.provider.litellm_provider import LiteLLMProvider
from xpeech.provider.schema import LLMParameters, ToolCallRequest


def named_tool(name: str):
    async def tool():
        return name

    tool.__name__ = name
    return tool


class FakeProvider:
    def __init__(self) -> None:
        self.support_image = False
        self.support_video = True
        self.registered: list[str] = []
        self.mcp_tools: list[tuple[dict[str, Any], Any]] = []

    def register_function_tool(self, tool):
        self.registered.append(tool.__name__)

    def register_mcp_tool(self, tool, *, tool_json):
        self.registered.append(tool.__name__)
        self.mcp_tools.append((tool_json, tool))


@pytest.mark.asyncio
@pytest.mark.parametrize("not_interactive", [False, True])
async def test_registers_supported_default_and_mcp_tools(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    not_interactive: bool,
):
    file_tools = tuple(
        named_tool(name) for name in ("read_image", "read_video", "read_file", "write_file", "edit_file")
    )
    monkeypatch.setattr(registry, "build_file_tools", lambda **_kwargs: file_tools)
    monkeypatch.setattr(registry, "build_shell_tools", lambda **_kwargs: named_tool("shell"))
    monkeypatch.setattr(registry, "web_fetch", named_tool("web_fetch"))
    for name in ("baidu_search", "bing_search"):
        monkeypatch.setattr(registry, name, named_tool(name))
    monkeypatch.setattr(registry, "build_browser_preview_tool", lambda **_kwargs: named_tool("browser_preview"))
    monkeypatch.setattr(registry, "build_file_message_tools", lambda **_kwargs: named_tool("send_file"))
    monkeypatch.setattr(registry, "ask_user_question", named_tool("ask_user_question"))

    registration = object()
    registration_calls = []

    async def get_registration(*args, **kwargs):
        registration_calls.append((args, kwargs))
        return registration

    monkeypatch.setattr(registry, "get_session_mcp_registration_from_config", get_registration)
    mcp_tools = [
        ({"type": "function", "function": {"name": name}}, named_tool(name))
        for name in ("mcp_demo_echo", "mcp_demo_ping")
    ]

    async def collect_tools(current_registration):
        assert current_registration is registration
        for tool_json, tool_func in mcp_tools:
            yield tool_json, tool_func, tool_func.__name__

    monkeypatch.setattr(registry, "collect_mcp_tool", collect_tools)
    provider = FakeProvider()
    config = ToolConfig.model_validate({"mcpServers": {"demo": {"command": "demo"}}})

    await registry.register_tools(
        user_id="session-1",
        provider=provider,
        workspace=tmp_path,
        config=config,
        not_interactive=not_interactive,
    )

    assert "read_image" not in provider.registered
    expected_tools = [
        "read_video",
        "read_file",
        "write_file",
        "edit_file",
        "shell",
        "web_fetch",
        "baidu_search",
        "bing_search",
        "arxiv_search",
        "arxiv_download",
        "browser_preview",
        "read_doc_file",
        "send_file",
        "ask_user_question",
        "list_sessions",
        "read_session",
        "feishu_schedule",
        "feishu_schedule_list",
        "feishu_schedule_cancel",
    ]
    if not_interactive:
        interactive_names = {"ask_user_question", "feishu_schedule", "feishu_schedule_list", "feishu_schedule_cancel"}
        expected_tools = [name for name in expected_tools if name not in interactive_names]
    assert provider.registered == expected_tools + [tool.__name__ for _, tool in mcp_tools]
    assert provider.mcp_tools == mcp_tools
    assert registration_calls == [(("session-1", "demo", config.mcp_servers["demo"]), {"workspace": tmp_path})]


@pytest.mark.asyncio
async def test_mcp_functions_call_the_bound_server_with_original_schema_and_arguments(tmp_path, monkeypatch):
    calls = []
    echo_schema = {
        "type": "object",
        "properties": {
            "text": {"type": "string", "enum": ["hello", "world"]},
            "style": {"type": "string", "default": "plain"},
        },
        "required": ["text"],
        "additionalProperties": True,
    }

    class FakeSession:
        def __init__(self, server_name):
            self.server_name = server_name

        async def list_tools(self):
            return SimpleNamespace(
                tools=[
                    {"name": "echo", "description": "Echo text", "inputSchema": echo_schema},
                    {"name": "ping", "inputSchema": {"type": "object", "properties": {}}},
                    {"name": "disabled"},
                ]
            )

        async def call_tool(self, name, arguments):
            calls.append((self.server_name, name, arguments))
            return {"content": [{"type": "text", "text": f"{self.server_name}:{name}"}]}

    async def connect(registration):
        registration._session = FakeSession(registration.config.server_name)

    monkeypatch.setattr(mcp_client.MCPServerRegistration, "_connect", connect)
    provider = LiteLLMProvider(
        api_key="test",
        api_base="https://example.test",
        default_model="test",
        parameters=LLMParameters(),
    )
    config = ToolConfig.model_validate(
        {
            "mcpServers": {
                "alpha": {"command": "demo", "enabled_tools": ["echo", "ping"]},
                "beta": {"command": "demo", "enabled_tools": ["mcp_beta_echo"]},
            }
        }
    )

    try:
        await registry.register_tools(
            user_id="mcp-binding-test",
            provider=provider,
            workspace=tmp_path,
            config=config,
        )

        mcp_tools = {name: tool for name, tool in provider.default_tools.items() if name.startswith("mcp_")}
        assert set(mcp_tools) == {"mcp_alpha_echo", "mcp_alpha_ping", "mcp_beta_echo"}
        assert mcp_tools["mcp_alpha_echo"].tool_json["function"]["parameters"] == echo_schema
        assert mcp_tools["mcp_beta_echo"].tool_json["function"]["parameters"] == echo_schema
        results = [
            result
            async for result in ToolExecutor(workspace=tmp_path, max_result_chars=10_000).execute(
                [
                    ToolCallRequest(id="1", name="mcp_alpha_echo", arguments={"text": "hello", "extra": 1}),
                    ToolCallRequest(id="2", name="mcp_alpha_ping", arguments={}),
                    ToolCallRequest(id="3", name="mcp_beta_echo", arguments={"text": "world", "style": None}),
                ],
                {name: tool.func for name, tool in mcp_tools.items()},
            )
        ]

        assert all(result.succeeded for result in results)
        assert {result.call.id: result.value for result in results} == {
            "1": "alpha:echo",
            "2": "alpha:ping",
            "3": "beta:echo",
        }
        assert sorted(calls) == [
            ("alpha", "echo", {"text": "hello", "extra": 1}),
            ("alpha", "ping", {}),
            ("beta", "echo", {"text": "world", "style": None}),
        ]
    finally:
        await mcp_client.close_session_mcp_connections("mcp-binding-test")
