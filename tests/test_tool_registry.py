from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from xpeech.agent.tool_executor import ToolExecutor
from xpeech.agent.tools import mcp_client, registry
from xpeech.agent.tools.execution_request import EXECUTION_REQUEST
from xpeech.agent.tools.helper import get_tool_model_cls
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
        self.function_tools: dict[str, Any] = {}
        self.mcp_tools: list[tuple[dict[str, Any], Any]] = []

    def register_function_tool(self, tool):
        self.registered.append(tool.__name__)
        self.function_tools[tool.__name__] = tool

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
        interactive_names = {
            "shell",
            "baidu_search",
            "bing_search",
            "ask_user_question",
            "feishu_schedule",
            "feishu_schedule_list",
            "feishu_schedule_cancel",
        }
        expected_tools = [name for name in expected_tools if name not in interactive_names]
    assert provider.registered == expected_tools + [tool.__name__ for _, tool in mcp_tools]
    for name in ("shell", "baidu_search", "bing_search"):
        if not_interactive:
            assert name not in provider.function_tools
        else:
            assert "execution_description" in get_tool_model_cls(provider.function_tools[name]).model_cls.model_fields
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
                "beta": {"command": "demo", "enabled_tools": ["echo"]},
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


_LONG_MCP_TOOL_NAME = "update.record-" + "details_" * 12


@pytest.mark.asyncio
@pytest.mark.parametrize("not_interactive", [False, True])
@pytest.mark.parametrize(
    ("requested", "expected_confirmation"),
    [
        ([], set()),
        (["echo"], {"echo"}),
        (["mcp_demo_echo"], set()),
        (["echo", "mcp_demo_ping"], {"echo"}),
        (["*"], {"echo", "ping", _LONG_MCP_TOOL_NAME}),
        (["disabled"], set()),
        (["unknown"], set()),
        ([_LONG_MCP_TOOL_NAME], {_LONG_MCP_TOOL_NAME}),
        (["e*"], {"echo"}),
        (["p?ng"], {"ping"}),
        (["[ep]*"], {"echo", "ping"}),
        (["mcp_demo_*"], set()),
        (["mcp_demo_p*"], set()),
        (["mcp_other_*"], set()),
        (["echo*", "mcp_demo_p?ng"], {"echo"}),
        (["update.record-*"], {_LONG_MCP_TOOL_NAME}),
        (["mcp_demo_update_record-*"], set()),
        (["E*"], set()),
        (["dis*"], set()),
    ],
)
async def test_mcp_confirmation_selection_at_registration(
    tmp_path,
    monkeypatch,
    not_interactive,
    requested,
    expected_confirmation,
):
    calls = []
    enabled_names = {"echo", "ping", _LONG_MCP_TOOL_NAME}

    class Session:
        async def list_tools(self):
            return SimpleNamespace(
                tools=[
                    {"name": name, "description": name, "inputSchema": {"type": "object", "properties": {}}}
                    for name in sorted(enabled_names | {"disabled"})
                ]
            )

        async def call_tool(self, name, arguments):
            calls.append((name, arguments))
            return {"content": [{"type": "text", "text": name}]}

    async def connect(registration):
        registration._session = Session()

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
                "demo": {
                    "command": "demo",
                    "enabled_tools": ["e*", "p?ng", "update.record-*"],
                    "execution_request_tools": requested,
                }
            }
        }
    )
    try:
        await registry.register_tools(
            user_id="mcp-confirmation-test",
            provider=provider,
            workspace=tmp_path,
            config=config,
            not_interactive=not_interactive,
        )
        tools = {name: tool for name, tool in provider.default_tools.items() if name.startswith("mcp_")}
        registered_names = enabled_names - expected_confirmation if not_interactive else enabled_names
        assert set(tools) == {mcp_client._mcp_tool_name("demo", name) for name in registered_names}
        for name in sorted(registered_names):
            tool_name = mcp_client._mcp_tool_name("demo", name)
            needs_confirmation = name in expected_confirmation
            parameters = tools[tool_name].tool_json["function"]["parameters"]
            assert ("execution_description" in parameters["properties"]) is needs_confirmation
            arguments = {"execution_description": f"调用 {name}"} if needs_confirmation else {}
            previous_calls = len(calls)
            [result] = [
                result
                async for result in ToolExecutor(workspace=tmp_path, max_result_chars=10_000).execute(
                    [ToolCallRequest(id="1", name=tool_name, arguments=arguments)],
                    {tool_name: tools[tool_name].func},
                )
            ]
            assert result.succeeded
            if needs_confirmation:
                assert result.value == EXECUTION_REQUEST
                assert len(calls) == previous_calls
                assert "execution_description" in parameters["required"]
            else:
                assert result.value == name
                assert calls[-1] == (name, {})

        registration = await mcp_client.get_session_mcp_registration_from_config(
            "mcp-confirmation-test",
            "demo",
            config.mcp_servers["demo"],
            workspace=tmp_path,
        )
        for binding in await registration._get_tool_bindings():
            assert "execution_description" not in binding.schema["function"]["parameters"]["properties"]
    finally:
        await mcp_client.close_session_mcp_connections("mcp-confirmation-test")
