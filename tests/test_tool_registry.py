import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from xpeech.agent.tool_executor import ToolExecutor
from xpeech.agent.tools import mcp_client, registry
from xpeech.agent.tools.shell import dcg as shell_module
from xpeech.agent.tools.execution_request import EXECUTION_REQUEST, ExecutionDecision
from xpeech.agent.tools.helper import get_tool_model_cls
from xpeech.agent.tools.shell.schema import ShellArgs
from xpeech.config.settings import ToolConfig
from xpeech.provider.litellm_provider import LiteLLMProvider
from xpeech.provider.schema import LLMParameters, ToolCallRequest


def named_tool(name: str):
    async def tool():
        return name

    tool.__name__ = name
    return tool


@pytest.fixture(autouse=True)
def stub_unrelated_shell_builder(monkeypatch):
    # Registry/MCP tests do not require the Linux-only shell implementation.
    monkeypatch.setattr(registry, "build_shell_tools", lambda **_kwargs: named_tool("shell"))


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
@pytest.mark.parametrize(
    ("returncode", "output", "expected"),
    [
        (0, b'{"decision":"allow"}', ExecutionDecision.ALLOW),
        (1, b'{"decision":"deny"}', ExecutionDecision.DENY),
        (0, b'{"decision":"deny"}', ExecutionDecision.DENY),
        (0, b'{"decision":"ask"}', ExecutionDecision.CONFIRM),
        (1, b'{"decision":"ask"}', ExecutionDecision.CONFIRM),
    ],
)
async def test_shell_dcg_returns_allow_confirm_or_reject(tmp_path, monkeypatch, returncode, output, expected):
    process = SimpleNamespace(returncode=returncode, communicate=AsyncMock(return_value=(output, b"")))
    spawn = AsyncMock(return_value=process)
    monkeypatch.setattr(shell_module.asyncio, "create_subprocess_exec", spawn)
    command = "echo '目录'; $(some-command)"

    assert await shell_module.shell_requires_confirmation(json.dumps({"command": command}), workspace=tmp_path) is expected

    process.communicate.assert_awaited_once_with(command.encode("utf-8"))
    argv = spawn.call_args.args
    assert argv[:3] == ("dcg", "test", "--stdin")
    assert command not in argv
    assert argv[argv.index("--dialect") + 1] == "posix"
    assert argv[argv.index("--config") + 1] == str(shell_module.DCG_POLICY_DIR / "config.toml")
    assert "--robot" in argv
    assert spawn.call_args.kwargs["env"]["DCG_CUSTOM_PATHS"] == str(shell_module.DCG_POLICY_DIR / "agent.yaml")
    assert spawn.call_args.kwargs["cwd"] == tmp_path.resolve()
    assert spawn.call_args.kwargs["env"]["HOME"] == str(tmp_path / "home")
    assert spawn.call_args.kwargs["env"]["PWD"] == str(tmp_path)


@pytest.mark.asyncio
async def test_shell_dcg_missing_binary_refuses_execution(tmp_path, monkeypatch):
    monkeypatch.setattr(shell_module.asyncio, "create_subprocess_exec", AsyncMock(side_effect=FileNotFoundError("dcg")))
    with pytest.raises(RuntimeError, match="危险命令检查失败，拒绝执行"):
        await shell_module.shell_requires_confirmation('{"command":"pwd"}', workspace=tmp_path)


@pytest.mark.asyncio
@pytest.mark.parametrize(("returncode", "output"), [
    (0, b'{"decision":"indeterminate"}'),
    (3, b'{"decision":"allow"}'),
    (1, b'{"decision":"allow"}'),
    (0, b'{"decision":"warn"}'),
    (0, b"{}"), (0, b"[]"), (0, b"not json"), (0, b"\xff"),
])
async def test_shell_dcg_invalid_results_refuse_execution(tmp_path, monkeypatch, returncode, output):
    process = SimpleNamespace(returncode=returncode, communicate=AsyncMock(return_value=(output, b"")))
    monkeypatch.setattr(shell_module.asyncio, "create_subprocess_exec", AsyncMock(return_value=process))
    with pytest.raises(RuntimeError, match="危险命令检查失败，拒绝执行"):
        await shell_module.shell_requires_confirmation('{"command":"pwd"}', workspace=tmp_path)


@pytest.mark.asyncio
async def test_shell_dcg_ignores_ambient_policy_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv("DCG_BYPASS", "1")
    monkeypatch.setenv("DCG_POLICY_DEFAULT_MODE", "log")
    monkeypatch.setenv("DCG_CUSTOM_PATHS", "/untrusted/pack.yaml")
    process = SimpleNamespace(returncode=0, communicate=AsyncMock(return_value=(b'{"decision":"allow"}', b"")))
    spawn = AsyncMock(return_value=process)
    monkeypatch.setattr(shell_module.asyncio, "create_subprocess_exec", spawn)
    assert await shell_module.shell_requires_confirmation('{"command":"pwd"}', workspace=tmp_path) is ExecutionDecision.ALLOW
    env = spawn.call_args.kwargs["env"]
    assert "DCG_BYPASS" not in env and "DCG_POLICY_DEFAULT_MODE" not in env
    assert env["DCG_CUSTOM_PATHS"] == str(shell_module.DCG_POLICY_DIR / "agent.yaml")


@pytest.mark.asyncio
async def test_shell_dcg_keeps_concurrent_workspaces_separate(tmp_path, monkeypatch):
    backend_cwd = Path.cwd()
    backend_home = os.environ.get("HOME")
    workspaces = [tmp_path / "first", tmp_path / "second"]
    for workspace in workspaces:
        workspace.mkdir()

    def process(*_args, **_kwargs):
        return SimpleNamespace(returncode=0, communicate=AsyncMock(return_value=(b'{"decision":"allow"}', b"")))

    spawn = AsyncMock(side_effect=process)
    monkeypatch.setattr(shell_module.asyncio, "create_subprocess_exec", spawn)
    results = await asyncio.gather(*(
        shell_module.shell_requires_confirmation('{"command":"pwd"}', workspace=workspace)
        for workspace in workspaces
    ))
    assert results == [ExecutionDecision.ALLOW, ExecutionDecision.ALLOW]
    contexts = {call.kwargs["cwd"]: call.kwargs["env"] for call in spawn.call_args_list}
    assert set(contexts) == set(workspaces)
    for workspace in workspaces:
        assert contexts[workspace]["HOME"] == str(workspace / "home")
        assert contexts[workspace]["PWD"] == str(workspace)
    assert Path.cwd() == backend_cwd
    assert os.environ.get("HOME") == backend_home


@pytest.mark.asyncio
async def test_shell_dcg_refuses_invalid_workspace_without_spawning(tmp_path, monkeypatch):
    spawn = AsyncMock()
    monkeypatch.setattr(shell_module.asyncio, "create_subprocess_exec", spawn)
    workspace = tmp_path / "missing"
    with pytest.raises(RuntimeError, match="危险命令检查失败，拒绝执行"):
        await shell_module.shell_requires_confirmation('{"command":"pwd"}', workspace=workspace)
    spawn.assert_not_awaited()
    assert not workspace.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancelled", [False, True])
async def test_shell_dcg_stops_and_reaps_check_on_timeout_or_cancellation(tmp_path, monkeypatch, cancelled):
    started = asyncio.Event()

    async def communicate(_command):
        started.set()
        await asyncio.Event().wait()

    process = SimpleNamespace(returncode=None, communicate=communicate, kill=Mock(), wait=AsyncMock())
    monkeypatch.setattr(shell_module.asyncio, "create_subprocess_exec", AsyncMock(return_value=process))
    monkeypatch.setattr(shell_module, "SHELL_CHECK_TIMEOUT", 1 if cancelled else 0.01)
    task = asyncio.create_task(shell_module.shell_requires_confirmation('{"command":"pwd"}', workspace=tmp_path))
    await asyncio.wait_for(started.wait(), timeout=1)
    if cancelled:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        with pytest.raises(RuntimeError, match="危险命令检查失败，拒绝执行"):
            await task
    process.kill.assert_called_once_with()
    process.wait.assert_awaited_once_with()


@pytest.mark.asyncio
@pytest.mark.parametrize("not_interactive", [False, True])
@pytest.mark.parametrize("decision", ["allow", "ask", "deny"])
async def test_registered_shell_uses_dcg_before_execution(tmp_path, monkeypatch, not_interactive, decision):
    executions = []

    async def shell(args: ShellArgs):
        """Execute a test command."""
        executions.append(args)
        return "shell output"

    monkeypatch.setattr(registry, "build_shell_tools", lambda **_kwargs: shell)
    process = SimpleNamespace(
        returncode=0 if decision == "allow" else 1,
        communicate=AsyncMock(return_value=(json.dumps({"decision": decision}).encode(), b"")),
    )
    spawn = AsyncMock(return_value=process)
    monkeypatch.setattr(shell_module.asyncio, "create_subprocess_exec", spawn)
    provider = FakeProvider()
    await registry.register_tools(
        user_id="shell-dcg-test",
        provider=provider,
        workspace=tmp_path,
        config=ToolConfig(),
        not_interactive=not_interactive,
    )
    command = {"allow": "pwd", "ask": "rm -rf ./build", "deny": "init 6"}[decision]
    [result] = [
        result
        async for result in ToolExecutor(workspace=tmp_path, max_result_chars=10_000).execute(
            [
                ToolCallRequest(
                    id="1", name="shell", arguments={"command": command, "execution_description": "测试检查"}
                )
            ],
            provider.function_tools,
        )
    ]
    process.communicate.assert_awaited_once_with(command.encode())
    assert spawn.call_args.kwargs["cwd"] == tmp_path.resolve()
    assert spawn.call_args.kwargs["env"]["HOME"] == str(tmp_path / "home")
    if decision == "allow":
        assert result.succeeded and result.value == "shell output"
        [args] = executions
        assert type(args) is ShellArgs and args.command == command
    else:
        assert executions == []
        if decision == "deny":
            assert not result.succeeded and "发现高危命令，直接拒绝执行" in result.value
        elif not_interactive:
            assert not result.succeeded and "拒绝执行" in result.value
        else:
            assert result.succeeded and result.value == EXECUTION_REQUEST


@pytest.mark.asyncio
@pytest.mark.parametrize("not_interactive", [False, True])
async def test_registers_supported_default_and_mcp_tools(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    not_interactive: bool,
):
    # This test uses an argument-free shell stub; exercise only registration
    # and leave command validation to the dcg tests above.
    monkeypatch.setattr(registry, "shell_requires_confirmation", lambda _, **_kwargs: ExecutionDecision.CONFIRM)
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

    registration = mcp_client.create_mcp_registration(server_name="demo", command="demo", workspace=tmp_path)
    registration_calls = []

    async def get_registration(*args, **kwargs):
        registration_calls.append((args, kwargs))
        return registration

    monkeypatch.setattr(registry, "get_session_mcp_registration_from_config", get_registration)
    mcp_tools = [
        ({"type": "function", "function": {"name": name}}, named_tool(name))
        for name in ("mcp_demo_echo", "mcp_demo_ping")
    ]

    async def collect_tools():
        registration._session = object()
        registration._bindings = [mcp_client.MCPToolBinding(schema=schema, func=func) for schema, func in mcp_tools]
        return registration._bindings

    monkeypatch.setattr(registration, "_get_tool_bindings", collect_tools)
    provider = FakeProvider()
    config = ToolConfig.model_validate({"mcpServers": {"demo": {"command": "demo"}}})

    await registry.register_tools(
        user_id="session-1",
        provider=provider,
        workspace=tmp_path,
        config=config,
        not_interactive=not_interactive,
    )
    await registration.start_tool_discovery()

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
        "query_mcp_status",
    ]
    if not_interactive:
        interactive_names = {
            "ask_user_question",
            "feishu_schedule",
            "feishu_schedule_list",
            "feishu_schedule_cancel",
        }
        expected_tools = [name for name in expected_tools if name not in interactive_names]
    assert provider.registered == expected_tools + [tool.__name__ for _, tool in mcp_tools]
    assert "execution_description" in get_tool_model_cls(provider.function_tools["shell"]).model_cls.model_fields
    for name in ("baidu_search", "bing_search"):
        assert "execution_description" not in get_tool_model_cls(provider.function_tools[name]).model_cls.model_fields
    [result] = [
        result
        async for result in ToolExecutor(workspace=tmp_path, max_result_chars=10_000).execute(
            [ToolCallRequest(id="shell-1", name="shell", arguments={"execution_description": "测试"})],
            provider.function_tools,
        )
    ]
    if not_interactive:
        assert not result.succeeded
        assert "拒绝执行" in result.value
    else:
        assert result.succeeded
        assert result.value == EXECUTION_REQUEST
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
        await asyncio.gather(*(
            registration.start_tool_discovery()
            for registration in mcp_client.MCP_CONNECTIONS["mcp-binding-test"].values()
        ))

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
        await asyncio.gather(*(
            registration.start_tool_discovery()
            for registration in mcp_client.MCP_CONNECTIONS["mcp-confirmation-test"].values()
        ))
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
            if needs_confirmation:
                if not_interactive:
                    assert not result.succeeded
                    assert "拒绝执行" in result.value
                else:
                    assert result.succeeded
                    assert result.value == EXECUTION_REQUEST
                assert len(calls) == previous_calls
                assert "execution_description" in parameters["required"]
            else:
                assert result.succeeded
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


@pytest.mark.asyncio
@pytest.mark.parametrize("async_check", [False, True])
@pytest.mark.parametrize("decision", list(ExecutionDecision))
@pytest.mark.parametrize("not_interactive", [False, True])
async def test_register_mcp_bindings_passes_confirmation_callback(
    tmp_path, async_check, decision, not_interactive,
):
    session = SimpleNamespace(
        list_tools=AsyncMock(return_value=SimpleNamespace(tools=[{
            "name": "echo",
            "inputSchema": {
                "type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"],
            },
        }])),
        call_tool=AsyncMock(return_value={"content": [{"type": "text", "text": "echo result"}]}),
    )
    registration = mcp_client.create_mcp_registration(server_name="demo", command="demo", workspace=tmp_path)
    registration._session = session
    task = registration.start_tool_discovery()
    await task
    check = (AsyncMock if async_check else Mock)(return_value=decision)
    provider = FakeProvider()
    mcp_client.register_mcp_bindings(
        task,
        registration=registration,
        provider=provider,
        execution_request_tools=["echo"],
        not_interactive=not_interactive,
        requires_confirmation=check,
    )
    [(schema, tool)] = provider.mcp_tools
    arguments = {"text": "hello", "execution_description": "测试回调"}
    [result] = [result async for result in ToolExecutor(workspace=tmp_path, max_result_chars=10_000).execute(
        [ToolCallRequest(id="1", name="mcp_demo_echo", arguments=arguments)], {tool.__name__: tool},
    )]
    check.assert_called_once()
    assert json.loads(check.call_args.args[0]) == arguments
    if async_check:
        check.assert_awaited_once()
    assert "execution_description" in schema["function"]["parameters"]["required"]
    if decision is not ExecutionDecision.ALLOW:
        session.call_tool.assert_not_called()
        if decision is ExecutionDecision.DENY or not_interactive:
            assert not result.succeeded and "拒绝执行" in result.value
        else:
            assert result.succeeded and result.value == EXECUTION_REQUEST
    else:
        assert result.succeeded and result.value == "echo result"
        session.call_tool.assert_awaited_once_with("echo", arguments={"text": "hello"})
    await registration.aclose()


@pytest.mark.asyncio
async def test_background_mcp_discovery_is_nonblocking_shared_and_visible_to_ai(tmp_path, monkeypatch):
    release_slow = asyncio.Event()
    started = {name: asyncio.Event() for name in ("slow", "fast")}
    attempts = []

    class Session:
        async def list_tools(self):
            return SimpleNamespace(tools=[{"name": "echo"}, {"name": "disabled"}])

    async def run_connection(registration, ready, close_event):
        name = registration.config.server_name
        attempts.append(name)
        started[name].set()
        try:
            if name == "slow":
                await release_slow.wait()
            registration._session = Session()
            ready.set_result(None)
            await close_event.wait()
        finally:
            registration._session = None

    monkeypatch.setattr(mcp_client.MCPServerRegistration, "_run_connection", run_connection)
    config = ToolConfig.model_validate({"mcpServers": {
        "slow": {"command": "demo", "enabled_tools": ["echo"], "description": "查询客户订单"},
        "fast": {"command": "demo", "enabled_tools": ["echo"], "description": "查询商品信息"},
        "private": {"command": "demo", "channel_tags": ["private"], "description": "查询内部薪资"},
    }})
    providers = [FakeProvider(), FakeProvider()]
    try:
        for provider in providers:
            await asyncio.wait_for(registry.register_tools(
                user_id="background-test", provider=provider, workspace=tmp_path, config=config,
            ), timeout=1)
        await asyncio.wait_for(asyncio.gather(*(event.wait() for event in started.values())), timeout=1)
        registrations = list(mcp_client.MCP_CONNECTIONS["background-test"].values())
        fast = next(registration for registration in registrations if registration.config.server_name == "fast")
        await asyncio.wait_for(fast.start_tool_discovery(), timeout=1)

        [result] = [result async for result in ToolExecutor(workspace=tmp_path, max_result_chars=10_000).execute(
            [ToolCallRequest(id="status", name="query_mcp_status", arguments={})], providers[0].function_tools,
        )]
        assert result.succeeded
        status = json.loads(result.value)["servers"]
        assert set(status) == {"slow", "fast"}
        assert status["slow"]["status"] == "connecting"
        assert status["slow"]["description"] == "查询客户订单"
        assert status["fast"]["status"] == "ready"
        assert status["fast"]["description"] == "查询商品信息"
        assert status["fast"]["tools"] == ["mcp_fast_echo"]
        for provider in providers:
            assert [tool.__name__ for _, tool in provider.mcp_tools] == ["mcp_fast_echo"]

        release_slow.set()
        await asyncio.wait_for(asyncio.gather(*(registration.start_tool_discovery() for registration in registrations)), 1)
        for provider in providers:
            assert [tool.__name__ for _, tool in provider.mcp_tools] == ["mcp_fast_echo", "mcp_slow_echo"]
            assert json.loads(await provider.function_tools["query_mcp_status"]())["servers"]["slow"]["status"] == "ready"

        cached_provider = FakeProvider()
        await registry.register_tools(
            user_id="background-test", provider=cached_provider, workspace=tmp_path, config=config,
        )
        assert len(cached_provider.mcp_tools) == 2
        assert sorted(attempts) == ["fast", "slow"]

        slow = next(registration for registration in registrations if registration.config.server_name == "slow")
        await slow.aclose()
        slow._start_retry_cooldown(ConnectionError("connection closed"))
        unavailable_provider = FakeProvider()
        await registry.register_tools(
            user_id="background-test", provider=unavailable_provider, workspace=tmp_path, config=config,
        )
        assert [tool.__name__ for _, tool in unavailable_provider.mcp_tools] == ["mcp_fast_echo"]
        status = json.loads(await unavailable_provider.function_tools["query_mcp_status"]())["servers"]
        assert status["slow"]["status"] == "unavailable"
    finally:
        await mcp_client.close_session_mcp_connections("background-test")


@pytest.mark.asyncio
async def test_mcp_status_tool_is_not_registered_without_servers(tmp_path):
    provider = FakeProvider()
    await registry.register_tools(user_id="no-mcp", provider=provider, workspace=tmp_path, config=ToolConfig())
    assert "query_mcp_status" not in provider.function_tools
    assert provider.mcp_tools == []


@pytest.mark.asyncio
async def test_ready_mcp_tools_are_sent_on_the_next_model_call(tmp_path, monkeypatch):
    release = asyncio.Event()

    class Session:
        async def list_tools(self):
            await release.wait()
            return SimpleNamespace(tools=[{"name": "echo"}, {"name": "ping"}])

    async def connect(registration):
        registration._session = Session()

    async def upstream():
        if False:
            yield

    monkeypatch.setattr(mcp_client.MCPServerRegistration, "_connect", connect)
    provider = LiteLLMProvider(
        api_key="test", api_base="https://example.test", default_model="test", parameters=LLMParameters(),
    )
    completion = Mock(side_effect=lambda **_kwargs: upstream())
    monkeypatch.setattr(provider._retry_client, "acompletion", completion)
    chats_scheduled_mid_registration = []
    register_mcp_tool = provider.register_mcp_tool

    def register_and_schedule_chat(tool, *, tool_json):
        register_mcp_tool(tool, tool_json=tool_json)
        if tool.__name__ == "mcp_demo_echo":
            # Schedule chat after the first insertion; it must only run once
            # the synchronous registration callback has published both tools.
            chats_scheduled_mid_registration.append(asyncio.create_task(provider.chat(messages=[])))

    monkeypatch.setattr(provider, "register_mcp_tool", register_and_schedule_chat)
    config = ToolConfig.model_validate({"mcpServers": {
        "demo": {"command": "demo", "description": "查询模型评测、对比模型能力和成本"},
    }})
    try:
        await registry.register_tools(user_id="next-model-call", provider=provider, workspace=tmp_path, config=config)
        first = await provider.chat(messages=[])
        assert "query_mcp_status" in first.mapping_tool_call_funcs
        status_schema = next(
            tool for tool in completion.call_args.kwargs["tools"]
            if tool["function"]["name"] == "query_mcp_status"
        )
        assert "demo" in status_schema["function"]["description"]
        assert "查询模型评测、对比模型能力和成本" in status_schema["function"]["description"]
        assert "是否需要等待" in status_schema["function"]["description"]
        assert "mcp_demo_echo" not in first.mapping_tool_call_funcs
        assert "mcp_demo_echo" not in {tool["function"]["name"] for tool in completion.call_args.kwargs["tools"]}
        assert [chunk async for chunk in first.iter_mix_chunks] == []

        release.set()
        [registration] = mcp_client.MCP_CONNECTIONS["next-model-call"].values()
        await asyncio.wait_for(registration.start_tool_discovery(), timeout=1)
        assert len(chats_scheduled_mid_registration) == 1
        [concurrent] = await asyncio.gather(*chats_scheduled_mid_registration)
        assert {"mcp_demo_echo", "mcp_demo_ping"} <= concurrent.mapping_tool_call_funcs.keys()
        assert {"mcp_demo_echo", "mcp_demo_ping"} <= {
            tool["function"]["name"] for tool in completion.call_args.kwargs["tools"]
        }
        assert [chunk async for chunk in concurrent.iter_mix_chunks] == []
        second = await provider.chat(messages=[])
        assert "mcp_demo_echo" in second.mapping_tool_call_funcs
        assert "mcp_demo_echo" in {tool["function"]["name"] for tool in completion.call_args.kwargs["tools"]}
        assert "mcp_demo_echo" not in first.mapping_tool_call_funcs
        assert [chunk async for chunk in second.iter_mix_chunks] == []
    finally:
        await mcp_client.close_session_mcp_connections("next-model-call")
