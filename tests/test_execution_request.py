import asyncio
import json
from copy import deepcopy
from functools import partial
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel, ConfigDict, Field

from xpeech.agent.skills.skill import SkillsLoader
from xpeech.agent import loop as loop_module
from xpeech.agent.loop import AgentLoop
from xpeech.agent.server.routes.chat import answer_question
from xpeech.agent.tool_executor import ToolExecutor
from xpeech.agent.tools.ask_user_question import QuestionArgs, ask_user_question
from xpeech.agent.tools.execution_request import (
    EXECUTION_REQUEST,
    ExecutionDecision,
    build_execution_request_tool,
    execution_request_schema,
)
from xpeech.agent.tools.helper import as_tool
from xpeech.agent.tools.mcp_client import create_mcp_registration
from xpeech.agent.tools.shell import tool as shell_module
from xpeech.provider.litellm_provider import LiteLLMProvider
from xpeech.provider.schema import LLMParameters, ToolCallRequest
from xpeech.subapp.channel.feishu.cards import build_feishu_question_card
from xpeech.utils.question import validate_confirm_question_json, validate_question_form_json
from xpeech.utils.security import dcg as dcg_module


class CommandArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    command: str
    timeout_ms: int = Field(default=100, gt=0)


def provider():
    return LiteLLMProvider(
        api_key="test", api_base="https://example.test", default_model="test", parameters=LLMParameters()
    )


def stream_for(tmp_path, current_provider, calls, history, *, tool_timeout=1):
    agent_loop = AgentLoop.__new__(AgentLoop)
    agent_loop.tool_executor = ToolExecutor(workspace=tmp_path, max_result_chars=10_000, max_tool_timeout=tool_timeout)
    response = SimpleNamespace(
        tool_calls=calls,
        mapping_tool_call_funcs={name: tool.func for name, tool in current_provider.default_tools.items()},
        content="",
    )
    return agent_loop.tool_call(
        response,
        history,
        0,
        "approval-user",
        "approval-session",
        "Alice",
        session_metadata={"source": "test"},
        channel="web",
    )


async def answer(value):
    result = await answer_question(
        "approval-user", "approval-session", value if isinstance(value, str) else json.dumps(value)
    )
    assert result == {"message": "Answer received"}


@pytest.mark.asyncio
async def test_placeholder_returns_marker_without_running_original_function(tmp_path):
    async def shell(args: CommandArgs):
        """Execute a command."""
        pytest.fail("placeholder executed the actual command")

    current_provider = provider()
    current_provider.register_function_tool(build_execution_request_tool(shell))
    call = ToolCallRequest(
        id="1", name="shell", arguments={"command": "echo hello", "execution_description": "输出 hello"}
    )
    [result] = [
        result
        async for result in ToolExecutor(workspace=tmp_path, max_result_chars=10_000).execute(
            [call],
            {"shell": current_provider.default_tools["shell"].func},
        )
    ]
    assert result.succeeded
    assert result.value == EXECUTION_REQUEST


@pytest.mark.parametrize("kwargs", [
    {"not_interactive": True},
    {"not_interactive": True, "requires_confirmation": None},
])
def test_noninteractive_wrapper_requires_confirmation_callback_at_registration(kwargs):
    async def ping():
        """Ping."""
        pytest.fail("registration must not execute the tool")

    with pytest.raises(ValueError, match="请在注册的时候就屏蔽掉，而不是在运行时屏蔽"):
        build_execution_request_tool(ping, **kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["shutdown -h now", "reboot", "poweroff", "init 6", "telinit 0"])
async def test_power_commands_are_rejected_without_prompting(tmp_path, monkeypatch, command):
    monkeypatch.setattr(shell_module.platform, "system", lambda: "Linux")
    monkeypatch.setattr(shell_module.shutil, "which", lambda name: f"/usr/bin/{name}")
    initialize = AsyncMock()
    execute = AsyncMock()
    monkeypatch.setattr(shell_module, "_ensure_workspace_uv_venv", initialize)
    monkeypatch.setattr(shell_module, "_run_wrapped_command", execute)
    process = SimpleNamespace(returncode=1, communicate=AsyncMock(return_value=(b'{"decision":"deny"}', b"")))
    monkeypatch.setattr(dcg_module.asyncio, "create_subprocess_exec", AsyncMock(return_value=process))
    current_provider = provider()
    current_provider.register_function_tool(
        build_execution_request_tool(
            shell_module.build_shell_tools(tmp_path, skills_loader=SkillsLoader(tmp_path)), partial(dcg_module.shell_requires_confirmation, workspace=tmp_path)
        )
    )
    call = ToolCallRequest(
        id="1", name="shell", arguments={"command": command, "execution_description": "测试硬拦截"}
    )
    events = []
    async for event in stream_for(tmp_path, current_provider, [call], []):
        events.append(event)
        if event["event"] == "question":
            await answer({"decision": "confirm"})
    assert [event["event"] for event in events] == ["tool_call", "tool_call_result"]
    assert "发现高危命令，直接拒绝执行" in json.loads(events[-1]["context"])["result"]
    initialize.assert_not_awaited()
    execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_shell_checks_once_before_confirmed_execution(tmp_path, monkeypatch):
    monkeypatch.setattr(shell_module.platform, "system", lambda: "Linux")
    monkeypatch.setattr(shell_module.shutil, "which", lambda name: f"/usr/bin/{name}")
    initialize = AsyncMock()
    execute = AsyncMock(return_value=shell_module.ShellCommandResult(b"removed", b"", 0))
    monkeypatch.setattr(shell_module, "_ensure_workspace_uv_venv", initialize)
    monkeypatch.setattr(shell_module, "_run_wrapped_command", execute)
    process = SimpleNamespace(returncode=1, communicate=AsyncMock(return_value=(b'{"decision":"ask"}', b"")))
    check = AsyncMock(return_value=process)
    monkeypatch.setattr(dcg_module.asyncio, "create_subprocess_exec", check)
    current_provider = provider()
    current_provider.register_function_tool(
        build_execution_request_tool(
            shell_module.build_shell_tools(tmp_path, skills_loader=SkillsLoader(tmp_path)), partial(dcg_module.shell_requires_confirmation, workspace=tmp_path)
        )
    )
    call = ToolCallRequest(
        id="1", name="shell", arguments={"command": "rm -rf ./build", "execution_description": "删除构建目录"}
    )
    events = []
    async for event in stream_for(tmp_path, current_provider, [call], []):
        events.append(event)
        if event["event"] == "question":
            execute.assert_not_awaited()
            await answer({"decision": "confirm"})
    assert [event["event"] for event in events] == ["tool_call", "question", "tool_call_result"]
    assert "removed" in json.loads(events[-1]["context"])["result"]
    check.assert_awaited_once()
    execute.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("async_check", [False, True])
@pytest.mark.parametrize("decision", list(ExecutionDecision))
@pytest.mark.parametrize("not_interactive", [False, True])
async def test_optional_confirmation_check_receives_json_and_controls_prompt(
    tmp_path, async_check, decision, not_interactive
):
    checks = []
    executions = []

    def check(payload: str) -> ExecutionDecision:
        checks.append(json.loads(payload))
        return decision

    async def async_checker(payload: str) -> ExecutionDecision:
        return check(payload)

    def shell(args: CommandArgs):
        """Execute a command."""
        executions.append(args)
        return "command output"

    current_provider = provider()
    current_provider.register_function_tool(
        build_execution_request_tool(
            shell,
            requires_confirmation=async_checker if async_check else check,
            not_interactive=not_interactive,
        )
    )
    arguments = {"command": "echo hello", "execution_description": "输出 hello"}
    call = ToolCallRequest(id="1", name="shell", arguments=arguments)
    events = []
    async for event in stream_for(tmp_path, current_provider, [call], []):
        events.append(event)
        if event["event"] == "question":
            assert not not_interactive
            assert executions == []
            await answer({"decision": "confirm"})

    assert checks == [arguments]
    assert [event["event"] for event in events] == (
        ["tool_call", "question", "tool_call_result"]
        if decision is ExecutionDecision.CONFIRM and not not_interactive
        else ["tool_call", "tool_call_result"]
    )
    result = json.loads(events[-1]["context"])["result"]
    if decision is ExecutionDecision.DENY or (decision is ExecutionDecision.CONFIRM and not_interactive):
        assert "PermissionError" in result and "拒绝执行" in result
        assert executions == []
        assert ("approval-user", "approval-session") not in AgentLoop.SESSION_QUESTION_EVENT
        return
    assert result == "command output"
    [args] = executions
    assert type(args) is CommandArgs
    assert args.command == "echo hello" and args.timeout_ms == 100


@pytest.mark.asyncio
@pytest.mark.parametrize("behavior", ["error", "timeout", "invalid_result", "deny"])
async def test_confirmation_check_failure_never_prompts_or_executes(tmp_path, behavior):
    async def ping():
        """Ping."""
        pytest.fail("failed check executed the actual tool")

    async def check(payload):
        if behavior == "error":
            raise RuntimeError("check failed")
        if behavior == "timeout":
            await asyncio.sleep(1)
        return ExecutionDecision.DENY if behavior == "deny" else "invalid"

    current_provider = provider()
    current_provider.register_function_tool(build_execution_request_tool(ping, check))
    call = ToolCallRequest(id="1", name="ping", arguments={"execution_description": "测试"})
    events = [event async for event in stream_for(tmp_path, current_provider, [call], [], tool_timeout=0.05)]
    assert [event["event"] for event in events] == ["tool_call", "tool_call_result"]
    result = json.loads(events[-1]["context"])["result"]
    if behavior == "error":
        assert result == "RuntimeError: check failed"
    elif behavior == "timeout":
        assert "timed out" in result
    elif behavior == "deny":
        assert result == "PermissionError: 发现高危命令，直接拒绝执行。"
    else:
        assert result == "TypeError: requires_confirmation must return an ExecutionDecision"


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_result", [None, True, False, "allow", "confirm", "deny"])
async def test_callback_requires_enum_and_never_executes_legacy_values(tmp_path, invalid_result):
    async def ping():
        """Ping."""
        pytest.fail("invalid decision executed the tool")

    current_provider = provider()
    current_provider.register_function_tool(build_execution_request_tool(ping, lambda _: invalid_result))
    call = ToolCallRequest(id="1", name="ping", arguments={"execution_description": "测试决策类型"})
    events = [event async for event in stream_for(tmp_path, current_provider, [call], [])]
    assert [event["event"] for event in events] == ["tool_call", "tool_call_result"]
    assert json.loads(events[-1]["context"])["result"] == (
        "TypeError: requires_confirmation must return an ExecutionDecision"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ({"decision": "confirm", "user_customization": ""}, "confirmed"),
        ("confirm", "confirmed"),
        ("answer=confirm", "confirmed"),
        ('{"decision":confirm}', "confirmed"),
        ({"decision": "cancel"}, "cancelled"),
        ("cancel", "cancelled"),
        ("confirm cancel", "cancelled"),
        ({"decision": "confirm", "user_customization": "cancel"}, "cancelled"),
        ({"decision": "cancel", "user_customization": "confirm"}, "cancelled"),
        ("invalid", "invalid"),
        ("", "invalid"),
        ({}, "invalid"),
    ],
)
async def test_waits_for_confirmation_using_the_entire_reply(tmp_path, reply, expected):
    executions = []

    async def shell(args: CommandArgs):
        """Execute a test command."""
        executions.append(args)
        return "command output"

    current_provider = provider()
    current_provider.register_function_tool(build_execution_request_tool(shell))
    call = ToolCallRequest(
        id="shell-1", name="shell", arguments={"command": "echo hello", "execution_description": "输出 hello"}
    )
    history = []
    events = []
    async for event in stream_for(tmp_path, current_provider, [call], history):
        events.append(event)
        if event["event"] == "question":
            assert executions == []
            form = validate_confirm_question_json(event["context"])
            assert "输出 hello" in form["description"]
            assert "echo hello" in form["description"]
            assert form["type"] == "confirm"
            assert "fields" not in form
            card = build_feishu_question_card(event["context"])
            assert card["header"]["subtitle"] == {"tag": "plain_text", "content": ""}
            [description, actions] = card["body"]["elements"]
            assert description == {
                "tag": "div",
                "text": {
                    "tag": "plain_text",
                    "content": form["description"],
                    "text_size": "normal",
                    "text_align": "left",
                    "text_color": "default",
                },
                "margin": "0px 0px 0px 0px",
            }
            buttons = [column["elements"][0] for column in actions["columns"]]
            assert [button["text"]["content"] for button in buttons] == ["确认", "取消"]
            assert [button["behaviors"][0]["value"] for button in buttons] == [
                {"decision": "confirm"}, {"decision": "cancel"},
            ]
            await answer(reply)

    assert [event["event"] for event in events] == ["tool_call", "question", "tool_call_result"]
    result = json.loads(events[-1]["context"])
    assert (result["id"], result["name"]) == ("shell-1", "shell")
    assert history[-1]["content"] == result["result"]
    assert ("approval-user", "approval-session") not in AgentLoop.SESSION_QUESTION_EVENT
    if expected == "confirmed":
        assert result["result"] == "command output"
        [args] = executions
        assert type(args) is CommandArgs
        assert args.command == "echo hello" and args.timeout_ms == 100
    elif expected == "cancelled":
        assert executions == []
        assert "取消" in result["result"]
    else:
        assert executions == []
        assert "格式异常" in result["result"]


@pytest.mark.asyncio
@pytest.mark.parametrize("conditional", [False, True])
@pytest.mark.parametrize(
    "arguments",
    [
        {"command": "ok"},
        {"command": "ok", "execution_description": "  "},
        {"command": "ok", "execution_description": "用途", "timeout_ms": -1},
    ],
)
async def test_invalid_request_never_prompts_or_executes(tmp_path, arguments, conditional):
    async def shell(args: CommandArgs):
        """Execute a command."""
        pytest.fail("invalid request executed")

    def check(payload):
        pytest.fail("invalid request reached confirmation check")

    current_provider = provider()
    current_provider.register_function_tool(build_execution_request_tool(shell, check if conditional else None))
    call = ToolCallRequest(id="1", name="shell", arguments=arguments)
    events = [event async for event in stream_for(tmp_path, current_provider, [call], [])]
    assert [event["event"] for event in events] == ["tool_call", "tool_call_result"]
    assert "ValidationError" in json.loads(events[-1]["context"])["result"]


@pytest.mark.asyncio
async def test_timeout_does_not_execute(tmp_path, monkeypatch):
    monkeypatch.setattr(loop_module, "USER_TIMEOUT", 0.01)

    async def ping():
        """Ping."""
        pytest.fail("timed out request executed")

    current_provider = provider()
    current_provider.register_function_tool(build_execution_request_tool(ping))
    call = ToolCallRequest(id="1", name="ping", arguments={"execution_description": "测试连通性"})
    events = [event async for event in stream_for(tmp_path, current_provider, [call], [])]
    assert "超时" in json.loads(events[-1]["context"])["result"]
    assert ("approval-user", "approval-session") not in AgentLoop.SESSION_QUESTION_EVENT


@pytest.mark.asyncio
async def test_multiple_requests_are_confirmed_individually(tmp_path):
    executions = []

    def ping():
        """Synchronous ping."""
        executions.append("ping")
        return "pong"

    current_provider = provider()
    current_provider.register_function_tool(build_execution_request_tool(ping))
    calls = [
        ToolCallRequest(id=str(i), name="ping", arguments={"execution_description": f"测试 {i}"}) for i in range(2)
    ]
    questions = []
    results = []
    async for event in stream_for(tmp_path, current_provider, calls, []):
        if event["event"] == "question":
            questions.append(event)
            assert executions == []
            await answer({"decision": "cancel" if len(questions) == 1 else "confirm"})
        elif event["event"] == "tool_call_result":
            results.append(json.loads(event["context"]))
    assert len(questions) == 2
    assert executions == ["ping"]
    assert "取消" in results[0]["result"]
    assert results[1]["result"] == "pong"


@pytest.mark.asyncio
async def test_cancelling_while_waiting_cleans_up_without_execution(tmp_path):
    async def ping():
        """Ping."""
        pytest.fail("cancelled request executed")

    current_provider = provider()
    current_provider.register_function_tool(build_execution_request_tool(ping))
    call = ToolCallRequest(id="1", name="ping", arguments={"execution_description": "测试"})
    question_received = asyncio.Event()

    async def consume():
        async for event in stream_for(tmp_path, current_provider, [call], []):
            if event["event"] == "question":
                question_received.set()

    task = asyncio.create_task(consume())
    await asyncio.wait_for(question_received.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert ("approval-user", "approval-session") not in AgentLoop.SESSION_QUESTION_EVENT


@pytest.mark.asyncio
async def test_confirmation_wait_does_not_consume_execution_timeout(tmp_path):
    async def ping():
        """Ping."""
        return "pong"

    current_provider = provider()
    current_provider.register_function_tool(build_execution_request_tool(ping))
    call = ToolCallRequest(id="1", name="ping", arguments={"execution_description": "测试"})
    async for event in stream_for(tmp_path, current_provider, [call], [], tool_timeout=0.01):
        if event["event"] == "question":
            await asyncio.sleep(0.03)
            await answer({"decision": "confirm"})
        elif event["event"] == "tool_call_result":
            assert json.loads(event["context"])["result"] == "pong"


@pytest.mark.asyncio
@pytest.mark.parametrize("needs_confirmation", [False, True])
@pytest.mark.parametrize("behavior", ["error", "timeout", "large_result"])
async def test_direct_execution_preserves_error_timeout_and_result_handling(
    tmp_path, behavior, needs_confirmation
):
    async def ping():
        """Ping."""
        if behavior == "error":
            raise RuntimeError("test failure")
        if behavior == "timeout":
            await asyncio.sleep(1)
        return "x" * 10_100

    current_provider = provider()
    current_provider.register_function_tool(build_execution_request_tool(
        ping, lambda _: ExecutionDecision.CONFIRM if needs_confirmation else ExecutionDecision.ALLOW
    ))
    call = ToolCallRequest(id="1", name="ping", arguments={"execution_description": "测试"})
    history = []
    async for event in stream_for(tmp_path, current_provider, [call], history, tool_timeout=0.05):
        if event["event"] == "question":
            await answer({"decision": "confirm"})
        elif event["event"] == "tool_call_result":
            result = json.loads(event["context"])["result"]
            assert history[-1]["content"] == result
            if behavior == "error":
                assert result == "RuntimeError: test failure"
            elif behavior == "timeout":
                assert ("TimeoutError" if needs_confirmation else "timed out") in result
            else:
                saved_path = result.split("Full result saved to: ")[1]
                assert (tmp_path / saved_path).read_text() == "x" * 10_100


@pytest.mark.asyncio
@pytest.mark.parametrize("needs_confirmation", [None, False, True])
@pytest.mark.parametrize("not_interactive", [False, True])
async def test_mcp_placeholder_preserves_schema_and_only_sends_original_arguments(
    tmp_path, needs_confirmation, not_interactive
):
    remote_calls = []

    class Session:
        async def list_tools(self):
            return SimpleNamespace(
                tools=[
                    {
                        "name": "echo",
                        "description": "Echo",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "text": {"type": "string", "enum": ["hello", "world"]},
                                "style": {"type": "string", "default": "plain"},
                            },
                            "required": ["text"],
                            "additionalProperties": True,
                        },
                    }
                ]
            )

        async def call_tool(self, name, arguments):
            remote_calls.append((name, arguments))
            return {"content": [{"type": "text", "text": "remote result"}]}

    registration = create_mcp_registration(server_name="demo", command="demo", enabled_tools=["*"], workspace=tmp_path, skills_loader=SkillsLoader(tmp_path))
    registration._session = Session()
    [binding] = await registration._discover_tool_bindings()
    original = deepcopy(binding.schema)
    wrapped_schema = execution_request_schema(binding.schema)
    assert binding.schema == original
    parameters = wrapped_schema["function"]["parameters"]
    assert parameters["required"] == ["text", "execution_description"]
    assert parameters["properties"]["text"]["enum"] == ["hello", "world"]
    assert parameters["additionalProperties"] is True

    current_provider = provider()
    checks = []

    def check(payload):
        checks.append(json.loads(payload))
        return ExecutionDecision.CONFIRM if needs_confirmation else ExecutionDecision.ALLOW

    if not_interactive and needs_confirmation is None:
        with pytest.raises(ValueError, match="请在注册的时候就屏蔽掉，而不是在运行时屏蔽"):
            build_execution_request_tool(binding.func, not_interactive=True)
        assert remote_calls == []
        return

    current_provider.register_mcp_tool(
        build_execution_request_tool(
            binding.func, check if needs_confirmation is not None else None, not_interactive=not_interactive
        ),
        tool_json=wrapped_schema,
    )
    call = ToolCallRequest(
        id="1",
        name="mcp_demo_echo",
        arguments={
            "text": "hello",
            "extra": 1,
            "execution_description": "测试 MCP 调用",
        },
    )
    denied = not_interactive and needs_confirmation is not False
    events = []
    async for event in stream_for(tmp_path, current_provider, [call], []):
        events.append(event)
        if event["event"] == "question":
            assert not not_interactive
            assert remote_calls == []
            await answer({"decision": "confirm"})
        elif event["event"] == "tool_call_result":
            result = json.loads(event["context"])["result"]
            if denied:
                assert "PermissionError" in result and "拒绝执行" in result
            else:
                assert result == "remote result"
    if not_interactive:
        assert [event["event"] for event in events] == ["tool_call", "tool_call_result"]
    assert remote_calls == ([] if denied else [("echo", {"text": "hello", "extra": 1})])
    assert checks == ([] if needs_confirmation is None else [call.arguments])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("existing_names", "custom_name"),
    [
        (["name"], "user_customization"),
        ([], "user_customization"),
        (["user_customization", "user_customization_1"], "user_customization_2"),
    ],
)
async def test_existing_ask_user_question_still_returns_user_answer(tmp_path, existing_names, custom_name):
    current_provider = provider()
    current_provider.register_function_tool(ask_user_question)
    fields = [{"name": name, "label": "名称", "type": "input"} for name in existing_names]
    args = QuestionArgs(question=json.dumps({"fields": fields}))
    expected_context = ask_user_question(args)
    call = ToolCallRequest(id="1", name="ask_user_question", arguments=args.model_dump())
    reply = {**dict.fromkeys(existing_names, "Alice"), custom_name: "补充内容"}
    async for event in stream_for(tmp_path, current_provider, [call], []):
        if event["event"] == "question":
            assert event["context"] == expected_context
            form = validate_question_form_json(event["context"])
            assert [field["name"] for field in form["fields"]] == [*existing_names, custom_name]
            assert form["fields"][-1]["type"] == "input"
            assert form["fields"][-1]["label"] == "补充说明"
            card = build_feishu_question_card(event["context"])
            inputs = [element for element in card["body"]["elements"][0]["elements"] if element["tag"] == "input"]
            assert [element["name"] for element in inputs] == [*existing_names, custom_name]
            await answer(reply)
        elif event["event"] == "tool_call_result":
            assert json.loads(json.loads(event["context"])["result"]) == reply


def test_placeholder_schema_keeps_original_constraints_and_hides_injected_context():
    async def shell(args: CommandArgs, session_id):
        """Run a command."""

    placeholder = build_execution_request_tool(shell)
    schema = as_tool(placeholder)["function"]["parameters"]
    assert placeholder.__name__ == "shell"
    assert set(schema["properties"]) == {"command", "timeout_ms", "execution_description"}
    assert "execution_description" in schema["required"]
    assert schema["properties"]["timeout_ms"]["exclusiveMinimum"] == 0
