import asyncio
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
from pydantic import BaseModel, ConfigDict, Field

from xpeech.agent import loop as loop_module
from xpeech.agent.loop import AgentLoop
from xpeech.agent.server.routes.chat import answer_question
from xpeech.agent.tool_executor import ToolExecutor
from xpeech.agent.tools.execution_request import (
    EXECUTION_REQUEST,
    build_execution_request_tool,
    execution_request_schema,
)
from xpeech.agent.tools.helper import as_tool
from xpeech.agent.tools.mcp_client import create_mcp_registration
from xpeech.agent.tools.question import QuestionArgs, ask_user_question, validate_question_json
from xpeech.provider.litellm_provider import LiteLLMProvider
from xpeech.provider.schema import LLMParameters, ToolCallRequest


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
            form = validate_question_json(event["context"])
            assert "输出 hello" in form["subtitle"]
            assert "echo hello" in form["subtitle"]
            assert form["fields"][0]["name"] == "decision"
            assert form["fields"][0]["options"] == [
                {"label": "确认", "value": "confirm"},
                {"label": "取消", "value": "cancel"},
            ]
            await answer(reply)

    assert [event["event"] for event in events] == ["tool_call", "question", "tool_call_result"]
    [result] = json.loads(events[-1]["context"])
    assert result[:2] == ["shell-1", "shell"]
    assert history[-1]["content"] == result[2]
    assert ("approval-user", "approval-session") not in AgentLoop.SESSION_QUESTION_EVENT
    if expected == "confirmed":
        assert result[2] == "command output"
        [args] = executions
        assert type(args) is CommandArgs
        assert args.command == "echo hello" and args.timeout_ms == 100
    elif expected == "cancelled":
        assert executions == []
        assert "取消" in result[2]
    else:
        assert executions == []
        assert "格式异常" in result[2]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "arguments",
    [
        {"command": "ok"},
        {"command": "ok", "execution_description": "  "},
        {"command": "ok", "execution_description": "用途", "timeout_ms": -1},
    ],
)
async def test_invalid_request_never_prompts_or_executes(tmp_path, arguments):
    async def shell(args: CommandArgs):
        """Execute a command."""
        pytest.fail("invalid request executed")

    current_provider = provider()
    current_provider.register_function_tool(build_execution_request_tool(shell))
    call = ToolCallRequest(id="1", name="shell", arguments=arguments)
    events = [event async for event in stream_for(tmp_path, current_provider, [call], [])]
    assert [event["event"] for event in events] == ["tool_call", "tool_call_result"]
    assert "ValidationError" in json.loads(events[-1]["context"])[0][2]


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
    assert "超时" in json.loads(events[-1]["context"])[0][2]
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
            results.extend(json.loads(event["context"]))
    assert len(questions) == 2
    assert executions == ["ping"]
    assert "取消" in results[0][2]
    assert results[1][2] == "pong"


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
            assert json.loads(event["context"])[0][2] == "pong"


@pytest.mark.asyncio
@pytest.mark.parametrize("behavior", ["error", "timeout", "large_result"])
async def test_direct_execution_preserves_error_timeout_and_result_handling(tmp_path, behavior):
    async def ping():
        """Ping."""
        if behavior == "error":
            raise RuntimeError("test failure")
        if behavior == "timeout":
            await asyncio.sleep(1)
        return "x" * 10_100

    current_provider = provider()
    current_provider.register_function_tool(build_execution_request_tool(ping))
    call = ToolCallRequest(id="1", name="ping", arguments={"execution_description": "测试"})
    history = []
    async for event in stream_for(tmp_path, current_provider, [call], history, tool_timeout=0.05):
        if event["event"] == "question":
            await answer({"decision": "confirm"})
        elif event["event"] == "tool_call_result":
            result = json.loads(event["context"])[0][2]
            assert history[-1]["content"] == result
            if behavior == "error":
                assert result == "RuntimeError: test failure"
            elif behavior == "timeout":
                assert "TimeoutError" in result
            else:
                saved_path = result.split("Full result saved to: ")[1]
                assert (tmp_path / saved_path).read_text() == "x" * 10_100


@pytest.mark.asyncio
async def test_mcp_placeholder_preserves_schema_and_only_sends_original_arguments(tmp_path):
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

    registration = create_mcp_registration(server_name="demo", command="demo", enabled_tools=["*"])
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
    current_provider.register_mcp_tool(build_execution_request_tool(binding.func), tool_json=wrapped_schema)
    call = ToolCallRequest(
        id="1",
        name="mcp_demo_echo",
        arguments={
            "text": "hello",
            "extra": 1,
            "execution_description": "测试 MCP 调用",
        },
    )
    async for event in stream_for(tmp_path, current_provider, [call], []):
        if event["event"] == "question":
            assert remote_calls == []
            await answer({"decision": "confirm"})
        elif event["event"] == "tool_call_result":
            assert json.loads(event["context"])[0][2] == "remote result"
    assert remote_calls == [("echo", {"text": "hello", "extra": 1})]


@pytest.mark.asyncio
async def test_existing_ask_user_question_still_returns_user_answer(tmp_path):
    current_provider = provider()
    current_provider.register_function_tool(ask_user_question)
    args = QuestionArgs(question=json.dumps({"fields": [{"name": "name", "label": "名称", "type": "input"}]}))
    call = ToolCallRequest(id="1", name="ask_user_question", arguments=args.model_dump())
    async for event in stream_for(tmp_path, current_provider, [call], []):
        if event["event"] == "question":
            await answer({"name": "Alice"})
        elif event["event"] == "tool_call_result":
            assert json.loads(json.loads(event["context"])[0][2]) == {"name": "Alice"}


def test_placeholder_schema_keeps_original_constraints_and_hides_injected_context():
    async def shell(args: CommandArgs, session_id):
        """Run a command."""

    placeholder = build_execution_request_tool(shell)
    schema = as_tool(placeholder)["function"]["parameters"]
    assert placeholder.__name__ == "shell"
    assert set(schema["properties"]) == {"command", "timeout_ms", "execution_description"}
    assert "execution_description" in schema["required"]
    assert schema["properties"]["timeout_ms"]["exclusiveMinimum"] == 0
