import asyncio
from datetime import date
from pathlib import Path
from typing import Annotated

import pytest
from openai import pydantic_function_tool
from pydantic import BaseModel, Field, ValidationError, create_model

from xpeech.agent.tool_executor import ToolExecutor
from xpeech.agent.tools import sandbox
from xpeech.agent.tools.helper import EmptyToolArgs, as_tool, get_tool_model_cls
from xpeech.agent.tools.mcp_client import _render_mcp_result
from xpeech.agent.tools.shell import (
    MAX_EXEC_TIMEOUT_MS,
    ShellArgs,
    _format_command_output,
    _guard_command,
    _run_wrapped_command,
)
from xpeech.config.settings import MCPServerSettings, ToolConfig
from xpeech.provider.schema import ToolCallRequest


class EchoArgs(BaseModel):
    text: Annotated[str, Field(description="Text to echo")]


@pytest.mark.parametrize("has_model", [False, True])
@pytest.mark.parametrize("has_metadata", [False, True])
def test_tool_model_info(has_model, has_metadata):
    def empty():
        pass

    def model_only(args: EchoArgs):
        pass

    def metadata_only(session_metadata):
        pass

    def both(args: EchoArgs, session_metadata):
        pass

    func = {
        (False, False): empty,
        (True, False): model_only,
        (False, True): metadata_only,
        (True, True): both,
    }[has_model, has_metadata]
    info = get_tool_model_cls(func)

    assert info.has_pydantic_param is has_model
    assert info.has_session_metadata is has_metadata
    assert info.has_session_id is False
    assert info.has_sender_name is False
    if has_model:
        assert info.model_cls is EchoArgs
    else:
        assert issubclass(info.model_cls, BaseModel)
        assert info.model_cls is EmptyToolArgs
        assert info.model_cls().model_dump() == {}


def test_shared_empty_model_matches_dynamic_model_parameter_constraints():
    shared = pydantic_function_tool(EmptyToolArgs, name="ping", description="Ping tool.")
    dynamic = pydantic_function_tool(create_model("pingArgs"), name="ping", description="Ping tool.")

    shared_params = shared["function"]["parameters"]
    dynamic_params = dynamic["function"]["parameters"]
    assert shared_params["title"] == "EmptyToolArgs"
    assert dynamic_params["title"] == "pingArgs"
    assert shared_params["description"] == EmptyToolArgs.__doc__
    assert "description" not in dynamic_params
    for params in (shared_params, dynamic_params):
        params.pop("title")
        params.pop("description", None)
    assert shared == dynamic
    assert shared_params == {
        "type": "object", "properties": {}, "required": [], "additionalProperties": False,
    }


def test_tools_sharing_empty_model_keep_independent_names_and_descriptions():
    def ping():
        """Ping tool."""

    def context(session_metadata):
        """Context tool."""

    ping_schema = as_tool(ping)
    context_schema = as_tool(context)
    assert ping_schema["function"]["name"] == "ping"
    assert ping_schema["function"]["description"] == "Ping tool."
    assert context_schema["function"]["name"] == "context"
    assert context_schema["function"]["description"] == "Context tool."
    assert ping_schema["function"]["strict"] is True
    assert context_schema["function"]["strict"] is True
    assert ping_schema["function"]["parameters"] == context_schema["function"]["parameters"]
    context_schema["function"]["parameters"]["properties"]["changed"] = {}
    assert ping_schema["function"]["parameters"]["properties"] == {}
    assert EmptyToolArgs.model_validate_json("{}").model_dump() == {}


class TestToolExecutor:
    def test_shell_guard_allows_format_url_query_parameter(self, tmp_path: Path):
        command = 'curl -s "wttr.in/Wuhan?lang=zh&format=%l:+%c+%t+%h+%w"'

        assert _guard_command(command, tmp_path) == command

    def test_result_limit_is_global_not_mcp_specific(self):
        assert ToolConfig().max_result_chars == 10_000
        assert ToolConfig().max_tool_timeout == 300
        with pytest.raises(ValidationError, match="max_result_chars"):
            MCPServerSettings(command="mcp-server", max_result_chars=1_000)
        with pytest.raises(ValidationError, match="max_tool_timeout"):
            ToolConfig(max_tool_timeout=0)

    def test_shell_output_is_not_truncated_before_executor(self):
        stdout = b"x" * 10_001

        result = _format_command_output(stdout, b"", 0)

        assert result.startswith("x" * 10_001)
        assert "truncated" not in result

    def test_shell_timeout_argument_is_bounded(self):
        assert ShellArgs(command="true").timeout_ms == 120_000
        assert ShellArgs(command="true", timeout_ms=MAX_EXEC_TIMEOUT_MS).timeout_ms == MAX_EXEC_TIMEOUT_MS
        with pytest.raises(ValidationError, match="timeout_ms"):
            ShellArgs(command="true", timeout_ms=0)
        with pytest.raises(ValidationError, match="timeout_ms"):
            ShellArgs(command="true", timeout_ms=MAX_EXEC_TIMEOUT_MS + 1)

    @pytest.mark.asyncio
    async def test_shell_timeout_terminates_command_and_preserves_partial_output(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(sandbox, "wrap_command", lambda command, workspace, env=None: command)

        result = await asyncio.wait_for(
            _run_wrapped_command("printf 'partial output\\n'; sleep 10 & wait", tmp_path, timeout_ms=500),
            timeout=2,
        )

        assert result.timed_out is True
        assert result.stdout == b"partial output\n"
        output = _format_command_output(
            result.stdout,
            result.stderr,
            result.returncode,
            timed_out=result.timed_out,
            timeout_ms=500,
        )
        assert "partial output" in output
        assert "Command timed out after 0.5s" in output

    @pytest.mark.asyncio
    async def test_shell_completed_command_is_not_marked_timed_out(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(sandbox, "wrap_command", lambda command, workspace, env=None: command)

        result = await _run_wrapped_command("printf done", tmp_path, timeout_ms=1_000)

        assert result.stdout == b"done"
        assert result.returncode == 0
        assert result.timed_out is False

    def test_mcp_output_is_not_truncated_before_executor(self):
        full_result = "x" * 50_001

        result = _render_mcp_result({"content": [{"type": "text", "text": full_result}]})

        assert result == full_result

    @pytest.mark.asyncio
    async def test_executes_tools_and_preserves_request_order(self, tmp_path: Path):
        async def echo(args: EchoArgs) -> str:
            return args.text

        calls = [
            ToolCallRequest(id="2", name="echo", arguments={"text": "second"}),
            ToolCallRequest(id="1", name="echo", arguments={"text": "first"}),
        ]

        results = await ToolExecutor(workspace=tmp_path, max_result_chars=10_000).execute(
            calls,
            {"echo": echo},
            loop_count=3,
        )

        assert [result.call.id for result in results] == ["2", "1"]
        assert [result.value for result in results] == ["second", "first"]
        assert all(result.succeeded for result in results)

    @pytest.mark.asyncio
    async def test_timed_out_tool_does_not_block_other_results(self, tmp_path: Path):
        cancelled = asyncio.Event()

        async def stuck_tool() -> str:
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        async def fast_tool() -> str:
            return "ready"

        calls = [
            ToolCallRequest(id="1", name="stuck_tool", arguments={}),
            ToolCallRequest(id="2", name="fast_tool", arguments={}),
        ]
        results = await ToolExecutor(
            workspace=tmp_path,
            max_result_chars=10_000,
            max_tool_timeout=0.02,
        ).execute(calls, {"stuck_tool": stuck_tool, "fast_tool": fast_tool})

        assert cancelled.is_set()
        assert [result.call.id for result in results] == ["1", "2"]
        assert results[0].succeeded is False
        assert results[0].value == "Tool call timed out after 0.02s"
        assert results[1].succeeded is True
        assert results[1].value == "ready"

    @pytest.mark.asyncio
    async def test_tool_raised_timeout_keeps_its_own_error(self, tmp_path: Path):
        async def network_tool() -> str:
            raise TimeoutError("read timed out")

        [result] = await ToolExecutor(
            workspace=tmp_path,
            max_result_chars=10_000,
            max_tool_timeout=300,
        ).execute(
            [ToolCallRequest(id="1", name="network_tool", arguments={})],
            {"network_tool": network_tool},
        )

        assert result.succeeded is False
        assert result.value == "TimeoutError: read timed out"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("metadata", [{"channel": "test"}, {}, None])
    async def test_injects_session_metadata_only_when_declared(self, tmp_path: Path, metadata):
        async def echo(args: EchoArgs, session_metadata: dict[str, str] | None) -> str:
            """Echo with session context."""
            assert session_metadata is metadata
            return args.text

        async def context_only(*, session_metadata):
            """Read session context."""
            assert session_metadata is metadata
            return "context"

        async def plain(args: EchoArgs) -> str:
            return args.text

        schema = as_tool(echo)["function"]["parameters"]
        assert set(schema["properties"]) == {"text"}
        assert as_tool(context_only)["function"]["parameters"]["properties"] == {}
        calls = [
            ToolCallRequest(id="1", name="echo", arguments={"text": "echo"}),
            ToolCallRequest(id="2", name="context", arguments={}),
            ToolCallRequest(id="3", name="plain", arguments={"text": "plain"}),
        ]
        results = await ToolExecutor(workspace=tmp_path, max_result_chars=10_000).execute(
            calls, {"echo": echo, "context": context_only, "plain": plain},
            session_metadata=metadata,
        )
        assert all(result.succeeded for result in results)
        assert [result.value for result in results] == ["echo", "context", "plain"]

    @pytest.mark.asyncio
    async def test_hides_and_injects_all_runtime_context(self, tmp_path: Path):
        async def context(
            args: EchoArgs,
            session_metadata,
            session_id,
            sender_name,
        ) -> str:
            """Return all runtime context."""
            return f"{args.text}:{session_id}:{sender_name}:{session_metadata['channel']}"

        schema = as_tool(context)["function"]["parameters"]
        assert set(schema["properties"]) == {"text"}
        [result] = await ToolExecutor(workspace=tmp_path, max_result_chars=10_000).execute(
            [ToolCallRequest(id="1", name="context", arguments={"text": "ok"})],
            {"context": context},
            session_metadata={"channel": "feishu"},
            session_id="session-1",
            sender_name="Alice",
        )
        assert result.value == "ok:session-1:Alice:feishu"

    @pytest.mark.asyncio
    async def test_returns_failure_for_unregistered_tool(self, tmp_path: Path):
        call = ToolCallRequest(id="1", name="missing", arguments={})

        [result] = await ToolExecutor(workspace=tmp_path, max_result_chars=10_000).execute([call], {})

        assert not result.succeeded
        assert result.value == "ValueError: Tool is not registered for this request: missing"

    @pytest.mark.asyncio
    async def test_saves_oversized_result_and_returns_prefix_with_path(self, tmp_path: Path):
        full_result = "abcdefghij" * 4

        async def large_result() -> str:
            return full_result

        call = ToolCallRequest(id="call/1", name="large_tool", arguments={})
        [result] = await ToolExecutor(workspace=tmp_path, max_result_chars=12).execute(
            [call],
            {"large_tool": large_result},
        )

        assert result.succeeded
        assert result.value.startswith(full_result[:12])
        result_path = self._saved_result_path(tmp_path, result.value)
        assert result_path.read_text(encoding="utf-8") == full_result
        assert result_path.parent.parent == tmp_path / "tool-results"
        assert result_path.parent.name == date.today().isoformat()
        assert result_path.name.startswith("large_tool-")

    @pytest.mark.asyncio
    async def test_does_not_offload_oversized_read_file_result(self, tmp_path: Path):
        full_result = "abcdefghij" * 4

        async def read_file() -> str:
            return full_result

        call = ToolCallRequest(id="1", name="read_file", arguments={})
        [result] = await ToolExecutor(workspace=tmp_path, max_result_chars=12).execute(
            [call],
            {"read_file": read_file},
        )

        assert result.succeeded
        assert result.value == full_result
        assert not (tmp_path / "tool-results").exists()

    @pytest.mark.asyncio
    async def test_does_not_limit_list_result(self, tmp_path: Path):
        image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}
        full_result = [
            {"type": "text", "text": "first block"},
            image,
            {"type": "text", "text": "second block"},
        ]

        async def multimodal_result() -> list[dict]:
            return full_result

        call = ToolCallRequest(id="1", name="multimodal", arguments={})
        [result] = await ToolExecutor(workspace=tmp_path, max_result_chars=8).execute(
            [call],
            {"multimodal": multimodal_result},
        )

        assert result.succeeded
        assert result.value == full_result
        assert not (tmp_path / "tool-results").exists()

    @pytest.mark.asyncio
    async def test_does_not_limit_tool_error(self, tmp_path: Path):
        async def failing_tool() -> str:
            raise RuntimeError("x" * 40)

        call = ToolCallRequest(id="1", name="failure", arguments={})
        [result] = await ToolExecutor(workspace=tmp_path, max_result_chars=16).execute(
            [call],
            {"failure": failing_tool},
        )

        assert not result.succeeded
        assert result.value == "RuntimeError: " + "x" * 40
        assert not (tmp_path / "tool-results").exists()

    @staticmethod
    def _saved_result_path(workspace: Path, limited_result: str) -> Path:
        marker = "Full result saved to: "
        assert marker in limited_result
        return workspace / limited_result.rsplit(marker, maxsplit=1)[1]
