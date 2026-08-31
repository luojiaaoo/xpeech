from pathlib import Path
from typing import Annotated

import pytest
from pydantic import BaseModel, Field, ValidationError

from xpeech.agent.tool_executor import ToolExecutor
from xpeech.agent.tools.mcp_client import _render_mcp_result
from xpeech.agent.tools.shell import _format_command_output
from xpeech.config.settings import MCPServerSettings, ToolConfig
from xpeech.provider.schema import ToolCallRequest


class EchoArgs(BaseModel):
    text: Annotated[str, Field(description="Text to echo")]


class TestToolExecutor:
    def test_result_limit_is_global_not_mcp_specific(self):
        assert ToolConfig().max_result_chars == 10_000
        with pytest.raises(ValidationError, match="max_result_chars"):
            MCPServerSettings(command="mcp-server", max_result_chars=1_000)

    def test_shell_output_is_not_truncated_before_executor(self):
        stdout = b"x" * 10_001

        result = _format_command_output(stdout, b"", 0)

        assert result.startswith("x" * 10_001)
        assert "truncated" not in result

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
        assert all(result.error is None for result in results)

    @pytest.mark.asyncio
    async def test_returns_failure_for_unregistered_tool(self, tmp_path: Path):
        call = ToolCallRequest(id="1", name="missing", arguments={})

        [result] = await ToolExecutor(workspace=tmp_path, max_result_chars=10_000).execute([call], {})

        assert not result.succeeded
        assert result.value == "ValueError: Tool is not registered for this request: missing"
        assert result.error == result.value

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
        assert result_path.parent.name == "tool-results"
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
        assert result.error == result.value
        assert result.value == "RuntimeError: " + "x" * 40
        assert not (tmp_path / "tool-results").exists()

    @staticmethod
    def _saved_result_path(workspace: Path, limited_result: str) -> Path:
        marker = "Full result saved to: "
        assert marker in limited_result
        return workspace / limited_result.rsplit(marker, maxsplit=1)[1]
