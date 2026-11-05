import asyncio
import platform
import shutil
from datetime import date
from pathlib import Path
from typing import Annotated

import pytest
import pytest_asyncio
from openai import pydantic_function_tool
from pydantic import BaseModel, Field, ValidationError, create_model

from xpeech.agent.skills.skill import SkillsLoader
from xpeech.agent.tool_executor import ToolExecutor
from xpeech.agent.tools import sandbox
from xpeech.agent.tools.helper import EmptyToolArgs, as_tool, get_tool_model_cls
from xpeech.agent.tools.mcp_client import _render_mcp_result
from xpeech.agent.tools.shell import build_shell_tools
from xpeech.agent.tools.shell import tool as shell_module
from xpeech.agent.tools.shell.schema import MAX_EXEC_TIMEOUT_MS, ShellArgs, ShellCommandResult
from xpeech.agent.tools.shell.session import SandboxManager
from xpeech.agent.tools.shell.tool import _format_command_output, _guard_command, _run_wrapped_command
from xpeech.config.settings import MCPServerSettings, ToolConfig
from xpeech.provider.schema import ToolCallRequest


@pytest_asyncio.fixture
async def shell_session_manager(monkeypatch):
    manager = SandboxManager()
    monkeypatch.setattr(shell_module, "sandbox_manager", manager)
    try:
        yield manager
    finally:
        await manager.close()


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
    assert info.has_user_id is False
    assert info.has_sender_name is False
    assert info.has_channel is False
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
    @pytest.mark.parametrize(
        "command",
        ["rm -rf ./test-dir", "dd if=input.bin of=output.bin", "mkfs --help"],
    )
    def test_shell_guard_leaves_dangerous_command_decisions_to_dcg(self, command: str):
        assert _guard_command(command) == command

    @pytest.mark.parametrize(
        "command",
        [
            "python --version",
            "pip --version",
            "python3.12 --version",
            "pip3.12 --version",
            "echo ready && python3 --version",
            "sudo -u root python3 --version",
            "env TEST=1 pip --version",
            "uv run python --version && python3 --version",
            "python 'unterminated",
        ],
    )
    def test_shell_guard_still_rejects_direct_python_and_pip(self, command: str):
        with pytest.raises(RuntimeError, match="run Python through 'uv run python/pip'"):
            _guard_command(command)

    @pytest.mark.parametrize(
        "command",
        ["uv run python --version", "uv pip install demo", 'echo "python3"', "command -v python3"],
    )
    def test_shell_guard_allows_uv_and_python_references(self, command: str):
        assert _guard_command(command) == command

    @pytest.mark.parametrize(
        "command",
        [
            "/usr/bin/ls /usr",
            "printf data > /tmp/output.txt",
            "echo ready > /dev/null",
            "cat /dev/fd/0",
            "cat ~/.npmrc",
            "cat ../secret.txt",
            r"cat ..\secret.txt",
            'printf "%s" "/outside/workspace"',
        ],
    )
    def test_shell_guard_leaves_path_access_to_sandbox(self, command: str):
        assert _guard_command(command) == command

    def test_shell_guard_uses_configured_callable_and_message(self, monkeypatch):
        checked_commands = []

        def is_blocked(command: str) -> bool:
            checked_commands.append(command)
            return command.endswith("blocked_test_command")

        monkeypatch.setitem(
            shell_module.SHELL_BLOCK_RULES,
            is_blocked,
            "Custom command is prohibited.",
        )
        with pytest.raises(RuntimeError, match="Custom command is prohibited"):
            _guard_command("env TEST=1 blocked_test_command")
        assert checked_commands == ["env TEST=1 blocked_test_command"]

    def test_shell_guard_allows_format_url_query_parameter(self):
        command = 'curl -s "wttr.in/Wuhan?lang=zh&format=%l:+%c+%t+%h+%w"'

        assert _guard_command(command) == command

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

    def test_shell_builder_does_not_expand_the_backend_home(self, tmp_path: Path, monkeypatch):
        workspace = tmp_path / "~" / "workspace"
        workspace.mkdir(parents=True)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("HOME", str(tmp_path / "backend-home"))
        monkeypatch.setattr(shell_module.platform, "system", lambda: "Linux")
        monkeypatch.setattr(shell_module.shutil, "which", lambda name: f"/usr/bin/{name}")

        assert build_shell_tools(workspace, skills_loader=SkillsLoader(workspace)).__name__ == "shell"

    @pytest.mark.skipif(
        platform.system() != "Linux" or any(shutil.which(name) is None for name in sandbox.SANDBOX_EXECUTABLES),
        reason="Requires Linux sandbox dependencies",
    )
    @pytest.mark.asyncio
    async def test_shell_timeout_terminates_command_and_preserves_partial_output(
        self, tmp_path: Path, monkeypatch, shell_session_manager,
    ):
        logged_errors: list[str] = []
        monkeypatch.setattr(
            shell_module.logger,
            "error",
            lambda message, *args: logged_errors.append(message.format(*args)),
        )

        result = await asyncio.wait_for(
            _run_wrapped_command("printf 'partial output\\n'; sleep 10 & wait", tmp_path, timeout_ms=500, skills_loader=SkillsLoader(tmp_path)),
            timeout=5,
        )

        assert result.timed_out is True
        assert result.stdout == b"partial output\n"
        assert len(logged_errors) == 1
        assert logged_errors[0].startswith("Shell command timed out after 500ms; terminating command process group")
        output = _format_command_output(
            result.stdout,
            result.stderr,
            result.returncode,
            timed_out=result.timed_out,
            timeout_ms=500,
        )
        assert output.startswith("ERROR: Command timed out after 0.5s")
        assert "partial output" in output
        assert "Command timed out after 0.5s" in output

    @pytest.mark.skipif(
        platform.system() != "Linux" or any(shutil.which(name) is None for name in sandbox.SANDBOX_EXECUTABLES),
        reason="Requires Linux sandbox dependencies",
    )
    @pytest.mark.asyncio
    async def test_shell_completed_command_is_not_marked_timed_out(self, tmp_path: Path, shell_session_manager):

        result = await _run_wrapped_command("printf done", tmp_path, timeout_ms=1_000, skills_loader=SkillsLoader(tmp_path))

        assert result.stdout == b"done"
        assert result.returncode == 0
        assert result.timed_out is False

    @pytest.mark.asyncio
    async def test_shell_timeout_result_is_limited_normally(self, tmp_path: Path, monkeypatch):
        async def skip_venv(_workspace: Path, **_kwargs) -> None:
            return None

        async def timed_out_command(*_args, **_kwargs) -> ShellCommandResult:
            return ShellCommandResult(
                stdout=b"partial output\n",
                stderr=b"still running\n",
                returncode=-9,
                timed_out=True,
            )

        monkeypatch.setattr(shell_module.platform, "system", lambda: "Linux")
        monkeypatch.setattr(shell_module.shutil, "which", lambda name: f"/usr/bin/{name}")
        monkeypatch.setattr(shell_module, "_ensure_workspace_uv_venv", skip_venv)
        monkeypatch.setattr(shell_module, "_run_wrapped_command", timed_out_command)
        shell = build_shell_tools(tmp_path, skills_loader=SkillsLoader(tmp_path))

        [result] = [
            result
            async for result in ToolExecutor(workspace=tmp_path, max_result_chars=32).execute(
                [ToolCallRequest(id="1", name="shell", arguments={"command": "sleep 10", "timeout_ms": 500})],
                {"shell": shell},
            )
        ]

        assert result.succeeded is True
        assert result.value.startswith("ERROR: Command timed out after 0")
        assert "Full result saved to:" in result.value
        full_result = self._saved_result_path(tmp_path, result.value).read_text(encoding="utf-8")
        assert full_result.startswith("ERROR: Command timed out after 0.5s")
        assert "partial output" in full_result
        assert "STDERR:\nstill running" in full_result
        assert "Command timed out after 0.5s" in full_result

    def test_mcp_output_is_not_truncated_before_executor(self):
        full_result = "x" * 50_001

        result = _render_mcp_result({"content": [{"type": "text", "text": full_result}]})

        assert result == full_result

    @pytest.mark.asyncio
    async def test_executes_each_tool_once(self, tmp_path: Path):
        async def echo(args: EchoArgs) -> str:
            return args.text

        calls = [
            ToolCallRequest(id="2", name="echo", arguments={"text": "second"}),
            ToolCallRequest(id="1", name="echo", arguments={"text": "first"}),
        ]

        results = [
            result
            async for result in ToolExecutor(workspace=tmp_path, max_result_chars=10_000).execute(
                calls,
                {"echo": echo},
                loop_count=3,
            )
        ]

        assert [result.call.id for result in results] == ["2", "1"]
        assert [result.value for result in results] == ["second", "first"]
        assert all(result.succeeded for result in results)

    @pytest.mark.asyncio
    async def test_yields_completed_tool_before_other_tools_finish(self, tmp_path: Path):
        release_slow = asyncio.Event()
        slow_started = asyncio.Event()

        async def slow_tool() -> str:
            slow_started.set()
            await release_slow.wait()
            return "slow"

        async def fast_tool() -> str:
            await slow_started.wait()
            return "fast"

        calls = [
            ToolCallRequest(id="slow", name="slow_tool", arguments={}),
            ToolCallRequest(id="fast", name="fast_tool", arguments={}),
        ]
        executor = ToolExecutor(workspace=tmp_path, max_result_chars=10_000)
        results = []
        async with asyncio.timeout(1):
            async for result in executor.execute(calls, {"slow_tool": slow_tool, "fast_tool": fast_tool}):
                results.append(result)
                if result.call.id == "fast":
                    assert not release_slow.is_set()
                    release_slow.set()

        assert [result.call.id for result in results] == ["fast", "slow"]
        assert [result.value for result in results] == ["fast", "slow"]

    @pytest.mark.asyncio
    async def test_cancelling_iteration_cancels_pending_tools(self, tmp_path: Path):
        started = asyncio.Event()
        cancelled = asyncio.Event()
        result_received = asyncio.Event()

        async def stuck_tool() -> str:
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        async def fast_tool() -> str:
            await started.wait()
            return "ready"

        executor = ToolExecutor(workspace=tmp_path, max_result_chars=10_000)

        async def consume():
            async for result in executor.execute(
                [
                    ToolCallRequest(id="stuck", name="stuck_tool", arguments={}),
                    ToolCallRequest(id="fast", name="fast_tool", arguments={}),
                ],
                {"stuck_tool": stuck_tool, "fast_tool": fast_tool},
            ):
                assert result.call.id == "fast"
                result_received.set()

        consumer = asyncio.create_task(consume())
        try:
            await asyncio.wait_for(result_received.wait(), timeout=1)
        finally:
            consumer.cancel()
            with pytest.raises(asyncio.CancelledError):
                await consumer

        assert cancelled.is_set()

    @pytest.mark.asyncio
    async def test_empty_tool_calls_end_iteration(self, tmp_path: Path):
        executor = ToolExecutor(workspace=tmp_path, max_result_chars=10_000)
        assert [result async for result in executor.execute([], {})] == []

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
        results = [
            result
            async for result in ToolExecutor(
                workspace=tmp_path,
                max_result_chars=10_000,
                max_tool_timeout=0.02,
            ).execute(calls, {"stuck_tool": stuck_tool, "fast_tool": fast_tool})
        ]

        assert cancelled.is_set()
        assert [result.call.id for result in results] == ["2", "1"]
        assert results[0].succeeded is True
        assert results[0].value == "ready"
        assert results[1].succeeded is False
        assert results[1].value == "Tool call timed out after 0.02s"

    @pytest.mark.asyncio
    async def test_tool_raised_timeout_keeps_its_own_error(self, tmp_path: Path):
        async def network_tool() -> str:
            raise TimeoutError("read timed out")

        [result] = [
            result
            async for result in ToolExecutor(
                workspace=tmp_path,
                max_result_chars=10_000,
                max_tool_timeout=300,
            ).execute(
                [ToolCallRequest(id="1", name="network_tool", arguments={})],
                {"network_tool": network_tool},
            )
        ]

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
        results = [
            result
            async for result in ToolExecutor(workspace=tmp_path, max_result_chars=10_000).execute(
                calls,
                {"echo": echo, "context": context_only, "plain": plain},
                session_metadata=metadata,
            )
        ]
        assert all(result.succeeded for result in results)
        assert [result.value for result in results] == ["echo", "context", "plain"]

    @pytest.mark.asyncio
    async def test_hides_and_injects_all_runtime_context(self, tmp_path: Path):
        async def context(
            args: EchoArgs,
            session_metadata,
            user_id,
            sender_name,
            channel,
        ) -> str:
            """Return all runtime context."""
            return f"{args.text}:{user_id}:{sender_name}:{channel}:{session_metadata['source']}"

        schema = as_tool(context)["function"]["parameters"]
        assert set(schema["properties"]) == {"text"}
        [result] = [
            result
            async for result in ToolExecutor(workspace=tmp_path, max_result_chars=10_000).execute(
                [ToolCallRequest(id="1", name="context", arguments={"text": "ok"})],
                {"context": context},
                session_metadata={"source": "test"},
                user_id="session-1",
                sender_name="Alice",
                channel="feishu",
            )
        ]
        assert result.value == "ok:session-1:Alice:feishu:test"

    @pytest.mark.asyncio
    async def test_injects_channel_only_when_declared(self, tmp_path: Path):
        async def with_channel(args: EchoArgs, channel: str | None) -> str:
            """Echo with channel context."""
            assert channel == "feishu"
            return args.text

        async def without_channel(args: EchoArgs) -> str:
            return args.text

        schema = as_tool(with_channel)["function"]["parameters"]
        assert set(schema["properties"]) == {"text"}
        results = [
            result
            async for result in ToolExecutor(workspace=tmp_path, max_result_chars=10_000).execute(
                [
                    ToolCallRequest(id="1", name="with_channel", arguments={"text": "a"}),
                    ToolCallRequest(id="2", name="without_channel", arguments={"text": "b"}),
                ],
                {"with_channel": with_channel, "without_channel": without_channel},
                channel="feishu",
            )
        ]
        assert all(result.succeeded for result in results)
        assert [result.value for result in results] == ["a", "b"]

    @pytest.mark.asyncio
    async def test_returns_failure_for_unregistered_tool(self, tmp_path: Path):
        call = ToolCallRequest(id="1", name="missing", arguments={})

        [result] = [
            result async for result in ToolExecutor(workspace=tmp_path, max_result_chars=10_000).execute([call], {})
        ]

        assert not result.succeeded
        assert result.value == "ValueError: Tool is not registered for this request: missing"

    @pytest.mark.asyncio
    async def test_saves_oversized_result_and_returns_prefix_with_path(self, tmp_path: Path):
        full_result = "abcdefghij" * 4

        async def large_result() -> str:
            return full_result

        call = ToolCallRequest(id="call/1", name="large_tool", arguments={})
        [result] = [
            result
            async for result in ToolExecutor(workspace=tmp_path, max_result_chars=12).execute(
                [call],
                {"large_tool": large_result},
            )
        ]

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
        [result] = [
            result
            async for result in ToolExecutor(workspace=tmp_path, max_result_chars=12).execute(
                [call],
                {"read_file": read_file},
            )
        ]

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
        [result] = [
            result
            async for result in ToolExecutor(workspace=tmp_path, max_result_chars=8).execute(
                [call],
                {"multimodal": multimodal_result},
            )
        ]

        assert result.succeeded
        assert result.value == full_result
        assert not (tmp_path / "tool-results").exists()

    @pytest.mark.asyncio
    async def test_does_not_limit_tool_error(self, tmp_path: Path):
        async def failing_tool() -> str:
            raise RuntimeError("x" * 40)

        call = ToolCallRequest(id="1", name="failure", arguments={})
        [result] = [
            result
            async for result in ToolExecutor(workspace=tmp_path, max_result_chars=16).execute(
                [call],
                {"failure": failing_tool},
            )
        ]

        assert not result.succeeded
        assert result.value == "RuntimeError: " + "x" * 40
        assert not (tmp_path / "tool-results").exists()

    @staticmethod
    def _saved_result_path(workspace: Path, limited_result: str) -> Path:
        marker = "Full result saved to: "
        assert marker in limited_result
        return workspace / limited_result.rsplit(marker, maxsplit=1)[1]
