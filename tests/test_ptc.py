import asyncio
import json

import pytest
from pydantic import BaseModel, ValidationError

from xpeech.agent.tool_executor import ToolExecutor
from xpeech.agent.tools.execution_request import build_execution_request_tool
from xpeech.agent.tools.helper import as_tool
from xpeech.agent.tools.interactions import ExecutionRequest, InteractionRequest, QuestionRequest, SendFileRequest
from xpeech.agent.tools.ptc import build_ptc_tool, patch
from xpeech.agent.tools.ptc import ptc as ptc_module
from xpeech.agent.tools.ptc.ptc import StarlarkArgs
from xpeech.provider.schema import ToolCallRequest
from xpeech.utils.file_io import LockedFile


class TextArgs(BaseModel):
    text: str


def executor(tmp_path, tools=None, *, max_tool_timeout=300.0, max_result_chars=10_000):
    ptc = build_ptc_tool(workspace=tmp_path)
    return ToolExecutor(
        workspace=tmp_path,
        max_result_chars=max_result_chars,
        mapping_tool_call_funcs={"run_ptc": ptc, **(tools or {})},
        max_tool_timeout=max_tool_timeout,
    )


async def run(ex, code=None, *, path=None, inputs=None, timeout_ms=None, **context):
    arguments = {"inputs": inputs or {}}
    if code is not None:
        arguments["code"] = code
    if path is not None:
        arguments["path"] = path
    if timeout_ms is not None:
        arguments["timeout_ms"] = timeout_ms
    result = await ex.execute_one(ToolCallRequest(id="test", name="run_ptc", arguments=arguments), **context)
    return json.loads(result.value) if result.succeeded else {"ok": False, "error": result.value}


@pytest.mark.asyncio
async def test_chain_transforms_results_and_preserves_context(tmp_path):
    calls = []

    async def echo(args: TextArgs, user_id, session_id, channel, session_metadata, sender_name):
        calls.append((args.text, user_id, session_id, channel, session_metadata, sender_name))
        return json.dumps({"text": args.text})

    ex = executor(tmp_path, {"echo": echo})
    result = await run(
        ex,
        """
def main(inputs):
    first = json.decode(call_tool("echo", {"text": inputs["text"]}))
    second = call_tool("echo", {"text": first["text"].strip().upper()})
    return json.decode(second)
""",
        inputs={"text": " hello "},
        user_id="u",
        session_id="s",
        channel="test",
        session_metadata={"org": "dev"},
        sender_name="Alice",
    )
    assert result["ok"] and result["result"] == {"text": "HELLO"}
    assert [call[0] for call in calls] == [" hello ", "HELLO"]
    assert all(call[1:] == ("u", "s", "test", {"org": "dev"}, "Alice") for call in calls)
    assert [entry["name"] for entry in result["tool_calls"]] == ["echo", "echo"]
    assert set(result) == {"ok", "result", "tool_calls"}


@pytest.mark.parametrize("arguments", [{}, {"code": "pass", "path": "script.star"}, {"code": ""}, {"path": ""}])
def test_exactly_one_nonempty_source_is_required(arguments):
    with pytest.raises(ValidationError):
        StarlarkArgs(**arguments)


@pytest.mark.asyncio
@pytest.mark.parametrize("absolute", [False, True])
async def test_workspace_file_matches_inline_execution_and_preserves_context(tmp_path, absolute):
    script = tmp_path / "scripts" / "normalize.star"
    script.parent.mkdir()
    source = 'def main(inputs):\n    return call_tool("echo", {"text": inputs["text"].strip()})\n'
    script.write_text(source, encoding="utf-8-sig")

    async def echo(args: TextArgs, user_id):
        return args.text + ":" + user_id

    ex = executor(tmp_path, {"echo": echo})
    path = str(script) if absolute else "scripts/normalize.star"
    result = await run(ex, path=path, inputs={"text": " hello "}, user_id="user-a")
    inline = await run(ex, source, inputs={"text": " hello "}, user_id="user-a")
    assert result["ok"] and result["result"] == inline["result"] == "hello:user-a"
    assert result["tool_calls"][0]["name"] == "echo"
    assert set(result) == {"ok", "result", "tool_calls"}
    # 文件内容每次重新读取，保存修改后的源码即可复用，无需维护额外的流程注册表。
    script.write_text('def main(inputs):\n    return "updated"', encoding="utf-8")
    assert (await run(ex, path=path))["result"] == "updated"


@pytest.mark.asyncio
@pytest.mark.parametrize("absolute", [False, True])
async def test_script_file_cannot_escape_workspace(tmp_path, absolute):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.star"
    outside.write_text('def main(inputs):\n    return "outside"', encoding="utf-8")
    path = str(outside) if absolute else "../outside.star"
    result = await run(executor(workspace), path=path)
    assert not result["ok"] and "Path escapes workspace" in result["error"]


@pytest.mark.asyncio
async def test_script_symlink_cannot_escape_workspace(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.star"
    outside.write_text('def main(inputs):\n    return "outside"', encoding="utf-8")
    link = workspace / "linked.star"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("Creating symlinks is not supported in this test environment")
    result = await run(executor(workspace), path="linked.star")
    assert not result["ok"] and "Path escapes workspace" in result["error"]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["missing", "directory", "invalid_utf8"])
async def test_invalid_script_file_is_rejected_before_tool_dispatch(tmp_path, kind):
    effects = []

    async def write():
        effects.append(1)
        return "ok"

    script = tmp_path / "script.star"
    if kind == "directory":
        script.mkdir()
    elif kind == "invalid_utf8":
        script.write_bytes(b"\xff\xfe\xfa")
    result = await run(executor(tmp_path, {"write": write}), path="script.star")
    assert not result["ok"] and not effects


@pytest.mark.asyncio
@pytest.mark.parametrize("from_file", [False, True])
async def test_large_source_runs_without_length_limit(tmp_path, from_file):
    source = "#" + "x" * 100_000 + '\ndef main(inputs):\n    return "complete"\n'
    if from_file:
        (tmp_path / "large.star").write_text(source, encoding="utf-8")
        result = await run(executor(tmp_path), path="large.star")
    else:
        result = await run(executor(tmp_path), source)
    assert result["ok"] and result["result"] == "complete"


@pytest.mark.asyncio
async def test_file_lock_wait_counts_towards_ptc_timeout(tmp_path):
    script = tmp_path / "script.star"
    script.write_text('def main(inputs):\n    return "ok"', encoding="utf-8")
    ex = executor(tmp_path)
    async with LockedFile(script).transaction():
        result = await asyncio.wait_for(run(ex, path="script.star", timeout_ms=50), 2)
    assert not result["ok"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "interaction",
    [InteractionRequest(), ExecutionRequest(), QuestionRequest(question="{}"), SendFileRequest(source="x")],
)
async def test_any_interaction_request_terminates_before_next_effect(tmp_path, interaction):
    effects = []

    async def interactive():
        return interaction

    async def write():
        effects.append("changed")
        return "written"

    result = await run(
        executor(tmp_path, {"interactive": interactive, "write": write}),
        """
def main(inputs):
    call_tool("interactive", {})
    return call_tool("write", {})
""",
    )
    assert not result["ok"]
    assert "InteractionRequest" in result["error"]
    assert effects == []
    assert len(result["tool_calls"]) == 1
    assert not result["tool_calls"][0]["succeeded"]
    assert result["tool_calls"][0]["interaction_type"] == type(interaction).__name__


@pytest.mark.asyncio
async def test_failed_tool_stops_chain(tmp_path):
    effects = []

    async def broken():
        raise ValueError("broken tool")

    async def write():
        effects.append(1)
        return "ok"

    result = await run(
        executor(tmp_path, {"broken": broken, "write": write}),
        """
def main(inputs):
    call_tool("broken", {})
    return call_tool("write", {})
""",
    )
    assert not result["ok"] and "broken tool" in result["error"]
    assert not effects


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "source",
    [
        'load("secret.star", "secret")\ndef main(inputs):\n    return secret',
        "import os\ndef main(inputs):\n    return os.getcwd()",
        'def main(inputs):\n    return open("secret", "w")',
        'def main(inputs):\n    return __import__("os")',
        'def main(inputs):\n    return print("not allowed")',
        "def main(inputs):\n    return call_tool.__class__",
        "def main(inputs):\n    return write()",
    ],
)
async def test_no_ambient_capabilities(tmp_path, source):
    effects = []

    async def write():
        effects.append(1)
        return "ok"

    result = await run(executor(tmp_path, {"write": write}), source)
    assert not result["ok"]
    assert not effects and not list(tmp_path.iterdir())


@pytest.mark.asyncio
async def test_injected_normalize_method_is_callable_without_tools(tmp_path):
    result = await run(
        executor(tmp_path),
        'def main(inputs):\n    return ptc_normalize_lines(inputs["lines"])',
        inputs={"lines": [" banana ", "apple", "banana", " "]},
    )
    assert result["ok"] and result["result"] == ["apple", "banana"]
    assert result["tool_calls"] == []
    assert not list(tmp_path.iterdir())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "name,expected",
    [
        ("run_ptc", "Recursive"),
        ("missing", "not registered"),
    ],
)
async def test_recursive_and_unregistered_tools_fail_before_dispatch(tmp_path, name, expected):
    result = await run(
        executor(tmp_path),
        'def main(inputs):\n    return call_tool(inputs["name"], {})',
        inputs={"name": name},
    )
    assert not result["ok"] and expected in result["error"]
    assert not result["tool_calls"]


@pytest.mark.asyncio
async def test_call_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(ptc_module, "PTC_MAX_TOOL_CALLS", 2)
    effects = []

    async def write():
        effects.append(1)
        return "ok"

    result = await run(
        executor(tmp_path, {"write": write}),
        """
def main(inputs):
    for i in range(5):
        call_tool("write", {})
""",
    )
    assert not result["ok"] and "budget" in result["error"]
    assert len(effects) == 2


@pytest.mark.asyncio
async def test_compute_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(ptc_module, "PTC_MAX_TICKS", 2)
    result = await run(
        executor(tmp_path),
        """
def main(inputs):
    n = 0
    for i in range(1000000):
        n += i
    return n
""",
    )
    assert not result["ok"] and "budget" in result["error"]


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_timeout_or_cancellation_revokes_pending_and_later_effects(tmp_path, cancel):
    started = asyncio.Event()
    finished = asyncio.Event()
    effects = []

    async def slow():
        started.set()
        try:
            await asyncio.sleep(5)
            effects.append("slow")
            return "ok"
        finally:
            finished.set()

    async def write():
        effects.append("write")
        return "ok"

    ex = executor(tmp_path, {"slow": slow, "write": write})
    task = asyncio.create_task(
        run(
            ex,
            """
def main(inputs):
    call_tool("slow", {})
    return call_tool("write", {})
""",
            timeout_ms=300,
        )
    )
    await asyncio.wait_for(started.wait(), 2)
    if cancel:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        result = await asyncio.wait_for(task, 2)
        assert not result["ok"] and "timed out" in result["error"]
        assert result["tool_calls"][0]["status"] == "cancelled"
    await asyncio.wait_for(finished.wait(), 2)
    assert not effects


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [200, 2000, 1_100_000])
async def test_final_report_uses_normal_result_offload(tmp_path, size):
    ex = executor(tmp_path, max_result_chars=1000)
    execution = await ex.execute_one(
        ToolCallRequest(
            id="report",
            name="run_ptc",
            arguments={
                "code": 'def main(inputs):\n    return "x" * inputs["size"]',
                "inputs": {"size": size},
            },
        )
    )
    assert execution.succeeded
    files = list(tmp_path.rglob("*.txt"))
    if size > 1000:
        assert len(files) == 1
        assert "Full result saved to:" in execution.value
        report = json.loads(files[0].read_text(encoding="utf-8"))
    else:
        assert not files
        report = json.loads(execution.value)
    assert report["ok"] and report["result"] == "x" * size


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [2000, 1_100_000])
async def test_intermediate_results_are_processed_in_full_without_automatic_files(tmp_path, size):
    async def large():
        return json.dumps({"text": "x" * size})

    result = await run(
        executor(tmp_path, {"large": large}, max_result_chars=1000),
        """
def main(inputs):
    data = json.decode(call_tool("large", {}))
    return len(data["text"])
""",
    )
    assert result["ok"] and result["result"] == size
    assert not list(tmp_path.iterdir())


@pytest.mark.asyncio
async def test_non_json_tool_objects_are_rejected(tmp_path):
    async def opaque():
        return object()

    result = await run(
        executor(tmp_path, {"opaque": opaque}),
        'def main(inputs):\n    return call_tool("opaque", {})',
    )
    assert not result["ok"] and "JSON serializable" in result["error"]


@pytest.mark.asyncio
async def test_large_inputs_tool_arguments_and_patch_data_are_not_size_limited(tmp_path, monkeypatch):
    size = 1_100_000

    def ptc_copy(value):
        return {"text": value}

    async def echo(args: TextArgs):
        assert len(args.text) == size
        return json.dumps({"text": args.text})

    monkeypatch.setattr(patch, "INJECTED_METHODS", {"ptc_copy": ptc_copy})
    result = await run(
        executor(tmp_path, {"echo": echo}),
        """
def main(inputs):
    copied = ptc_copy(inputs["text"])
    raw = call_tool("echo", {"text": copied["text"]})
    return len(json.decode(raw)["text"])
""",
        inputs={"text": "x" * size},
    )
    assert result["ok"] and result["result"] == size
    assert not list(tmp_path.iterdir())


def test_schema_hides_injected_executor_and_context(tmp_path):
    schema = as_tool(build_ptc_tool(workspace=tmp_path))["function"]["parameters"]
    assert set(schema["properties"]) == {"code", "path", "inputs", "timeout_ms"}
    timeout = schema["properties"]["timeout_ms"]
    assert timeout["default"] == 120_000
    assert timeout["maximum"] == 240_000
    assert timeout["exclusiveMinimum"] == 0
    with pytest.raises(ValidationError):
        StarlarkArgs(code="x", allowed_tools=["echo"])


@pytest.mark.parametrize("timeout_ms", [0, -1, 240_001])
def test_timeout_rejects_values_outside_model_limit(timeout_ms):
    with pytest.raises(ValidationError):
        StarlarkArgs(code="def main(inputs):\n    return 1", timeout_ms=timeout_ms)


@pytest.mark.asyncio
async def test_timeout_accepts_maximum_duration(tmp_path):
    result = await run(executor(tmp_path), "def main(inputs):\n    return 1", timeout_ms=240_000)
    assert result["ok"] and result["result"] == 1


@pytest.mark.asyncio
async def test_requested_timeout_is_capped_by_outer_executor(tmp_path):
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def slow():
        started.set()
        try:
            await asyncio.sleep(5)
            return "unexpected"
        finally:
            cancelled.set()

    ex = executor(tmp_path, {"slow": slow}, max_tool_timeout=0.2)
    result = await asyncio.wait_for(
        run(
            ex,
            'def main(inputs):\n    return call_tool("slow", {})',
            timeout_ms=10_000,
        ),
        2,
    )
    assert started.is_set()
    assert not result["ok"] and "timed out" in result["error"]
    await asyncio.wait_for(cancelled.wait(), 2)


@pytest.mark.asyncio
async def test_patch_methods_get_detached_data_and_support_keyword_arguments(tmp_path, monkeypatch):
    shared = ["original"]

    def ptc_custom(values, suffix):
        """Append a suffix to values."""
        values.append(suffix)
        return {"values": values, "shared": shared}

    monkeypatch.setattr(patch, "INJECTED_METHODS", {"ptc_custom": ptc_custom})
    tool = build_ptc_tool(workspace=tmp_path)
    assert "ptc_custom" in tool.__doc__ and "Append a suffix" in tool.__doc__
    ex = ToolExecutor(workspace=tmp_path, max_result_chars=1000, mapping_tool_call_funcs={"run_ptc": tool})
    result = await run(
        ex,
        """
def main(inputs):
    result = ptc_custom(inputs["values"], suffix="added")
    result["shared"].append("local")
    return {"inputs": inputs, "result": result}
""",
        inputs={"values": ["first"]},
    )
    assert result["ok"]
    assert result["result"]["inputs"]["values"] == ["first"]
    assert result["result"]["result"]["values"] == ["first", "added"]
    assert shared == ["original"]


@pytest.mark.parametrize("name", ["_hidden", "not-valid", "for"])
def test_patch_rejects_invalid_method_names(monkeypatch, name, tmp_path):
    monkeypatch.setattr(patch, "INJECTED_METHODS", {name: lambda: None})
    with pytest.raises(ValueError):
        build_ptc_tool(workspace=tmp_path)


@pytest.mark.asyncio
async def test_prefixed_patch_method_leaves_builtin_available(tmp_path, monkeypatch):
    def ptc_len(value):
        return 42

    monkeypatch.setattr(patch, "INJECTED_METHODS", {"ptc_len": ptc_len})
    result = await run(executor(tmp_path), 'def main(inputs):\n    return [len("text"), ptc_len("text")]')
    assert result["ok"] and result["result"] == [4, 42]


def test_patch_rejects_async_methods(monkeypatch, tmp_path):
    async def ptc_custom():
        return "value"

    monkeypatch.setattr(patch, "INJECTED_METHODS", {"ptc_custom": ptc_custom})
    with pytest.raises(TypeError, match="synchronous"):
        build_ptc_tool(workspace=tmp_path)


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [InteractionRequest(), object()])
async def test_patch_rejects_host_objects_and_interaction_and_stops_chain(tmp_path, monkeypatch, value):
    effects = []

    def ptc_custom():
        return value

    async def write():
        effects.append(1)
        return "ok"

    monkeypatch.setattr(patch, "INJECTED_METHODS", {"ptc_custom": ptc_custom})
    result = await run(
        executor(tmp_path, {"write": write}),
        """
def main(inputs):
    ptc_custom()
    return call_tool("write", {})
""",
    )
    assert not result["ok"]
    assert (
        "InteractionRequest" in result["error"]
        if isinstance(value, InteractionRequest)
        else "JSON serializable" in result["error"]
    )
    assert not effects


@pytest.mark.asyncio
async def test_confirmation_wrapper_cannot_be_bypassed(tmp_path):
    effects = []

    async def dangerous():
        """A tool requiring confirmation."""
        effects.append("executed")
        return "ok"

    wrapped = build_execution_request_tool(dangerous)
    result = await run(
        executor(tmp_path, {"dangerous": wrapped}),
        """
def main(inputs):
    return call_tool("dangerous", {"execution_description": "test confirmation"})
""",
    )
    assert not result["ok"] and "InteractionRequest" in result["error"]
    assert not effects


@pytest.mark.asyncio
async def test_provider_registration_and_tool_wrapper(tmp_path):
    from xpeech.provider.litellm_provider import LiteLLMProvider
    from xpeech.provider.schema import LLMParameters

    provider = LiteLLMProvider(
        api_key="test", api_base="https://example.test", default_model="test", parameters=LLMParameters()
    )

    async def echo(args: TextArgs):
        """Echo text."""
        return args.text

    provider.register_function_tool(echo)
    provider.register_function_tool(build_ptc_tool(workspace=tmp_path))
    ex = ToolExecutor(
        workspace=tmp_path, max_result_chars=1_000, mapping_tool_call_funcs=provider.mapping_tool_call_funcs
    )
    result = await run(ex, 'def main(inputs):\n    return call_tool("echo", {"text": "hello"})')
    assert result["ok"] and result["result"] == "hello"


@pytest.mark.asyncio
async def test_tool_names_do_not_shadow_starlark_functions(tmp_path):
    async def length_tool():
        return "tool result"

    result = await run(
        executor(tmp_path, {"len": length_tool}),
        """
def main(inputs):
    return [len("text"), call_tool("len", {})]
""",
    )
    assert result["ok"] and result["result"] == [4, "tool result"]
