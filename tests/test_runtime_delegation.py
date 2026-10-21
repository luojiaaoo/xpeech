import asyncio
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID

import httpx
import anyio
import pytest
from fastapi import FastAPI

from xpeech.agent import runner as gateway
from xpeech.agent.delegation import (
    ACTIVE_RUNTIME_LEASES,
    ACTIVE_USER_RUNS,
    new_runtime_lease,
    release_runtime_lease,
)
from xpeech.agent.server.routes import runtime as callbacks
from xpeech.agent.server.schema import InboundMessage, InputText
from xpeech.config.settings import settings


def message(user_id="runtime-test"):
    return InboundMessage(
        user_id=user_id, sender_name="Alice",
        session_metadata={"channel": "feishu", "open_id": "ou-alice"},
        content=[InputText(text="Make a file and check it")],
        timestamp=datetime.now(), files=[],
    )


@pytest.fixture
def config(tmp_path):
    config = settings.model_copy(update={
        "path": settings.path.model_copy(deep=True),
        "tool": settings.tool.model_copy(deep=True),
        "sandbox": settings.sandbox.model_copy(deep=True),
    })
    root = tmp_path / "data"
    for name, value in {
        "workspace_base_path": root / "workspace_base",
        "session_history_path": root / "session/history",
        "cache_path": root / "cache",
        "sandbox_home_path": root / "sandbox-home",
    }.items():
        setattr(config.path, name, value)
    config.tool.browser_preview.browser_preview_path = root / "browser_preview"
    config.sandbox.host_data_path = root
    return config


def test_gateway_import_does_not_load_agent_runtime():
    result = subprocess.run(
        [sys.executable, "-c", "import sys; import xpeech.agent.server.api; "
         "assert 'xpeech.agent.loop' not in sys.modules; "
         "assert 'xpeech.provider.litellm_provider' not in sys.modules; "
         "assert 'xpeech.agent.tools.registry' not in sys.modules"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.asyncio
async def test_runtime_bootstraps_from_scoped_environment_without_backend_config(config, tmp_path):
    runner = await gateway.AgentRunner.create("runtime-test", config=config)
    lease = new_runtime_lease("runtime-test")
    try:
        volumes, env = runner._mounts_and_env(lease)
        # Import in a fresh process without conf.toml/.env: verifies the actual
        # container bootstrap contract rather than reusing test globals.
        script = (
            "from xpeech.config.settings import settings; "
            "import xpeech_runtime.service; "
            "assert settings.llm.api_key; "
            "assert settings.path.session_history_path.name == 'runtime-test'; "
            "assert settings.feishu.app_secret == 'unused'; "
            "print('bootstrap-ok')"
        )
        result = await asyncio.to_thread(
            subprocess.run, [sys.executable, "-c", script],
            cwd=tmp_path, env={**os.environ, **env},
            text=True, capture_output=True, timeout=30,
        )
        assert result.returncode == 0, result.stderr
        assert "bootstrap-ok" in result.stdout
    finally:
        release_runtime_lease(lease)


@pytest.mark.asyncio
async def test_gateway_provisions_full_runtime_and_forwards_events(config, monkeypatch):
    runner = await gateway.AgentRunner.create("runtime-test", config=config)
    destroyed = AsyncMock()
    captured = {}

    async def create(image, **kwargs):
        captured.update(image=image, **kwargs)
        endpoint = SimpleNamespace(
            endpoint="runtime.test", build_request_headers=lambda config: {"routing": "test"}
        )
        return SimpleNamespace(
            get_endpoint=AsyncMock(return_value=endpoint),
            connection_config=SimpleNamespace(), destroy=destroyed,
        )

    def serve(request):
        assert request.headers["routing"] == "test"
        body = json.loads(request.content)
        assert body["message"]["user_id"] == "runtime-test"
        assert body["background"] is True
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=(
            'data: {"event":"assistant","context":"done"}\n\n'
            'data: {"event":"assistant_end","context":""}\n\n'
            'event: runtime_done\ndata: {}\n\n'
        ))

    original_client = httpx.AsyncClient
    monkeypatch.setattr(gateway.Sandbox, "create", create)
    monkeypatch.setattr(gateway.httpx, "AsyncClient", lambda **kwargs: original_client(
        transport=httpx.MockTransport(serve), **kwargs
    ))
    assert await runner.run_once(message(), background=True) == "done"
    assert captured["entrypoint"][2] == "xpeech_runtime"
    assert json.loads(captured["env"]["XPEECH_LLM"])["api_key"]
    assert settings.jwt.secret_key not in json.dumps(captured["env"])
    assert settings.feishu.app_secret not in json.dumps(captured["env"])
    assert "mcpServers" in json.loads(captured["env"]["XPEECH_TOOL"])
    preview_path = Path(json.loads(captured["env"]["XPEECH_TOOL"])["browser_preview"]["browser_preview_path"])
    assert preview_path.name == str(UUID(preview_path.name))
    assert all(volume.host.path.startswith(str(config.sandbox.host_data_path)) for volume in captured["volumes"])
    assert next(volume for volume in captured["volumes"] if volume.name == "home-config").read_only
    destroyed.assert_awaited_once()
    assert "runtime-test" not in ACTIVE_USER_RUNS
    assert not ACTIVE_RUNTIME_LEASES


@pytest.mark.asyncio
async def test_cancellation_destroys_container_and_revokes_callbacks(config, monkeypatch):
    runner = await gateway.AgentRunner.create("runtime-test", config=config)
    started = asyncio.Event()
    destroyed = AsyncMock()

    async def start(lease):
        lease.sandbox = SimpleNamespace(destroy=destroyed)
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(runner, "_start", start)

    async def consume():
        return [event async for event in runner.run(message())]

    task = asyncio.create_task(consume())
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    destroyed.assert_awaited_once()
    assert not ACTIVE_RUNTIME_LEASES
    assert not ACTIVE_USER_RUNS


@pytest.mark.asyncio
async def test_stream_disconnect_cancellation_does_not_interrupt_deletion(config, monkeypatch):
    runner = await gateway.AgentRunner.create("runtime-test", config=config)
    started = asyncio.Event()
    deleted = asyncio.Event()

    async def destroy():
        await asyncio.sleep(0.02)
        deleted.set()

    async def start(lease):
        lease.sandbox = SimpleNamespace(destroy=destroy)
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(runner, "_start", start)

    async def consume():
        return [event async for event in runner.run(message())]

    # Match Starlette/FastAPI's cancellation scope, not just task.cancel().
    async with anyio.create_task_group() as group:
        group.start_soon(consume)
        await started.wait()
        group.cancel_scope.cancel()
    assert deleted.is_set()
    assert not ACTIVE_RUNTIME_LEASES
    assert not ACTIVE_USER_RUNS


@pytest.mark.asyncio
async def test_callbacks_bind_owner_and_are_idempotent(monkeypatch):
    app = FastAPI()
    app.include_router(callbacks.router)
    append = AsyncMock()
    monkeypatch.setattr(callbacks, "SqliteConversationRecordRepository", lambda: SimpleNamespace(append=append))
    lease = new_runtime_lease("runtime-test")
    lease.message = message()
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gateway") as client:
            payload = {
                "action": "record", "call_id": "record-1", "payload": {
                    "user_id": "victim", "sender_name": "forged", "user_question": "q",
                    "model_response": "a", "input_tokens": 1, "output_tokens": 2,
                    "model_call_count": 3, "duration_s": 1,
                },
            }
            url = "/internal/runtime/callback"
            assert (await client.post(url, json=payload)).status_code == 401
            headers = {"X-Xpeech-Runtime-Token": lease.token}
            assert (await client.post(url, headers=headers, json=payload)).status_code == 200
            assert (await client.post(url, headers=headers, json=payload)).status_code == 200
            append.assert_awaited_once()
            record = append.await_args.args[0]
            assert (record.user_id, record.sender_name) == ("runtime-test", "Alice")
            release_runtime_lease(lease)
            assert (await client.post(url, headers=headers, json=payload)).status_code == 401
    finally:
        release_runtime_lease(lease)


@pytest.mark.asyncio
async def test_runtime_schedule_tools_cross_http_boundary_and_keep_gateway_owner(tmp_path, monkeypatch):
    from xpeech.agent import background
    from xpeech.agent.tools import schedule

    app = FastAPI()
    app.include_router(callbacks.router)
    lease = new_runtime_lease("runtime-test")
    lease.message = message()
    background.start_background_scheduler(tmp_path / "schedule.db")
    original_client = httpx.AsyncClient
    monkeypatch.setenv("XPEECH_RUNTIME_TOKEN", lease.token)
    monkeypatch.setenv("XPEECH_RUNTIME_CALLBACK_URL", "http://gateway/internal/runtime")
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: original_client(
        transport=httpx.ASGITransport(app=app), **kwargs,
    ))
    try:
        created = json.loads(await schedule.feishu_schedule(
            schedule.FeishuScheduleArgs(prompt="report", run_at="2099-01-01T10:00:00+08:00"),
            session_metadata=lease.message.session_metadata,
            sender_name="forged-name", user_id="forged-user",
        ))
        listed = json.loads(await schedule.feishu_schedule_list(user_id="forged-user"))
        assert listed[0]["job_id"] == created["job_id"]
        assert background.list_scheduled_tasks("forged-user") == []
        job = background.get_background_scheduler().get_job(created["job_id"])
        assert job.kwargs["user_id"] == "runtime-test"
        assert job.kwargs["sender_name"] == "Alice"
        await schedule.feishu_schedule_cancel(
            schedule.FeishuScheduleCancelArgs(job_id=created["job_id"]), user_id="forged-user",
        )
        assert background.list_scheduled_tasks("runtime-test") == []
    finally:
        background.stop_background_scheduler()
        release_runtime_lease(lease)


@pytest.mark.asyncio
async def test_clear_does_not_reimport_legacy_history(config):
    runner = await gateway.AgentRunner.create("runtime-test", config=config)
    config.path.session_history_path.mkdir(parents=True, exist_ok=True)
    legacy = config.path.session_history_path / "runtime-test.yaml"
    legacy.write_text("- role: user\n  content: old\n")
    lease = new_runtime_lease("runtime-test")
    try:
        runner._mounts_and_env(lease)
        migrated = config.path.session_history_path / "runtime-test/runtime-test.yaml"
        assert migrated.read_text() == legacy.read_text()
        migrated.unlink()
        runner._mounts_and_env(lease)
        assert not migrated.exists()
    finally:
        release_runtime_lease(lease)


@pytest.mark.asyncio
async def test_runtime_service_executes_multiple_model_and_tool_iterations(tmp_path, monkeypatch):
    from xpeech_runtime import service
    from xpeech_runtime import callbacks as runtime_callbacks
    from xpeech.agent import runtime_runner, loop as loop_module
    from xpeech.provider.schema import LLMResponse, ToolCallChunk

    monkeypatch.setenv("XPEECH_RUNTIME_TOKEN", "test-token")
    monkeypatch.setenv("XPEECH_RUNTIME_USER_ID", "runtime-test")
    monkeypatch.setenv("XPEECH_RUNTIME_SHELL", "container")
    monkeypatch.setattr(settings.path, "workspace_base_path", tmp_path / "workspaces")
    monkeypatch.setattr(settings.path, "session_history_path", tmp_path / "history")
    monkeypatch.setattr(settings.path, "cache_path", tmp_path / "cache")
    monkeypatch.setattr(settings.tool, "mcp_servers", {})
    tool_names = ("write_file", "shell", "read_file", "ask_user_question")
    monkeypatch.setattr(settings.llm, "default_tools", list(tool_names))
    monkeypatch.setattr(loop_module, "token_counter", AsyncMock(return_value=10))
    sent_records = AsyncMock(return_value=None)
    monkeypatch.setattr(runtime_callbacks, "gateway_call", sent_records)
    seen = []

    async def chat(self, *, messages, **kwargs):
        step = len(seen)
        seen.append(messages.copy())

        async def chunks():
            if step == 0:
                yield "tool_calls", ToolCallChunk(
                    index=0, id="write-1", name="write_file",
                    arguments=json.dumps({"path": "result.txt", "content": "42"}),
                )
            elif step == 1:
                yield "tool_calls", ToolCallChunk(
                    index=0, id="shell-1", name="shell",
                    arguments=json.dumps({"command": "printf '43' > result.txt"}),
                )
            elif step == 2:
                yield "tool_calls", ToolCallChunk(
                    index=0, id="read-1", name="read_file",
                    arguments=json.dumps({"path": "result.txt"}),
                )
            elif step == 3:
                yield "tool_calls", ToolCallChunk(
                    index=0, id="question-1", name="ask_user_question",
                    arguments=json.dumps({"question": json.dumps({
                        "fields": [{"type": "input", "name": "confirmation", "label": "Confirm"}],
                    })}),
                )
            else:
                assert any(item.get("role") == "tool" and "43" in str(item) for item in messages)
                assert any(item.get("role") == "tool" and item.get("content") == "confirmed" for item in messages)
                yield "content", "The file contains 43; confirmed"

        return LLMResponse(iter_mix_chunks=chunks(), mapping_tool_call_funcs={
            name: self.default_tools[name].func for name in tool_names
        })

    monkeypatch.setattr(runtime_runner.LiteLLMProvider, "chat", chat)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=service.app), base_url="http://runtime") as client:
        assert (await client.get("/health")).status_code == 401
        headers = {"X-Xpeech-Runtime-Token": "test-token"}

        async def answer_question():
            async with asyncio.timeout(15):
                while "runtime-test" not in loop_module.AgentLoop.SESSION_QUESTION_EVENT:
                    await asyncio.sleep(0.01)
                response = await client.post("/answer", headers=headers, json={"answer": "confirmed"})
                assert response.json() == {"received": True}

        answer_task = asyncio.create_task(answer_question())
        try:
            response = await client.post("/run", headers=headers, json={
                "message": message().model_dump(mode="json"),
            })
            await answer_task
        finally:
            if not answer_task.done():
                answer_task.cancel()
    assert response.status_code == 200
    assert "runtime_error" not in response.text
    assert "runtime_done" in response.text
    assert "The file contains 43; confirmed" in response.text
    assert len(seen) == 5
    assert (tmp_path / "workspaces/runtime-test/result.txt").read_text() == "43"
    assert (tmp_path / "history/runtime-test.yaml").exists()
    assert sent_records.await_args.args[0] == "record"
    assert sent_records.await_args.args[1]["model_call_count"] == 5


@pytest.mark.asyncio
async def test_runtime_heartbeat_keeps_loop_alive_and_disconnect_cleans_up(monkeypatch):
    from xpeech_runtime import service

    monkeypatch.setenv("XPEECH_RUNTIME_USER_ID", "runtime-test")
    monkeypatch.setattr(service, "HEARTBEAT_SECONDS", 0.01)
    cancelled = asyncio.Event()
    cleanup = AsyncMock()
    monkeypatch.setattr(service, "close_all_mcp_connections", cleanup)

    class WaitingRunner:
        _agent_loop = SimpleNamespace()

        async def run(self, *_args, **_kwargs):
            try:
                await asyncio.Event().wait()
                yield {}
            finally:
                cancelled.set()

    monkeypatch.setattr(service.AgentRunner, "create", AsyncMock(return_value=WaitingRunner()))
    response = await service.run(service.RunRequest(message=message()))
    stream = response.body_iterator
    async with asyncio.timeout(1):
        assert await anext(stream) == ": keepalive\n\n"
        assert await anext(stream) == ": keepalive\n\n"
    assert not cancelled.is_set()
    assert service._run_lock.locked()
    await stream.aclose()
    assert cancelled.is_set()
    assert not service._run_lock.locked()
    cleanup.assert_awaited_once()


@pytest.mark.asyncio
async def test_shutdown_revokes_capabilities_even_when_one_container_fails():
    from xpeech.agent.server.server import _close_runtime

    leases = [new_runtime_lease(f"shutdown-{index}") for index in range(2)]
    for lease in leases:
        ACTIVE_USER_RUNS[lease.user_id] = lease
        lease.sandbox = SimpleNamespace(destroy=AsyncMock())
    leases[0].sandbox.destroy.side_effect = RuntimeError("control plane unavailable")
    try:
        await asyncio.gather(*(_close_runtime(lease) for lease in leases))
        for lease in leases:
            lease.sandbox.destroy.assert_awaited_once()
            assert lease.token not in ACTIVE_RUNTIME_LEASES
            assert lease.user_id not in ACTIVE_USER_RUNS
    finally:
        for lease in leases:
            release_runtime_lease(lease)
