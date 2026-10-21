"""Opt-in acceptance test against a real OpenSandbox Server and runtime image.

The fake model and MCP server are real HTTP services; neither Sandbox.create
nor any agent/provider/tool implementation is mocked. All credentials are
test-specific, and the model endpoint never calls an external model.
"""

import asyncio
import json
import os
import socket
import tempfile
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, UTC
from pathlib import Path
from uuid import uuid4
from urllib.parse import urlsplit

import httpx
import pytest
import uvicorn
from fastapi import Depends, FastAPI, Request
from fastapi.responses import StreamingResponse
from httpx_sse import aconnect_sse
from mcp.server.fastmcp import FastMCP
from opensandbox import Sandbox, SandboxManager
from opensandbox.config import ConnectionConfig
from opensandbox.exceptions import SandboxApiException
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from xpeech.agent.background import start_background_scheduler, stop_background_scheduler
from xpeech.agent import background
from xpeech.agent.delegation import ACTIVE_USER_RUNS
from xpeech.agent.record import (
    ConversationRecord,
    SqliteConversationRecordRepository,
    create_db_and_tables,
)
from xpeech.agent.server.auth import require_jwt
from xpeech.agent.server.routes import chat, files, public_file, runtime as callbacks
from xpeech.config.settings import JWTConfig, LLMConfig, MCPServerSettings, settings
from xpeech.utils.jwt_auth import create_access_token


pytestmark = pytest.mark.skipif(
    not os.environ.get("XPEECH_TEST_SANDBOX_DOMAIN"),
    reason="Set XPEECH_TEST_SANDBOX_DOMAIN to a real OpenSandbox server",
)


class ChannelAuth(httpx.Auth):
    def auth_flow(self, request):
        # Like real Channels, mint a short-lived JWT for each new request.
        request.headers["Authorization"] = "Bearer " + create_access_token()
        yield request


@asynccontextmanager
async def serve_http(app):
    sock = socket.socket()
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning", lifespan="off"))
    task = asyncio.create_task(server.serve(sockets=[sock]))
    try:
        async with asyncio.timeout(10):
            while not server.started:
                if task.done():
                    await task
                await asyncio.sleep(0.02)
        # On bridge-network Docker, use a host address reachable by containers.
        host = os.environ.get("XPEECH_TEST_CALLBACK_HOST", "127.0.0.1")
        yield f"http://{host}:{port}", f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        try:
            async with asyncio.timeout(10):
                await task
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            sock.close()


def model_stream(*, name=None, arguments=None, content=None):
    async def chunks():
        delta = {"content": content} if content is not None else {"tool_calls": [{
            "index": 0, "id": "call-" + uuid4().hex, "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)},
        }]}
        base = {"id": "chatcmpl-test", "object": "chat.completion.chunk", "created": 1, "model": "gpt-4o"}
        yield "data: " + json.dumps({**base, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}) + "\n\n"
        yield "data: " + json.dumps({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop" if content else "tool_calls"}],
                                     "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}) + "\n\n"
        yield "data: [DONE]\n\n"
    return StreamingResponse(chunks(), media_type="text/event-stream")


@pytest.mark.asyncio
async def test_complete_loop_in_real_container(monkeypatch):
    root = Path(os.environ["XPEECH_TEST_DATA_ROOT"]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="acceptance-", dir=root) as directory:
        data = Path(directory)
        user_id = "acceptance-" + uuid4().hex[:12]
        paths = settings.path.model_copy(update={
            "session_path": data / "session", "session_history_path": data / "session/history",
            "session_record_path": data / "session/records.db", "session_schedule_path": data / "session/schedule.db",
            "workspace_base_path": data / "workspace_base", "sandbox_home_path": data / "sandbox-home",
            "cache_path": data / "cache", "log_path": data / "logs",
        })
        monkeypatch.setattr(settings, "path", paths)
        monkeypatch.setattr(settings, "jwt", JWTConfig(secret_key="acceptance-only-jwt-key-not-a-real-secret"))
        for path in (data / "session", paths.workspace_base_path, paths.cache_path, paths.sandbox_home_path):
            path.mkdir(parents=True, exist_ok=True)
        other_workspace = paths.workspace_base_path / "another-user"
        other_workspace.mkdir()
        (other_workspace / "private.txt").write_text("must-not-be-visible-in-runtime")

        engine = create_async_engine(f"sqlite+aiosqlite:///{paths.session_record_path}")
        await create_db_and_tables(engine)
        repository = SqliteConversationRecordRepository(engine)
        monkeypatch.setattr(callbacks, "SqliteConversationRecordRepository", lambda: repository)
        start_background_scheduler(paths.session_schedule_path)
        created = []
        real_create = Sandbox.create

        async def track_create(*args, **kwargs):
            sandbox = await real_create(*args, **kwargs)
            created.append(sandbox.id)
            destroy = sandbox.destroy

            async def capture_and_destroy():
                try:
                    # The gateway correctly deletes failed containers too;
                    # capture diagnostics before that removes their logs.
                    logs = await sandbox.get_diagnostic_logs("all")
                    print(f"Runtime diagnostics {sandbox.id}:\n{logs.content}", flush=True)
                except Exception as exc:
                    print(f"Runtime diagnostics unavailable: {exc}", flush=True)
                finally:
                    await destroy()

            sandbox.destroy = capture_and_destroy
            try:
                # Probe the container's mounts independently of the shell
                # tool's additional workspace-path guard.
                probe = await sandbox.commands.run(
                    f"if test -e '{other_workspace}/private.txt'; then printf isolation-failed; else printf isolation-ok; fi"
                )
                assert "isolation-ok" in "".join(item.text for item in probe.logs.stdout)
            except BaseException:
                await sandbox.destroy()
                raise
            return sandbox

        monkeypatch.setattr(Sandbox, "create", track_create)

        app = FastAPI()
        app.include_router(chat.router, dependencies=[Depends(require_jwt)])
        app.include_router(files.router, dependencies=[Depends(require_jwt)])
        app.include_router(public_file.router)
        app.include_router(callbacks.router)
        mcp = FastMCP("acceptance", stateless_http=True, json_response=True, streamable_http_path="/")

        @mcp.tool()
        def add(left: int, right: int) -> int:
            return left + right

        app.mount("/mcp", mcp.streamable_http_app())
        calls = []
        workspace = paths.workspace_base_path / user_id

        @app.post("/llm/v1/chat/completions")
        async def model(request: Request):
            body = await request.json()
            messages = body["messages"]
            question = next(item for item in reversed(messages) if item["role"] == "user")
            followup = "verify-persistence" in str(question.get("content"))
            tool_results = [item for item in messages if item["role"] == "tool"]
            if "acceptance scheduled prompt" in str(question.get("content")):
                return model_stream(content="Background execution passed")
            if "verify-cancellation" in str(question.get("content")):
                return model_stream(name="shell", arguments={"command": "printf running > cancellation.txt; sleep 120"})
            if followup:
                assert any(item["role"] == "assistant" and "Complete runtime acceptance passed" in str(item.get("content"))
                           for item in messages), "The previous container's conversation history was not restored"
                if messages[-1]["role"] != "tool":
                    return model_stream(name="read_file", arguments={"path": "result.txt"})
                assert "43" in str(messages[-1])
                return model_stream(content="Persistence verified: 43")

            step = len(calls)
            calls.append(body)
            attachment = next(workspace.glob("attachments/*/input.txt"))
            plan = [
                ("read_file", {"path": str(attachment.relative_to(workspace))}),
                ("write_file", {"path": "result.txt", "content": "42"}),
                ("shell", {"command": "printf 43 > result.txt"}),
                ("read_file", {"path": "result.txt"}),
                ("echo", {"content": "custom-tool-ok"}),
                ("mcp_acceptance_add", {"left": 20, "right": 23}),
                ("ask_user_question", {"question": json.dumps({"fields": [{"type": "input", "name": "confirm", "label": "Confirm"}]})}),
                ("write_file", {"path": "index.html", "content": "<h1>Acceptance preview</h1>"}),
                ("create_browser_preview", {"path": "index.html"}),
                ("send_file", {"source": "result.txt"}),
                ("feishu_schedule", {"prompt": "acceptance scheduled prompt", "run_at": (datetime.now(UTC) + timedelta(days=1)).isoformat()}),
            ]
            if step < len(plan):
                name, arguments = plan[step]
                return model_stream(name=name, arguments=arguments)
            assert len(tool_results) >= len(plan)
            assert "attachment-ok" in str(tool_results[0])
            assert "Exit code: 0" in str(tool_results[2])
            assert "43" in str(tool_results[3])
            assert "custom-tool-ok" in str(tool_results[4])
            assert "43" in str(tool_results[5])
            assert tool_results[6]["content"] == "accepted"
            assert "job_id" in str(tool_results[-1])
            return model_stream(content="Complete runtime acceptance passed")

        sandbox_config = settings.sandbox.model_copy(update={
            "domain": os.environ["XPEECH_TEST_SANDBOX_DOMAIN"],
            "api_key": os.environ.get("XPEECH_TEST_SANDBOX_API_KEY"),
            "image": os.environ.get("XPEECH_TEST_RUNTIME_IMAGE", "xpeech-runtime:local"),
            "host_data_path": data, "ready_timeout_seconds": 120,
            "run_timeout_seconds": 240, "ttl_seconds": 600,
        })
        monkeypatch.setattr(settings, "sandbox", sandbox_config)
        try:
            async with mcp.session_manager.run(), serve_http(app) as (external, local):
                settings.sandbox.callback_url = external + "/internal/runtime"
                tool_config = settings.tool.model_copy(deep=True)
                tool_config.browser_preview.browser_preview_path = data / "browser_preview"
                tool_config.browser_preview.browser_preview_base_url = external + "/browser_preview"
                tool_config.mcp_servers = {"acceptance": MCPServerSettings(url=external + "/mcp/")}
                monkeypatch.setattr(settings, "tool", tool_config)
                llm_values = settings.llm.model_dump()
                llm_values.update(api_key="acceptance-only", api_base=external + "/llm/v1",
                                  default_model="openai/gpt-4o", tools_python_package="custom_tools",
                                  # Built-ins/MCP are registered by Runtime;
                                  # this setting names extra custom callables.
                                  default_tools=["echo"],
                                  support_image=False, support_video=False, max_iterations=20)
                monkeypatch.setattr(settings, "llm", LLMConfig.model_validate(llm_values))
                headers = {"x-user-id": user_id, "x-sender-name": "Acceptance"}
                async with httpx.AsyncClient(base_url=local, auth=ChannelAuth(), timeout=httpx.Timeout(30, read=None), trust_env=False) as client:
                    events = []
                    async with asyncio.timeout(360):
                        async with aconnect_sse(client, "POST", "/chat", headers=headers,
                                                data={"content": '[{"text":"Run acceptance"}]', "session_metadata": '{"channel":"feishu","open_id":"acceptance-open-id"}'},
                                                files={"files": ("input.txt", b"attachment-ok", "text/plain")}) as source:
                            source.response.raise_for_status()
                            async for event in source.aiter_sse():
                                if not event.data:
                                    continue
                                payload = json.loads(event.data)
                                events.append(payload)
                                if payload["event"] == "question":
                                    answer = await client.post("/answer_question", headers=headers, data={"answer": "accepted"})
                                    assert answer.json() == {"message": "Answer received"}
                    assert any(item["event"] == "assistant" and "acceptance passed" in item["context"] for item in events)
                    assert any(item["event"] == "send_file" for item in events)
                    artifact = await client.get(f"/sessions/{user_id}/files", headers=headers, params={"path": "result.txt"})
                    assert artifact.text == "43"
                    assert any((data / "browser_preview").glob("*/*/index.html"))
                    preview_urls = [item[2] for event in events if event["event"] == "tool_call_result"
                                    for item in json.loads(event["context"]) if item[1] == "create_browser_preview"]
                    assert len(preview_urls) == 1
                    preview = await client.get(urlsplit(preview_urls[0]).path)
                    assert preview.status_code == 200
                    assert "Acceptance preview" in preview.text
                    async with aconnect_sse(client, "POST", "/chat", headers=headers,
                                            data={"content": '[{"text":"verify-persistence"}]'}) as source:
                        followup = [json.loads(item.data) async for item in source.aiter_sse() if item.data]
                    assert any(item["event"] == "assistant" and "Persistence verified" in item["context"] for item in followup)

                    job = background.get_background_scheduler().get_jobs()[0]
                    monkeypatch.setattr(background.random, "uniform", lambda *_: 0)
                    await job.func(**job.kwargs)
                    delivered = await client.get("/background_message", headers=headers, params={"channel": "feishu"})
                    assert delivered.json()["content"] == "Background execution passed"

                    async with aconnect_sse(client, "POST", "/chat", headers=headers,
                                            data={"content": '[{"text":"verify-cancellation"}]'}) as source:
                        async for event in source.aiter_sse():
                            if event.data and json.loads(event.data)["event"] == "tool_call":
                                async with asyncio.timeout(15):
                                    while not (workspace / "cancellation.txt").exists():
                                        await asyncio.sleep(0.02)
                                break
                    async with asyncio.timeout(30):
                        while user_id in ACTIVE_USER_RUNS:
                            await asyncio.sleep(0.05)

            async with AsyncSession(engine) as session:
                records = (await session.exec(select(ConversationRecord))).all()
                assert len(records) == 3
                assert records[0].model_call_count == 12
            assert user_id not in ACTIVE_USER_RUNS
            assert len(created) == 4
            connection = ConnectionConfig(domain=settings.sandbox.domain, api_key=settings.sandbox.api_key, use_server_proxy=True)
            manager = await SandboxManager.create(connection_config=connection)
            try:
                for sandbox_id in created:
                    try:
                        info = await manager.get_sandbox_info(sandbox_id)
                    except SandboxApiException as exc:
                        assert exc.status_code == 404
                    else:
                        assert info.status.state in {"Stopping", "Terminated"}
            finally:
                await manager.close()
        finally:
            stop_background_scheduler()
            # Assertions above must detect leaks; this fallback only prevents a
            # failed test from leaving containers behind in the test daemon.
            manager = None
            try:
                async with asyncio.timeout(30):
                    manager = await SandboxManager.create(connection_config=ConnectionConfig(
                        domain=settings.sandbox.domain, api_key=settings.sandbox.api_key,
                        use_server_proxy=True,
                    ))
                    for sandbox_id in created:
                        try:
                            await manager.kill_sandbox(sandbox_id)
                        except SandboxApiException as exc:
                            if exc.status_code != 404:
                                raise
            finally:
                if manager is not None:
                    await manager.close()
                await engine.dispose()
