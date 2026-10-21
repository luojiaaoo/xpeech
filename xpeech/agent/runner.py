"""Gateway facade: delegate the entire agent loop to an OpenSandbox runtime."""

import asyncio
import json
import secrets
import shutil
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Self
from uuid import uuid4
from datetime import timedelta

import httpx
import anyio
from httpx_sse import aconnect_sse
from opensandbox import Sandbox
from opensandbox.config import ConnectionConfig
from opensandbox.models.sandboxes import Host, Volume

from ..config.settings import Settings, settings
from ..utils.helper import ensure_path_async
from ..utils.user_id import validate_user_id
from .delegation import (
    ACTIVE_USER_RUNS,
    RuntimeLease,
    new_runtime_lease,
    release_runtime_lease,
)
from .server.schema import InboundMessage


class AgentRunner:
    """Own one container while it runs all model iterations and all tools."""

    def __init__(self, user_id: str, *, config: Settings) -> None:
        self.user_id = validate_user_id(user_id)
        self.config = config
        self.workspace = (config.path.workspace_base_path / self.user_id).resolve()
        self._lease: RuntimeLease | None = None
        self._initialized = False

    @classmethod
    async def create(cls, user_id: str, *, config: Settings = settings) -> Self:
        runner = cls(user_id, config=config)
        await ensure_path_async(runner.workspace)
        runner._initialized = True
        return runner

    def _mounts_and_env(self, lease: RuntimeLease) -> tuple[list[Volume], dict[str, str]]:
        config = self.config
        paths = config.path
        data_root = paths.workspace_base_path.expanduser().resolve().parent
        host_root = config.sandbox.host_data_path.expanduser().resolve()
        history = paths.session_history_path.expanduser().resolve() / self.user_id
        cache = paths.cache_path.expanduser().resolve() / self.user_id
        # The public route parses this segment as UUID and uses its canonical
        # (hyphenated) spelling to locate the directory on disk.
        preview = config.tool.browser_preview.browser_preview_path.expanduser().resolve() / str(uuid4())
        home_config = paths.sandbox_home_path.expanduser().resolve()
        volumes = []
        for name, local_path, read_only in (
            ("workspace", self.workspace, False),
            ("history", history, False),
            ("cache", cache, False),
            ("preview", preview, False),
            ("home-config", home_config, True),
        ):
            local_path.mkdir(parents=True, exist_ok=True)
            if not local_path.is_relative_to(data_root):
                raise ValueError(f"Sandbox mount {name} must be inside data directory {data_root}")
            volumes.append(Volume(
                name=name,
                host=Host(path=str(host_root / local_path.relative_to(data_root))),
                mount_path=str(local_path),
                read_only=read_only,
            ))
        # Preserve existing histories without letting a sandbox read other users.
        legacy = paths.session_history_path.expanduser().resolve() / f"{self.user_id}.yaml"
        destination = history / legacy.name
        migrated = history / ".legacy-migrated"
        if not migrated.exists():
            if legacy.is_file() and not destination.exists():
                shutil.copy2(legacy, destination)
            migrated.touch()

        runtime_paths = paths.model_dump(mode="json")
        runtime_paths.update(
            workspace_base_path=str(self.workspace.parent),
            session_history_path=str(history),
            session_path=str(self.workspace / ".xpeech"),
            session_record_path=str(self.workspace / ".xpeech" / "unused-records.db"),
            session_schedule_path=str(self.workspace / ".xpeech" / "unused-schedule.db"),
            log_path="/tmp/xpeech-logs",
            sandbox_home_path=str(home_config),
            cache_path=str(cache.parent),
        )
        llm = config.llm.model_dump(mode="json")
        llm["api_key"] = config.llm.api_key
        tool = config.tool.model_dump(mode="json", by_alias=True)
        tool["mcpServers"] = tool.pop("mcp_servers", {})
        tool["browser_preview"]["browser_preview_path"] = str(preview)
        tool["browser_preview"]["browser_preview_base_url"] = (
            config.tool.browser_preview.browser_preview_base_url + "/" + preview.name
        )
        env = {
            **config.sandbox.runtime_env,
            "XPEECH_PATH": json.dumps(runtime_paths),
            "XPEECH_LLM": json.dumps(llm),
            "XPEECH_TOOL": json.dumps(tool),
            "XPEECH_FEISHU": json.dumps({"app_id": "runtime", "app_secret": "unused"}),
            "XPEECH_JWT": json.dumps({"secret_key": secrets.token_urlsafe(32)}),
            "XPEECH_WEB_CLIENT": json.dumps({"database_path": "/tmp/unused-web-client.db"}),
            "XPEECH_RUNTIME_TOKEN": lease.token,
            "XPEECH_RUNTIME_USER_ID": self.user_id,
            "XPEECH_RUNTIME_CALLBACK_URL": config.sandbox.callback_url,
            "XPEECH_RUNTIME_SHELL": "container",
        }
        return volumes, env

    async def _start(self, lease: RuntimeLease) -> None:
        config = self.config.sandbox
        volumes, env = await asyncio.to_thread(self._mounts_and_env, lease)
        connection = ConnectionConfig(
            domain=config.domain,
            protocol=config.protocol,
            api_key=config.api_key,
            use_server_proxy=config.use_server_proxy,
            request_timeout=timedelta(seconds=config.ready_timeout_seconds),
        )

        async def ready(sandbox: Sandbox) -> bool:
            endpoint = await sandbox.get_endpoint(config.runtime_port)
            url = endpoint.endpoint
            if not url.startswith(("https://", "http://")):
                url = f"{config.protocol}://{url}"
            headers = endpoint.build_request_headers(sandbox.connection_config)
            headers["X-Xpeech-Runtime-Token"] = lease.token
            try:
                async with httpx.AsyncClient(timeout=2) as client:
                    response = await client.get(url + "/health", headers=headers)
                return response.status_code == 200
            except httpx.RequestError:
                return False

        lease.sandbox = await Sandbox.create(
            config.image,
            entrypoint=["/app/.venv/bin/python", "-m", "xpeech_runtime", "--port", str(config.runtime_port)],
            connection_config=connection,
            timeout=timedelta(seconds=config.ttl_seconds),
            ready_timeout=timedelta(seconds=config.ready_timeout_seconds),
            env=env,
            volumes=volumes,
            resource=config.resource,
            metadata={"app": "xpeech", "user_id": self.user_id},
            health_check=ready,
        )
        endpoint = await lease.sandbox.get_endpoint(config.runtime_port)
        lease.base_url = endpoint.endpoint.rstrip("/")
        if not lease.base_url.startswith(("http://", "https://")):
            lease.base_url = f"{config.protocol}://{lease.base_url}"
        lease.headers = endpoint.build_request_headers(lease.sandbox.connection_config)
        lease.headers["X-Xpeech-Runtime-Token"] = lease.token

    async def run(self, message: InboundMessage, *, background: bool = False) -> AsyncIterator[dict[str, Any]]:
        if message.user_id != self.user_id:
            raise ValueError("Inbound message user_id does not match the runner user")
        if not self._initialized:
            raise RuntimeError("AgentRunner must be created with AgentRunner.create()")
        if self.user_id in ACTIVE_USER_RUNS:
            raise RuntimeError("An agent run is already active for this user")
        lease = new_runtime_lease(self.user_id)
        lease.message = message
        self._lease = lease
        ACTIVE_USER_RUNS[self.user_id] = lease
        try:
            await self._start(lease)
            async with asyncio.timeout(self.config.sandbox.run_timeout_seconds):
                async with httpx.AsyncClient(timeout=httpx.Timeout(30, read=None)) as client:
                    async with aconnect_sse(
                        client, "POST", lease.base_url + "/run",
                        headers=lease.headers,
                        json={"message": message.model_dump(mode="json"), "background": background},
                    ) as source:
                        source.response.raise_for_status()
                        async for event in source.aiter_sse():
                            if event.event == "runtime_error":
                                raise RuntimeError(f"Agent container failed: {event.data}")
                            if event.event == "runtime_done":
                                return
                            if event.data:
                                yield json.loads(event.data)
                        raise RuntimeError("Agent runtime disconnected before completion")
        finally:
            # Streaming responses use AnyIO level cancellation: after a client
            # disconnect every await can be cancelled again. asyncio.shield
            # alone does not protect the following cleanup await in that scope.
            with anyio.CancelScope(shield=True):
                cleanup = asyncio.create_task(self.aclose())
                cancelled = False
                while not cleanup.done():
                    try:
                        await asyncio.shield(cleanup)
                    except asyncio.CancelledError:
                        cancelled = True
                cleanup.result()
                if cancelled:
                    raise asyncio.CancelledError

    async def aclose(self) -> None:
        lease, self._lease = self._lease, None
        if lease is None:
            return
        try:
            if lease.sandbox is not None:
                async with asyncio.timeout(20):
                    await lease.sandbox.destroy()
        finally:
            release_runtime_lease(lease)

    async def run_once(self, message: InboundMessage, *, background: bool = False) -> str:
        current = None
        final = None
        async for event in self.run(message, background=background):
            if event.get("event") == "assistant":
                if current is None:
                    current = []
                current.append(event.get("context", ""))
            elif event.get("event") == "assistant_end" and current is not None:
                final, current = "".join(current), None
        if not final or not final.strip():
            raise RuntimeError("Agent run completed without model output")
        return final


async def answer_runtime_question(user_id: str, answer: str) -> bool:
    lease = ACTIVE_USER_RUNS.get(user_id)
    if lease is None or not lease.base_url:
        return False
    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.post(lease.base_url + "/answer", headers=lease.headers, json={"answer": answer})
    response.raise_for_status()
    return response.json()["received"]
