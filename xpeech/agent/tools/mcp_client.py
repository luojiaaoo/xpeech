"""MCP client integration for xpeech tools.

Default MCP servers are configured in ``conf.toml``:

    [tool.mcpServers.filesystem]
    command = "npx"
    args = ["-y", "@modelcontextprotocol/server-filesystem", "."]
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import platform
import re
import shutil
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from contextlib import AsyncExitStack, suppress
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from fnmatch import fnmatchcase
from functools import partial
from pathlib import Path
from typing import Any, Literal

import httpx
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, create_model

from ...config.settings import MCPServerSettings, settings
from ...utils.command_template import execute_command_template
from ...utils.filter_expr import parse_filter
from ...utils.security.sandbox import wrap_sandbox_command
from ...utils.security.sandbox.isolation import SANDBOX_EXECUTABLES
from ..skills.skill import SkillsLoader
from .execution_request import ExecutionDecision, build_execution_request_tool, execution_request_schema

_TRANSIENT_EXC_NAMES: frozenset[str] = frozenset(
    (
        "ClosedResourceError",
        "BrokenResourceError",
        "EndOfStream",
        "BrokenPipeError",
        "ConnectionResetError",
        "ConnectionRefusedError",
        "ConnectionAbortedError",
        "ConnectionError",
    )
)
_SANITIZE_RE = re.compile(r"_+")
_MAX_TOOL_NAME_LENGTH = 64
_HASH_LENGTH = 8
MCPTransport = Literal["stdio", "sse", "streamable-http"]
MCP_CONNECTION_TTL_SECONDS = 20 * 60
MCP_CONNECT_TIMEOUT_SECONDS = 30
MCP_AUTHORIZATION_TOKEN_TIMEOUT_SECONDS = 10
MCPConnectionScope = tuple[str | None, Path]
# None identifies shared connections; user-scoped connections also include their workspace.
MCP_CONNECTIONS: dict[MCPConnectionScope, dict[tuple[Any, ...], MCPServerRegistration]] = {}

_SESSION_EXPIRY_HANDLES: dict[MCPConnectionScope, asyncio.TimerHandle] = {}
_MCP_CONNECTIONS_LOCK: asyncio.Lock | None = None

_USER_ID_PLACEHOLDER_RE = re.compile(r"\$(?:\{user_id\}|user_id(?![a-zA-Z0-9_]))")
_AUTHORIZATION_TOKEN_PLACEHOLDER_RE = re.compile(
    r"\$(?:\{authorization_token\}|authorization_token(?![a-zA-Z0-9_]))"
)


class _MCPRegistrationRetiredError(RuntimeError):
    pass


def _sanitize_name(name: str) -> str:
    return _SANITIZE_RE.sub("_", re.sub(r"[^a-zA-Z0-9_-]", "_", name)).strip("_")


def _limit_tool_name(name: str, max_length: int = _MAX_TOOL_NAME_LENGTH) -> str:
    if len(name) <= max_length:
        return name
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:_HASH_LENGTH]
    prefix_length = max_length - _HASH_LENGTH - 1
    return f"{name[:prefix_length]}_{digest}"


def _mcp_tool_name(server_name: str, tool_name: str) -> str:
    return _limit_tool_name(_sanitize_name(f"mcp_{server_name}_{tool_name}"))


def _is_transient(exc: BaseException) -> bool:
    return type(exc).__name__ in _TRANSIENT_EXC_NAMES


def _is_session_terminated(exc: BaseException) -> bool:
    if _is_transient(exc):
        return True
    messages = [str(exc)]
    error = getattr(exc, "error", None)
    if error is not None:
        messages.append(str(getattr(error, "message", "")))
    return any(
        marker in message.lower()
        for marker in ("session terminated", "connection closed")
        for message in messages
    )


def _mcp_connections_lock() -> asyncio.Lock:
    global _MCP_CONNECTIONS_LOCK
    if _MCP_CONNECTIONS_LOCK is None:
        _MCP_CONNECTIONS_LOCK = asyncio.Lock()
    return _MCP_CONNECTIONS_LOCK


def _normalize_transport(
    command: str | None,
    url: str | None,
    transport: str | None,
) -> MCPTransport:
    if transport is None:
        normalized: MCPTransport = (
            "stdio"
            if command
            else "sse"
            if url and url.rstrip("/").endswith("/sse")
            else "streamable-http"
        )
    elif transport == "stdio":
        normalized = "stdio"
    elif transport == "sse":
        normalized = "sse"
    elif transport == "streamable-http":
        normalized = "streamable-http"
    else:
        raise ValueError(f"Unsupported MCP transport: {transport}")
    if command and normalized != "stdio":
        raise ValueError("An MCP command requires the stdio transport")
    if url and normalized == "stdio":
        raise ValueError("An MCP URL requires the sse or streamable-http transport")
    return normalized


async def _resolve_http_headers(
    headers: Mapping[str, str] | None,
    command_authorization_token: str | None,
    user_id: str | None,
) -> dict[str, str]:
    def replace_placeholder(
        value: str,
        pattern: re.Pattern[str],
        replacement: str,
    ) -> str:
        return pattern.sub(lambda _match: replacement, value)

    async def resolve_authorization_token(command_template: str, current_user_id: str) -> str:
        token = await execute_command_template(
            command_template,
            {"user_id": current_user_id},
            operation="MCP command_authorization_token",
            timeout=MCP_AUTHORIZATION_TOKEN_TIMEOUT_SECONDS,
        )
        if not token:
            raise RuntimeError("MCP authorization token command returned an empty token")
        return token

    resolved = dict(headers or {})
    if user_id is None and any(
        _USER_ID_PLACEHOLDER_RE.search(value) for value in resolved.values()
    ):
        raise ValueError("MCP header user_id placeholders require a user-scoped registration")
    if user_id is not None:
        resolved = {
            name: replace_placeholder(value, _USER_ID_PLACEHOLDER_RE, user_id)
            for name, value in resolved.items()
        }

    has_authorization_token_placeholder = any(
        _AUTHORIZATION_TOKEN_PLACEHOLDER_RE.search(value) for value in resolved.values()
    )
    if has_authorization_token_placeholder:
        if command_authorization_token is None:
            raise ValueError(
                "MCP header authorization_token placeholders require command_authorization_token"
            )
        if user_id is None:
            raise ValueError(
                "MCP command_authorization_token requires a user-scoped registration"
            )
        authorization_token = await resolve_authorization_token(
            command_authorization_token,
            user_id,
        )
        resolved = {
            name: replace_placeholder(
                value,
                _AUTHORIZATION_TOKEN_PLACEHOLDER_RE,
                authorization_token,
            )
            for name, value in resolved.items()
        }
    return resolved


def _extract_nullable_branch(options: Any) -> tuple[dict[str, Any], bool] | None:
    if not isinstance(options, list):
        return None

    non_null: list[dict[str, Any]] = []
    saw_null = False
    for option in options:
        if not isinstance(option, dict):
            return None
        if option.get("type") == "null":
            saw_null = True
            continue
        non_null.append(option)

    if saw_null and len(non_null) == 1:
        return non_null[0], True
    return None


def _normalize_schema_for_openai(schema: Any) -> dict[str, Any]:
    if not isinstance(schema, dict):
        return {"type": "object", "properties": {}}

    normalized = dict(schema)
    raw_type = normalized.get("type")
    if isinstance(raw_type, list):
        non_null = [item for item in raw_type if item != "null"]
        if "null" in raw_type and len(non_null) == 1:
            normalized["type"] = non_null[0]
            normalized["nullable"] = True

    for key in ("oneOf", "anyOf"):
        nullable_branch = _extract_nullable_branch(normalized.get(key))
        if nullable_branch is not None:
            branch, _ = nullable_branch
            merged = {k: v for k, v in normalized.items() if k != key}
            merged.update(branch)
            normalized = merged
            normalized["nullable"] = True
            break

    if "properties" in normalized and isinstance(normalized["properties"], dict):
        normalized["properties"] = {
            name: _normalize_schema_for_openai(prop) if isinstance(prop, dict) else prop
            for name, prop in normalized["properties"].items()
        }

    if "items" in normalized and isinstance(normalized["items"], dict):
        normalized["items"] = _normalize_schema_for_openai(normalized["items"])

    if normalized.get("type") != "object":
        return normalized

    normalized.setdefault("properties", {})
    normalized.setdefault("required", [])
    return normalized


def _schema_to_model(tool_name: str, schema: dict[str, Any]) -> type[BaseModel]:
    """Create a permissive pydantic model used only for loop argument binding."""
    properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
    required = set(schema.get("required") or [])
    fields: dict[str, tuple[Any, Any]] = {}

    for name, prop_schema in properties.items():
        if not isinstance(name, str) or not name.isidentifier():
            continue
        description = prop_schema.get("description") if isinstance(prop_schema, dict) else None
        default = ... if name in required else None
        fields[name] = (Any, Field(default, description=description))

    model_name = re.sub(r"[^a-zA-Z0-9_]", "_", _sanitize_name(tool_name)).title().replace("_", "")
    if not model_name or not re.match(r"^[a-zA-Z_]", model_name):
        model_name = f"MCP{model_name}"

    return create_model(
        f"{model_name}Args",
        __config__=ConfigDict(extra="allow"),
        **fields,
    )


def _iter_tools(list_tools_result: Any) -> list[Any]:
    tools = getattr(list_tools_result, "tools", None)
    if isinstance(tools, list):
        return tools

    found: list[Any] = []
    try:
        for item in list_tools_result:
            if isinstance(item, tuple) and len(item) == 2 and item[0] == "tools":
                found.extend(item[1])
    except TypeError:
        pass
    return found


def _get_attr_or_key(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _content_block_to_message(block: Any) -> dict[str, Any] | None:
    block_type = _get_attr_or_key(block, "type")
    if block_type == "text":
        text = _get_attr_or_key(block, "text", "")
        return {"type": "text", "text": str(text)}
    if block_type == "image":
        data = _get_attr_or_key(block, "data")
        mime = _get_attr_or_key(block, "mimeType") or _get_attr_or_key(block, "mime_type") or "image/png"
        if data:
            return {
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{data}"},
                "_meta": "The MCP tool returned this image.",
            }
    return None


def _render_mcp_result(result: Any) -> str | list[dict[str, Any]]:
    content = _get_attr_or_key(result, "content", None)
    structured = _get_attr_or_key(result, "structuredContent", None)
    if structured is None:
        structured = _get_attr_or_key(result, "structured_content", None)
    is_error = bool(_get_attr_or_key(result, "isError", False) or _get_attr_or_key(result, "is_error", False))

    messages: list[dict[str, Any]] = []
    if isinstance(content, list):
        for block in content:
            message = _content_block_to_message(block)
            if message is not None:
                messages.append(message)
            else:
                messages.append({"type": "text", "text": json.dumps(block, ensure_ascii=False, default=str)})
    elif content is not None:
        messages.append({"type": "text", "text": str(content)})

    if structured is not None:
        messages.append(
            {
                "type": "text",
                "text": "Structured content:\n" + json.dumps(structured, ensure_ascii=False, default=str),
            }
        )

    if not messages:
        messages.append({"type": "text", "text": "(MCP tool returned no content)"})

    if is_error:
        messages.insert(0, {"type": "text", "text": "(MCP tool returned an error)"})

    if all(message["type"] == "text" for message in messages):
        return "\n".join(message["text"] for message in messages)
    return messages


@dataclass(frozen=True)
class MCPServerConfig:
    server_name: str
    workspace: Path
    skills_loader: SkillsLoader
    command: str | None = None
    args: tuple[str, ...] = ()
    env: Mapping[str, str] | None = None
    url: str | None = None
    transport: MCPTransport = "stdio"
    command_authorization_token: str | None = None
    user_scoped: bool = False
    headers: Mapping[str, str] | None = None
    verify_tls: bool = True
    enabled_tools: tuple[str, ...] = ("*",)
    tool_timeout: float = 30.0
    connect_timeout: float = MCP_CONNECT_TIMEOUT_SECONDS
    filter: str | None = None
    description: str = ""

    def __post_init__(self) -> None:
        if not self.command and not self.url:
            raise ValueError("MCP server must define either command or url")
        if self.command and self.url:
            raise ValueError("MCP server cannot define both command and url")
        if self.env is not None:
            if not self.command or self.transport != "stdio":
                raise ValueError("MCP env is only supported for stdio servers")
            for name in ("PATH", "HOME"):
                if name in self.env:
                    raise ValueError(f"MCP env cannot configure {name}; it is managed by the sandbox")
        if self.command_authorization_token is not None and not self.url:
            raise ValueError("MCP command_authorization_token requires a remote URL")
        if self.command_authorization_token is not None and not self.command_authorization_token.strip():
            raise ValueError("MCP command_authorization_token cannot be empty")
        if self.filter is not None:
            parse_filter(self.filter)


@dataclass
class MCPToolBinding:
    schema: dict[str, Any]
    func: Callable[[BaseModel], Any]


@dataclass
class MCPServerRegistration:
    config: MCPServerConfig
    user_id: str | None = None
    _session: Any | None = dataclass_field(default=None, init=False, repr=False)
    _bindings: list[MCPToolBinding] | None = dataclass_field(default=None, init=False, repr=False)
    _lock: asyncio.Lock = dataclass_field(default_factory=asyncio.Lock, init=False, repr=False)
    _owner_task: asyncio.Task[None] | None = dataclass_field(default=None, init=False, repr=False)
    _close_event: asyncio.Event | None = dataclass_field(default=None, init=False, repr=False)
    _discovery_task: asyncio.Task[list[MCPToolBinding]] | None = dataclass_field(
        default=None, init=False, repr=False
    )
    _last_error: str | None = dataclass_field(default=None, init=False, repr=False)
    _retired: bool = dataclass_field(default=False, init=False, repr=False)
    _wrapped_command: list[str] | None = dataclass_field(default=None, init=False, repr=False)
    _active_calls: int = dataclass_field(default=0, init=False, repr=False)
    _drained: asyncio.Condition = dataclass_field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._drained = asyncio.Condition(self._lock)

    @property
    def connection_scope(self) -> MCPConnectionScope:
        return self.user_id, self.config.workspace

    def retire(self) -> None:
        """Prevent stale agent references from reopening an evicted registration."""
        self._retired = True
        self._bindings = None

    def start_tool_discovery(self) -> asyncio.Task[list[MCPToolBinding]]:
        """Start shared background discovery without waiting for the server."""
        task = self._discovery_task
        if not self._retired and task is not None and not task.cancelled():
            if not task.done():
                return task
            if self._bindings is not None and self._session is not None and not self.config.command:
                return task
        if self._session is None:
            self._bindings = None

        async def discover() -> list[MCPToolBinding]:
            return [
                MCPToolBinding(schema=schema, func=func)
                async for schema, func, _ in collect_mcp_tool(self)
            ]

        self._discovery_task = asyncio.create_task(
            discover(), name=f"mcp-{self.config.server_name}-discovery"
        )
        return self._discovery_task

    def get_status(self) -> dict[str, Any]:
        """Read connection/discovery state without performing MCP I/O."""
        task = self._discovery_task
        if self._retired:
            status = "disconnected"
        elif task is not None and not task.done():
            status = "connecting"
        elif self._session is not None and self._bindings is not None:
            status = "ready"
        elif self._last_error is not None:
            status = "unavailable"
        else:
            status = "disconnected"
        tools = [binding.func.__name__ for binding in self._bindings or []] if status == "ready" else []
        return {
            "description": self.config.description,
            "status": status,
            "transport": self.config.transport,
            "tool_count": len(tools),
            "tools": tools,
            "error": self._last_error,
        }

    def _mark_unavailable(self, exc: BaseException) -> None:
        self._bindings = None
        # Exception messages can contain credentials or authorization headers.
        self._last_error = type(exc).__name__

    async def _get_tool_bindings(self) -> list[MCPToolBinding]:
        async with self._lock:
            if self._retired:
                return []
            try:
                await self._ensure_session_locked()
                if self._bindings is not None:
                    return self._bindings
                self._bindings = await self._discover_tool_bindings()
                self._last_error = None
                await _refresh_session_expiry(self)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._mark_unavailable(exc)
                # anyio may report transport shutdown as a BaseExceptionGroup.
                # Cleanup must not replace the original connection failure.
                try:
                    await self._close_connection()
                except BaseException as cleanup_exc:
                    logger.debug(
                        "MCP server '{}' cleanup after connection failure: {}",
                        self.config.server_name,
                        cleanup_exc,
                    )
                raise

            return self._bindings

    async def call_tool(self, tool_name: str, arguments: Mapping[str, Any]) -> str | list[dict[str, Any]]:
        try:
            async with self._lock:
                await self._ensure_session_locked()
                session = self._session
                if session is None or self._retired:
                    return "(MCP server is temporarily unavailable)"
                # Reserve this session under the lifecycle lock so a rebuild
                # cannot close it between validation and the actual tool call.
                self._active_calls += 1
        except _MCPRegistrationRetiredError:
            return "(MCP server is temporarily unavailable)"
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception(
                "MCP server '{}' failed to reconnect: {}",
                self.config.server_name,
                exc,
            )
            return "(MCP server is temporarily unavailable)"

        try:
            result = await asyncio.wait_for(
                session.call_tool(tool_name, arguments=dict(arguments)),
                timeout=self.config.tool_timeout,
            )
            await _refresh_session_expiry(self)
            return _render_mcp_result(result)
        except TimeoutError:
            logger.warning(
                "MCP tool '{}.{}' timed out after {}s",
                self.config.server_name,
                tool_name,
                self.config.tool_timeout,
            )
            return f"(MCP tool call timed out after {self.config.tool_timeout}s)"
        except asyncio.CancelledError:
            task = asyncio.current_task()
            if task is not None and task.cancelling() > 0:
                raise
            logger.warning(
                "MCP tool '{}.{}' was cancelled by the server or MCP SDK",
                self.config.server_name,
                tool_name,
            )
            return "(MCP tool call was cancelled)"
        except Exception as exc:
            if _is_session_terminated(exc):
                logger.warning(
                    "MCP server '{}' session ended ({}); the next request can reconnect",
                    self.config.server_name,
                    type(exc).__name__,
                )
                async with self._lock:
                    if self._session is session:
                        self._mark_unavailable(exc)
                        await self._close_connection()
                return "(MCP server is temporarily unavailable)"
            logger.exception(
                "MCP tool '{}.{}' failed: {}: {}",
                self.config.server_name,
                tool_name,
                type(exc).__name__,
                exc,
            )
            return f"(MCP tool call failed: {type(exc).__name__})"
        finally:
            async with self._lock:
                self._active_calls -= 1
                self._drained.notify_all()

    async def aclose(self) -> None:
        discovery_task = self._discovery_task
        if discovery_task is not None and discovery_task is not asyncio.current_task():
            if not discovery_task.done():
                discovery_task.cancel()
                await asyncio.gather(discovery_task, return_exceptions=True)
        await self._close_connection()

    async def _close_connection(self) -> None:
        """Close transport state without cancelling a new conversation's discovery."""
        owner_task = self._owner_task
        close_event = self._close_event
        self._owner_task = None
        self._close_event = None
        self._session = None
        self._wrapped_command = None
        if close_event is not None:
            close_event.set()
        if owner_task is not None and owner_task is not asyncio.current_task():
            await owner_task

    async def _ensure_session(self) -> None:
        async with self._lock:
            await self._ensure_session_locked()

    async def _ensure_session_locked(self) -> None:
        """Validate sandbox configuration and refresh the connection under the lock."""
        if self._retired:
            raise _MCPRegistrationRetiredError("MCP registration has been retired")
        if self.config.command:
            command = await wrap_sandbox_command(
                [self.config.command, *self.config.args], self.config.workspace,
                network_isolation=False, skills_loader=self.config.skills_loader,
            )
            # Concurrent calls with unchanged configuration keep sharing the
            # connection. Configuration changes wait for existing calls.
            while (
                self._session is not None
                and self._wrapped_command is not None
                and self._wrapped_command != command
                and self._active_calls
            ):
                await self._drained.wait()
            if self._retired:
                raise _MCPRegistrationRetiredError("MCP registration has been retired")
            if self._wrapped_command is not None and self._wrapped_command != command:
                await self._close_connection()
        if self._session is None:
            try:
                await self._connect()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._mark_unavailable(exc)
                with suppress(BaseException):
                    await self._close_connection()
                raise

    async def _connect(self) -> None:
        await self._close_connection()
        if self._retired:
            raise _MCPRegistrationRetiredError("MCP registration has been retired")
        # A replacement session may expose different tools and schemas.
        self._bindings = None
        loop = asyncio.get_running_loop()
        ready: asyncio.Future[None] = loop.create_future()
        close_event = asyncio.Event()
        owner_task = asyncio.create_task(
            self._run_connection(ready, close_event),
            name=f"mcp-{self.config.server_name}-connection",
        )
        self._owner_task = owner_task
        self._close_event = close_event
        try:
            await asyncio.wait_for(ready, timeout=self.config.connect_timeout)
            if self._retired:
                raise _MCPRegistrationRetiredError("MCP registration has been retired")
            self._last_error = None
            await _refresh_session_expiry(self)
        except BaseException:
            close_event.set()
            owner_task.cancel()
            with suppress(BaseException):
                await owner_task
            if self._owner_task is owner_task:
                self._owner_task = None
                self._close_event = None
            raise

    async def _run_connection(
        self,
        ready: asyncio.Future[None],
        close_event: asyncio.Event,
    ) -> None:
        cfg = self.config
        workspace = cfg.workspace.resolve()
        session = None
        command = None
        try:
            async with AsyncExitStack() as stack:
                from mcp import ClientSession, StdioServerParameters, types
                from mcp.client.sse import sse_client
                from mcp.client.stdio import stdio_client
                from mcp.client.streamable_http import streamable_http_client

                if cfg.command:
                    if platform.system() != "Linux" or shutil.which("bwrap") is None:
                        raise RuntimeError("Stdio MCP requires Linux with bubblewrap")
                    missing = [name for name in SANDBOX_EXECUTABLES if shutil.which(name) is None]
                    if missing:
                        raise RuntimeError(f"Stdio MCP requires executable(s) on PATH: {', '.join(missing)}")
                    # MCP servers may need host-local services; keep the host
                    # network while retaining filesystem and process isolation.
                    command = await wrap_sandbox_command(
                        [cfg.command, *cfg.args], workspace, network_isolation=False,
                        skills_loader=cfg.skills_loader,
                    )
                    server_params = StdioServerParameters(
                        command=command[0],
                        args=command[1:],
                        env=dict(cfg.env) if cfg.env is not None else None,
                        cwd=workspace,
                    )
                    read, write = await stack.enter_async_context(stdio_client(server_params))
                else:
                    headers = await _resolve_http_headers(
                        cfg.headers,
                        cfg.command_authorization_token,
                        self.user_id,
                    )
                    if cfg.transport == "sse":
                        read, write = await stack.enter_async_context(
                            sse_client(
                                cfg.url,
                                headers=headers or None,
                                timeout=cfg.connect_timeout,
                                httpx_client_factory=partial(
                                    httpx.AsyncClient,
                                    follow_redirects=True,
                                    verify=cfg.verify_tls,
                                ),
                            )
                        )
                    elif cfg.transport == "streamable-http":
                        http_client = await stack.enter_async_context(
                            httpx.AsyncClient(
                                headers=headers or None,
                                follow_redirects=True,
                                verify=cfg.verify_tls,
                                timeout=httpx.Timeout(
                                    cfg.tool_timeout,
                                    connect=cfg.connect_timeout,
                                    read=300.0,
                                ),
                            )
                        )
                        transport_result = await stack.enter_async_context(
                            streamable_http_client(cfg.url, http_client=http_client)
                        )
                        read, write = transport_result[:2]
                    else:
                        raise ValueError(f"Unsupported MCP transport: {cfg.transport}")

                async def list_roots_callback(_context: Any) -> types.ListRootsResult:
                    return types.ListRootsResult(
                        roots=[types.Root(uri=workspace.as_uri(), name=workspace.name)]
                    )

                session = await stack.enter_async_context(
                    ClientSession(
                        read,
                        write,
                        list_roots_callback=list_roots_callback,
                    )
                )
                await asyncio.wait_for(session.initialize(), timeout=cfg.tool_timeout)
                self._session = session
                self._wrapped_command = command
                logger.info("MCP server '{}' connected", cfg.server_name)
                ready.set_result(None)
                await close_event.wait()
        except BaseException as exc:
            if not ready.done():
                ready.set_exception(exc)
            elif self._owner_task is asyncio.current_task() and not isinstance(exc, asyncio.CancelledError):
                self._mark_unavailable(exc)
                logger.exception("MCP server '{}' connection owner failed", cfg.server_name)
        finally:
            if self._session is session:
                self._session = None
                self._wrapped_command = None
            if self._owner_task is asyncio.current_task():
                self._owner_task = None
                self._close_event = None

    async def _discover_tool_bindings(self) -> list[MCPToolBinding]:
        tools_result = await asyncio.wait_for(
            self._session.list_tools(),
            timeout=self.config.tool_timeout,
        )
        tool_defs = _iter_tools(tools_result)
        bindings: list[MCPToolBinding] = []

        for tool_def in tool_defs:
            original_name = _get_attr_or_key(tool_def, "name")
            if not original_name:
                continue
            wrapped_name = _mcp_tool_name(self.config.server_name, str(original_name))
            if not any(fnmatchcase(str(original_name), pattern) for pattern in self.config.enabled_tools):
                continue

            description = _get_attr_or_key(tool_def, "description") or f"MCP tool {original_name}"
            raw_schema = (
                _get_attr_or_key(tool_def, "inputSchema", None)
                or _get_attr_or_key(tool_def, "input_schema", None)
                or {"type": "object", "properties": {}}
            )
            schema = _normalize_schema_for_openai(raw_schema)
            model_cls = _schema_to_model(wrapped_name, schema)
            tool_func = self._build_tool_func(str(original_name), wrapped_name, str(description), model_cls)
            bindings.append(
                MCPToolBinding(
                    schema={
                        "type": "function",
                        "function": {
                            "name": wrapped_name,
                            "description": str(description),
                            "parameters": schema,
                        },
                    },
                    func=tool_func,
                )
            )

        logger.info(
            "MCP server '{}' discovered {} enabled tools",
            self.config.server_name,
            len(bindings),
        )
        return bindings

    def _build_tool_func(
        self,
        original_name: str,
        wrapped_name: str,
        description: str,
        model_cls: type[BaseModel],
    ) -> Callable[[BaseModel], Any]:
        async def call_mcp_tool(args: BaseModel) -> str | list[dict[str, Any]]:
            fields_set = getattr(args, "model_fields_set", set())
            arguments = args.model_dump(mode="json", include=fields_set)
            return await self.call_tool(original_name, arguments)

        call_mcp_tool.__name__ = wrapped_name
        call_mcp_tool.mcp_original_name = original_name
        call_mcp_tool.__qualname__ = wrapped_name
        call_mcp_tool.__doc__ = description
        call_mcp_tool.__annotations__ = {"args": model_cls, "return": str | list[dict[str, Any]]}
        call_mcp_tool.__signature__ = inspect.Signature(
            parameters=[
                inspect.Parameter(
                    "args",
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=model_cls,
                )
            ],
            return_annotation=str | list[dict[str, Any]],
        )
        return call_mcp_tool


def create_mcp_registration(
    *,
    server_name: str,
    workspace: Path,
    skills_loader: SkillsLoader,
    description: str = "",
    command: str | None = None,
    args: Sequence[str] | None = None,
    env: Mapping[str, str] | None = None,
    url: str | None = None,
    transport: str | None = None,
    command_authorization_token: str | None = None,
    user_scoped: bool = False,
    headers: Mapping[str, str] | None = None,
    verify_tls: bool = True,
    enabled_tools: Sequence[str] | None = None,
    tool_timeout: float = 30.0,
    connect_timeout: float | None = None,
    filter: str | None = None,
) -> MCPServerRegistration:
    """Create an MCP server registration from configuration."""

    sanitized_server_name = _sanitize_name(server_name)
    if not sanitized_server_name:
        raise ValueError("MCP server name cannot be empty")
    normalized_transport = _normalize_transport(command, url, transport)
    return MCPServerRegistration(
        MCPServerConfig(
            server_name=sanitized_server_name,
            description=description,
            workspace=workspace.resolve(),
            command=command,
            args=tuple(args or ()),
            env=dict(env) if env is not None else None,
            url=url,
            transport=normalized_transport,
            command_authorization_token=command_authorization_token,
            user_scoped=user_scoped,
            headers=dict(headers) if headers is not None else None,
            verify_tls=verify_tls,
            enabled_tools=("*",) if enabled_tools is None else tuple(enabled_tools),
            tool_timeout=tool_timeout,
            connect_timeout=(
                MCP_CONNECT_TIMEOUT_SECONDS if connect_timeout is None else connect_timeout
            ),
            filter=filter,
            skills_loader=skills_loader,
        )
    )


def create_mcp_registration_from_config(
    server_name: str,
    config: MCPServerSettings,
    workspace: Path,
    *,
    skills_loader: SkillsLoader,
) -> MCPServerRegistration:
    return create_mcp_registration(
        server_name=server_name,
        description=config.description,
        workspace=workspace,
        command=config.command,
        args=config.args,
        env=config.env,
        url=config.url,
        transport=config.transport,
        command_authorization_token=config.command_authorization_token,
        user_scoped=config.user_scoped,
        headers=config.headers,
        verify_tls=config.verify_tls,
        enabled_tools=config.enabled_tools,
        tool_timeout=config.tool_timeout,
        connect_timeout=config.connect_timeout,
        filter=config.filter,
        skills_loader=skills_loader,
    )


def _hashable_mapping(value: Mapping[str, str] | None) -> tuple[tuple[str, str], ...] | None:
    if value is None:
        return None
    return tuple(sorted((str(key), str(item)) for key, item in value.items()))


def _persistent_registration_key(config: MCPServerConfig) -> tuple[Any, ...]:
    loader = config.skills_loader if config.command else None
    return (
        config.server_name,
        config.description,
        config.workspace,
        config.command,
        config.args,
        _hashable_mapping(config.env),
        config.url,
        config.transport,
        config.command_authorization_token,
        config.user_scoped,
        _hashable_mapping(config.headers),
        config.verify_tls,
        config.enabled_tools,
        float(config.tool_timeout),
        float(config.connect_timeout),
        config.filter,
        loader.channel_tag if loader is not None else None,
        loader.user_organization if loader is not None else None,
        loader.not_interactive if loader is not None else False,
    )


async def get_session_mcp_registration_from_config(
    user_id: str,
    server_name: str,
    config: MCPServerSettings,
    workspace: Path,
    *,
    skills_loader: SkillsLoader,
) -> MCPServerRegistration:
    """Always isolate stdio; share remote connections unless authorization/config requires isolation."""
    if not user_id:
        raise ValueError("user_id cannot be empty")
    transport = _normalize_transport(config.command, config.url, config.transport)
    user_scoped = (
        transport == "stdio"
        or config.command_authorization_token is not None
        or config.user_scoped
    )
    if not user_scoped:
        # Shared transports and roots must never inherit the first user's workspace.
        workspace = settings.path.cache_path / "mcp" / _sanitize_name(server_name)
    registration = create_mcp_registration_from_config(
        server_name, config, workspace=workspace, skills_loader=skills_loader,
    )
    if user_scoped:
        registration.user_id = user_id
    key = _persistent_registration_key(registration.config)
    async with _mcp_connections_lock():
        session_connections = MCP_CONNECTIONS.setdefault(registration.connection_scope, {})
        cached = session_connections.get(key)
        if cached is not None:
            return cached
        if not user_scoped:
            registration.config.workspace.mkdir(parents=True, exist_ok=True)
        session_connections[key] = registration
        _schedule_session_expiry_locked(registration.connection_scope)
        return registration


def _schedule_session_expiry_locked(scope: MCPConnectionScope) -> None:
    if scope in _SESSION_EXPIRY_HANDLES:
        return

    handle: asyncio.TimerHandle

    def expire() -> None:
        asyncio.create_task(
            _expire_session(scope, handle),
            name=f"mcp-expiry-{scope}",
        )

    handle = asyncio.get_running_loop().call_later(MCP_CONNECTION_TTL_SECONDS, expire)
    _SESSION_EXPIRY_HANDLES[scope] = handle


async def _refresh_session_expiry(registration: MCPServerRegistration) -> None:
    """Extend a pooled connection scope's idle lifetime after successful MCP I/O."""
    scope = registration.connection_scope
    async with _mcp_connections_lock():
        registrations = MCP_CONNECTIONS.get(scope, {})
        if registrations.get(_persistent_registration_key(registration.config)) is not registration:
            return
        expiry_handle = _SESSION_EXPIRY_HANDLES.pop(scope, None)
        if expiry_handle is not None:
            expiry_handle.cancel()
        _schedule_session_expiry_locked(scope)


async def _expire_session(scope: MCPConnectionScope, handle: asyncio.TimerHandle) -> None:
    async with _mcp_connections_lock():
        if _SESSION_EXPIRY_HANDLES.get(scope) is not handle:
            return
        registrations = MCP_CONNECTIONS.pop(scope, {})
        _SESSION_EXPIRY_HANDLES.pop(scope, None)
        for registration in registrations.values():
            registration.retire()

    logger.info(
        "MCP session '{}' expired after {}s without successful MCP activity",
        scope,
        MCP_CONNECTION_TTL_SECONDS,
    )
    await asyncio.gather(
        *(registration.aclose() for registration in registrations.values()),
        return_exceptions=True,
    )


async def close_session_mcp_connections(user_id: str, workspace: Path | None = None) -> None:
    """Close one user's connections, optionally limited to a workspace."""
    if workspace is not None:
        workspace = workspace.resolve()
    async with _mcp_connections_lock():
        scopes = [
            scope for scope in MCP_CONNECTIONS
            if scope[0] == user_id and (workspace is None or scope[1] == workspace)
        ]
        registrations = [
            registration
            for scope in scopes
            for registration in MCP_CONNECTIONS.pop(scope).values()
        ]
        for registration in registrations:
            registration.retire()
        for scope in scopes:
            expiry_handle = _SESSION_EXPIRY_HANDLES.pop(scope, None)
            if expiry_handle is not None:
                expiry_handle.cancel()
    await asyncio.gather(
        *(registration.aclose() for registration in registrations),
        return_exceptions=True,
    )


async def close_all_mcp_connections() -> None:
    async with _mcp_connections_lock():
        registrations = [
            registration
            for session_connections in MCP_CONNECTIONS.values()
            for registration in session_connections.values()
        ]
        for registration in registrations:
            registration.retire()
        MCP_CONNECTIONS.clear()
        for handle in _SESSION_EXPIRY_HANDLES.values():
            handle.cancel()
        _SESSION_EXPIRY_HANDLES.clear()

    await asyncio.gather(
        *(registration.aclose() for registration in registrations),
        return_exceptions=True,
    )


async def collect_mcp_tool(
    registration: MCPServerRegistration,
) -> AsyncIterator[tuple[dict[str, Any], Callable[[BaseModel], Any], str]]:
    try:
        bindings = await registration._get_tool_bindings()
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.exception(
            "MCP server '{}' failed to connect/discover tools: {}",
            registration.config.server_name,
            exc,
        )
        return

    for binding in bindings:
        yield binding.schema, binding.func, binding.func.__name__


def get_mcp_status(registrations: Mapping[str, MCPServerRegistration]) -> dict[str, Any]:
    """Return a non-blocking snapshot of the MCP servers visible to this agent."""
    return {"servers": {name: registration.get_status() for name, registration in registrations.items()}}


def build_mcp_status_tool(*, registrations: Mapping[str, MCPServerRegistration]):
    async def query_mcp_status() -> str:
        """查询 MCP 状态及可用工具。connecting 表示后台连接中；ready 后工具会自动出现在后续模型调用中。

        根据配置的服务用途判断当前任务是否需要等待相关 MCP；不依赖其能力时可继续处理任务。
        本工具立即返回，不等待连接；unavailable 表示上次连接失败，下次对话会立即在后台重试。
        """
        return json.dumps(get_mcp_status(registrations), ensure_ascii=False)

    if registrations:
        descriptions = {
            name: registration.config.description or "未配置用途说明"
            for name, registration in registrations.items()
        }
        query_mcp_status.__doc__ = (
            f"{inspect.cleandoc(query_mcp_status.__doc__)}\n\n"
            f"当前通道的 MCP 服务及用途：\n{json.dumps(descriptions, ensure_ascii=False)}"
        )
    return query_mcp_status


def register_mcp_bindings(
    task: asyncio.Task[list[MCPToolBinding]],
    *,
    registration: MCPServerRegistration,
    provider: Any,
    execution_request_tools: list[str],
    not_interactive: bool,
    requires_confirmation: Callable[[str], ExecutionDecision | Awaitable[ExecutionDecision]] | None = None,
) -> None:
    if task.cancelled():
        return
    bindings = task.result()
    if registration.get_status()["status"] != "ready":
        return
    # Keep this callback synchronous: chat must see either none or all of this
    # server's tools, with schemas and callables drawn from the same snapshot.
    for binding in bindings:
        tool_json, tool_func = binding.schema, binding.func
        if any(
            fnmatchcase(tool_func.mcp_original_name, pattern)
            for pattern in execution_request_tools
        ):
            if not_interactive and requires_confirmation is None:
                continue
            tool_func = build_execution_request_tool(
                tool_func,
                requires_confirmation=requires_confirmation,
                not_interactive=not_interactive,
            )
            tool_json = execution_request_schema(tool_json)
        provider.register_mcp_tool(tool_func, tool_json=tool_json)
