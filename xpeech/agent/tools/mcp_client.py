"""MCP client integration for xpeech tools.

Default MCP servers are configured in ``conf.toml``:

    [tool.mcpServers.filesystem]
    command = "npx"
    args = ["-y", "@modelcontextprotocol/server-filesystem", "workspace_base"]
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import os
import re
import shutil
import urllib.parse
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from contextlib import AsyncExitStack, suppress
from dataclasses import dataclass, field as dataclass_field
from typing import Any, Literal

import httpx
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, create_model


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
_WINDOWS_SHELL_LAUNCHERS: frozenset[str] = frozenset(("npx", "npm", "pnpm", "yarn", "bunx"))
_SANITIZE_RE = re.compile(r"_+")
_MAX_TOOL_NAME_LENGTH = 64
_HASH_LENGTH = 8
_PERSISTENT_REGISTRATION_CACHE: dict[tuple[Any, ...], "MCPServerRegistration"] = {}
_PERSISTENT_REGISTRATION_LOCK: asyncio.Lock | None = None


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


async def _probe_http_url(url: str, timeout: float = 3.0) -> bool:
    parsed = urllib.parse.urlparse(url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        _, writer = await asyncio.wait_for(asyncio.open_connection(host, port), timeout=timeout)
        writer.close()
        with suppress(OSError, asyncio.TimeoutError):
            await asyncio.wait_for(writer.wait_closed(), timeout=0.2)
        return True
    except (OSError, asyncio.TimeoutError):
        return False


def _persistent_registration_lock() -> asyncio.Lock:
    global _PERSISTENT_REGISTRATION_LOCK
    if _PERSISTENT_REGISTRATION_LOCK is None:
        _PERSISTENT_REGISTRATION_LOCK = asyncio.Lock()
    return _PERSISTENT_REGISTRATION_LOCK


def _windows_command_basename(command: str) -> str:
    return command.replace("\\", "/").rsplit("/", maxsplit=1)[-1].lower()


def _normalize_windows_stdio_command(
    command: str,
    args: Sequence[str] | None,
    env: Mapping[str, str] | None,
) -> tuple[str, list[str], dict[str, str] | None]:
    normalized_args = list(args or [])
    normalized_env = dict(env) if env is not None else None
    if os.name != "nt":
        return command, normalized_args, normalized_env

    basename = _windows_command_basename(command)
    if basename in {"cmd", "cmd.exe", "powershell", "powershell.exe", "pwsh", "pwsh.exe"}:
        return command, normalized_args, normalized_env
    if basename.endswith((".exe", ".com")):
        return command, normalized_args, normalized_env

    resolved = shutil.which(command, path=(normalized_env or {}).get("PATH")) or command
    resolved_basename = _windows_command_basename(resolved)
    should_wrap = (
        basename in _WINDOWS_SHELL_LAUNCHERS
        or basename.endswith((".cmd", ".bat"))
        or resolved_basename.endswith((".cmd", ".bat"))
    )
    if not should_wrap:
        return command, normalized_args, normalized_env

    comspec = (normalized_env or {}).get("COMSPEC") or os.environ.get("COMSPEC") or "cmd.exe"
    return comspec, ["/d", "/c", command, *normalized_args], normalized_env


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


def _truncate_mcp_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    marker = f"\n\n... [MCP result truncated: {len(text)} characters total] ...\n\n"
    available = max(0, max_chars - len(marker))
    head_chars = int(available * 0.6)
    tail_chars = available - head_chars
    return text[:head_chars] + marker + (text[-tail_chars:] if tail_chars else "")


def _truncate_mcp_messages(messages: list[dict[str, Any]], max_chars: int) -> list[dict[str, Any]]:
    text = "\n".join(message["text"] for message in messages if message.get("type") == "text")
    if len(text) <= max_chars:
        return messages

    truncated_text = _truncate_mcp_text(text, max_chars)
    non_text_messages = [message for message in messages if message.get("type") != "text"]
    return [{"type": "text", "text": truncated_text}, *non_text_messages]


def _render_mcp_result(result: Any, max_chars: int) -> str | list[dict[str, Any]]:
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

    messages = _truncate_mcp_messages(messages, max_chars)
    if all(message["type"] == "text" for message in messages):
        return "\n".join(message["text"] for message in messages)
    return messages


@dataclass(frozen=True)
class MCPServerConfig:
    server_name: str
    command: str | None = None
    args: tuple[str, ...] = ()
    env: Mapping[str, str] | None = None
    url: str | None = None
    transport: Literal["stdio", "sse", "streamable_http", "http"] = "stdio"
    headers: Mapping[str, str] | None = None
    enabled_tools: tuple[str, ...] = ("*",)
    tool_timeout: float = 30.0
    max_result_chars: int = 50_000

    def __post_init__(self) -> None:
        if not self.command and not self.url:
            raise ValueError("MCP server must define either command or url")
        if self.command and self.url:
            raise ValueError("MCP server cannot define both command and url")
        if self.max_result_chars < 1_000:
            raise ValueError("MCP server max_result_chars must be at least 1000")


@dataclass
class MCPToolBinding:
    schema: dict[str, Any]
    func: Callable[[BaseModel], Any]


@dataclass
class MCPServerRegistration:
    config: MCPServerConfig
    _session: Any | None = dataclass_field(default=None, init=False, repr=False)
    _bindings: list[MCPToolBinding] | None = dataclass_field(default=None, init=False, repr=False)
    _lock: asyncio.Lock = dataclass_field(default_factory=asyncio.Lock, init=False, repr=False)
    _owner_task: asyncio.Task[None] | None = dataclass_field(default=None, init=False, repr=False)
    _close_event: asyncio.Event | None = dataclass_field(default=None, init=False, repr=False)

    async def _get_tool_bindings(self) -> list[MCPToolBinding]:
        async with self._lock:
            if self._bindings is not None:
                return self._bindings
            await self._connect()
            self._bindings = await self._discover_tool_bindings()
            return self._bindings

    async def call_tool(self, tool_name: str, arguments: Mapping[str, Any]) -> str | list[dict[str, Any]]:
        for attempt in range(2):
            await self._ensure_session()
            try:
                result = await asyncio.wait_for(
                    self._session.call_tool(tool_name, arguments=dict(arguments)),
                    timeout=self.config.tool_timeout,
                )
                return _render_mcp_result(result, self.config.max_result_chars)
            except asyncio.TimeoutError:
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
                if attempt == 0 and _is_session_terminated(exc):
                    logger.warning(
                        "MCP server '{}' session ended ({}), reconnecting once",
                        self.config.server_name,
                        type(exc).__name__,
                    )
                    await self.aclose()
                    await asyncio.sleep(1)
                    continue
                logger.exception(
                    "MCP tool '{}.{}' failed: {}: {}",
                    self.config.server_name,
                    tool_name,
                    type(exc).__name__,
                    exc,
                )
                return f"(MCP tool call failed: {type(exc).__name__})"
        return "(MCP tool call failed)"

    async def aclose(self) -> None:
        owner_task = self._owner_task
        close_event = self._close_event
        self._owner_task = None
        self._close_event = None
        self._session = None
        if close_event is not None:
            close_event.set()
        if owner_task is not None and owner_task is not asyncio.current_task():
            await owner_task

    async def _ensure_session(self) -> None:
        if self._session is None:
            async with self._lock:
                if self._session is None:
                    await self._connect()

    async def _connect(self) -> None:
        await self.aclose()
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
            await ready
        except BaseException:
            close_event.set()
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
        stack = AsyncExitStack()
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.sse import sse_client
            from mcp.client.stdio import stdio_client
            from mcp.client.streamable_http import streamable_http_client

            if cfg.command:
                command, args, env = _normalize_windows_stdio_command(cfg.command, cfg.args, cfg.env)
                server_params = StdioServerParameters(command=command, args=args, env=env)
                read, write = await stack.enter_async_context(stdio_client(server_params))
            else:
                if not await _probe_http_url(cfg.url):
                    raise ConnectionError(f"MCP server '{cfg.server_name}' is unreachable")
                headers = dict(cfg.headers or {})
                transport = "streamable_http" if cfg.transport == "http" else cfg.transport
                if transport == "sse":
                    read, write = await stack.enter_async_context(sse_client(cfg.url, headers=headers or None))
                elif transport == "streamable_http":
                    http_client = await stack.enter_async_context(
                        httpx.AsyncClient(
                            headers=headers or None,
                            follow_redirects=True,
                            timeout=httpx.Timeout(30.0, connect=10.0, read=300.0),
                        )
                    )
                    transport_result = await stack.enter_async_context(
                        streamable_http_client(cfg.url, http_client=http_client)
                    )
                    read, write = transport_result[:2]
                else:
                    raise ValueError(f"Unsupported MCP transport: {cfg.transport}")

            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            self._session = session
            logger.info("MCP server '{}' connected", cfg.server_name)
            ready.set_result(None)
            await close_event.wait()
        except BaseException as exc:
            if not ready.done():
                ready.set_exception(exc)
            elif not isinstance(exc, asyncio.CancelledError):
                logger.exception("MCP server '{}' connection owner failed", cfg.server_name)
        finally:
            self._session = None
            try:
                await stack.aclose()
            except (RuntimeError, BaseExceptionGroup) as exc:
                logger.debug(
                    "MCP server '{}' cleanup error ignored: {}",
                    cfg.server_name,
                    exc,
                )
            except Exception:
                logger.exception("MCP server '{}' cleanup failed", cfg.server_name)
            if self._owner_task is asyncio.current_task():
                self._owner_task = None
                self._close_event = None

    async def _discover_tool_bindings(self) -> list[MCPToolBinding]:
        tools_result = await self._session.list_tools()
        tool_defs = _iter_tools(tools_result)
        enabled_tools = set(self.config.enabled_tools)
        allow_all = "*" in enabled_tools
        bindings: list[MCPToolBinding] = []

        for tool_def in tool_defs:
            original_name = _get_attr_or_key(tool_def, "name")
            if not original_name:
                continue
            wrapped_name = _mcp_tool_name(self.config.server_name, str(original_name))
            if not allow_all and original_name not in enabled_tools and wrapped_name not in enabled_tools:
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
    command: str | None = None,
    args: Sequence[str] | None = None,
    env: Mapping[str, str] | None = None,
    url: str | None = None,
    headers: Mapping[str, str] | None = None,
    enabled_tools: Sequence[str] | None = None,
    tool_timeout: float = 30.0,
    max_result_chars: int = 50_000,
) -> MCPServerRegistration:
    """Create an MCP server registration from configuration."""

    sanitized_server_name = _sanitize_name(server_name)
    if not sanitized_server_name:
        raise ValueError("MCP server name cannot be empty")
    transport: Literal["stdio", "sse", "streamable_http"] = (
        "stdio" if command else "sse" if url and url.rstrip("/").endswith("/sse") else "streamable_http"
    )
    return MCPServerRegistration(
        MCPServerConfig(
            server_name=sanitized_server_name,
            command=command,
            args=tuple(args or ()),
            env=dict(env) if env is not None else None,
            url=url,
            transport=transport,
            headers=dict(headers) if headers is not None else None,
            enabled_tools=("*",) if enabled_tools is None else tuple(enabled_tools),
            tool_timeout=tool_timeout,
            max_result_chars=max_result_chars,
        )
    )


def _get_mcp_config_value(config: Any, name: str, default: Any = None, attr_name: str | None = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, attr_name or name, default)


def create_mcp_registration_from_config(server_name: str, config: Any) -> MCPServerRegistration:
    env = _get_mcp_config_value(config, "env")
    headers = _get_mcp_config_value(config, "headers")
    return create_mcp_registration(
        server_name=server_name,
        command=_get_mcp_config_value(config, "command"),
        args=_get_mcp_config_value(config, "args", []),
        env=dict(env) if env is not None else None,
        url=_get_mcp_config_value(config, "url"),
        headers=dict(headers) if headers is not None else None,
        enabled_tools=_get_mcp_config_value(config, "enabled_tools", ["*"], attr_name="enabled_tools"),
        tool_timeout=_get_mcp_config_value(config, "tool_timeout", 30.0, attr_name="tool_timeout"),
        max_result_chars=_get_mcp_config_value(
            config,
            "max_result_chars",
            50_000,
            attr_name="max_result_chars",
        ),
    )


def _hashable_mapping(value: Mapping[str, str] | None) -> tuple[tuple[str, str], ...] | None:
    if value is None:
        return None
    return tuple(sorted((str(key), str(item)) for key, item in value.items()))


def _persistent_registration_key(config: MCPServerConfig) -> tuple[Any, ...]:
    return (
        config.server_name,
        config.command,
        config.args,
        _hashable_mapping(config.env),
        config.url,
        config.transport,
        _hashable_mapping(config.headers),
        config.enabled_tools,
        float(config.tool_timeout),
        int(config.max_result_chars),
    )


async def get_persistent_mcp_registration_from_config(
    server_name: str,
    config: Any,
) -> MCPServerRegistration:
    registration = create_mcp_registration_from_config(server_name, config)
    key = _persistent_registration_key(registration.config)
    async with _persistent_registration_lock():
        cached = _PERSISTENT_REGISTRATION_CACHE.get(key)
        if cached is not None:
            return cached
        _PERSISTENT_REGISTRATION_CACHE[key] = registration
        return registration


async def close_persistent_mcp_registrations() -> None:
    async with _persistent_registration_lock():
        registrations = list(_PERSISTENT_REGISTRATION_CACHE.values())
        _PERSISTENT_REGISTRATION_CACHE.clear()

    for registration in registrations:
        await registration.aclose()


async def connect_persistent_mcp_registrations(mcp_servers: Mapping[str, Any]) -> None:
    async def connect_one(server_name: str, config: Any) -> None:
        try:
            registration = await get_persistent_mcp_registration_from_config(server_name, config)
            bindings = await registration._get_tool_bindings()
        except Exception as exc:
            logger.exception(
                "MCP server '{}' failed to connect during application startup: {}",
                server_name,
                exc,
            )
            return
        logger.info(
            "MCP server '{}' ready during application startup with {} enabled tools",
            server_name,
            len(bindings),
        )

    await asyncio.gather(*(connect_one(name, config) for name, config in mcp_servers.items()))


async def collect_mcp_tool(
    registration: MCPServerRegistration,
) -> AsyncIterator[tuple[dict[str, Any], Callable[[BaseModel], Any], str]]:
    try:
        bindings = await registration._get_tool_bindings()
    except Exception as exc:
        logger.exception(
            "MCP server '{}' failed to connect/discover tools: {}",
            registration.config.server_name,
            exc,
        )
        return

    for binding in bindings:
        yield binding.schema, binding.func, binding.func.__name__
