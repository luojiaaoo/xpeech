import base64
import binascii
import codecs
import json
import re
from collections.abc import Awaitable, Callable, Mapping
from types import SimpleNamespace
from typing import Annotated, Any
from urllib.parse import unquote, urlsplit

import anyio
import httpx
from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    Header,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.responses import JSONResponse, Response
from loguru import logger
from yarl import URL

from .....config.settings import settings
from .....exceptions import PathProtectionError
from .....utils.channel_jwt import UserJWTAuth
from .....utils.ids import validate_session_id
from .....utils.internal_jwt import create_internal_access_token
from ...helper import BackendAuthParams, BackendJWTParams, BackendUserParams, fetch_file, proxy_get
from ..backend import chat_status, open_chat, open_chat_sync, stop_chat, submit_question
from ..dao import User
from ..history import JsonlEventHistoryRepository

UserDependency = Callable[..., Awaitable[User | UserJWTAuth] | User | UserJWTAuth]


def _proxy_response(
    upstream: httpx.Response,
    response_headers: Mapping[str, str] | None = None,
) -> Response:
    """Convert a buffered backend response at the web route boundary."""
    headers = {
        name: upstream.headers[name]
        for name in ("cache-control", "content-disposition")
        if name in upstream.headers
    }
    if response_headers:
        headers.update({name.lower(): value for name, value in response_headers.items()})
    return Response(
        upstream.content,
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type"),
        headers=headers,
    )


class _SSEEventCollector:
    """在透传字节的同时增量解析 SSE，不让历史记录改变响应内容。"""

    def __init__(self) -> None:
        self._decoder = codecs.getincrementaldecoder("utf-8")()
        self._buffer = ""
        self.events: list[dict] = []

    def _collect(self, block: str) -> None:
        data = "\n".join(
            line[5:].lstrip(" ")
            for line in block.splitlines()
            if line.startswith("data:")
        )
        if not data:
            return
        try:
            event = json.loads(data)
        except json.JSONDecodeError:
            return
        if isinstance(event, dict):
            self.events.append(event)

    def feed(self, chunk: bytes) -> None:
        self._buffer += self._decoder.decode(chunk)
        blocks = re.split(r"\r?\n\r?\n", self._buffer)
        self._buffer = blocks.pop()
        for block in blocks:
            self._collect(block)

    def finish(self) -> None:
        self._buffer += self._decoder.decode(b"", final=True)
        if self._buffer.strip():
            self._collect(self._buffer)
        self._buffer = ""


def _history_identity(user: User | UserJWTAuth) -> tuple[str, str]:
    if isinstance(user, UserJWTAuth):
        return user.user_id, user.channel
    return user.user_id, user.channel


def _validated_chat_session_id(session_id: str) -> str:
    try:
        return validate_session_id(session_id)
    except PathProtectionError:
        raise HTTPException(422, "x-session-id 必须是规范的小写 UUIDv7") from None


def _validated_sender_name(user: User | UserJWTAuth, sender_name: str | None) -> str:
    if sender_name is None:
        resolved = user.username.strip() if isinstance(user, User) else ""
    else:
        try:
            resolved = unquote(sender_name, errors="strict").strip()
        except UnicodeDecodeError:
            raise HTTPException(422, "Invalid UTF-8 encoding in x-sender-name header") from None
    if not resolved:
        raise HTTPException(422, "x-sender-name header must not be blank")
    return resolved


def _backend_auth(user: User | UserJWTAuth) -> BackendAuthParams:
    if isinstance(user, UserJWTAuth):
        return BackendJWTParams(jwt_token=user.token)
    return BackendUserParams(user_id=user.user_id, channel=user.channel)


def _cookie_websocket_origin_allowed(websocket: WebSocket) -> bool:
    """Cookie-authenticated WebSockets must originate from this web client's own origin."""
    origin = websocket.headers.get("origin")
    if origin is None:
        return True  # Non-browser clients do not send Origin and still require a valid cookie.
    try:
        return urlsplit(origin).netloc == websocket.headers.get("host", "")
    except ValueError:
        return False


def _user_history_event(content: str, files: list[Any] | None) -> dict:
    try:
        inputs = json.loads(content)
    except json.JSONDecodeError:
        inputs = []
    text = "\n".join(
        item["text"].strip()
        for item in inputs
        if isinstance(item, dict) and isinstance(item.get("text"), str) and item["text"].strip()
    ) if isinstance(inputs, list) else ""
    attachments = [f"📎 {file.filename or 'attachment'}" for file in files or ()]
    return {
        "event": "user",
        "context": "\n".join(part for part in (text, *attachments) if part),
    }


def create_proxy_router(
    backend_url: str,
    current_user_dependency: UserDependency,
    admin_user_dependency: UserDependency,
    history: JsonlEventHistoryRepository,
    cookie_name: str,
) -> APIRouter:
    router = APIRouter(prefix="/api")
    CurrentUser = Annotated[User | UserJWTAuth, Depends(current_user_dependency)]
    AdminUser = Annotated[User, Depends(admin_user_dependency)]

    def chat_session_id(
        session_id: Annotated[str, Header(alias="x-session-id")],
    ) -> str:
        return _validated_chat_session_id(session_id)

    ChatSessionId = Annotated[str, Depends(chat_session_id)]

    def request_backend_auth(user: CurrentUser) -> BackendAuthParams:
        return _backend_auth(user)

    RequestBackendAuth = Annotated[BackendAuthParams, Depends(request_backend_auth)]

    @router.get("/history/sessions")
    async def list_history_sessions(user: CurrentUser):
        user_id, channel = _history_identity(user)
        sessions = await history.list_channel_sessions(user_id, channel=channel)
        return JSONResponse(sessions, headers={"Cache-Control": "no-store"})

    @router.get("/history/sessions/{session_id}")
    async def load_history_session(session_id: str, user: CurrentUser):
        try:
            session_id = validate_session_id(session_id)
        except PathProtectionError:
            raise HTTPException(422, "session_id 必须是规范的小写 UUIDv7") from None
        user_id, channel = _history_identity(user)
        events = await history.load(user_id, session_id, channel=channel)
        return JSONResponse(
            {"session_id": session_id, "channel": channel, "events": events},
            headers={"Cache-Control": "no-store"},
        )

    @router.get("/chat/status")
    async def proxy_chat_status(session_id: ChatSessionId, backend_auth: RequestBackendAuth):
        upstream = await chat_status(backend_auth, backend_url, session_id=session_id)
        return Response(
            upstream.content,
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type"),
            headers={"Cache-Control": "no-store"},
        )

    @router.post("/chat/stop")
    async def proxy_stop_chat(session_id: ChatSessionId, backend_auth: RequestBackendAuth):
        upstream = await stop_chat(backend_auth, backend_url, session_id=session_id)
        return Response(
            upstream.content,
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type"),
        )

    @router.post("/chat_sync")
    async def proxy_chat_sync(
        content: Annotated[
            str,
            Form(
                description='消息内容列表，JSON 格式字符串，例如 [{"text":"你好"}]。'
                "在本次消息的 text 中加入完整标记 `::请严格以JSON格式回复::` 可强制模型输出 JSON 对象；"
                "前后的 `::` 均不可省略，仅本次请求生效。模型需配置支持 JSON 输出，否则返回 HTTP 400。"
            ),
        ],
        user: CurrentUser,
        session_id: ChatSessionId,
        backend_auth: RequestBackendAuth,
        sender_name: Annotated[str | None, Header(alias="x-sender-name", min_length=1)] = None,
        session_metadata: Annotated[str, Form()] = "{}",
        files: Annotated[list[UploadFile] | None, File()] = None,
    ):
        sender_name = _validated_sender_name(user, sender_name)
        try:
            metadata = json.loads(session_metadata)
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="Invalid JSON format",
            )
        if not isinstance(metadata, dict):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="session_metadata must be a JSON object",
            )

        metadata.update(await settings.extra_user_info.resolve_session_metadata(user.user_id))
        upload_data = []
        for file in files or ():
            upload_data.append(
                (
                    "files",
                    (
                        file.filename or "attachment",
                        await file.read(),
                        file.content_type,
                    ),
                )
            )
        upstream = await open_chat_sync(
            backend_auth,
            sender_name,
            content,
            json.dumps(metadata, ensure_ascii=False),
            str(URL(backend_url) / "chat_sync"),
            files=upload_data,
        )
        return Response(
            upstream.content,
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type"),
        )

    @router.websocket("/chat/ws")
    async def proxy_chat_websocket(websocket: WebSocket):
        """Forward one session stream; the socket itself correlates all events."""
        await websocket.accept()
        session_id = ""

        async def send_error(detail: str, error_status: int = 500) -> None:
            await websocket.send_json({
                "type": "error",
                "session_id": session_id,
                "status": error_status,
                "detail": detail,
            })

        try:
            payload = await websocket.receive_json()
            if not isinstance(payload, dict) or payload.get("type") != "start":
                await send_error("无效的聊天请求", 422)
                return
            if isinstance(payload.get("session_id"), str):
                session_id = payload["session_id"]
            try:
                token = payload.get("token")
                if token is not None and not isinstance(token, str):
                    raise HTTPException(401, "嵌入 JWT 无效或已过期")
                if token is None:
                    if not _cookie_websocket_origin_allowed(websocket):
                        raise HTTPException(403, "不允许跨站使用登录 Cookie 建立聊天连接")
                    cookie = websocket.cookies.get(cookie_name)
                    authorization = None
                else:
                    cookie = None
                    authorization = f"Bearer {token}"
                user = await current_user_dependency(token=cookie, authorization=authorization)
                session_id = _validated_chat_session_id(session_id)
                sender_name = payload.get("sender_name")
                if sender_name is not None and not isinstance(sender_name, str):
                    raise HTTPException(422, "sender_name 必须是字符串")
                sender_name = _validated_sender_name(user, sender_name)
                content = payload.get("content")
                if not isinstance(content, str):
                    raise HTTPException(422, "content 必须是 JSON 字符串")
                metadata = payload.get("session_metadata", {})
                if not isinstance(metadata, dict):
                    raise HTTPException(422, "session_metadata 必须是 JSON 对象")
                metadata.update(await settings.extra_user_info.resolve_session_metadata(user.user_id))
                file_payloads = payload.get("files", [])
                if not isinstance(file_payloads, list):
                    raise HTTPException(422, "files 必须是数组")
                files = []
                upload_data = []
                for file_payload in file_payloads:
                    if not isinstance(file_payload, dict):
                        raise HTTPException(422, "文件参数必须是对象")
                    filename = file_payload.get("name")
                    media_type = file_payload.get("type", "")
                    encoded = file_payload.get("data")
                    if not isinstance(filename, str) or not isinstance(media_type, str) or not isinstance(encoded, str):
                        raise HTTPException(422, "文件名、类型和内容必须是字符串")
                    try:
                        data = base64.b64decode(encoded, validate=True)
                    except (binascii.Error, ValueError):
                        raise HTTPException(422, "文件内容必须是有效的 base64") from None
                    filename = filename or "attachment"
                    files.append(SimpleNamespace(filename=filename))
                    upload_data.append(("files", (filename, data, media_type)))
            except HTTPException as error:
                await send_error(str(error.detail), error.status_code)
                return

            async with anyio.create_task_group() as tasks:
                async def watch_disconnect() -> None:
                    # Keep reading even while the backend only sends SSE pings.
                    while True:
                        message = await websocket.receive()
                        if message["type"] == "websocket.disconnect":
                            tasks.cancel_scope.cancel()
                            return

                tasks.start_soon(watch_disconnect)
                try:
                    try:
                        chat_stream = await open_chat(
                            _backend_auth(user),
                            sender_name,
                            content,
                            json.dumps(metadata, ensure_ascii=False),
                            str(URL(backend_url) / "chat"),
                            files=upload_data,
                            session_id=session_id,
                        )
                    except Exception:  # noqa: BLE001 - 后端连接失败通过当前 WebSocket 返回
                        logger.exception("WebSocket chat could not open the backend stream")
                        await send_error("无法连接聊天后端")
                        return

                    upstream = chat_stream.response
                    if upstream.status_code >= 400:
                        try:
                            body = await upstream.aread()
                        finally:
                            await chat_stream.aclose()
                        await send_error(
                            body.decode("utf-8", errors="replace") or f"聊天后端返回 {upstream.status_code}",
                            upstream.status_code,
                        )
                        return

                    history_user_id, history_channel = _history_identity(user)
                    user_event = _user_history_event(content, files)
                    collector = _SSEEventCollector()
                    sent_events = 0

                    async def emit_events() -> None:
                        nonlocal sent_events
                        for event in collector.events[sent_events:]:
                            await websocket.send_json({
                                "type": "event",
                                "session_id": session_id,
                                "event": event,
                            })
                        sent_events = len(collector.events)

                    try:
                        async for chunk in upstream.aiter_bytes():
                            try:
                                collector.feed(chunk)
                            except (UnicodeDecodeError, ValueError):
                                logger.exception("Web event history could not parse an SSE chunk")
                            await emit_events()
                        try:
                            collector.finish()
                        except (UnicodeDecodeError, ValueError):
                            logger.exception("Web event history could not finish SSE parsing")
                        await emit_events()
                    finally:
                        with anyio.CancelScope(shield=True):
                            try:
                                await chat_stream.aclose()
                            finally:
                                try:
                                    await history.append(
                                        history_user_id,
                                        session_id,
                                        [user_event, *collector.events],
                                        channel=history_channel,
                                    )
                                except Exception:  # noqa: BLE001 - 历史失败不能改变聊天响应
                                    logger.exception("Web event history could not be saved")
                    await websocket.send_json({"type": "done", "session_id": session_id})
                except (WebSocketDisconnect, RuntimeError):
                    pass
                finally:
                    tasks.cancel_scope.cancel()
        except (WebSocketDisconnect, RuntimeError):
            pass
        except Exception:  # noqa: BLE001 - 将异常转换为当前会话的错误消息
            logger.exception("WebSocket chat request failed")
            try:
                await send_error("聊天请求执行失败")
            except (WebSocketDisconnect, RuntimeError):
                pass
        finally:
            try:
                await websocket.close()
            except RuntimeError:
                pass

    @router.post("/answer_question")
    async def proxy_answer(
        answer: Annotated[str, Form()],
        session_id: ChatSessionId,
        backend_auth: RequestBackendAuth,
    ):
        upstream = await submit_question(
            backend_auth,
            answer,
            backend_url,
            session_id=session_id,
        )
        return Response(
            upstream.content,
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type"),
        )

    @router.get("/chat_command")
    @router.get("/files/list")
    @router.get("/files/search")
    async def proxy_chat_command_choices(request: Request, backend_auth: RequestBackendAuth):
        endpoint = request.url.path.removeprefix("/api/")
        upstream = await proxy_get(
            backend_auth,
            f"{backend_url.rstrip('/')}/{endpoint}",
            params=request.query_params.multi_items(),
        )
        return _proxy_response(upstream, {"Cache-Control": "no-store"})

    @router.get("/files")
    async def proxy_file(
        path: str,
        backend_auth: RequestBackendAuth,
    ):
        upstream = await fetch_file(
            backend_auth,
            path,
            backend_url,
        )
        headers = {}
        if disposition := upstream.headers.get("content-disposition"):
            headers["Content-Disposition"] = disposition
        return Response(
            upstream.content,
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type"),
            headers=headers,
        )

    @router.get("/statistics")
    @router.get("/statistics/{statistics_path:path}")
    async def proxy_statistics(
        request: Request,
        user: AdminUser,
        statistics_path: str = "",
    ):
        """将已登录 Web 用户的统计查询转发到后端统计接口。"""
        upstream_url = str(URL(backend_url) / "statistics")
        if statistics_path:
            upstream_url = f"{upstream_url}/{statistics_path}"
        upstream = await proxy_get(
            BackendJWTParams(jwt_token=create_internal_access_token()),
            upstream_url,
            params=request.query_params.multi_items(),
        )
        return _proxy_response(upstream)

    return router
