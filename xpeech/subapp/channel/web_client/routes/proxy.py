import json
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Annotated
from urllib.parse import unquote

import httpx
from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, Request, UploadFile, status
from fastapi.responses import Response, StreamingResponse
from yarl import URL

from .....exceptions import PathProtectionError
from .....utils.channel_jwt import UserJWTAuth
from .....utils.ids import DEFAULT_SESSION_ID, validate_session_id
from .....utils.internal_jwt import create_internal_access_token
from ...helper import BackendAuthParams, BackendJWTParams, BackendUserParams, fetch_file
from ..backend import open_chat, open_chat_sync, submit_question
from ..dao import User

UserDependency = Callable[..., Awaitable[User | UserJWTAuth] | User | UserJWTAuth]


def create_proxy_router(
    backend_url: str,
    current_user_dependency: UserDependency,
    admin_user_dependency: UserDependency,
) -> APIRouter:
    router = APIRouter(prefix="/api")
    CurrentUser = Annotated[User | UserJWTAuth, Depends(current_user_dependency)]
    AdminUser = Annotated[User, Depends(admin_user_dependency)]

    def chat_session_id(
        user: CurrentUser,
        session_id: Annotated[str | None, Header(alias="x-session-id")] = None,
    ) -> str | None:
        if not isinstance(user, UserJWTAuth):
            if session_id not in (None, DEFAULT_SESSION_ID):
                raise HTTPException(422, f"非嵌入聊天只支持主会话（{DEFAULT_SESSION_ID}）")
            return session_id
        if session_id is None:
            raise HTTPException(422, "嵌入聊天必须提供 UUIDv7 格式的 x-session-id")
        try:
            if session_id == DEFAULT_SESSION_ID:
                raise PathProtectionError("Explicit default session is not allowed")
            return validate_session_id(session_id)
        except PathProtectionError:
            raise HTTPException(422, "x-session-id 必须是规范的小写 UUIDv7") from None

    ChatSessionId = Annotated[str | None, Depends(chat_session_id)]

    def request_backend_auth(user: CurrentUser) -> BackendAuthParams:
        if isinstance(user, UserJWTAuth):
            return BackendJWTParams(jwt_token=user.token)
        return BackendUserParams(user_id=user.user_id, channel="web_client")

    RequestBackendAuth = Annotated[BackendAuthParams, Depends(request_backend_auth)]

    @router.post("/chat")
    @router.post("/chat_sync")
    async def proxy_chat(
        request: Request,
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
        if sender_name is None and isinstance(user, UserJWTAuth):
            raise HTTPException(422, "嵌入聊天必须提供 x-sender-name")
        try:
            sender_name = user.username if sender_name is None else unquote(sender_name, errors="strict").strip()
        except UnicodeDecodeError:
            raise HTTPException(422, "Invalid UTF-8 encoding in x-sender-name header") from None
        if not sender_name:
            raise HTTPException(422, "x-sender-name header must not be blank")
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
        synchronous = request.url.path.endswith("/chat_sync")
        if synchronous:
            # 同步执行：后端为每次调用生成独立会话，不传 x-session-id，返回完整 JSON 事件数组。
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

        chat_stream = await open_chat(
            backend_auth,
            sender_name,
            content,
            json.dumps(metadata, ensure_ascii=False),
            str(URL(backend_url) / "chat"),
            files=upload_data,
            session_id=session_id,
        )
        upstream = chat_stream.response
        if upstream.status_code >= 400:
            try:
                body = await upstream.aread()
            finally:
                await chat_stream.aclose()
            return Response(
                body,
                status_code=upstream.status_code,
                media_type=upstream.headers.get("content-type"),
            )

        async def stream() -> AsyncIterator[bytes]:
            try:
                async for chunk in upstream.aiter_bytes():
                    if await request.is_disconnected():
                        break
                    yield chunk
            finally:
                await chat_stream.aclose()

        return StreamingResponse(
            stream(),
            media_type=upstream.headers.get("content-type", "text/event-stream"),
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

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
        async with httpx.AsyncClient(timeout=30) as client:
            upstream = await client.get(
                upstream_url,
                headers={
                    "authorization": f"Bearer {create_internal_access_token()}",
                },
                params=request.query_params.multi_items(),
            )

        headers = {}
        if cache_control := upstream.headers.get("cache-control"):
            headers["Cache-Control"] = cache_control
        if content_disposition := upstream.headers.get("content-disposition"):
            headers["Content-Disposition"] = content_disposition
        return Response(
            upstream.content,
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type"),
            headers=headers,
        )

    return router
