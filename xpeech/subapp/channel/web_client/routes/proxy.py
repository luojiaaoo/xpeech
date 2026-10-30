import hashlib
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Annotated

import httpx
from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, Request, UploadFile, status
from fastapi.responses import Response, StreamingResponse
from yarl import URL

from .....exceptions import PathProtectionError
from .....utils.channel_jwt import ChannelJWTAuth
from .....utils.ids import DEFAULT_SESSION_ID, validate_session_id
from ...helper import _backend_headers, fetch_file, open_chat_stream, submit_question
from ..dao import User, WebClientDAO
from ..session import EMBED_TOKEN_PREFIX, embedded_channel_auth, session_token

UserDependency = Callable[..., Awaitable[User] | User]


def create_proxy_router(
    backend_url: str,
    backend_auth: ChannelJWTAuth,
    current_user_dependency: UserDependency,
    admin_user_dependency: UserDependency,
    dao: WebClientDAO,
) -> APIRouter:
    router = APIRouter(prefix="/api")
    CurrentUser = Annotated[User, Depends(current_user_dependency)]
    AdminUser = Annotated[User, Depends(admin_user_dependency)]

    def chat_session_id(
        request: Request,
        session_id: Annotated[str | None, Header(alias="x-session-id")] = None,
    ) -> str | None:
        token = session_token(None, request.headers.get("authorization"))
        if session_id is None:
            if token and token.startswith(EMBED_TOKEN_PREFIX):
                raise HTTPException(422, "嵌入聊天必须提供 UUIDv7 格式的 x-session-id")
            return None
        try:
            if session_id == DEFAULT_SESSION_ID:
                raise PathProtectionError("Explicit default session is not allowed")
            return validate_session_id(session_id)
        except PathProtectionError:
            raise HTTPException(422, "x-session-id 必须是规范的小写 UUIDv7") from None

    ChatSessionId = Annotated[str | None, Depends(chat_session_id)]

    async def request_backend_auth(
        request: Request,
        channel_name: Annotated[str | None, Header(alias="x-channel-name")] = None,
    ) -> ChannelJWTAuth:
        token = session_token(None, request.headers.get("authorization"))
        if token and token.startswith(EMBED_TOKEN_PREFIX):
            bound_channel = await dao.get_session_channel(hashlib.sha256(token.encode()).hexdigest())
            if bound_channel is None or (channel_name is not None and channel_name != bound_channel):
                raise HTTPException(403, "嵌入请求的通道与已验证的 JWT 不一致")
            return embedded_channel_auth(bound_channel)
        if channel_name is not None:
            raise HTTPException(403, "仅嵌入会话可以指定 channel_name")
        return backend_auth

    RequestBackendAuth = Annotated[ChannelJWTAuth, Depends(request_backend_auth)]

    @router.post("/chat")
    async def proxy_chat(
        request: Request,
        content: Annotated[str, Form()],
        user: CurrentUser,
        session_id: ChatSessionId,
        channel_auth: RequestBackendAuth,
        session_metadata: Annotated[str, Form()] = "{}",
        files: Annotated[list[UploadFile] | None, File()] = None,
        json_output: Annotated[
            bool,
            Form(
                description='为大模型请求添加 response_format={"type":"json_object"}，不修改提示词；'
                "模型未配置支持 JSON 输出时返回 HTTP 400，成功响应仍使用 SSE 流"
            ),
        ] = False,
    ):
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
        chat_stream = await open_chat_stream(
            user.user_id,
            user.username,
            content,
            json.dumps(metadata, ensure_ascii=False),
            str(URL(backend_url) / "chat"),
            channel_auth,
            files=upload_data,
            json_output=json_output,
            session_id=session_id,
        )
        upstream = chat_stream.response
        if upstream.status_code >= 400:
            body = await upstream.aread()
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
        user: CurrentUser,
        session_id: ChatSessionId,
        channel_auth: RequestBackendAuth,
    ):
        upstream = await submit_question(
            user.user_id,
            answer,
            backend_url,
            channel_auth,
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
        user: CurrentUser,
        channel_auth: RequestBackendAuth,
    ):
        upstream = await fetch_file(
            user.user_id,
            path,
            backend_url,
            channel_auth,
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
        channel_auth: RequestBackendAuth,
        statistics_path: str = "",
    ):
        """将已登录 Web 用户的统计查询转发到后端统计接口。"""
        upstream_url = str(URL(backend_url) / "statistics")
        if statistics_path:
            upstream_url = f"{upstream_url}/{statistics_path}"
        async with httpx.AsyncClient(timeout=30) as client:
            upstream = await client.get(
                upstream_url,
                headers=_backend_headers(user.user_id, channel_auth),
                params=request.query_params.multi_items(),
            )

        headers = {}
        if cache_control := upstream.headers.get("cache-control"):
            headers["Cache-Control"] = cache_control
        return Response(
            upstream.content,
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type"),
            headers=headers,
        )

    return router
