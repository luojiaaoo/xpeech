from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Annotated

import httpx
from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from fastapi.responses import Response, StreamingResponse
from yarl import URL

from ...helper import _backend_headers, fetch_file, open_chat_stream, submit_question
from ..dao import User


UserDependency = Callable[..., Awaitable[User] | User]


def create_proxy_router(
    backend_url: str,
    current_user_dependency: UserDependency,
    admin_user_dependency: UserDependency,
) -> APIRouter:
    router = APIRouter(prefix="/api")
    CurrentUser = Annotated[User, Depends(current_user_dependency)]
    AdminUser = Annotated[User, Depends(admin_user_dependency)]

    @router.post("/chat")
    async def proxy_chat(
        request: Request,
        content: Annotated[str, Form()],
        user: CurrentUser,
        session_metadata: Annotated[str, Form()] = "{}",
        timestamp: Annotated[str | None, Form()] = None,
        files: Annotated[list[UploadFile] | None, File()] = None,
    ):
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
            user.session_id,
            user.username,
            content,
            session_metadata,
            str(URL(backend_url) / "chat"),
            timestamp=timestamp,
            files=upload_data,
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
    ):
        upstream = await submit_question(user.session_id, answer, backend_url)
        return Response(
            upstream.content,
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type"),
        )

    @router.get("/files")
    async def proxy_file(
        path: str,
        user: CurrentUser,
    ):
        upstream = await fetch_file(user.session_id, path, backend_url)
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
                headers=_backend_headers(user.session_id),
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
