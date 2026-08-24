import argparse
import hashlib
import hmac
import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated

import httpx
import uvicorn
from fastapi import Cookie, Depends, FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from yarl import URL

from ..helper import _backend_headers, fetch_file, open_chat_stream, submit_question
from .dao import (
    DuplicateSessionIdError,
    ProtectedAdminDeletionError,
    ProtectedAdminIdentityError,
    User,
    WebClientDAO,
)

SESSION_DAYS = 7
PBKDF2_ITERATIONS = 600_000
XPEECH_FAVICON = Path(__file__).resolve().parents[3] / "assets" / "favicon.ico"


def _configured_system_name() -> str:
    from ...config.settings import settings

    return settings.llm.system_name.strip() or "AI 助手"


@dataclass(frozen=True)
class WebConfig:
    backend_url: str
    database_path: Path
    static_dir: Path
    system_name: str
    cookie_name: str = "xpeech_session"


class LoginBody(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)
    password: str = Field(min_length=1, max_length=256)


class PasswordChangeBody(BaseModel):
    new_password: str = Field(min_length=8, max_length=256)


class UserBody(BaseModel):
    session_id: str = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[\w@+-][\w.@+-]*$",
    )
    username: str = Field(min_length=1, max_length=64, pattern=r"^[\w.@+-]+$")
    password: str = Field(min_length=8, max_length=256)
    is_admin: bool = False


class UserUpdateBody(BaseModel):
    username: str | None = Field(
        default=None,
        min_length=1,
        max_length=64,
        pattern=r"^[\w.@+-]+$",
    )
    session_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        pattern=r"^[\w@+-][\w.@+-]*$",
    )
    password: str | None = Field(default=None, min_length=8, max_length=256)
    is_admin: bool | None = None
    is_active: bool | None = None


def _password_hash(password: str, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, PBKDF2_ITERATIONS)
    return f"pbkdf2_sha256${PBKDF2_ITERATIONS}${salt.hex()}${digest.hex()}"


def _password_matches(password: str, encoded: str) -> bool:
    try:
        algorithm, iterations, salt_hex, expected_hex = encoded.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        actual = hashlib.pbkdf2_hmac(
            "sha256", password.encode(), bytes.fromhex(salt_hex), int(iterations)
        )
        return hmac.compare_digest(actual, bytes.fromhex(expected_hex))
    except (ValueError, TypeError):
        return False


def _public_user(user: User) -> dict[str, object]:
    return {
        "id": user.id,
        "session_id": user.session_id,
        "username": user.username,
        "is_admin": user.is_admin,
        "is_active": user.is_active,
        "created_at": user.created_at.isoformat(),
    }


def _web_session_id(user: User) -> str:
    return user.session_id


def create_app(config: WebConfig) -> FastAPI:
    dao = WebClientDAO(config.database_path)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        await dao.initialize(lambda: _password_hash("admin123456"))
        try:
            yield
        finally:
            await dao.close()

    app = FastAPI(
        title=f"{config.system_name} Web",
        docs_url=None,
        redoc_url=None,
        lifespan=lifespan,
    )

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon():
        return FileResponse(
            XPEECH_FAVICON,
            media_type="image/x-icon",
            headers={"Cache-Control": "public, max-age=86400"},
        )

    @app.get("/api/config")
    async def public_config():
        return {"system_name": config.system_name}

    async def current_user(
        token: Annotated[str | None, Cookie(alias=config.cookie_name)] = None,
    ) -> User:
        if not token:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "请先登录")
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        user = await dao.get_user_for_session(token_hash)
        if user is None:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "登录已失效")
        return user

    CurrentUser = Annotated[User, Depends(current_user)]

    def admin_user(user: CurrentUser) -> User:
        if not user.is_admin:
            raise HTTPException(status.HTTP_403_FORBIDDEN, "需要管理员权限")
        return user

    AdminUser = Annotated[User, Depends(admin_user)]

    @app.post("/api/auth/login")
    async def login(body: LoginBody, response: Response):
        user = await dao.get_user_by_session_id(body.session_id)
        if (
            user is None
            or not user.is_active
            or not _password_matches(body.password, user.password_hash)
        ):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "会话 ID 或密码错误")
        if user.id is None:
            raise RuntimeError("数据库用户缺少主键")
        token = secrets.token_urlsafe(32)
        expires = datetime.now(UTC) + timedelta(days=SESSION_DAYS)
        await dao.create_session(
            token_hash=hashlib.sha256(token.encode()).hexdigest(),
            user_id=user.id,
            expires_at=expires,
        )
        response.set_cookie(
            config.cookie_name,
            token,
            max_age=SESSION_DAYS * 86400,
            httponly=True,
            secure=False,
            samesite="lax",
            path="/",
        )
        return _public_user(user)

    @app.post("/api/auth/logout", status_code=204)
    async def logout(
        response: Response,
        token: Annotated[str | None, Cookie(alias=config.cookie_name)] = None,
    ):
        if token:
            await dao.delete_session(hashlib.sha256(token.encode()).hexdigest())
        response.delete_cookie(config.cookie_name, path="/")

    @app.get("/api/auth/me")
    async def me(user: CurrentUser):
        return _public_user(user)

    @app.patch("/api/auth/password", status_code=204)
    async def change_password(
        body: PasswordChangeBody,
        user: CurrentUser,
        token: Annotated[str | None, Cookie(alias=config.cookie_name)] = None,
    ):
        if user.id is None or token is None:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "登录已失效")
        changed = await dao.change_password(
            user.id,
            password_hash=_password_hash(body.new_password),
            keep_token_hash=hashlib.sha256(token.encode()).hexdigest(),
        )
        if not changed:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "用户不存在")

    @app.get("/api/admin/users")
    async def list_users(_admin: AdminUser):
        return [_public_user(user) for user in await dao.list_users()]

    @app.post("/api/admin/users", status_code=201)
    async def create_user(body: UserBody, _admin: AdminUser):
        try:
            user = await dao.create_user(
                username=body.username,
                session_id=body.session_id,
                password_hash=_password_hash(body.password),
                is_admin=body.is_admin,
            )
        except DuplicateSessionIdError:
            raise HTTPException(status.HTTP_409_CONFLICT, "会话 ID 已存在")
        return _public_user(user)

    @app.patch("/api/admin/users/{user_id}")
    async def update_user(
        user_id: int,
        body: UserUpdateBody,
        admin: AdminUser,
    ):
        values = body.model_dump(exclude_none=True)
        if user_id == admin.id and values.get("is_active") is False:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "不能停用当前管理员")
        if not values:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "没有可更新字段")
        try:
            user = await dao.update_user(
                user_id,
                username=values.get("username"),
                session_id=values.get("session_id"),
                password_hash=(
                    _password_hash(values["password"])
                    if "password" in values
                    else None
                ),
                is_admin=values.get("is_admin"),
                is_active=values.get("is_active"),
            )
        except DuplicateSessionIdError:
            raise HTTPException(status.HTTP_409_CONFLICT, "会话 ID 已存在")
        except ProtectedAdminIdentityError:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                "默认管理员的用户名和会话 ID 不可修改",
            )
        if user is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "用户不存在")
        return _public_user(user)

    @app.delete("/api/admin/users/{user_id}", status_code=204)
    async def delete_user(user_id: int, admin: AdminUser):
        if user_id == admin.id:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "不能删除当前管理员")
        try:
            deleted = await dao.delete_user(user_id)
        except ProtectedAdminDeletionError:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                "默认管理员不可删除",
            )
        if not deleted:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "用户不存在")

    @app.post("/api/chat")
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
            upload_data.append(("files", (file.filename or "attachment", await file.read(), file.content_type)))
        chat_stream = await open_chat_stream(
            _web_session_id(user),
            user.username,
            content,
            session_metadata,
            str(URL(config.backend_url) / "chat"),
            timestamp=timestamp,
            files=upload_data,
        )
        upstream = chat_stream.response
        if upstream.status_code >= 400:
            body = await upstream.aread()
            await chat_stream.aclose()
            return Response(body, status_code=upstream.status_code, media_type=upstream.headers.get("content-type"))

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

    @app.post("/api/answer_question")
    async def proxy_answer(
        answer: Annotated[str, Form()],
        user: CurrentUser,
    ):
        upstream = await submit_question(_web_session_id(user), answer, config.backend_url)
        return Response(upstream.content, status_code=upstream.status_code, media_type=upstream.headers.get("content-type"))

    @app.get("/api/files")
    async def proxy_file(
        path: str,
        user: CurrentUser,
    ):
        session_id = _web_session_id(user)
        upstream = await fetch_file(session_id, path, config.backend_url)
        headers = {}
        if disposition := upstream.headers.get("content-disposition"):
            headers["Content-Disposition"] = disposition
        return Response(
            upstream.content,
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type"),
            headers=headers,
        )

    @app.get("/api/statistics")
    @app.get("/api/statistics/{statistics_path:path}")
    async def proxy_statistics(
        request: Request,
        user: AdminUser,
        statistics_path: str = "",
    ):
        """将已登录 Web 用户的统计查询转发到后端统计接口。"""
        upstream_url = str(URL(config.backend_url) / "statistics")
        if statistics_path:
            upstream_url = f"{upstream_url}/{statistics_path}"
        async with httpx.AsyncClient(timeout=30) as client:
            upstream = await client.get(
                upstream_url,
                headers=_backend_headers(_web_session_id(user)),
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

    if config.static_dir.exists():
        assets = config.static_dir / "assets"
        if assets.exists():
            app.mount("/assets", StaticFiles(directory=assets), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        async def spa(path: str):
            candidate = (config.static_dir / path).resolve()
            if path and candidate.is_relative_to(config.static_dir.resolve()) and candidate.is_file():
                return FileResponse(candidate)
            return FileResponse(config.static_dir / "index.html")
    else:
        @app.get("/", include_in_schema=False)
        async def missing_frontend():
            return JSONResponse(
                {"detail": "前端尚未构建，请先在 web_client/frontend 运行 npm install && npm run build"},
                status_code=503,
            )

    return app


def run(
    host: str = "0.0.0.0",
    port: int = 7939,
    backend_url: str = "http://127.0.0.1:7878",
    dev_frontend: bool = False,
) -> None:
    from ...config.settings import settings

    frontend = Path(__file__).parent / "frontend"
    static_dir = frontend / ("dist" if not dev_frontend else "dist")
    config = WebConfig(
        backend_url=backend_url.rstrip("/"),
        database_path=settings.web_client.database_path.resolve(),
        static_dir=static_dir.resolve(),
        system_name=_configured_system_name(),
        cookie_name=settings.web_client.cookie_name,
    )
    uvicorn.run(create_app(config), host=host, port=port)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"Run the authenticated {_configured_system_name()} web client."
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7939)
    parser.add_argument("--backend-url", default="http://127.0.0.1:7878")
    args = parser.parse_args()
    run(args.host, args.port, args.backend_url)


if __name__ == "__main__":
    main()
