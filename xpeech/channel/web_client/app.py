import argparse
import base64
import hashlib
import hmac
import html
import re
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
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    Response,
    StreamingResponse,
)
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
OAUTH2_LOGIN_MINUTES = 5
OAUTH2_MAX_ATTEMPTS = 1_000
PBKDF2_ITERATIONS = 600_000
SESSION_ID_PATTERN = r"^[\w@+-][\w.@+-]*$"
XPEECH_FAVICON = Path(__file__).resolve().parents[3] / "assets" / "favicon.ico"


def _configured_system_name() -> str:
    from ...config.settings import settings

    return settings.llm.system_name.strip() or "AI 助手"


@dataclass(frozen=True)
class OAuth2WebConfig:
    provider_name: str
    client_id: str
    client_secret: str
    authorization_url: str
    token_url: str
    userinfo_url: str
    redirect_uri: str | None
    scopes: tuple[str, ...]
    session_id_claim: str
    username_claim: str
    use_pkce: bool
    token_auth_method: str
    extra_authorization_params: dict[str, str]
    auto_create_users: bool = False
    display_type: str = "qrcode"


@dataclass
class OAuth2LoginAttempt:
    state: str
    poll_token_hash: str
    redirect_uri: str
    code_verifier: str | None
    expires_at: datetime
    user_session_id: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class WebConfig:
    backend_url: str
    database_path: Path
    static_dir: Path
    system_name: str
    cookie_name: str = "xpeech_session"
    oauth2: OAuth2WebConfig | None = None


class LoginBody(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)
    password: str = Field(min_length=1, max_length=256)


class PasswordChangeBody(BaseModel):
    new_password: str = Field(min_length=8, max_length=256)


class OAuth2PollBody(BaseModel):
    login_id: str = Field(min_length=1, max_length=128)
    poll_token: str = Field(min_length=1, max_length=256)


class UserBody(BaseModel):
    session_id: str = Field(
        min_length=1,
        max_length=128,
        pattern=SESSION_ID_PATTERN,
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
        pattern=SESSION_ID_PATTERN,
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


def _oauth2_claim(payload: dict[str, object], path: str) -> object | None:
    """Resolve a dotted OAuth2 claim path such as ``data.employee_no``."""
    value: object = payload
    for part in path.split("."):
        if not part or not isinstance(value, dict):
            return None
        value = value.get(part)
        if value is None:
            return None
    return value


def _web_session_id(user: User) -> str:
    return user.session_id


def create_app(config: WebConfig) -> FastAPI:
    dao = WebClientDAO(config.database_path)
    oauth2_attempts: dict[str, OAuth2LoginAttempt] = {}
    oauth2_state_index: dict[str, str] = {}

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

    def discard_oauth2_attempt(login_id: str) -> None:
        attempt = oauth2_attempts.pop(login_id, None)
        if attempt is not None:
            oauth2_state_index.pop(attempt.state, None)

    def prune_oauth2_attempts() -> None:
        now = datetime.now(UTC)
        for login_id, attempt in list(oauth2_attempts.items()):
            if attempt.expires_at <= now:
                discard_oauth2_attempt(login_id)

    async def create_login_session(user: User, response: Response) -> None:
        if user.id is None:
            raise RuntimeError("数据库用户缺少主键")
        token = secrets.token_urlsafe(32)
        await dao.create_session(
            token_hash=hashlib.sha256(token.encode()).hexdigest(),
            user_id=user.id,
            expires_at=datetime.now(UTC) + timedelta(days=SESSION_DAYS),
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

    def oauth2_result_page(success: bool, detail: str) -> HTMLResponse:
        title = "授权成功" if success else "授权失败"
        safe_title = html.escape(title)
        safe_detail = html.escape(detail)
        content = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>{safe_title}</title><style>
body{{margin:0;min-height:100vh;display:grid;place-items:center;font-family:system-ui,sans-serif;background:#f5f7ff;color:#18213d}}
main{{width:min(420px,calc(100% - 40px));padding:36px;text-align:center;border:1px solid #e1e6f5;border-radius:20px;background:#fff;box-shadow:0 18px 60px #30456f1a}}
h1{{font-size:24px;margin:0 0 12px}}p{{margin:0;color:#646b78;line-height:1.7}}
</style></head><body><main><h1>{safe_title}</h1><p>{safe_detail}</p></main></body></html>"""
        return HTMLResponse(
            content,
            status_code=status.HTTP_200_OK if success else status.HTTP_400_BAD_REQUEST,
            headers={
                "Cache-Control": "no-store",
                "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'",
            },
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
        return {
            "system_name": config.system_name,
            "oauth2": (
                {
                    "enabled": True,
                    "provider_name": config.oauth2.provider_name,
                    "display_type": config.oauth2.display_type,
                }
                if config.oauth2 is not None
                else {
                    "enabled": False,
                    "provider_name": "OAuth2",
                    "display_type": "qrcode",
                }
            ),
        }

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
        await create_login_session(user, response)
        return _public_user(user)

    @app.post("/api/auth/oauth2/qr")
    async def create_oauth2_qr(request: Request):
        oauth2 = config.oauth2
        if oauth2 is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "OAuth2 登录未启用")

        prune_oauth2_attempts()
        if len(oauth2_attempts) >= OAUTH2_MAX_ATTEMPTS:
            oldest_login_id = min(
                oauth2_attempts,
                key=lambda current_id: oauth2_attempts[current_id].expires_at,
            )
            discard_oauth2_attempt(oldest_login_id)
        login_id = secrets.token_urlsafe(24)
        poll_token = secrets.token_urlsafe(32)
        state_value = secrets.token_urlsafe(32)
        code_verifier = secrets.token_urlsafe(64) if oauth2.use_pkce else None
        redirect_uri = oauth2.redirect_uri or str(request.url_for("oauth2_callback"))
        oauth2_attempts[login_id] = OAuth2LoginAttempt(
            state=state_value,
            poll_token_hash=hashlib.sha256(poll_token.encode()).hexdigest(),
            redirect_uri=redirect_uri,
            code_verifier=code_verifier,
            expires_at=datetime.now(UTC) + timedelta(minutes=OAUTH2_LOGIN_MINUTES),
        )
        oauth2_state_index[state_value] = login_id

        params = {
            **oauth2.extra_authorization_params,
            "response_type": "code",
            "client_id": oauth2.client_id,
            "redirect_uri": redirect_uri,
            "state": state_value,
        }
        if oauth2.scopes:
            params["scope"] = " ".join(oauth2.scopes)
        if code_verifier is not None:
            challenge = hashlib.sha256(code_verifier.encode()).digest()
            params["code_challenge"] = (
                base64.urlsafe_b64encode(challenge).rstrip(b"=").decode()
            )
            params["code_challenge_method"] = "S256"

        return {
            "authorization_url": str(URL(oauth2.authorization_url).update_query(params)),
            "login_id": login_id,
            "poll_token": poll_token,
            "expires_in": OAUTH2_LOGIN_MINUTES * 60,
        }

    @app.get("/api/auth/oauth2/callback", name="oauth2_callback")
    async def oauth2_callback(
        state: str | None = None,
        code: str | None = None,
        error: str | None = None,
        error_description: str | None = None,
    ):
        oauth2 = config.oauth2
        if oauth2 is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "OAuth2 登录未启用")

        prune_oauth2_attempts()
        login_id = oauth2_state_index.get(state or "")
        attempt = oauth2_attempts.get(login_id or "")
        if attempt is None:
            return oauth2_result_page(False, "登录请求已失效，请返回登录页重试。")
        if attempt.user_session_id is not None:
            return oauth2_result_page(True, "授权已完成，请返回原设备。")
        if attempt.error is not None:
            return oauth2_result_page(False, attempt.error)
        if error or not code:
            attempt.error = error_description or error or "OAuth2 授权未完成"
            return oauth2_result_page(False, attempt.error)

        token_data = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": attempt.redirect_uri,
        }
        token_auth: tuple[str, str] | None = None
        if oauth2.token_auth_method == "client_secret_basic":
            token_auth = (oauth2.client_id, oauth2.client_secret)
        else:
            token_data["client_id"] = oauth2.client_id
            token_data["client_secret"] = oauth2.client_secret
        if attempt.code_verifier is not None:
            token_data["code_verifier"] = attempt.code_verifier

        try:
            async with httpx.AsyncClient(timeout=15) as client:
                token_response = await client.post(
                    oauth2.token_url,
                    data=token_data,
                    headers={"Accept": "application/json"},
                    auth=token_auth,
                )
                if token_response.status_code >= 400:
                    raise ValueError("OAuth2 token endpoint rejected the request")
                token_payload = token_response.json()
                if not isinstance(token_payload, dict):
                    raise ValueError("OAuth2 token response must be an object")
                access_token = token_payload.get("access_token")
                if not isinstance(access_token, str) or not access_token:
                    raise ValueError("OAuth2 token response is missing access_token")
                userinfo_response = await client.get(
                    oauth2.userinfo_url,
                    headers={
                        "Accept": "application/json",
                        "Authorization": f"Bearer {access_token}",
                    },
                )
                if userinfo_response.status_code >= 400:
                    raise ValueError("OAuth2 userinfo endpoint rejected the request")
                userinfo = userinfo_response.json()
                if not isinstance(userinfo, dict):
                    raise ValueError("OAuth2 userinfo response must be an object")
        except (httpx.HTTPError, TypeError, ValueError):
            attempt.error = "OAuth2 服务请求失败，请返回登录页重试。"
            return oauth2_result_page(False, attempt.error)

        claim = _oauth2_claim(userinfo, oauth2.session_id_claim)
        session_id = str(claim).strip() if claim is not None else ""
        if (
            not session_id
            or len(session_id) > 128
            or re.fullmatch(SESSION_ID_PATTERN, session_id) is None
        ):
            attempt.error = f"OAuth2 用户信息缺少有效的 {oauth2.session_id_claim}"
            return oauth2_result_page(False, attempt.error)

        user = await dao.get_user_by_session_id(session_id)
        if user is None and oauth2.auto_create_users:
            username_claim = _oauth2_claim(userinfo, oauth2.username_claim)
            username = re.sub(
                r"[^\w.@+-]+",
                "_",
                str(username_claim or session_id).strip(),
            ).strip("_")[:64] or session_id
            try:
                user = await dao.create_user(
                    username=username,
                    session_id=session_id,
                    password_hash=_password_hash(secrets.token_urlsafe(32)),
                    is_admin=False,
                )
            except DuplicateSessionIdError:
                user = await dao.get_user_by_session_id(session_id)
        if user is None:
            attempt.error = "OAuth2 账号未绑定本地账号"
            return oauth2_result_page(False, attempt.error)
        if not user.is_active:
            attempt.error = "账号已停用"
            return oauth2_result_page(False, attempt.error)

        attempt.user_session_id = user.session_id
        return oauth2_result_page(True, "请返回原设备，登录页将自动进入系统。")

    @app.post("/api/auth/oauth2/poll")
    async def poll_oauth2_login(body: OAuth2PollBody, response: Response):
        oauth2 = config.oauth2
        if oauth2 is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "OAuth2 登录未启用")

        prune_oauth2_attempts()
        attempt = oauth2_attempts.get(body.login_id)
        supplied_poll_token_hash = hashlib.sha256(body.poll_token.encode()).hexdigest()
        if attempt is None or not hmac.compare_digest(
            supplied_poll_token_hash,
            attempt.poll_token_hash,
        ):
            raise HTTPException(status.HTTP_404_NOT_FOUND, "二维码已失效")
        if attempt.error is not None:
            detail = attempt.error
            discard_oauth2_attempt(body.login_id)
            raise HTTPException(status.HTTP_400_BAD_REQUEST, detail)
        if attempt.user_session_id is None:
            return {"status": "pending"}

        user = await dao.get_user_by_session_id(attempt.user_session_id)
        if user is None or not user.is_active:
            discard_oauth2_attempt(body.login_id)
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "账号不存在或已停用")
        await create_login_session(user, response)
        discard_oauth2_attempt(body.login_id)
        return {"status": "approved", "user": _public_user(user)}

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
    oauth2_settings = settings.web_client.oauth2
    oauth2_config = (
        OAuth2WebConfig(
            provider_name=oauth2_settings.provider_name,
            display_type=oauth2_settings.display_type,
            client_id=oauth2_settings.client_id,
            client_secret=oauth2_settings.client_secret,
            authorization_url=oauth2_settings.authorization_url,
            token_url=oauth2_settings.token_url,
            userinfo_url=oauth2_settings.userinfo_url,
            redirect_uri=oauth2_settings.redirect_uri,
            scopes=tuple(oauth2_settings.scopes),
            session_id_claim=oauth2_settings.session_id_claim,
            username_claim=oauth2_settings.username_claim,
            auto_create_users=oauth2_settings.auto_create_users,
            use_pkce=oauth2_settings.use_pkce,
            token_auth_method=oauth2_settings.token_auth_method,
            extra_authorization_params=dict(
                oauth2_settings.extra_authorization_params
            ),
        )
        if oauth2_settings.enabled
        else None
    )
    config = WebConfig(
        backend_url=backend_url.rstrip("/"),
        database_path=settings.web_client.database_path.resolve(),
        static_dir=static_dir.resolve(),
        system_name=_configured_system_name(),
        cookie_name=settings.web_client.cookie_name,
        oauth2=oauth2_config,
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
