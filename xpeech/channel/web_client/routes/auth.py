import asyncio
import base64
import hashlib
import hmac
import html
import re
import secrets
import shlex
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Annotated

import httpx
from fastapi import APIRouter, Cookie, Depends, HTTPException, Query, Request, Response, status
from fastapi.responses import HTMLResponse, RedirectResponse
from yarl import URL

from ..dao import DuplicateSessionIdError, User, WebClientDAO
from ..models import (
    INJECT_PROMPT_STATE_MAX_LENGTH,
    INJECT_PROMPT_STATE_MIN_LENGTH,
    INJECT_PROMPT_STATE_PATTERN,
    SESSION_ID_PATTERN,
    InjectPromptWebConfig,
    LoginBody,
    OAuth2CreateBody,
    OAuth2PollBody,
    PasswordChangeBody,
    WebConfig,
    public_user,
)

SESSION_DAYS = 7
OAUTH2_LOGIN_MINUTES = 5
OAUTH2_MAX_ATTEMPTS = 1_000

PasswordHash = Callable[[str], str]
PasswordMatches = Callable[[str, str], bool]
CurrentUserDependency = Callable[..., Awaitable[User]]


@dataclass
class OAuth2LoginAttempt:
    state: str
    poll_token_hash: str
    redirect_uri: str
    code_verifier: str | None
    expires_at: datetime
    inject_prompt_state: str | None = None
    user_session_id: str | None = None
    error: str | None = None


def oauth2_claim(payload: dict[str, object], path: str) -> object | None:
    """Resolve a dotted OAuth2 claim path such as ``data.employee_no``."""
    value: object = payload
    for part in path.split("."):
        if not part or not isinstance(value, dict):
            return None
        value = value.get(part)
        if value is None:
            return None
    return value


async def resolve_injected_prompt(
    config: InjectPromptWebConfig,
    state: str,
) -> str:
    try:
        command = shlex.split(config.command_template)
    except ValueError as error:
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY,
            "inject_prompt.command_template 格式无效",
        ) from error
    if not command:
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY,
            "inject_prompt.command_template 不能为空",
        )

    has_state_placeholder = any("${state}" in argument or "$state" in argument for argument in command)
    if not has_state_placeholder:
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY,
            "inject_prompt.command_template 缺少 ${state} 或 $state",
        )
    # Replace placeholders after splitting arguments and never invoke a shell, so
    # state cannot add options, pipelines, redirects, or another command.
    command = [argument.replace("${state}", state).replace("$state", state) for argument in command]
    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except OSError as error:
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY,
            "提示词命令无法启动",
        ) from error

    try:
        stdout, _ = await asyncio.wait_for(process.communicate(), timeout=10)
    except TimeoutError as error:
        process.kill()
        await process.wait()
        raise HTTPException(
            status.HTTP_504_GATEWAY_TIMEOUT,
            "提示词命令执行超时",
        ) from error
    if process.returncode != 0:
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY,
            "提示词命令执行失败",
        )
    try:
        prompt = stdout.decode("utf-8").strip()
    except UnicodeDecodeError as error:
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY,
            "提示词命令必须输出 UTF-8 文本",
        ) from error
    if not prompt:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            "未找到对应的提示词",
        )
    return prompt


def _oauth2_result_page(success: bool, detail: str) -> HTMLResponse:
    title = "授权成功" if success else "授权失败"
    safe_title = html.escape(title)
    safe_detail = html.escape(detail)
    close_script = ""
    countdown = ""
    content_security_policy = "default-src 'none'; style-src 'unsafe-inline'"
    if success:
        script_nonce = secrets.token_urlsafe(16)
        countdown = '<p id="close-countdown" aria-live="polite">3 秒后自动关闭</p>'
        close_script = (
            f'<script nonce="{script_nonce}">'
            "let seconds=3;"
            "const countdown=document.getElementById('close-countdown');"
            "const timer=window.setInterval(()=>{"
            "seconds-=1;"
            "if(seconds>0){countdown.textContent=`${seconds} 秒后自动关闭`;return;}"
            "window.clearInterval(timer);window.close();"
            "},1000);"
            "</script>"
        )
        content_security_policy += f"; script-src 'nonce-{script_nonce}'"
    content = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>{safe_title}</title><style>
body{{margin:0;min-height:100vh;display:grid;place-items:center;font-family:system-ui,sans-serif;background:#f5f7ff;color:#18213d}}
main{{width:min(420px,calc(100% - 40px));padding:36px;text-align:center;border:1px solid #e1e6f5;border-radius:20px;background:#fff;box-shadow:0 18px 60px #30456f1a}}
h1{{font-size:24px;margin:0 0 12px}}p{{margin:0;color:#646b78;line-height:1.7}}#close-countdown{{margin-top:12px;font-variant-numeric:tabular-nums}}
</style></head><body><main><h1>{safe_title}</h1><p>{safe_detail}</p>{countdown}</main>{close_script}</body></html>"""
    return HTMLResponse(
        content,
        status_code=status.HTTP_200_OK if success else status.HTTP_400_BAD_REQUEST,
        headers={
            "Cache-Control": "no-store",
            "Content-Security-Policy": content_security_policy,
        },
    )


def create_auth_router(
    config: WebConfig,
    dao: WebClientDAO,
    current_user_dependency: CurrentUserDependency,
    password_hash: PasswordHash,
    password_matches: PasswordMatches,
) -> APIRouter:
    router = APIRouter(prefix="/api/auth")
    oauth2_attempts: dict[str, OAuth2LoginAttempt] = {}
    oauth2_state_index: dict[str, str] = {}
    CurrentUser = Annotated[User, Depends(current_user_dependency)]

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

    @router.post("/login")
    async def login(body: LoginBody, response: Response):
        user = await dao.get_user_by_session_id(body.session_id)
        if user is None or not user.is_active or not password_matches(body.password, user.password_hash):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "会话 ID 或密码错误")
        await create_login_session(user, response)
        return public_user(user)

    @router.get("/inject-prompt")
    async def inject_prompt(
        state_value: Annotated[
            str,
            Query(
                alias="state",
                min_length=INJECT_PROMPT_STATE_MIN_LENGTH,
                max_length=INJECT_PROMPT_STATE_MAX_LENGTH,
                pattern=INJECT_PROMPT_STATE_PATTERN,
            ),
        ],
        response: Response,
        user: CurrentUser,
    ):
        del user
        if not config.inject_prompt.enabled:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "提示词注入未启用")
        response.headers["Cache-Control"] = "no-store"
        return {
            "user_prefix": await resolve_injected_prompt(
                config.inject_prompt,
                state_value,
            )
        }

    @router.post("/oauth2/qr")
    async def create_oauth2_qr(
        request: Request,
        body: OAuth2CreateBody | None = None,
    ):
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
        inject_prompt_state = body.state if body is not None else None
        if inject_prompt_state is not None and not config.inject_prompt.enabled:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "提示词注入未启用")
        # 入口 state 只用于查询提示词；随机 token 后缀保证每次 OAuth2 授权都有唯一的回调 state。
        state_value = (
            f"{inject_prompt_state}_-_{secrets.token_urlsafe(32)}"
            if inject_prompt_state is not None
            else secrets.token_urlsafe(32)
        )

        login_id = secrets.token_urlsafe(24)
        poll_token = secrets.token_urlsafe(32)
        code_verifier = secrets.token_urlsafe(64) if oauth2.use_pkce else None
        redirect_uri = oauth2.redirect_uri or str(request.url_for("oauth2_callback"))
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
            params["code_challenge"] = base64.urlsafe_b64encode(challenge).rstrip(b"=").decode()
            params["code_challenge_method"] = "S256"

        authorization_url = str(URL(oauth2.authorization_url).update_query(params))
        oauth2_attempts[login_id] = OAuth2LoginAttempt(
            state=state_value,
            poll_token_hash=hashlib.sha256(poll_token.encode()).hexdigest(),
            redirect_uri=redirect_uri,
            code_verifier=code_verifier,
            expires_at=datetime.now(UTC) + timedelta(minutes=OAUTH2_LOGIN_MINUTES),
            inject_prompt_state=inject_prompt_state,
        )
        oauth2_state_index[state_value] = login_id

        return {
            "authorization_url": authorization_url,
            "login_id": login_id,
            "poll_token": poll_token,
            "expires_in": OAUTH2_LOGIN_MINUTES * 60,
        }

    @router.get("/oauth2/callback", name="oauth2_callback")
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
            return _oauth2_result_page(False, "登录请求已失效，请返回登录页重试。")
        if attempt.user_session_id is not None:
            return _oauth2_result_page(True, "授权已完成，请返回原设备。")
        if attempt.error is not None:
            return _oauth2_result_page(False, attempt.error)
        if error or not code:
            attempt.error = error_description or error or "OAuth2 授权未完成"
            return _oauth2_result_page(False, attempt.error)

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
                    raise TypeError("OAuth2 token response must be an object")
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
                    raise TypeError("OAuth2 userinfo response must be an object")
        except (httpx.HTTPError, TypeError, ValueError):
            attempt.error = "OAuth2 服务请求失败，请返回登录页重试。"
            return _oauth2_result_page(False, attempt.error)

        claim = oauth2_claim(userinfo, oauth2.session_id_claim)
        session_id = str(claim).strip() if claim is not None else ""
        if not session_id or re.fullmatch(SESSION_ID_PATTERN, session_id) is None:
            attempt.error = f"OAuth2 用户信息缺少有效的 {oauth2.session_id_claim}"
            return _oauth2_result_page(False, attempt.error)

        user = await dao.get_user_by_session_id(session_id)
        if user is None and oauth2.auto_create_users:
            username_claim = oauth2_claim(userinfo, oauth2.username_claim)
            username = (
                re.sub(
                    r"[^\w.@+-]+",
                    "_",
                    str(username_claim or session_id).strip(),
                ).strip("_")[:64]
                or session_id
            )
            try:
                user = await dao.create_user(
                    username=username,
                    session_id=session_id,
                    password_hash=password_hash(secrets.token_urlsafe(32)),
                    is_admin=False,
                )
            except DuplicateSessionIdError:
                user = await dao.get_user_by_session_id(session_id)
        if user is None:
            attempt.error = "OAuth2 账号未绑定本地账号"
            return _oauth2_result_page(False, attempt.error)
        if not user.is_active:
            attempt.error = "账号已停用"
            return _oauth2_result_page(False, attempt.error)

        if oauth2.display_type == "link":
            target_url = (
                str(URL("/").update_query(state=attempt.inject_prompt_state)) if attempt.inject_prompt_state else "/"
            )
            callback_response = RedirectResponse(
                url=target_url,
                status_code=status.HTTP_303_SEE_OTHER,
            )
            await create_login_session(user, callback_response)
            discard_oauth2_attempt(login_id or "")
            return callback_response

        attempt.user_session_id = user.session_id
        return _oauth2_result_page(True, "请返回原设备，登录页将自动进入系统。")

    @router.post("/oauth2/poll")
    async def poll_oauth2_login(body: OAuth2PollBody, response: Response):
        if config.oauth2 is None:
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
        return {
            "status": "approved",
            "user": public_user(user),
        }

    @router.post("/logout", status_code=204)
    async def logout(
        response: Response,
        token: Annotated[str | None, Cookie(alias=config.cookie_name)] = None,
    ):
        if token:
            await dao.delete_session(hashlib.sha256(token.encode()).hexdigest())
        response.delete_cookie(config.cookie_name, path="/")

    @router.get("/me")
    async def me(user: CurrentUser):
        return public_user(user)

    @router.patch("/password", status_code=204)
    async def change_password(
        body: PasswordChangeBody,
        user: CurrentUser,
        token: Annotated[str | None, Cookie(alias=config.cookie_name)] = None,
    ):
        if user.id is None or token is None:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "登录已失效")
        changed = await dao.change_password(
            user.id,
            password_hash=password_hash(body.new_password),
            keep_token_hash=hashlib.sha256(token.encode()).hexdigest(),
        )
        if not changed:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "用户不存在")

    return router
