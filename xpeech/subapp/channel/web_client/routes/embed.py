import hashlib
import math
import re
import secrets
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

import jwt
from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, ConfigDict, Field

from .....config.settings import settings
from ..dao import DuplicateUserIdError, WebClientDAO
from ..models import USER_ID_PATTERN, WebConfig, public_user
from ..session import EMBED_TOKEN_PREFIX, embedded_channel_auth, session_token


class EmbedSessionBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    jwt: str = Field(min_length=1, max_length=16384, repr=False)


def verify_embed_jwt(token: str) -> tuple[str, str, datetime]:
    """Only a verified business-server assertion may select the local user/channel."""
    try:
        header = jwt.get_unverified_header(token)
        channel_name = header.get("kid")
        if (
            not isinstance(channel_name, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,63}", channel_name) is None
        ):
            raise jwt.InvalidTokenError("Invalid key ID")
        channel_auth = embedded_channel_auth(channel_name)
        claims = jwt.decode(
            token,
            channel_auth.secret_key,
            algorithms=[settings.jwt.algorithm],
            options={"require": ["sub", "exp"], "verify_aud": False},
        )
        user_id = claims["sub"]
        if (
            not isinstance(user_id, str)
            or not 1 <= len(user_id) <= 128
            or re.fullmatch(USER_ID_PATTERN, user_id) is None
        ):
            raise jwt.InvalidTokenError("Invalid subject")
        expires_at = claims["exp"]
        if (
            isinstance(expires_at, bool)
            or not isinstance(expires_at, (int, float))
            or not math.isfinite(expires_at)
        ):
            raise jwt.InvalidTokenError("Invalid expiration")
        return user_id, channel_name, datetime.fromtimestamp(expires_at, UTC)
    except (jwt.InvalidTokenError, HTTPException, ValueError, TypeError, OverflowError, OSError):
        raise HTTPException(401, "嵌入 JWT 无效或已过期") from None


def create_embed_router(
    config: WebConfig, dao: WebClientDAO, password_hash: Callable[[str], str],
) -> APIRouter:
    router = APIRouter(prefix="/api/auth/embed")

    def require_origin(request: Request) -> str:
        origin = request.headers.get("origin", "")
        if origin not in config.embed.allowed_origins:
            raise HTTPException(403, "不允许此站点嵌入")
        return origin

    async def revoke(request: Request) -> None:
        token = session_token(None, request.headers.get("authorization"))
        if token and token.startswith(EMBED_TOKEN_PREFIX):
            await dao.delete_session(hashlib.sha256(token.encode()).hexdigest())

    @router.post("/session")
    async def create_session(request: Request, body: EmbedSessionBody, response: Response):
        require_origin(request)
        user_id, channel_name, jwt_expires_at = verify_embed_jwt(body.jwt)
        now = datetime.now(UTC)
        expires_at = min(jwt_expires_at, now + timedelta(seconds=settings.jwt.access_token_expire_seconds))
        expires_in = int((expires_at - now).total_seconds())
        if expires_in < 1:
            raise HTTPException(401, "嵌入 JWT 无效或已过期")
        user = await dao.get_user_by_user_id(user_id)
        if user is None:
            try:
                user = await dao.create_user(
                    user_id=user_id,
                    username=user_id[:64],
                    password_hash=password_hash(secrets.token_urlsafe(32)),
                    is_admin=False,
                )
            except DuplicateUserIdError:
                user = await dao.get_user_by_user_id(user_id)
        if user is None or not user.is_active:
            raise HTTPException(403, "账号已停用")
        token = EMBED_TOKEN_PREFIX + secrets.token_urlsafe(32)
        await dao.create_session(
            token_hash=hashlib.sha256(token.encode()).hexdigest(),
            user_id=user.id,
            expires_at=expires_at,
            channel_name=channel_name,
        )
        await revoke(request)
        response.headers["Cache-Control"] = "no-store"
        return {"token": token, "expires_in": expires_in, "channel_name": channel_name, "user": public_user(user)}

    @router.delete("/session", status_code=204)
    async def delete_session(request: Request):
        require_origin(request)
        await revoke(request)

    return router
