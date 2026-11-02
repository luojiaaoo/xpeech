import hashlib
import math
import secrets
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

import jwt
from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, ConfigDict, Field

from .....config.settings import settings
from .....utils.channel_jwt import UserJWTAuth
from ..dao import DuplicateUserIdError, WebClientDAO
from ..models import public_user
from ..session import EMBED_TOKEN_PREFIX, embedded_channel, session_token


class EmbedSessionBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    jwt: str = Field(min_length=1, max_length=16384, repr=False)


def verify_embed_jwt(token: str) -> tuple[str, str, datetime]:
    """Only a verified business-server assertion may select the local user/channel."""
    try:
        auth = UserJWTAuth(token)
        channel_name = embedded_channel(auth.channel)
        user_id = auth.user_id
        if not user_id.strip():
            raise jwt.InvalidTokenError("Empty user identity")
        claims = auth.claims
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
    dao: WebClientDAO, password_hash: Callable[[str], str],
) -> APIRouter:
    router = APIRouter(prefix="/api/auth/embed")

    async def revoke(request: Request) -> None:
        token = session_token(None, request.headers.get("authorization"))
        if token and token.startswith(EMBED_TOKEN_PREFIX):
            await dao.delete_session(hashlib.sha256(token.encode()).hexdigest())

    @router.post("/session")
    async def create_session(request: Request, body: EmbedSessionBody, response: Response):
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
        await revoke(request)

    return router
